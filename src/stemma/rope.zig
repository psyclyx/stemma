//! The rope: a persistent, snapshot-able UTF-8 text buffer.
//!
//! Structure: a B-tree whose leaves hold UTF-8 chunks and whose nodes cache
//! aggregate *summaries* (byte / scalar / UTF-16 / newline counts). Metric
//! queries and coordinate conversions fold into a single O(log n) traversal
//! over those dimensions. Nodes are immutable-when-shared via reference
//! counts: `snapshot()` is O(1) (structural sharing), edits mutate in place
//! when a node is uniquely owned and copy only what is shared otherwise.
//!
//! Leaves come in two flavors:
//! - `owned`   — an inline chunk the rope allocated (all edited text);
//! - `borrowed`— a span of caller-provided immutable backing (e.g. an mmap'd
//!   file via `fromBacking`), never copied until an edit splinters it.
//!
//! Allocation is explicit and unmanaged: a `Rope` never stores an allocator;
//! the *same* allocator must be used for a rope and every snapshot derived
//! from it, and it must be thread-safe if snapshots cross threads.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const geometry = @import("geometry.zig");
const Point = geometry.Point;
const PointUtf16 = geometry.PointUtf16;
const Range = geometry.Range;
const Edit = geometry.Edit;

/// Expensive precondition checks (scalar-boundary and UTF-8 validity) run only
/// in safety-checked builds; release-fast trusts the caller, as contracted.
const runtime_safety = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    .ReleaseFast, .ReleaseSmall => false,
};

/// Compile-time configuration for `RopeWith`. The defaults are the library's
/// blessed instantiation (`Rope`); specialize to shed dimensions or atomics
/// you don't use — disabled dimensions cost zero bytes and zero work.
pub const RopeOptions = struct {
    /// Max bytes per owned (heap) leaf chunk. Default set by benchmark sweep
    /// (dev/bench): vs 256, 512 is ~17% faster on random edits, ~2.8× on
    /// chunk scans, +23% on bulk load, for ~15% slower point conversions
    /// (still sub-µs) — the right trade for editor workloads.
    chunk_capacity: usize = 512,
    /// Max bytes per borrowed (caller-backing) leaf span. Large on purpose:
    /// borrowed leaves are zero-copy views, so fewer, bigger spans keep node
    /// count (and load memory) tiny for huge files; in-leaf scans stay fast.
    borrowed_capacity: usize = 1 << 20,
    /// Max children per internal node. 16 beat 8 on conversions and scans in
    /// the same sweep with no edit-path cost.
    branch_factor: usize = 16,
    /// Atomic reference counts, so snapshots may cross threads.
    thread_safe: bool = true,
    /// Track Unicode scalar counts (offsetToScalar/scalarToOffset, scalarLen).
    track_scalars: bool = true,
    /// Track UTF-16 code-unit counts (LSP coordinates).
    track_utf16: bool = true,
    /// Track newline counts (Point conversions, lineRange, lineCount).
    track_lines: bool = true,
};

/// A `Rope` type specialized for `opts`. See `RopeOptions`; `Rope` (the default
/// instantiation) is what most callers want.
pub fn RopeWith(comptime opts: RopeOptions) type {
    comptime {
        assert(opts.chunk_capacity >= 8);
        assert(opts.branch_factor >= 3);
        assert(opts.borrowed_capacity >= opts.chunk_capacity);
    }
    return struct {
        const RopeT = @This();

        /// Shared, refcounted tree root. `null` is the canonical empty rope.
        root: ?*Node = null,
        /// Sticky: true iff this rope MAY contain unrealized holes (set by
        /// `fromUnrealized`, inherited through snapshot/split/append; never
        /// cleared). Ropes that never touched lazy content skip all hole
        /// checks on hot paths.
        may_have_holes: bool = false,

        /// The empty rope. Costs nothing and needs no `deinit` until it has
        /// been edited into a non-empty state.
        pub const empty: RopeT = .{ .root = null };

        /// Errors any allocating operation may return. Out-of-bounds offsets,
        /// offsets that split a scalar, and invalid-UTF-8 input are asserted
        /// preconditions (programmer errors), not returned errors.
        pub const Error = Allocator.Error;

        // ════════════════════════════════════════════════════════════════
        // Summary: per-node aggregate metrics
        // ════════════════════════════════════════════════════════════════

        /// Integer type for a metric dimension: zero-sized when disabled, so
        /// disabled dimensions cost nothing in nodes or in scans.
        fn Dim(comptime enabled: bool) type {
            return if (enabled) usize else u0;
        }

        pub const Summary = struct {
            bytes: usize = 0,
            scalars: Dim(opts.track_scalars) = 0,
            utf16: Dim(opts.track_utf16) = 0,
            newlines: Dim(opts.track_lines) = 0,

            fn add(a: Summary, b: Summary) Summary {
                return .{
                    .bytes = a.bytes + b.bytes,
                    .scalars = a.scalars + b.scalars,
                    .utf16 = a.utf16 + b.utf16,
                    .newlines = a.newlines + b.newlines,
                };
            }

            fn sub(a: Summary, b: Summary) Summary {
                return .{
                    .bytes = a.bytes - b.bytes,
                    .scalars = a.scalars - b.scalars,
                    .utf16 = a.utf16 - b.utf16,
                    .newlines = a.newlines - b.newlines,
                };
            }

            /// One-pass scan of a UTF-8 chunk. Disabled dimensions compile to
            /// nothing.
            fn of(bytes: []const u8) Summary {
                var s: Summary = .{ .bytes = bytes.len };
                if (opts.track_scalars or opts.track_utf16) {
                    var scalars: usize = 0;
                    var supplementary: usize = 0; // 4-byte scalars → 2 UTF-16 units
                    for (bytes) |b| {
                        scalars += @intFromBool(b & 0xC0 != 0x80);
                        supplementary += @intFromBool(b >= 0xF0);
                    }
                    if (opts.track_scalars) s.scalars = scalars;
                    if (opts.track_utf16) s.utf16 = scalars + supplementary;
                }
                if (opts.track_lines) s.newlines = std.mem.count(u8, bytes, "\n");
                return s;
            }
        };

        // ════════════════════════════════════════════════════════════════
        // Nodes: refcounted, immutable-when-shared
        // ════════════════════════════════════════════════════════════════

        const Refs = if (opts.thread_safe) std.atomic.Value(u32) else u32;

        const max_height = 48; // log_3(2^64) < 41; generous stack bound

        const Node = struct {
            refs: Refs,
            height: u8, // 0 = leaf; discriminates `data`
            summary: Summary,
            data: Data,

            const Data = union {
                leaf: Leaf,
                internal: Internal,
            };

            fn isLeaf(n: *const Node) bool {
                return n.height == 0;
            }
        };

        const Leaf = union(enum) {
            owned: Owned,
            borrowed: []const u8,
            /// Unrealized content: byte length known (node summary), bytes
            /// absent. Structural operations work; content access panics —
            /// fetch and `realize()` first (`isRealized`/`unrealized` tell
            /// you what to fetch).
            hole,

            fn bytes(l: *const Leaf) []const u8 {
                return switch (l.*) {
                    .owned => |*o| o.buf[0..o.len],
                    .borrowed => |b| b,
                    .hole => @panic("stemma.Rope: content access to an unrealized range — fetch and realize() first"),
                };
            }
        };

        const Owned = struct {
            len: u32,
            buf: [opts.chunk_capacity]u8,
        };

        const Internal = struct {
            count: u32,
            children: [opts.branch_factor]*Node,

            fn slice(i: *const Internal) []const *Node {
                return i.children[0..i.count];
            }
        };

        fn refsInit() Refs {
            return if (opts.thread_safe) Refs.init(1) else 1;
        }

        fn retain(n: *Node) *Node {
            if (opts.thread_safe) {
                _ = n.refs.fetchAdd(1, .monotonic);
            } else {
                n.refs += 1;
            }
            return n;
        }

        fn release(gpa: Allocator, n: *Node) void {
            if (opts.thread_safe) {
                if (n.refs.fetchSub(1, .release) == 1) {
                    _ = n.refs.load(.acquire);
                    destroyNode(gpa, n);
                }
            } else {
                n.refs -= 1;
                if (n.refs == 0) destroyNode(gpa, n);
            }
        }

        fn releaseOpt(gpa: Allocator, n: ?*Node) void {
            if (n) |p| release(gpa, p);
        }

        fn destroyNode(gpa: Allocator, n: *Node) void {
            if (!n.isLeaf()) {
                for (n.data.internal.slice()) |c| release(gpa, c);
            }
            gpa.destroy(n);
        }

        fn isUnique(n: *const Node) bool {
            return if (opts.thread_safe) n.refs.load(.acquire) == 1 else n.refs == 1;
        }

        /// New owned leaf from up to two byte slices (concatenated).
        fn newLeafOwned2(gpa: Allocator, a: []const u8, b: []const u8) Error!*Node {
            assert(a.len + b.len <= opts.chunk_capacity);
            assert(a.len + b.len > 0);
            const n = try gpa.create(Node);
            n.* = .{
                .refs = refsInit(),
                .height = 0,
                .summary = Summary.add(Summary.of(a), Summary.of(b)),
                .data = .{ .leaf = .{ .owned = .{ .len = @intCast(a.len + b.len), .buf = undefined } } },
            };
            @memcpy(n.data.leaf.owned.buf[0..a.len], a);
            @memcpy(n.data.leaf.owned.buf[a.len..][0..b.len], b);
            return n;
        }

        fn newLeafOwned(gpa: Allocator, bytes: []const u8) Error!*Node {
            return newLeafOwned2(gpa, bytes, &.{});
        }

        fn newLeafHole(gpa: Allocator, len: usize) Error!*Node {
            assert(len > 0);
            const n = try gpa.create(Node);
            n.* = .{
                .refs = refsInit(),
                .height = 0,
                .summary = .{ .bytes = len },
                .data = .{ .leaf = .hole },
            };
            return n;
        }

        fn newLeafBorrowed(gpa: Allocator, span: []const u8) Error!*Node {
            assert(span.len > 0);
            const n = try gpa.create(Node);
            n.* = .{
                .refs = refsInit(),
                .height = 0,
                .summary = Summary.of(span),
                .data = .{ .leaf = .{ .borrowed = span } },
            };
            return n;
        }

        /// New internal node over `children` (all the same height). Consumes
        /// the child refs; on error the children are released.
        fn newInternal(gpa: Allocator, children: []const *Node) Error!*Node {
            assert(children.len >= 2 and children.len <= opts.branch_factor);
            if (runtime_safety) {
                for (children[1..]) |c| assert(c.height == children[0].height);
            }
            const n = gpa.create(Node) catch |e| {
                for (children) |c| release(gpa, c);
                return e;
            };
            var summary: Summary = .{};
            for (children) |c| summary = Summary.add(summary, c.summary);
            n.* = .{
                .refs = refsInit(),
                .height = children[0].height + 1,
                .summary = summary,
                .data = .{ .internal = .{ .count = @intCast(children.len), .children = undefined } },
            };
            @memcpy(n.data.internal.children[0..children.len], children);
            return n;
        }

        /// Uniquely-owned handle on an internal node: `n` itself if we hold
        /// the only ref, else a shallow copy with retained children. Consumes
        /// the incoming ref; on error it is released.
        fn makeMutableInternal(gpa: Allocator, n: *Node) Error!*Node {
            assert(!n.isLeaf());
            if (isUnique(n)) return n;
            const m = gpa.create(Node) catch |e| {
                release(gpa, n);
                return e;
            };
            m.* = n.*;
            m.refs = refsInit();
            for (m.data.internal.slice()) |c| _ = retain(c);
            release(gpa, n);
            return m;
        }

        // ════════════════════════════════════════════════════════════════
        // Structural engine: concat and split
        // ════════════════════════════════════════════════════════════════

        const Pair = struct { first: *Node, second: ?*Node };

        /// Concatenate two trees of arbitrary heights into one or two nodes of
        /// height `max(l.height, r.height)`. Consumes both refs; on error both
        /// (and any intermediates) are released.
        fn concatNodes(gpa: Allocator, l: *Node, r: *Node) Error!Pair {
            if (l.height == r.height) {
                if (l.isLeaf()) {
                    // Adjacent holes coalesce (keeps sparse trees tiny).
                    if (l.data.leaf == .hole and r.data.leaf == .hole) {
                        if (isUnique(l)) {
                            l.summary.bytes += r.summary.bytes;
                            release(gpa, r);
                            return .{ .first = l, .second = null };
                        }
                        const m = newLeafHole(gpa, l.summary.bytes + r.summary.bytes) catch |e| {
                            release(gpa, l);
                            release(gpa, r);
                            return e;
                        };
                        release(gpa, l);
                        release(gpa, r);
                        return .{ .first = m, .second = null };
                    }
                    if (l.data.leaf == .hole or r.data.leaf == .hole) {
                        return .{ .first = l, .second = r };
                    }
                    const lb = l.data.leaf.bytes();
                    const rb = r.data.leaf.bytes();
                    const both_owned = l.data.leaf == .owned and r.data.leaf == .owned;
                    if (both_owned and lb.len + rb.len <= opts.chunk_capacity) {
                        if (isUnique(l)) {
                            const o = &l.data.leaf.owned;
                            @memcpy(o.buf[o.len..][0..rb.len], rb);
                            o.len += @intCast(rb.len);
                            l.summary = Summary.add(l.summary, r.summary);
                            release(gpa, r);
                            return .{ .first = l, .second = null };
                        }
                        const m = newLeafOwned2(gpa, lb, rb) catch |e| {
                            release(gpa, l);
                            release(gpa, r);
                            return e;
                        };
                        release(gpa, l);
                        release(gpa, r);
                        return .{ .first = m, .second = null };
                    }
                    return .{ .first = l, .second = r };
                }
                const lc = l.data.internal.count;
                const rc = r.data.internal.count;
                if (lc + rc <= opts.branch_factor) {
                    const lm = makeMutableInternal(gpa, l) catch |e| {
                        release(gpa, r);
                        return e;
                    };
                    const li = &lm.data.internal;
                    for (r.data.internal.slice(), 0..) |c, k| {
                        li.children[li.count + k] = retain(c);
                    }
                    li.count += rc;
                    lm.summary = Summary.add(lm.summary, r.summary);
                    release(gpa, r);
                    return .{ .first = lm, .second = null };
                }
                return .{ .first = l, .second = r };
            }

            if (l.height > r.height) {
                const lm = makeMutableInternal(gpa, l) catch |e| {
                    release(gpa, r);
                    return e;
                };
                const li = &lm.data.internal;
                // Detach the last child and fold `r` into it.
                li.count -= 1;
                const last = li.children[li.count];
                const pair = concatNodes(gpa, last, r) catch |e| {
                    destroyMutableShell(gpa, lm);
                    return e;
                };
                return reattach(gpa, lm, pair, .back);
            }

            // r.height > l.height: symmetric, folding `l` into r's first child.
            const rm = makeMutableInternal(gpa, r) catch |e| {
                release(gpa, l);
                return e;
            };
            const ri = &rm.data.internal;
            const first = ri.children[0];
            std.mem.copyForwards(*Node, ri.children[0 .. ri.count - 1], ri.children[1..ri.count]);
            ri.count -= 1;
            const pair = concatNodes(gpa, l, first) catch |e| {
                destroyMutableShell(gpa, rm);
                return e;
            };
            return reattach(gpa, rm, pair, .front);
        }

        /// Release a uniquely-owned internal shell and its remaining children.
        fn destroyMutableShell(gpa: Allocator, n: *Node) void {
            assert(isUnique(n));
            destroyNode(gpa, n);
        }

        /// Insert `pair` (1–2 nodes one level below `n`) at the front or back
        /// of uniquely-owned internal `n`, splitting `n` if it overflows.
        /// Consumes `n` and the pair refs.
        fn reattach(gpa: Allocator, n: *Node, pair: Pair, comptime side: enum { front, back }) Error!Pair {
            const ni = &n.data.internal;
            const extra: u32 = if (pair.second == null) 1 else 2;
            if (ni.count + extra <= opts.branch_factor) {
                switch (side) {
                    .back => {
                        ni.children[ni.count] = pair.first;
                        if (pair.second) |s| ni.children[ni.count + 1] = s;
                    },
                    .front => {
                        std.mem.copyBackwards(*Node, ni.children[extra .. ni.count + extra], ni.children[0..ni.count]);
                        ni.children[0] = pair.first;
                        if (pair.second) |s| ni.children[1] = s;
                    },
                }
                ni.count += extra;
                recomputeSummary(n);
                return .{ .first = n, .second = null };
            }
            // Overflow: distribute all children across two fresh internals.
            var all: [opts.branch_factor + 2]*Node = undefined;
            const total = ni.count + extra;
            switch (side) {
                .back => {
                    @memcpy(all[0..ni.count], ni.children[0..ni.count]);
                    all[ni.count] = pair.first;
                    if (pair.second) |s| all[ni.count + 1] = s;
                },
                .front => {
                    all[0] = pair.first;
                    if (pair.second) |s| all[1] = s;
                    @memcpy(all[extra..][0..ni.count], ni.children[0..ni.count]);
                },
            }
            // Children refs have been moved out of `n`; free the bare shell.
            ni.count = 0;
            assert(isUnique(n));
            gpa.destroy(n);
            const half = (total + 1) / 2;
            const a = newInternal(gpa, all[0..half]) catch |e| {
                for (all[half..total]) |c| release(gpa, c);
                return e;
            };
            const b = newInternal(gpa, all[half..total]) catch |e| {
                release(gpa, a);
                return e;
            };
            return .{ .first = a, .second = b };
        }

        fn recomputeSummary(n: *Node) void {
            assert(!n.isLeaf());
            var s: Summary = .{};
            for (n.data.internal.slice()) |c| s = Summary.add(s, c.summary);
            n.summary = s;
        }

        /// Concatenate two optional roots. Consumes both refs; on error both
        /// are released.
        fn concatRoots(gpa: Allocator, l: ?*Node, r: ?*Node) Error!?*Node {
            const ln = l orelse return r;
            const rn = r orelse return ln;
            const pair = try concatNodes(gpa, ln, rn);
            const second = pair.second orelse return pair.first;
            return try newInternal(gpa, &.{ pair.first, second });
        }

        /// 0/1/n children → null / the child itself / a fresh internal.
        /// Consumes the child refs; on error they are released.
        fn nodeFromChildren(gpa: Allocator, children: []const *Node) Error!?*Node {
            return switch (children.len) {
                0 => null,
                1 => children[0],
                else => try newInternal(gpa, children),
            };
        }

        const SplitResult = struct { left: ?*Node, right: ?*Node };

        /// Split a tree at `offset` (0 < offset < total handled generally;
        /// boundary offsets short-circuit). Consumes `n`; on error everything
        /// held is released.
        fn splitNode(gpa: Allocator, n: *Node, offset: usize) Error!SplitResult {
            assert(offset <= n.summary.bytes);
            if (offset == 0) return .{ .left = null, .right = n };
            if (offset == n.summary.bytes) return .{ .left = n, .right = null };

            if (n.isLeaf()) {
                if (n.data.leaf == .hole) {
                    // Splitting a hole is pure arithmetic.
                    const total = n.summary.bytes;
                    const left = newLeafHole(gpa, offset) catch |e| {
                        release(gpa, n);
                        return e;
                    };
                    const right = newLeafHole(gpa, total - offset) catch |e| {
                        release(gpa, left);
                        release(gpa, n);
                        return e;
                    };
                    release(gpa, n);
                    return .{ .left = left, .right = right };
                }
                const b = n.data.leaf.bytes();
                const mk = struct {
                    fn mk(g: Allocator, leaf: *const Leaf, part: []const u8) Error!*Node {
                        return switch (leaf.*) {
                            .owned => newLeafOwned(g, part),
                            .borrowed => newLeafBorrowed(g, part),
                            .hole => unreachable, // handled above
                        };
                    }
                }.mk;
                const left = mk(gpa, &n.data.leaf, b[0..offset]) catch |e| {
                    release(gpa, n);
                    return e;
                };
                const right = mk(gpa, &n.data.leaf, b[offset..]) catch |e| {
                    release(gpa, left);
                    release(gpa, n);
                    return e;
                };
                release(gpa, n);
                return .{ .left = left, .right = right };
            }

            // Locate the boundary child.
            const cs = n.data.internal.slice();
            var acc: usize = 0;
            var i: usize = 0;
            while (i < cs.len) : (i += 1) {
                if (offset < acc + cs[i].summary.bytes) break;
                acc += cs[i].summary.bytes;
            }
            assert(i < cs.len); // offset < total ensured above

            // Take our own refs on the parts we keep, then drop `n`.
            var left_cs: [opts.branch_factor]*Node = undefined;
            const left_n = i;
            for (cs[0..i], 0..) |c, k| left_cs[k] = retain(c);
            var right_cs: [opts.branch_factor]*Node = undefined;
            const right_n = cs.len - i - 1;
            for (cs[i + 1 ..], 0..) |c, k| right_cs[k] = retain(c);
            const boundary = retain(cs[i]);
            release(gpa, n);

            const sub = splitNode(gpa, boundary, offset - acc) catch |e| {
                for (left_cs[0..left_n]) |c| release(gpa, c);
                for (right_cs[0..right_n]) |c| release(gpa, c);
                return e;
            };
            const left_tree = nodeFromChildren(gpa, left_cs[0..left_n]) catch |e| {
                releaseOpt(gpa, sub.left);
                releaseOpt(gpa, sub.right);
                for (right_cs[0..right_n]) |c| release(gpa, c);
                return e;
            };
            const left_full = concatRoots(gpa, left_tree, sub.left) catch |e| {
                releaseOpt(gpa, sub.right);
                for (right_cs[0..right_n]) |c| release(gpa, c);
                return e;
            };
            const right_tree = nodeFromChildren(gpa, right_cs[0..right_n]) catch |e| {
                releaseOpt(gpa, left_full);
                releaseOpt(gpa, sub.right);
                return e;
            };
            const right_full = concatRoots(gpa, sub.right, right_tree) catch |e| {
                releaseOpt(gpa, left_full);
                return e;
            };
            return .{ .left = left_full, .right = right_full };
        }

        fn splitRoot(gpa: Allocator, root: ?*Node, offset: usize) Error!SplitResult {
            const n = root orelse {
                assert(offset == 0);
                return .{ .left = null, .right = null };
            };
            return splitNode(gpa, n, offset);
        }

        // ════════════════════════════════════════════════════════════════
        // Bulk construction
        // ════════════════════════════════════════════════════════════════

        /// Back `end` off to a scalar boundary (never past `start`).
        fn scalarFloor(bytes: []const u8, start: usize, end_in: usize) usize {
            var end = end_in;
            if (end < bytes.len) {
                while (end > start and bytes[end] & 0xC0 == 0x80) end -= 1;
            }
            assert(end > start);
            return end;
        }

        /// Build a balanced tree bottom-up from a level of equal-height nodes.
        /// Consumes all node refs (also on error).
        fn buildFromLevel(gpa: Allocator, first_level: []*Node) Error!?*Node {
            if (first_level.len == 0) return null;
            var level: std.ArrayList(*Node) = .empty;
            defer level.deinit(gpa);
            level.appendSlice(gpa, first_level) catch |e| {
                for (first_level) |c| release(gpa, c);
                return e;
            };
            var next: std.ArrayList(*Node) = .empty;
            defer next.deinit(gpa);

            while (level.items.len > 1) {
                next.clearRetainingCapacity();
                var i: usize = 0;
                while (i < level.items.len) {
                    const rem = level.items.len - i;
                    const take = if (rem <= opts.branch_factor)
                        rem
                    else if (rem == opts.branch_factor + 1)
                        (opts.branch_factor + 1) / 2
                    else
                        opts.branch_factor;
                    const parent = if (take == 1)
                        level.items[i]
                    else
                        newInternal(gpa, level.items[i..][0..take]) catch |e| {
                            // newInternal released its inputs; drop the rest.
                            for (level.items[i + take ..]) |c| release(gpa, c);
                            for (next.items) |c| release(gpa, c);
                            return e;
                        };
                    next.append(gpa, parent) catch |e| {
                        release(gpa, parent);
                        for (level.items[i + take ..]) |c| release(gpa, c);
                        for (next.items) |c| release(gpa, c);
                        return e;
                    };
                    i += take;
                }
                std.mem.swap(std.ArrayList(*Node), &level, &next);
            }
            return level.items[0];
        }

        const LeafKind = enum { owned, borrowed };

        fn leavesFromBytes(
            gpa: Allocator,
            list: *std.ArrayList(*Node),
            bytes: []const u8,
            comptime kind: LeafKind,
        ) Error!void {
            const cap = switch (kind) {
                .owned => opts.chunk_capacity,
                .borrowed => opts.borrowed_capacity,
            };
            var i: usize = 0;
            while (i < bytes.len) {
                const end = scalarFloor(bytes, i, @min(i + cap, bytes.len));
                const leaf = switch (kind) {
                    .owned => try newLeafOwned(gpa, bytes[i..end]),
                    .borrowed => try newLeafBorrowed(gpa, bytes[i..end]),
                };
                list.append(gpa, leaf) catch |e| {
                    release(gpa, leaf);
                    return e;
                };
                i = end;
            }
        }

        fn treeFromBytes(
            gpa: Allocator,
            bytes: []const u8,
            comptime kind: LeafKind,
        ) Error!?*Node {
            var leaves: std.ArrayList(*Node) = .empty;
            defer leaves.deinit(gpa);
            leavesFromBytes(gpa, &leaves, bytes, kind) catch |e| {
                for (leaves.items) |c| release(gpa, c);
                return e;
            };
            return buildFromLevel(gpa, leaves.items);
        }

        // ════════════════════════════════════════════════════════════════
        // Construction / teardown
        // ════════════════════════════════════════════════════════════════

        /// Build a rope from a UTF-8 byte slice (bulk load, copies). The
        /// caller may free `bytes` afterwards. Precondition: valid UTF-8.
        pub fn fromSlice(gpa: Allocator, bytes: []const u8) Error!RopeT {
            if (runtime_safety) assert(std.unicode.utf8ValidateSlice(bytes));
            return .{ .root = try treeFromBytes(gpa, bytes, .owned) };
        }

        /// Build a rope over caller-owned immutable backing (e.g. an mmap'd
        /// file) without copying: leaves borrow spans of `backing`. The caller
        /// must keep `backing` alive and unchanged for the lifetime of this
        /// rope and every snapshot/split derived from it; edited regions are
        /// copied out as they are touched. Precondition: valid UTF-8.
        pub fn fromBacking(gpa: Allocator, backing: []const u8) Error!RopeT {
            if (runtime_safety) assert(std.unicode.utf8ValidateSlice(backing));
            return .{ .root = try treeFromBytes(gpa, backing, .borrowed) };
        }

        /// Build a rope by streaming from `reader` until end of stream,
        /// without materializing the whole input. Precondition: the byte
        /// stream is valid UTF-8.
        pub fn fromReader(gpa: Allocator, reader: *std.Io.Reader) (Error || error{ReadFailed})!RopeT {
            var leaves: std.ArrayList(*Node) = .empty;
            defer leaves.deinit(gpa);
            errdefer for (leaves.items) |c| release(gpa, c);

            var buf: [opts.chunk_capacity]u8 = undefined;
            var pending: usize = 0; // carried bytes of a scalar split across reads
            while (true) {
                const n = try reader.readSliceShort(buf[pending..]);
                const total = pending + n;
                if (total == 0) break;
                const at_eof = n < buf.len - pending;
                // A scalar may straddle the read seam: carry its lead bytes.
                const cut = if (at_eof) total else blk: {
                    var p = total;
                    while (p > 0 and buf[p - 1] & 0xC0 == 0x80) p -= 1;
                    assert(p > 0); // ≥4 consecutive continuations = invalid UTF-8
                    const lead = p - 1;
                    const l = std.unicode.utf8ByteSequenceLength(buf[lead]) catch unreachable;
                    break :blk if (lead + l <= total) lead + l else lead;
                };
                if (runtime_safety) assert(std.unicode.utf8ValidateSlice(buf[0..cut]));
                const leaf = try newLeafOwned(gpa, buf[0..cut]);
                leaves.append(gpa, leaf) catch |e| {
                    release(gpa, leaf);
                    return e;
                };
                std.mem.copyForwards(u8, buf[0 .. total - cut], buf[cut..total]);
                pending = total - cut;
                if (at_eof) {
                    assert(pending == 0); // truncated scalar at EOF = invalid UTF-8
                    break;
                }
            }
            const items = leaves.items;
            leaves.items.len = 0; // ownership moves to buildFromLevel
            return .{ .root = try buildFromLevel(gpa, items) };
        }

        /// Release this handle's share of the tree. Nodes still referenced by
        /// a live snapshot survive; only the last handle frees. Must use the
        /// same allocator that built/edited the rope.
        // ── Lazy / unrealized content ────────────────────────────────────
        // The rope does no I/O. A rope over remote or not-yet-fetched data
        // starts as an unrealized "hole" of known byte length; the CALLER
        // fetches windows (its transport, its cache policy) and `realize`s
        // them. Byte-domain operations (seek arithmetic, split, delete,
        // insert around/into holes) work on unrealized content; content
        // access (chunks, cursors, search, conversions *inside* a hole)
        // panics deterministically — use `isRealized`/`unrealized` to
        // sequence fetches. Content metrics (lines/scalars/UTF-16) count
        // realized content only and converge as realization proceeds.
        // `realize` does not change offsets: anchors and versions are
        // unaffected (it is not an edit).

        /// A rope of `len` unrealized bytes. O(1); no content is read.
        pub fn fromUnrealized(gpa: Allocator, len: usize) Error!RopeT {
            if (len == 0) return .empty;
            return .{ .root = try newLeafHole(gpa, len), .may_have_holes = true };
        }

        /// Fill `[byte_offset, byte_offset + content.len)` — which must be
        /// entirely unrealized — with fetched bytes (copied; the caller may
        /// free `content` afterwards). Preconditions: the range is in
        /// bounds and fully unrealized; `content` is valid UTF-8 whose
        /// edges coincide with scalar boundaries of the true underlying
        /// data (the caller vouches — neighbors may be unfetched). On
        /// allocation failure the rope is unchanged.
        pub fn realize(self: *RopeT, gpa: Allocator, byte_offset: usize, content: []const u8) Error!void {
            return self.realizeImpl(gpa, byte_offset, content, .owned);
        }

        /// As `realize`, but zero-copy: leaves borrow `backing`, which the
        /// caller must keep alive and unchanged for the lifetime of this
        /// rope and everything derived from it.
        pub fn realizeBacking(self: *RopeT, gpa: Allocator, byte_offset: usize, backing: []const u8) Error!void {
            return self.realizeImpl(gpa, byte_offset, backing, .borrowed);
        }

        fn realizeImpl(self: *RopeT, gpa: Allocator, byte_offset: usize, content: []const u8, comptime kind: LeafKind) Error!void {
            if (content.len == 0) return;
            assert(byte_offset + content.len <= self.byteLen());
            // Checked in all build modes: silently overwriting realized
            // content would corrupt the document.
            const target: Range = .{ .start = byte_offset, .end = byte_offset + content.len };
            if (self.holeSpanBytes(target) != content.len) {
                @panic("stemma.Rope.realize: target range is not entirely unrealized");
            }
            if (runtime_safety) assert(std.unicode.utf8ValidateSlice(content));

            const mid = try treeFromBytes(gpa, content, kind);
            const saved = if (self.root) |r| retain(r) else null;
            errdefer self.root = saved;
            const halves = splitRoot(gpa, self.root, byte_offset) catch |e| {
                releaseOpt(gpa, mid);
                return e;
            };
            const tail = splitRoot(gpa, halves.right, content.len) catch |e| {
                releaseOpt(gpa, mid);
                releaseOpt(gpa, halves.left);
                return e;
            };
            releaseOpt(gpa, tail.left); // the hole span being replaced
            const left = concatRoots(gpa, halves.left, mid) catch |e| {
                releaseOpt(gpa, tail.right);
                return e;
            };
            self.root = try concatRoots(gpa, left, tail.right);
            releaseOpt(gpa, saved);
        }

        /// Whether every byte of `range` is realized (readable).
        pub fn isRealized(self: RopeT, range: Range) bool {
            var it = self.unrealized(range);
            return it.next() == null;
        }

        /// Total unrealized bytes within `range`.
        fn holeSpanBytes(self: RopeT, range: Range) usize {
            var n: usize = 0;
            var it = self.unrealized(range);
            while (it.next()) |r| n += r.len();
            return n;
        }

        /// Iterate the unrealized runs intersecting `range` — the fetch
        /// list. Same invalidation rule as `Cursor`.
        pub fn unrealized(self: RopeT, range: Range) UnrealizedIterator {
            assert(range.end <= self.byteLen());
            return .{ .rope = self, .pos = range.start, .end = range.end };
        }

        pub const UnrealizedIterator = struct {
            rope: RopeT,
            pos: usize,
            end: usize,

            /// Next unrealized run (clipped to the query range), or null.
            pub fn next(self: *UnrealizedIterator) ?Range {
                while (self.pos < self.end) {
                    const span = self.rope.leafSpanAt(self.pos);
                    const span_end = @min(span.end, self.end);
                    if (span.hole) {
                        const r: Range = .{ .start = self.pos, .end = span_end };
                        self.pos = span.end;
                        return r;
                    }
                    self.pos = span.end;
                }
                return null;
            }
        };

        /// The leaf span containing byte `offset` (its absolute extent and
        /// whether it is a hole). O(log n) descent; never touches content.
        fn leafSpanAt(self: RopeT, offset: usize) struct { start: usize, end: usize, hole: bool } {
            var n = self.root.?;
            var local = offset;
            var abs: usize = 0;
            while (!n.isLeaf()) {
                for (n.data.internal.slice()) |c| {
                    if (local < c.summary.bytes) {
                        n = c;
                        break;
                    }
                    local -= c.summary.bytes;
                    abs += c.summary.bytes;
                } else unreachable;
            }
            return .{ .start = abs, .end = abs + n.summary.bytes, .hole = n.data.leaf == .hole };
        }

        pub fn deinit(self: *RopeT, gpa: Allocator) void {
            releaseOpt(gpa, self.root);
            self.root = null;
        }

        // ════════════════════════════════════════════════════════════════
        // Snapshots
        // ════════════════════════════════════════════════════════════════

        /// An independent handle onto the current contents. O(1): bumps the
        /// root refcount; no text is copied. Hand it to a background thread
        /// and keep editing `self` — the snapshot observes the contents as of
        /// this call. The caller owns the result and must `deinit` it (same
        /// allocator).
        pub fn snapshot(self: RopeT) RopeT {
            return .{
                .root = if (self.root) |r| retain(r) else null,
                .may_have_holes = self.may_have_holes,
            };
        }

        // ════════════════════════════════════════════════════════════════
        // Metrics (O(1) from the root summary)
        // ════════════════════════════════════════════════════════════════

        pub fn byteLen(self: RopeT) usize {
            return if (self.root) |r| r.summary.bytes else 0;
        }

        /// Total Unicode scalar values (UTF-8 codepoints).
        pub fn scalarLen(self: RopeT) usize {
            comptime if (!opts.track_scalars) @compileError("scalars dimension disabled");
            return if (self.root) |r| r.summary.scalars else 0;
        }

        /// Total UTF-16 code units (what LSP positions count).
        pub fn utf16Len(self: RopeT) usize {
            comptime if (!opts.track_utf16) @compileError("utf16 dimension disabled");
            return if (self.root) |r| r.summary.utf16 else 0;
        }

        /// Number of lines: newline count + 1 (an empty rope has 1 line).
        pub fn lineCount(self: RopeT) usize {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            return 1 + if (self.root) |r| @as(usize, r.summary.newlines) else 0;
        }

        pub fn isEmpty(self: RopeT) bool {
            return self.root == null;
        }

        // ════════════════════════════════════════════════════════════════
        // Editing
        // ════════════════════════════════════════════════════════════════

        fn assertScalarBoundary(self: RopeT, offset: usize) void {
            if (!runtime_safety) return;
            const len = self.byteLen();
            assert(offset <= len);
            if (offset == 0 or offset == len) return;
            // Inside unrealized content the boundary is unverifiable — the
            // caller vouches for offsets into holes.
            if (self.may_have_holes and self.leafSpanAt(offset).hole) return;
            assert(self.byteAt(offset) & 0xC0 != 0x80);
        }

        /// Keystroke fast path: when the spine down to the target leaf is
        /// uniquely owned and the leaf is an owned chunk with room, mutate in
        /// place — zero allocation, zero copying of untouched nodes.
        fn tryFastInsert(n: *Node, offset: usize, text: []const u8, delta: Summary) bool {
            if (!isUnique(n)) return false;
            if (n.isLeaf()) {
                switch (n.data.leaf) {
                    .borrowed, .hole => return false,
                    .owned => |*o| {
                        if (o.len + text.len > opts.chunk_capacity) return false;
                        std.mem.copyBackwards(u8, o.buf[offset + text.len ..][0 .. o.len - offset], o.buf[offset..o.len]);
                        @memcpy(o.buf[offset..][0..text.len], text);
                        o.len += @intCast(text.len);
                        n.summary = Summary.add(n.summary, delta);
                        return true;
                    },
                }
            }
            var acc: usize = 0;
            for (n.data.internal.slice()) |c| {
                if (offset <= acc + c.summary.bytes) {
                    if (tryFastInsert(c, offset - acc, text, delta)) {
                        n.summary = Summary.add(n.summary, delta);
                        return true;
                    }
                    return false;
                }
                acc += c.summary.bytes;
            }
            unreachable;
        }

        fn tryFastDelete(n: *Node, start: usize, count: usize, delta: Summary) bool {
            if (!isUnique(n)) return false;
            if (n.isLeaf()) {
                switch (n.data.leaf) {
                    .borrowed, .hole => return false,
                    .owned => |*o| {
                        if (count >= o.len) return false; // don't empty a leaf in place
                        std.mem.copyForwards(u8, o.buf[start .. o.len - count], o.buf[start + count .. o.len]);
                        o.len -= @intCast(count);
                        n.summary = Summary.sub(n.summary, delta);
                        return true;
                    },
                }
            }
            var acc: usize = 0;
            for (n.data.internal.slice()) |c| {
                if (start >= acc and start + count <= acc + c.summary.bytes) {
                    if (tryFastDelete(c, start - acc, count, delta)) {
                        n.summary = Summary.sub(n.summary, delta);
                        return true;
                    }
                    return false;
                }
                acc += c.summary.bytes;
            }
            return false; // range spans children
        }

        /// Insert `text` at `byte_offset`. Preconditions: `byte_offset <=
        /// byteLen()`, on a scalar boundary; `text` valid UTF-8. Returns the
        /// `Edit` delta for anchor shifting. On allocation failure the rope is
        /// unchanged.
        pub fn insert(self: *RopeT, gpa: Allocator, byte_offset: usize, text: []const u8) Error!Edit {
            const edit: Edit = .{ .offset = byte_offset, .removed = 0, .inserted = text.len };
            if (text.len == 0) return edit;
            assert(byte_offset <= self.byteLen());
            self.assertScalarBoundary(byte_offset);
            if (runtime_safety) assert(std.unicode.utf8ValidateSlice(text));

            const delta = Summary.of(text);
            if (self.root) |r| {
                if (tryFastInsert(r, byte_offset, text, delta)) return edit;
            }

            // General path: build the middle first (rope state untouched if
            // that fails), then split + concat with the old root retained so
            // any later failure restores it. Once splitRoot runs, our root
            // ref is consumed — error paths must only reinstate `saved`.
            const mid = try treeFromBytes(gpa, text, .owned);
            const saved = if (self.root) |r| retain(r) else null;
            errdefer self.root = saved;
            const halves = splitRoot(gpa, self.root, byte_offset) catch |e| {
                releaseOpt(gpa, mid);
                return e;
            };
            const left = concatRoots(gpa, halves.left, mid) catch |e| {
                releaseOpt(gpa, halves.right);
                return e;
            };
            self.root = try concatRoots(gpa, left, halves.right);
            releaseOpt(gpa, saved);
            return edit;
        }

        /// Delete `range`. Preconditions: within bounds, both ends on scalar
        /// boundaries. Returns the `Edit` delta. On allocation failure the
        /// rope is unchanged.
        pub fn delete(self: *RopeT, gpa: Allocator, range: Range) Error!Edit {
            const edit: Edit = .{ .offset = range.start, .removed = range.len(), .inserted = 0 };
            if (range.isEmpty()) return edit;
            assert(range.end <= self.byteLen());
            self.assertScalarBoundary(range.start);
            self.assertScalarBoundary(range.end);

            if (self.root) |r| fast: {
                if (range.len() >= r.summary.bytes) break :fast;
                var scratch: [scratch_len]u8 = undefined;
                if (range.len() <= scratch.len and (!self.may_have_holes or self.isRealized(range))) {
                    self.copyRange(scratch[0..range.len()], range);
                    const delta = Summary.of(scratch[0..range.len()]);
                    if (tryFastDelete(r, range.start, range.len(), delta)) return edit;
                }
            }

            const saved = if (self.root) |r| retain(r) else null;
            errdefer self.root = saved;
            const a = try splitRoot(gpa, self.root, range.start);
            const b = splitRoot(gpa, a.right, range.len()) catch |e| {
                releaseOpt(gpa, a.left);
                return e;
            };
            releaseOpt(gpa, b.left); // the removed span
            self.root = try concatRoots(gpa, a.left, b.right);
            releaseOpt(gpa, saved);
            return edit;
        }

        const scratch_len = @min(opts.chunk_capacity, 512);

        fn tryFastReplace(n: *Node, start: usize, count: usize, text: []const u8, delta_sub: Summary, delta_add: Summary) bool {
            if (!isUnique(n)) return false;
            if (n.isLeaf()) {
                switch (n.data.leaf) {
                    .borrowed, .hole => return false,
                    .owned => |*o| {
                        const new_len = o.len - count + text.len;
                        if (new_len == 0 or new_len > opts.chunk_capacity) return false;
                        const tail = o.buf[start + count .. o.len];
                        if (text.len < count) {
                            std.mem.copyForwards(u8, o.buf[start + text.len ..][0..tail.len], tail);
                        } else if (text.len > count) {
                            std.mem.copyBackwards(u8, o.buf[start + text.len ..][0..tail.len], tail);
                        }
                        @memcpy(o.buf[start..][0..text.len], text);
                        o.len = @intCast(new_len);
                        n.summary = Summary.add(Summary.sub(n.summary, delta_sub), delta_add);
                        return true;
                    },
                }
            }
            var acc: usize = 0;
            for (n.data.internal.slice()) |c| {
                if (start >= acc and start + count <= acc + c.summary.bytes) {
                    if (tryFastReplace(c, start - acc, count, text, delta_sub, delta_add)) {
                        n.summary = Summary.add(Summary.sub(n.summary, delta_sub), delta_add);
                        return true;
                    }
                    return false;
                }
                acc += c.summary.bytes;
            }
            return false; // range spans children
        }

        /// Replace `range` with `text`: one call, one combined `Edit`. On
        /// allocation failure the rope is unchanged.
        pub fn replace(self: *RopeT, gpa: Allocator, range: Range, text: []const u8) Error!Edit {
            const edit: Edit = .{ .offset = range.start, .removed = range.len(), .inserted = text.len };
            if (range.isEmpty()) {
                _ = try self.insert(gpa, range.start, text);
                return edit;
            }
            if (text.len == 0) {
                _ = try self.delete(gpa, range);
                return edit;
            }
            assert(range.end <= self.byteLen());
            self.assertScalarBoundary(range.start);
            self.assertScalarBoundary(range.end);
            if (runtime_safety) assert(std.unicode.utf8ValidateSlice(text));

            if (self.root) |r| fast: {
                if (range.len() >= r.summary.bytes) break :fast;
                var scratch: [scratch_len]u8 = undefined;
                if (range.len() <= scratch.len and (!self.may_have_holes or self.isRealized(range))) {
                    self.copyRange(scratch[0..range.len()], range);
                    const delta_sub = Summary.of(scratch[0..range.len()]);
                    const delta_add = Summary.of(text);
                    if (tryFastReplace(r, range.start, range.len(), text, delta_sub, delta_add)) return edit;
                }
            }

            // General path: delete + insert, made atomic by holding a ref on
            // the pre-replace root. delete/insert each restore themselves on
            // failure, so `self.root` is always a valid tree here; on error we
            // release whichever state we're in and reinstate the original.
            const saved = if (self.root) |r| retain(r) else null;
            errdefer {
                releaseOpt(gpa, self.root);
                self.root = saved;
            }
            _ = try self.delete(gpa, range);
            _ = try self.insert(gpa, range.start, text);
            releaseOpt(gpa, saved);
            return edit;
        }

        // ════════════════════════════════════════════════════════════════
        // Structural operations
        // ════════════════════════════════════════════════════════════════

        /// Split off and return the suffix `[byte_offset, byteLen)`; `self`
        /// keeps the prefix. O(log n) + spine rebuilding; shared subtrees are
        /// not copied. On allocation failure `self` is unchanged.
        pub fn split(self: *RopeT, gpa: Allocator, byte_offset: usize) Error!RopeT {
            self.assertScalarBoundary(byte_offset);
            const saved = if (self.root) |r| retain(r) else null;
            errdefer self.root = saved;
            const halves = try splitRoot(gpa, self.root, byte_offset);
            self.root = halves.left;
            releaseOpt(gpa, saved);
            return .{ .root = halves.right, .may_have_holes = self.may_have_holes };
        }

        /// Append `other`'s contents to `self`. O(log n). On success `other`
        /// is drained (becomes empty); on allocation failure both ropes are
        /// unchanged.
        pub fn append(self: *RopeT, gpa: Allocator, other: *RopeT) Error!void {
            const l = if (self.root) |r| retain(r) else null;
            const r = if (other.root) |n| retain(n) else null;
            const merged = try concatRoots(gpa, l, r);
            releaseOpt(gpa, self.root);
            releaseOpt(gpa, other.root);
            other.root = null;
            self.root = merged;
            self.may_have_holes = self.may_have_holes or other.may_have_holes;
        }

        // ════════════════════════════════════════════════════════════════
        // Reading
        // ════════════════════════════════════════════════════════════════

        /// The byte at `offset`. Precondition: `offset < byteLen()`.
        pub fn byteAt(self: RopeT, offset: usize) u8 {
            var n = self.root.?;
            var local = offset;
            while (!n.isLeaf()) {
                for (n.data.internal.slice()) |c| {
                    if (local < c.summary.bytes) {
                        n = c;
                        break;
                    }
                    local -= c.summary.bytes;
                } else unreachable;
            }
            return n.data.leaf.bytes()[local];
        }

        /// Iterate the contiguous UTF-8 chunks backing `range`, borrowing the
        /// rope's storage (no copy). Chunk boundaries never split a scalar.
        /// Invalidated by any edit to this rope (snapshots are unaffected).
        pub fn slice(self: RopeT, range: Range) Chunks {
            assert(range.end <= self.byteLen());
            return .{ .cursor = self.cursorAt(range.start), .end = range.end };
        }

        /// Copy `range` into `dest`. Precondition: `dest.len == range.len()`.
        pub fn copyRange(self: RopeT, dest: []u8, range: Range) void {
            assert(dest.len == range.len());
            var chunks = self.slice(range);
            var i: usize = 0;
            while (chunks.next()) |c| {
                @memcpy(dest[i..][0..c.len], c);
                i += c.len;
            }
            assert(i == dest.len);
        }

        /// Materialize the whole buffer into a freshly allocated slice the
        /// caller owns.
        pub fn toOwnedSlice(self: RopeT, gpa: Allocator) Error![]u8 {
            const out = try gpa.alloc(u8, self.byteLen());
            self.copyRange(out, .{ .start = 0, .end = out.len });
            return out;
        }

        /// Stream the whole buffer to `writer` without an intermediate copy.
        pub fn writeTo(self: RopeT, writer: *std.Io.Writer) error{WriteFailed}!void {
            var chunks = self.slice(.{ .start = 0, .end = self.byteLen() });
            while (chunks.next()) |c| try writer.writeAll(c);
        }

        /// Content equality. O(1) when the ropes share a root (snapshots of
        /// an unedited buffer); otherwise a lockstep chunk compare with an
        /// O(1) length short-circuit.
        pub fn eql(self: RopeT, other: RopeT) bool {
            if (self.root == other.root) return true;
            const len = self.byteLen();
            if (len != other.byteLen()) return false;
            var ca = self.cursorAt(0);
            var cb = other.cursorAt(0);
            var a: []const u8 = &.{};
            var b: []const u8 = &.{};
            while (true) {
                if (a.len == 0) a = ca.nextChunk() orelse return true;
                if (b.len == 0) b = cb.nextChunk() orelse unreachable; // same length
                const m = @min(a.len, b.len);
                if (!std.mem.eql(u8, a[0..m], b[0..m])) return false;
                a = a[m..];
                b = b[m..];
            }
        }

        /// A `std.Io.Reader` over `range`, streaming borrowed chunks with no
        /// intermediate copy — feed the rope to any Reader-consuming API
        /// (parsers, hashers, save pipelines). Same invalidation rule as
        /// `Cursor`. `buffer` may be empty for pure streaming use; size it if
        /// the consumer peeks.
        pub fn streamReader(self: RopeT, range: Range, buffer: []u8) StreamReader {
            assert(range.end <= self.byteLen());
            return .{
                .cursor = self.cursorAt(range.start),
                .end = range.end,
                .interface = .{
                    .vtable = &.{ .stream = StreamReader.stream },
                    .buffer = buffer,
                    .seek = 0,
                    .end = 0,
                },
            };
        }

        pub const StreamReader = struct {
            cursor: Cursor,
            end: usize,
            interface: std.Io.Reader,

            fn stream(io_r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
                const self: *StreamReader = @alignCast(@fieldParentPtr("interface", io_r));
                const cur = &self.cursor;
                if (cur.offset >= self.end) return error.EndOfStream;
                if (cur.offset - cur.leaf_start == cur.leaf.len) cur.advanceLeaf();
                const pos = cur.offset - cur.leaf_start;
                const avail = @min(cur.leaf.len - pos, self.end - cur.offset);
                const chunk = limit.sliceConst(cur.leaf[pos..][0..avail]);
                const n = try w.write(chunk);
                cur.offset += n; // stays within the current leaf
                return n;
            }
        };

        // ════════════════════════════════════════════════════════════════
        // Search (byte-literal; regex and case folding layer above)
        // ════════════════════════════════════════════════════════════════

        /// True iff the rope bytes starting at `abs` match `rest` (which must
        /// fit before `limit`).
        fn matchesFrom(self: RopeT, abs: usize, rest: []const u8, limit: usize) bool {
            if (abs + rest.len > limit) return false;
            var cur = self.cursorAt(abs);
            var r = rest;
            while (r.len > 0) {
                const c = cur.nextChunk() orelse return false;
                const m = @min(c.len, r.len);
                if (!std.mem.eql(u8, c[0..m], r[0..m])) return false;
                r = r[m..];
            }
            return true;
        }

        /// Byte offset of the first occurrence of `needle` in `range`, or
        /// `null`. Matches may straddle chunk boundaries. `needle` is a byte
        /// pattern; if both haystack and needle are valid UTF-8 (the rope
        /// always is), matches land on scalar boundaries for free.
        pub fn find(self: RopeT, range: Range, needle: []const u8) ?usize {
            assert(needle.len > 0);
            assert(range.end <= self.byteLen());
            if (range.len() < needle.len) return null;
            var cur = self.cursorAt(range.start);
            while (cur.offset < range.end) {
                const chunk_start = cur.offset;
                const raw = cur.nextChunk() orelse break;
                const c = if (chunk_start + raw.len > range.end)
                    raw[0 .. range.end - chunk_start]
                else
                    raw;
                // Wholly-in-chunk match is the earliest possible in this chunk.
                if (std.mem.indexOf(u8, c, needle)) |p| return chunk_start + p;
                // Straddle candidates: tail positions whose chunk suffix is a
                // needle prefix, verified across subsequent chunks.
                const first_tail = if (c.len >= needle.len) c.len - needle.len + 1 else 0;
                for (first_tail..c.len) |p| {
                    const head = c.len - p;
                    if (std.mem.eql(u8, c[p..], needle[0..head]) and
                        self.matchesFrom(chunk_start + c.len, needle[head..], range.end))
                    {
                        return chunk_start + p;
                    }
                }
            }
            return null;
        }

        /// Byte offset of the last occurrence of `needle` in `range`, or
        /// `null`.
        pub fn findLast(self: RopeT, range: Range, needle: []const u8) ?usize {
            assert(needle.len > 0);
            assert(range.end <= self.byteLen());
            if (range.len() < needle.len) return null;
            var cur = self.cursorAt(range.end);
            while (cur.offset > range.start) {
                const chunk_end = cur.offset;
                const raw = cur.prevChunk() orelse break;
                const chunk_start = chunk_end - raw.len;
                const lo = if (chunk_start < range.start) range.start - chunk_start else 0;
                const c = raw[lo..];
                const abs = chunk_start + lo;
                var best: ?usize = null;
                if (std.mem.lastIndexOf(u8, c, needle)) |p| {
                    if (abs + p + needle.len <= range.end) best = abs + p;
                }
                // Straddle candidates start in this chunk's tail and extend
                // rightward; any match starting further right was already
                // checked in a previous (later) chunk iteration.
                const first_tail = if (c.len >= needle.len) c.len - needle.len + 1 else 0;
                var p = c.len;
                while (p > first_tail) {
                    p -= 1;
                    if (best != null and best.? >= abs + p) break;
                    const head = c.len - p;
                    if (head < needle.len and std.mem.eql(u8, c[p..], needle[0..head]) and
                        self.matchesFrom(chunk_start + raw.len, needle[head..], range.end))
                    {
                        best = @max(best orelse 0, abs + p);
                        break; // rightmost straddle in this chunk
                    }
                }
                if (best) |b| return b;
            }
            return null;
        }

        /// Iterator over non-overlapping occurrences of `needle` in `range`,
        /// left to right. Same invalidation rule as `Cursor`.
        pub fn findIterator(self: RopeT, range: Range, needle: []const u8) FindIterator {
            assert(needle.len > 0);
            return .{ .rope = self, .pos = range.start, .end = range.end, .needle = needle };
        }

        pub const FindIterator = struct {
            rope: RopeT,
            pos: usize,
            end: usize,
            needle: []const u8,

            pub fn next(self: *FindIterator) ?usize {
                if (self.pos >= self.end) return null;
                const hit = self.rope.find(.{ .start = self.pos, .end = self.end }, self.needle) orelse {
                    self.pos = self.end;
                    return null;
                };
                self.pos = hit + self.needle.len;
                return hit;
            }
        };

        // ════════════════════════════════════════════════════════════════
        // Line iteration (amortized O(1) per line vs O(log n) lineRange)
        // ════════════════════════════════════════════════════════════════

        /// Iterate line ranges from `start_row` to the last line. Each range
        /// excludes its terminating newline (`slice()` it for the bytes).
        /// Same invalidation rule as `Cursor`.
        pub fn lineIterator(self: RopeT, start_row: usize) LineIterator {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            assert(start_row < self.lineCount());
            const start = if (start_row == 0) 0 else offsetAfterNewlines(self.root, start_row);
            return .{ .cursor = self.cursorAt(start), .line_start = start, .done = false };
        }

        pub const LineIterator = struct {
            cursor: Cursor,
            line_start: usize,
            done: bool,

            pub fn next(self: *LineIterator) ?Range {
                if (self.done) return null;
                const cur = &self.cursor;
                while (cur.nextChunk()) |raw| {
                    const chunk_start = cur.offset - raw.len;
                    if (std.mem.indexOfScalar(u8, raw, '\n')) |p| {
                        const nl = chunk_start + p;
                        const r: Range = .{ .start = self.line_start, .end = nl };
                        self.line_start = nl + 1;
                        // Rewind within the current leaf: nl+1 is inside or at
                        // the end of it, both valid cursor positions.
                        cur.offset = nl + 1;
                        return r;
                    }
                }
                self.done = true;
                return .{ .start = self.line_start, .end = cur.len };
            }
        };

        /// Exhaustive O(n) structural check: uniform heights, fanout bounds,
        /// non-empty leaves within capacity, per-node summaries consistent
        /// with contents, live refcounts. Panics on violation; intended for
        /// tests and debugging, not production paths.
        pub fn validate(self: RopeT) void {
            const r = self.root orelse return;
            const s = validateNode(r);
            assert(s.bytes == r.summary.bytes);
        }

        fn validateNode(n: *Node) Summary {
            const live_refs = if (opts.thread_safe) n.refs.load(.acquire) else n.refs;
            assert(live_refs >= 1);
            if (n.isLeaf()) {
                if (n.data.leaf == .hole) {
                    // Holes: length-only summary, no content invariants.
                    assert(n.summary.bytes > 0);
                    assert(std.meta.eql(Summary{ .bytes = n.summary.bytes }, n.summary));
                    return n.summary;
                }
                const b = n.data.leaf.bytes();
                assert(b.len > 0);
                if (n.data.leaf == .owned) assert(b.len <= opts.chunk_capacity);
                assert(std.unicode.utf8ValidateSlice(b));
                const s = Summary.of(b);
                assert(std.meta.eql(s, n.summary));
                return s;
            }
            const cs = n.data.internal.slice();
            assert(cs.len >= 2 and cs.len <= opts.branch_factor);
            var s: Summary = .{};
            for (cs) |c| {
                assert(c.height == n.height - 1);
                s = Summary.add(s, validateNode(c));
            }
            assert(std.meta.eql(s, n.summary));
            return s;
        }

        // ════════════════════════════════════════════════════════════════
        // Coordinate conversion (O(log n) + one in-leaf scan)
        // ════════════════════════════════════════════════════════════════

        const DimField = enum { scalars, utf16, newlines };

        /// Count of `dim` in the byte prefix `[0, offset)`.
        fn prefixCount(root: ?*Node, offset: usize, comptime dim: DimField) usize {
            var n = root orelse return 0;
            var local = offset;
            var acc: usize = 0;
            while (!n.isLeaf()) {
                if (local == 0) return acc;
                for (n.data.internal.slice()) |c| {
                    if (local >= c.summary.bytes) {
                        acc += @field(c.summary, @tagName(dim));
                        local -= c.summary.bytes;
                    } else {
                        n = c;
                        break;
                    }
                } else return acc; // local == 0 exactly at end
            }
            if (n.data.leaf == .hole) return acc; // unrealized content contributes 0
            const partial = Summary.of(n.data.leaf.bytes()[0..local]);
            return acc + @field(partial, @tagName(dim));
        }

        /// Byte offset of the boundary where the prefix contains exactly
        /// `value` of `dim` (for scalars/utf16: offset of the value-th unit's
        /// start). Precondition: `value` ≤ total, and for utf16 the value must
        /// not land inside a surrogate pair. Newlines use
        /// `offsetAfterNewlines` instead.
        fn offsetOfCount(root: ?*Node, value: usize, comptime dim: DimField) usize {
            comptime assert(dim != .newlines);
            var n = root orelse {
                assert(value == 0);
                return 0;
            };
            var v = value;
            var acc: usize = 0;
            while (!n.isLeaf()) {
                if (v == 0) return acc;
                for (n.data.internal.slice()) |c| {
                    if (v >= @field(c.summary, @tagName(dim))) {
                        v -= @field(c.summary, @tagName(dim));
                        acc += c.summary.bytes;
                    } else {
                        n = c;
                        break;
                    }
                } else return acc;
            }
            const b = n.data.leaf.bytes();
            var pos: usize = 0;
            while (v > 0) {
                const l = std.unicode.utf8ByteSequenceLength(b[pos]) catch unreachable;
                const units: usize = switch (dim) {
                    .scalars => 1,
                    .utf16 => if (l == 4) @as(usize, 2) else 1,
                    .newlines => unreachable,
                };
                assert(v >= units); // utf16: no mid-surrogate positions
                v -= units;
                pos += l;
            }
            return acc + pos;
        }

        /// Byte offset just past the `k`-th newline (1-based `k`).
        fn offsetAfterNewlines(root: ?*Node, k: usize) usize {
            assert(k >= 1);
            var n = root.?;
            var v = k;
            var acc: usize = 0;
            while (!n.isLeaf()) {
                for (n.data.internal.slice()) |c| {
                    if (v > @field(c.summary, "newlines")) {
                        v -= c.summary.newlines;
                        acc += c.summary.bytes;
                    } else {
                        n = c;
                        break;
                    }
                } else unreachable;
            }
            const b = n.data.leaf.bytes();
            var pos: usize = 0;
            while (true) {
                const nl = std.mem.indexOfScalarPos(u8, b, pos, '\n').?;
                v -= 1;
                pos = nl + 1;
                if (v == 0) return acc + pos;
            }
        }

        pub fn offsetToPoint(self: RopeT, byte_offset: usize) Point {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            assert(byte_offset <= self.byteLen());
            const row = prefixCount(self.root, byte_offset, .newlines);
            const line_start = if (row == 0) 0 else offsetAfterNewlines(self.root, row);
            return .{ .row = row, .col = byte_offset - line_start };
        }

        pub fn pointToOffset(self: RopeT, point: Point) usize {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            const start = if (point.row == 0) 0 else offsetAfterNewlines(self.root, point.row);
            const offset = start + point.col;
            assert(offset <= self.byteLen());
            return offset;
        }

        /// The byte range of line `row`, excluding its terminating newline.
        /// Precondition: `row < lineCount()`.
        pub fn lineRange(self: RopeT, row: usize) Range {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            const newlines: usize = if (self.root) |r| r.summary.newlines else 0;
            assert(row <= newlines);
            const start = if (row == 0) 0 else offsetAfterNewlines(self.root, row);
            const end = if (row < newlines) offsetAfterNewlines(self.root, row + 1) - 1 else self.byteLen();
            return .{ .start = start, .end = end };
        }

        pub fn offsetToScalar(self: RopeT, byte_offset: usize) usize {
            comptime if (!opts.track_scalars) @compileError("scalars dimension disabled");
            assert(byte_offset <= self.byteLen());
            return prefixCount(self.root, byte_offset, .scalars);
        }

        pub fn scalarToOffset(self: RopeT, scalar_index: usize) usize {
            comptime if (!opts.track_scalars) @compileError("scalars dimension disabled");
            assert(scalar_index <= self.scalarLen());
            return offsetOfCount(self.root, scalar_index, .scalars);
        }

        pub fn offsetToUtf16(self: RopeT, byte_offset: usize) usize {
            comptime if (!opts.track_utf16) @compileError("utf16 dimension disabled");
            assert(byte_offset <= self.byteLen());
            return prefixCount(self.root, byte_offset, .utf16);
        }

        /// Precondition: `utf16_offset` does not land between the two units of
        /// a surrogate pair.
        pub fn utf16ToOffset(self: RopeT, utf16_offset: usize) usize {
            comptime if (!opts.track_utf16) @compileError("utf16 dimension disabled");
            assert(utf16_offset <= self.utf16Len());
            return offsetOfCount(self.root, utf16_offset, .utf16);
        }

        /// Byte offset → LSP-style (row, UTF-16 col within line).
        pub fn offsetToPointUtf16(self: RopeT, byte_offset: usize) PointUtf16 {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            comptime if (!opts.track_utf16) @compileError("utf16 dimension disabled");
            assert(byte_offset <= self.byteLen());
            const row = prefixCount(self.root, byte_offset, .newlines);
            const line_start = if (row == 0) 0 else offsetAfterNewlines(self.root, row);
            const col = prefixCount(self.root, byte_offset, .utf16) -
                prefixCount(self.root, line_start, .utf16);
            return .{ .row = row, .col = col };
        }

        /// LSP-style (row, UTF-16 col) → byte offset. Preconditions: `row` is
        /// a valid line; `col` lands on a scalar boundary within it (not
        /// mid-surrogate).
        pub fn pointUtf16ToOffset(self: RopeT, point: PointUtf16) usize {
            comptime if (!opts.track_lines) @compileError("lines dimension disabled");
            comptime if (!opts.track_utf16) @compileError("utf16 dimension disabled");
            const line_start = if (point.row == 0) 0 else offsetAfterNewlines(self.root, point.row);
            const base = prefixCount(self.root, line_start, .utf16);
            const offset = offsetOfCount(self.root, base + point.col, .utf16);
            assert(offset <= self.byteLen());
            return offset;
        }

        // ════════════════════════════════════════════════════════════════
        // Cursors and chunk iteration
        // ════════════════════════════════════════════════════════════════

        /// A cursor positioned at `byte_offset` for sequential scans without
        /// repeated root descents. Borrows the rope's tree: invalidated by any
        /// edit to this rope (take a `snapshot()` to read while editing).
        pub fn cursorAt(self: RopeT, byte_offset: usize) Cursor {
            assert(byte_offset <= self.byteLen());
            var c: Cursor = .{ .root = self.root, .len = self.byteLen() };
            c.seekTo(byte_offset);
            return c;
        }

        pub const Cursor = struct {
            root: ?*Node,
            len: usize,
            offset: usize = 0,
            leaf: []const u8 = &.{},
            leaf_start: usize = 0,
            depth: usize = 0,
            stack: [max_height]Frame = undefined,

            const Frame = struct { node: *Node, idx: usize };

            pub fn byteOffset(self: *const Cursor) usize {
                return self.offset;
            }

            /// Row/column of the current position. O(log n).
            pub fn point(self: *const Cursor) Point {
                comptime if (!opts.track_lines) @compileError("lines dimension disabled");
                const row = prefixCount(self.root, self.offset, .newlines);
                const line_start = if (row == 0) 0 else offsetAfterNewlines(self.root, row);
                return .{ .row = row, .col = self.offset - line_start };
            }

            /// UTF-16 offset of the current position. O(log n).
            pub fn utf16Offset(self: *const Cursor) usize {
                comptime if (!opts.track_utf16) @compileError("utf16 dimension disabled");
                return prefixCount(self.root, self.offset, .utf16);
            }

            /// Advance one scalar and return it, or `null` at end of buffer.
            pub fn next(self: *Cursor) ?u21 {
                if (self.offset >= self.len) return null;
                if (self.offset - self.leaf_start == self.leaf.len) self.advanceLeaf();
                const pos = self.offset - self.leaf_start;
                const l = std.unicode.utf8ByteSequenceLength(self.leaf[pos]) catch unreachable;
                const cp = std.unicode.utf8Decode(self.leaf[pos..][0..l]) catch unreachable;
                self.offset += l;
                return cp;
            }

            /// Advance to the end of the current chunk and return the bytes
            /// from the current position, or `null` at end of buffer.
            pub fn nextChunk(self: *Cursor) ?[]const u8 {
                if (self.offset >= self.len) return null;
                if (self.offset - self.leaf_start == self.leaf.len) self.advanceLeaf();
                const pos = self.offset - self.leaf_start;
                const c = self.leaf[pos..];
                self.offset = self.leaf_start + self.leaf.len;
                return c;
            }

            /// Step back one scalar and return it, or `null` at the start.
            pub fn prev(self: *Cursor) ?u21 {
                if (self.offset == 0) return null;
                const target = self.offset - 1;
                if (target < self.leaf_start or target >= self.leaf_start + self.leaf.len) {
                    self.descendContaining(target);
                }
                var p = target - self.leaf_start;
                // Chunks never split scalars, so the lead byte is in this leaf.
                while (self.leaf[p] & 0xC0 == 0x80) p -= 1;
                const l = std.unicode.utf8ByteSequenceLength(self.leaf[p]) catch unreachable;
                const cp = std.unicode.utf8Decode(self.leaf[p..][0..l]) catch unreachable;
                self.offset = self.leaf_start + p;
                return cp;
            }

            /// Step back to the start of the current chunk and return the
            /// bytes from there to the current position, or `null` at the
            /// start of the buffer.
            pub fn prevChunk(self: *Cursor) ?[]const u8 {
                if (self.offset == 0) return null;
                const target = self.offset - 1;
                if (target < self.leaf_start or target >= self.leaf_start + self.leaf.len) {
                    self.descendContaining(target);
                }
                const c = self.leaf[0 .. self.offset - self.leaf_start];
                self.offset = self.leaf_start;
                return c;
            }

            /// Reposition (backward or forward). O(log n).
            pub fn seekTo(self: *Cursor, byte_offset: usize) void {
                assert(byte_offset <= self.len);
                self.offset = byte_offset;
                if (self.root == null or byte_offset == self.len) {
                    // Empty rope or end position: park on an empty tail;
                    // next() returns null, prev() re-descends.
                    self.depth = 0;
                    self.leaf = &.{};
                    self.leaf_start = self.len;
                    return;
                }
                self.descendContaining(byte_offset);
            }

            /// Point the leaf cache and descent stack at the leaf containing
            /// byte `target`. Does not touch `self.offset`.
            fn descendContaining(self: *Cursor, target: usize) void {
                assert(target < self.len);
                self.depth = 0;
                var n = self.root.?;
                var local = target;
                var abs_start: usize = 0;
                while (!n.isLeaf()) {
                    var idx: usize = 0;
                    for (n.data.internal.slice()) |c| {
                        if (local < c.summary.bytes) break;
                        local -= c.summary.bytes;
                        abs_start += c.summary.bytes;
                        idx += 1;
                    }
                    self.stack[self.depth] = .{ .node = n, .idx = idx };
                    self.depth += 1;
                    n = n.data.internal.children[idx];
                }
                self.leaf = n.data.leaf.bytes();
                self.leaf_start = abs_start;
            }

            fn advanceLeaf(self: *Cursor) void {
                // Pop exhausted frames, step right, descend leftmost.
                while (self.depth > 0) {
                    const f = &self.stack[self.depth - 1];
                    const internal = &f.node.data.internal;
                    if (f.idx + 1 < internal.count) {
                        f.idx += 1;
                        var n = internal.children[f.idx];
                        while (!n.isLeaf()) {
                            self.stack[self.depth] = .{ .node = n, .idx = 0 };
                            self.depth += 1;
                            n = n.data.internal.children[0];
                        }
                        self.leaf_start += self.leaf.len;
                        self.leaf = n.data.leaf.bytes();
                        return;
                    }
                    self.depth -= 1;
                }
                unreachable; // offset < len guarantees a successor leaf
            }
        };

        /// Iterator over the contiguous UTF-8 chunks backing a range (borrowed
        /// storage, no copy). Same invalidation rule as `Cursor`.
        pub const Chunks = struct {
            cursor: Cursor,
            end: usize,

            /// Next chunk, or `null` when exhausted.
            pub fn next(self: *Chunks) ?[]const u8 {
                if (self.cursor.offset >= self.end) return null;
                const c = self.cursor.nextChunk() orelse return null;
                if (self.cursor.offset > self.end) {
                    const over = self.cursor.offset - self.end;
                    return c[0 .. c.len - over];
                }
                return c;
            }
        };
    };
}

/// The blessed instantiation: all dimensions tracked, thread-safe refcounts.
pub const Rope = RopeWith(.{});

test {
    std.testing.refAllDecls(Rope);
}
