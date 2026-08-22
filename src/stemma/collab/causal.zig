//! The causal event graph: append-only storage of operations with their
//! causal parents, agent identity, version frontiers, and the two-frontier
//! diff that powers both replay (retreat/advance sets) and sync (which
//! events does the remote lack).
//!
//! Type-generic over the operation payload `Op` — the DAG neither knows nor
//! cares what an event means; materializers (text first) interpret them.
//!
//! Local versions (`Lv`) are indices into the event array. Events are only
//! appended after their parents exist, so ascending `Lv` is always a valid
//! topological order — replay and diff lean on this.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

/// Local (per-replica) index of an agent's interned name. NOT stable across
/// replicas — cross-peer ordering uses the name bytes.
pub const AgentId = enum(u32) { _ };

/// Globally unique event identity: (agent, per-agent sequence number).
/// Sequence numbers count operation *units*, so runs RLE naturally later.
pub const EventId = struct {
    agent: AgentId,
    seq: u64,
};

/// Local version: index into this replica's event array. Different replicas
/// assign different Lvs to the same EventId; never serialize an Lv.
pub const Lv = u32;

pub const max_events = std.math.maxInt(Lv) - 1;

/// Causal relation between two versions, from the first version's
/// perspective: `.ancestor` means the first is strictly before the second.
pub const VersionOrder = enum { equal, ancestor, descendant, concurrent };

pub fn EventGraph(comptime Op: type) type {
    return struct {
        const Self = @This();

        pub const Event = struct {
            id: EventId,
            op: Op,
            parents_start: u32,
            parents_len: u32,
        };

        const Agent = struct {
            name_start: u32,
            name_len: u32,
            /// Sequence numbers below this are compacted: known to have
            /// happened (duplicate detection) but no longer stored.
            seq_base: u64 = 0,
            /// (seq - seq_base) → Lv (unit events arrive per-agent in order).
            lv_by_seq: std.ArrayList(Lv) = .empty,
        };

        events: std.ArrayList(Event) = .empty,
        parents_pool: std.ArrayList(Lv) = .empty,
        frontier: std.ArrayList(Lv) = .empty,
        agents: std.ArrayList(Agent) = .empty,
        names: std.ArrayList(u8) = .empty,

        pub const empty: Self = .{};

        pub fn deinit(self: *Self, gpa: Allocator) void {
            for (self.agents.items) |*a| a.lv_by_seq.deinit(gpa);
            self.agents.deinit(gpa);
            self.events.deinit(gpa);
            self.parents_pool.deinit(gpa);
            self.frontier.deinit(gpa);
            self.names.deinit(gpa);
            self.* = .{};
        }

        pub fn eventCount(self: *const Self) usize {
            return self.events.items.len;
        }

        // ── Agents ──────────────────────────────────────────────────────

        /// Intern `name`, returning the existing id if already registered.
        /// Agent count is expected to be small; lookup is linear.
        pub fn registerAgent(self: *Self, gpa: Allocator, name: []const u8) Allocator.Error!AgentId {
            assert(name.len > 0);
            for (self.agents.items, 0..) |a, i| {
                if (std.mem.eql(u8, self.names.items[a.name_start..][0..a.name_len], name)) {
                    return @enumFromInt(i);
                }
            }
            const start: u32 = @intCast(self.names.items.len);
            try self.names.appendSlice(gpa, name);
            errdefer self.names.items.len = start;
            try self.agents.append(gpa, .{ .name_start = start, .name_len = @intCast(name.len) });
            return @enumFromInt(self.agents.items.len - 1);
        }

        pub fn agentName(self: *const Self, id: AgentId) []const u8 {
            const a = self.agents.items[@intFromEnum(id)];
            return self.names.items[a.name_start..][0..a.name_len];
        }

        /// Look up an agent by name without registering it.
        pub fn findAgent(self: *const Self, name: []const u8) ?AgentId {
            for (self.agents.items, 0..) |a, i| {
                if (std.mem.eql(u8, self.names.items[a.name_start..][0..a.name_len], name)) {
                    return @enumFromInt(i);
                }
            }
            return null;
        }

        /// Next unused sequence number for `agent`.
        pub fn nextSeq(self: *const Self, agent: AgentId) u64 {
            const a = self.agents.items[@intFromEnum(agent)];
            return a.seq_base + a.lv_by_seq.items.len;
        }

        /// Resolve a global id to this replica's Lv. Null for unknown ids
        /// AND for compacted ones — distinguish with `isKnown`.
        pub fn lvOf(self: *const Self, id: EventId) ?Lv {
            const a = self.agents.items[@intFromEnum(id.agent)];
            if (id.seq < a.seq_base) return null; // compacted
            const rel = id.seq - a.seq_base;
            if (rel >= a.lv_by_seq.items.len) return null;
            return a.lv_by_seq.items[rel];
        }

        /// Whether this replica has ever seen the event (stored OR
        /// compacted). Known-but-null-Lv means compacted.
        pub fn isKnown(self: *const Self, id: EventId) bool {
            return id.seq < self.nextSeq(id.agent);
        }

        pub fn idOf(self: *const Self, lv: Lv) EventId {
            return self.events.items[lv].id;
        }

        pub fn opOf(self: *const Self, lv: Lv) Op {
            return self.events.items[lv].op;
        }

        pub fn parentsOf(self: *const Self, lv: Lv) []const Lv {
            const e = self.events.items[lv];
            return self.parents_pool.items[e.parents_start..][0..e.parents_len];
        }

        // ── Adding events ───────────────────────────────────────────────

        /// Append an event. Preconditions: `id.seq == nextSeq(id.agent)`
        /// (per-agent contiguity) and every parent Lv already exists. The
        /// graph frontier advances past the event's parents.
        pub fn add(self: *Self, gpa: Allocator, id: EventId, parents: []const Lv, op: Op) Allocator.Error!Lv {
            assert(self.events.items.len < max_events);
            assert(id.seq == self.nextSeq(id.agent));
            const lv: Lv = @intCast(self.events.items.len);
            if (std.debug.runtime_safety) for (parents) |p| assert(p < lv);

            const pstart: u32 = @intCast(self.parents_pool.items.len);
            try self.parents_pool.appendSlice(gpa, parents);
            errdefer self.parents_pool.items.len = pstart;
            try self.events.append(gpa, .{
                .id = id,
                .op = op,
                .parents_start = pstart,
                .parents_len = @intCast(parents.len),
            });
            errdefer _ = self.events.pop();
            try self.agents.items[@intFromEnum(id.agent)].lv_by_seq.append(gpa, lv);
            errdefer _ = self.agents.items[@intFromEnum(id.agent)].lv_by_seq.pop();

            try advanceFrontier(gpa, &self.frontier, lv, parents);
            return lv;
        }

        /// Record a local event: parents are the current frontier.
        pub fn addLocal(self: *Self, gpa: Allocator, agent: AgentId, op: Op) Allocator.Error!Lv {
            // add() mutates the frontier while reading `parents`; snapshot it.
            var parents_buf: [8]Lv = undefined;
            var parents: []const Lv = undefined;
            var owned: ?[]Lv = null;
            defer if (owned) |o| gpa.free(o);
            if (self.frontier.items.len <= parents_buf.len) {
                @memcpy(parents_buf[0..self.frontier.items.len], self.frontier.items);
                parents = parents_buf[0..self.frontier.items.len];
            } else {
                owned = try gpa.dupe(Lv, self.frontier.items);
                parents = owned.?;
            }
            return self.add(gpa, .{ .agent = agent, .seq = self.nextSeq(agent) }, parents, op);
        }

        /// frontier := (frontier \ parents) ∪ {lv}. Sorted ascending.
        fn advanceFrontier(gpa: Allocator, frontier: *std.ArrayList(Lv), lv: Lv, parents: []const Lv) Allocator.Error!void {
            var w: usize = 0;
            for (frontier.items) |f| {
                if (std.mem.indexOfScalar(Lv, parents, f) == null) {
                    frontier.items[w] = f;
                    w += 1;
                }
            }
            frontier.items.len = w;
            try frontier.append(gpa, lv);
            std.mem.sort(Lv, frontier.items, {}, std.sort.asc(Lv));
        }

        // ── Two-frontier diff ───────────────────────────────────────────

        pub const Diff = struct {
            /// Reachable from `a` but not `b`, ascending Lv.
            a_only: std.ArrayList(Lv) = .empty,
            /// Reachable from `b` but not `a`, ascending Lv.
            b_only: std.ArrayList(Lv) = .empty,

            pub fn deinit(self: *Diff, gpa: Allocator) void {
                self.a_only.deinit(gpa);
                self.b_only.deinit(gpa);
            }
        };

        const Flags = u2; // bit 0: reachable from a; bit 1: from b

        /// Partition history reachable from frontier `a` vs frontier `b`.
        /// Walks only back to the common ancestry, not to genesis.
        pub fn diff(self: *const Self, gpa: Allocator, a: []const Lv, b: []const Lv) Allocator.Error!Diff {
            var out: Diff = .{};
            errdefer out.deinit(gpa);

            var pending: std.AutoHashMapUnmanaged(Lv, Flags) = .empty;
            defer pending.deinit(gpa);
            var heap: Heap = .{};
            defer heap.deinit(gpa);
            var not_both: usize = 0;

            for (a) |lv| try diffPush(gpa, &pending, &heap, &not_both, lv, 1);
            for (b) |lv| try diffPush(gpa, &pending, &heap, &not_both, lv, 2);

            while (not_both > 0) {
                const lv = heap.pop().?;
                const flags = (pending.fetchRemove(lv) orelse continue).value;
                if (flags != 3) {
                    not_both -= 1;
                    const list = if (flags == 1) &out.a_only else &out.b_only;
                    try list.append(gpa, lv);
                }
                for (self.parentsOf(lv)) |p| {
                    try diffPush(gpa, &pending, &heap, &not_both, p, flags);
                }
            }
            std.mem.reverse(Lv, out.a_only.items); // popped descending → ascending
            std.mem.reverse(Lv, out.b_only.items);
            return out;
        }

        fn diffPush(
            gpa: Allocator,
            pending: *std.AutoHashMapUnmanaged(Lv, Flags),
            heap: *Heap,
            not_both: *usize,
            lv: Lv,
            flags: Flags,
        ) Allocator.Error!void {
            const gop = try pending.getOrPut(gpa, lv);
            if (!gop.found_existing) {
                gop.value_ptr.* = flags;
                if (flags != 3) not_both.* += 1;
                try heap.push(gpa, lv);
            } else {
                const old = gop.value_ptr.*;
                const new = old | flags;
                if (old != 3 and new == 3) not_both.* -= 1;
                gop.value_ptr.* = new;
            }
        }

        /// Events the holder of `remote_frontier` (as Lvs of *known* events)
        /// lacks from us, ascending (i.e. causally valid send order).
        pub fn missingFrom(self: *const Self, gpa: Allocator, remote_frontier: []const Lv) Allocator.Error!std.ArrayList(Lv) {
            var d = try self.diff(gpa, self.frontier.items, remote_frontier);
            d.b_only.deinit(gpa);
            return d.a_only;
        }

        /// Causal relation between two frontiers.
        pub fn compareFrontiers(self: *const Self, gpa: Allocator, a: []const Lv, b: []const Lv) Allocator.Error!VersionOrder {
            var d = try self.diff(gpa, a, b);
            defer d.deinit(gpa);
            const a_extra = d.a_only.items.len > 0;
            const b_extra = d.b_only.items.len > 0;
            if (!a_extra and !b_extra) return .equal;
            if (!a_extra) return .ancestor;
            if (!b_extra) return .descendant;
            return .concurrent;
        }
    };
}

// ── Compaction: graph-level rebuild ─────────────────────────────────────
// The part of "collapse pre-history into a frozen base" that is pure
// `EventGraph(Op)` shape — raising per-agent watermarks and renumbering
// retained events — with NO knowledge of what `Op` means. Originally
// written once, inline, in `TextDoc.compact` (see
// `app/weft/doc/stemma-unification.md` §3 step 4); extracted here verbatim
// (same agents-loop, same lv_map loop) because it never inspected `TextOp`,
// only graph shape (`parentsOf`, `add`, per-agent `lv_by_seq`). Lives in
// `causal.zig`, not `SeqWalker.zig`: it has nothing to do with sequences —
// `ObjectDoc.compact` (below) calls it too, over its own (non-sequence)
// `ObjectOp` graph, and `causal.zig` is already "the shared
// `EventGraph(Op)` implementation, including whichever of its own
// operations are op-agnostic" (compare `diff`/`missingFrom`, also here).
//
// `TextDoc.compact`'s discipline (`in_base` = the ENTIRE causal past of the
// stable point `s`, nothing else retained from before `s`) is the ONE
// case this was designed against, and `compactGraph` reproduces it exactly
// when called that way: zero behavior change, `text_tests.zig`'s
// compaction battery is the gate.
//
// `ObjectDoc.compact` (delta 2, stemma-unification.md §3 step 4) calls the
// SAME function with a DIFFERENT, narrower `in_base`: only `text_ins`/
// `text_del` events in the causal past of `s` — `map_set`/`map_del`/
// `list_ins`/`list_del` are excluded even when causally before `s` (see
// `ObjectDoc.zig`'s compaction doc comment for the two independent reasons
// why, discovered only while implementing: `compactGraph`'s own per-agent
// watermark constraint, described next, and two things a text-only base
// doesn't have to solve — object identity and bootstrap-from-serialize).
// This is why `compactGraph` takes `in_base` as a caller-supplied array
// rather than computing it from a single diff against `s` itself: the two
// callers disagree about what "in the base" means, but agree completely
// about how to rebuild the graph once that predicate is decided.
//
// Two invariants `compactGraph` needs to be sound — callers must establish
// BOTH before calling in, and `compactGraph` now enforces the first one
// itself (a `std.debug.assert`, not merely documentation — see below):
// 1. Every RETAINED event's in_base ancestors must be exactly {} or {s}
//    (the stable point) — never any OTHER `in_base` event. This is NOT
//    the same as "nothing outside `in_base` is causally concurrent to the
//    cut" (TextDoc's and ObjectDoc's `compact` each check that too, over
//    the causal-past-of-`s` set, before calling in) — it is a SEPARATE,
//    sharper requirement, and conflating the two was a real bug found by
//    review: a RETAINED event strictly inside the causal past of `s` (not
//    "outside" anything) can still reach another retained event ONLY
//    THROUGH a folded one — e.g. retained `map_set` P, folded `text_ins`
//    T, retained `map_set` Q, where Q's only causal edge back to P routes
//    through T. Dropping T's edge (the rule below) silently drops Q's
//    dependency on P too — Q and P then look CONCURRENT instead of
//    Q-supersedes-P, corrupting MV-register conflict resolution with no
//    error raised anywhere. `ObjectDoc.compact` now guards this
//    explicitly (checked over `in_base`, catching exactly this shape);
//    this `assert` is the second, cheaper line of defense — it fires if
//    ANY caller (this one or a future one) ever calls in with an
//    `in_base` that violates the invariant, so the failure is loud
//    (a crash, this file's own responsibility) rather than the silent
//    causal corruption above. It cannot do the caller's job for it
//    (proving the invariant holds requires the same reachability walk
//    the caller already did — `compactGraph` doesn't have `s`'s
//    causal-past set, only `in_base` and `s` themselves), only catch
//    violations of it.
// 2. For EVERY agent, `in_base` restricted to that agent's own events must
//    be a PREFIX (their earliest N, nothing after) — this is how
//    `new_graph.agents[i].seq_base` gets raised, and `new_graph.add`
//    asserts on it. TextDoc's `in_base` (the whole causal past of `s`) is
//    automatically such a prefix for every agent; ObjectDoc's narrower
//    `in_base` is NOT automatic — `ObjectDoc.compact` explicitly
//    validates it per agent and rejects (`error.NotCompactable`) rather
//    than violating it.
pub fn CompactResult(comptime Graph: type) type {
    return struct {
        graph: Graph,
        /// Old `Lv` → new `Lv`, valid only for indices where `in_base` was
        /// false (an `in_base` event has no successor — it was folded away).
        /// Sized to the OLD event count. Caller-owned.
        lv_map: []Lv,

        pub fn deinit(self: *@This(), gpa: Allocator) void {
            self.graph.deinit(gpa);
            gpa.free(self.lv_map);
        }
    };
}

/// Rebuild `history` keeping only the events where `in_base[lv]` is false:
/// same agents in the same order (so an `AgentId` obtained before compaction
/// stays valid after), each retained agent's `seq_base` raised by its own
/// share of `in_base`, retained events re-added in causal (ascending `Lv`)
/// order with their parent lists rewritten through `lv_map`. `s` is the
/// stable point every `in_base` event is (by the caller's own proof)
/// reachable from — a retained event's parent may be `in_base` ONLY when
/// that parent IS `s` itself (dropped: implicit "depends on the base");
/// any OTHER `in_base` parent asserts (see the module doc comment above —
/// this is the caller-verified invariant, checked here as a second line
/// of defense, not derived). `in_base.len` must equal
/// `history.eventCount()`. Never inspects `Op`.
pub fn compactGraph(
    gpa: Allocator,
    history: anytype, // *const EventGraph(Op)
    in_base: []const bool,
    s: Lv,
) Allocator.Error!CompactResult(@TypeOf(history.*)) {
    const Graph = @TypeOf(history.*);
    const n = history.eventCount();
    assert(in_base.len == n);

    var new_graph: Graph = .empty;
    errdefer new_graph.deinit(gpa);
    for (history.agents.items, 0..) |a, i| {
        const name = history.names.items[a.name_start..][0..a.name_len];
        const aid = try new_graph.registerAgent(gpa, name);
        assert(@intFromEnum(aid) == i);
        var compacted: u64 = 0;
        for (a.lv_by_seq.items) |lv| {
            if (in_base[lv]) compacted += 1;
        }
        new_graph.agents.items[i].seq_base = a.seq_base + compacted;
    }

    const lv_map = try gpa.alloc(Lv, n);
    errdefer gpa.free(lv_map);
    var parent_buf: std.ArrayList(Lv) = .empty;
    defer parent_buf.deinit(gpa);
    for (0..n) |old_lv| {
        if (in_base[old_lv]) continue;
        parent_buf.clearRetainingCapacity();
        for (history.parentsOf(@intCast(old_lv))) |p| {
            if (in_base[p]) {
                // Implicit: depends on the base. Sound ONLY when p == s —
                // see the module doc comment's invariant 1. The caller
                // must have already guaranteed this; this assert exists
                // so a caller that didn't gets a loud crash here instead
                // of a silently corrupted graph.
                assert(p == s);
                continue;
            }
            try parent_buf.append(gpa, lv_map[p]);
        }
        const e = history.events.items[old_lv];
        lv_map[old_lv] = try new_graph.add(gpa, e.id, parent_buf.items, e.op);
    }

    return .{ .graph = new_graph, .lv_map = lv_map };
}

/// Minimal binary max-heap of Lvs (duplicates tolerated; callers skip
/// already-resolved entries).
const Heap = struct {
    items: std.ArrayList(Lv) = .empty,

    fn deinit(self: *Heap, gpa: Allocator) void {
        self.items.deinit(gpa);
    }

    fn push(self: *Heap, gpa: Allocator, v: Lv) Allocator.Error!void {
        try self.items.append(gpa, v);
        var i = self.items.items.len - 1;
        while (i > 0) {
            const p = (i - 1) / 2;
            if (self.items.items[p] >= self.items.items[i]) break;
            std.mem.swap(Lv, &self.items.items[p], &self.items.items[i]);
            i = p;
        }
    }

    fn pop(self: *Heap) ?Lv {
        const items = self.items.items;
        if (items.len == 0) return null;
        const top = items[0];
        items[0] = items[items.len - 1];
        self.items.items.len -= 1;
        var i: usize = 0;
        const n = self.items.items.len;
        while (true) {
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            var m = i;
            if (l < n and self.items.items[l] > self.items.items[m]) m = l;
            if (r < n and self.items.items[r] > self.items.items[m]) m = r;
            if (m == i) break;
            std.mem.swap(Lv, &self.items.items[i], &self.items.items[m]);
            i = m;
        }
        return top;
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

const t = std.testing;
const TestGraph = EventGraph(u8);

test "linear history: frontier tracks head, diff is empty" {
    const gpa = t.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "alice");
    for (0..5) |_| _ = try g.addLocal(gpa, a, 'x');
    try t.expectEqualSlices(Lv, &.{4}, g.frontier.items);

    var d = try g.diff(gpa, g.frontier.items, g.frontier.items);
    defer d.deinit(gpa);
    try t.expectEqual(@as(usize, 0), d.a_only.items.len);
    try t.expectEqual(@as(usize, 0), d.b_only.items.len);
}

test "divergent branches: diff partitions both sides, stops at ancestor" {
    const gpa = t.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const alice = try g.registerAgent(gpa, "alice");
    const bob = try g.registerAgent(gpa, "bob");

    // Common: a0 → a1. Branch A: a2, a3 (parents a1). Branch B: b0, b1.
    _ = try g.addLocal(gpa, alice, 'c');
    const a1 = try g.addLocal(gpa, alice, 'c');
    const a2 = try g.add(gpa, .{ .agent = alice, .seq = 2 }, &.{a1}, 'a');
    const a3 = try g.add(gpa, .{ .agent = alice, .seq = 3 }, &.{a2}, 'a');
    const b0 = try g.add(gpa, .{ .agent = bob, .seq = 0 }, &.{a1}, 'b');
    const b1 = try g.add(gpa, .{ .agent = bob, .seq = 1 }, &.{b0}, 'b');

    try t.expectEqualSlices(Lv, &.{ a3, b1 }, g.frontier.items);

    var d = try g.diff(gpa, &.{a3}, &.{b1});
    defer d.deinit(gpa);
    try t.expectEqualSlices(Lv, &.{ a2, a3 }, d.a_only.items);
    try t.expectEqualSlices(Lv, &.{ b0, b1 }, d.b_only.items);

    // missingFrom the bob-only view: everything on alice's branch.
    var missing = try g.missingFrom(gpa, &.{b1});
    defer missing.deinit(gpa);
    try t.expectEqualSlices(Lv, &.{ a2, a3 }, missing.items);
}

test "lvOf / idOf roundtrip and per-agent contiguity" {
    const gpa = t.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");
    const lv0 = try g.addLocal(gpa, a, 'p');
    const lv1 = try g.addLocal(gpa, a, 'q');
    try t.expectEqual(lv0, g.lvOf(.{ .agent = a, .seq = 0 }).?);
    try t.expectEqual(lv1, g.lvOf(.{ .agent = a, .seq = 1 }).?);
    try t.expectEqual(@as(?Lv, null), g.lvOf(.{ .agent = a, .seq = 2 }));
    try t.expectEqual(@as(u64, 2), g.nextSeq(a));
    try t.expectEqual(@as(u8, 'q'), g.opOf(lv1));
    // Re-registering returns the same id.
    try t.expectEqual(a, try g.registerAgent(gpa, "a"));
}
