//! The eg-walker prepare/effect discipline for sequence-shaped ops,
//! shared by `TextDoc`'s single character sequence and every list/text
//! object inside `ObjectDoc`'s tree (see `Sequence.zig` for the FugueMax
//! ordering itself, `causal.zig` for the graph this walks).
//!
//! Three pieces, usable independently:
//! - `movePrepareTo`: the diff-based retreat/advance loop — retreat
//!   (newest-first) events no longer causally included, advance
//!   (oldest-first) newly-included ones. Generic over `history` (needs
//!   only `.diff`) and `ctx` (needs only `toggle(history, lv, on)`), so
//!   it works whether `ctx` dispatches to one sequence (`TextDoc.Replay`)
//!   or fans out across many sequences plus non-sequence state
//!   (`ObjectDoc`'s `Walker`, across its tree of objects and map
//!   registers).
//! - `SeqWalker(storage)`: one `Sequence` plus the bookkeeping to apply/
//!   retreat/advance its events by global `Lv` — the reusable unit
//!   `TextDoc` instantiates once (its one sequence) and `ObjectDoc`
//!   instantiates once per list/text object.
//! - `anchorAt`/`resolveAnchors`: portable identity positions (agent name
//!   + seq + side) over an already-caught-up `SeqWalker` instance —
//!   `TextDoc.anchorAt`/`resolveAnchors` are thin call-throughs against
//!   its one `SeqWalker`; `ObjectDoc.objectAnchorAt`/`resolveObjectAnchors`
//!   call the same functions against one per-text-object `SeqWalker`
//!   (delta 3, `app/weft/doc/stemma-unification.md` §3 step 3).
//!
//! Neither `movePrepareTo` nor `SeqWalker` touches `Op`: the graph is
//! passed in generically, and callers already know (from their own op
//! tag) which of a `SeqWalker` instance's methods to call and at what
//! position — that dispatch is irreducibly caller-specific (TextDoc:
//! always yes, the one sequence; ObjectDoc: `Walker.resolveObj` + op
//! tag). `resolveAnchors` needs one caller-specific bit `movePrepareTo`
//! doesn't: given an `Op`, is it THIS `SeqWalker`'s own insert kind (for
//! `TextDoc`, any `.ins`; for `ObjectDoc`, a `.text_ins`/`.list_ins`
//! targeting *this* object, not some other node sharing the same graph)
//! — supplied via `ctx.isInsert(op) bool`, the same anytype-duck-typed
//! shape `movePrepareTo`'s `ctx` already uses.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const Lv = causal.Lv;
const EventId = causal.EventId;
const Sequence = @import("Sequence.zig");

/// Move `prep_frontier` to `target`: diff the two, retreat (newest-first)
/// events only in the old frontier's past via `ctx.toggle(history, lv,
/// false)`, then advance (oldest-first) events only in `target`'s via
/// `ctx.toggle(history, lv, true)`. A no-op if the frontiers already
/// match. `history` needs only `diff(gpa, a, b) -> Diff{a_only,b_only}`
/// (any `causal.EventGraph(Op)` qualifies); `ctx` needs only `toggle`.
pub fn movePrepareTo(
    gpa: Allocator,
    history: anytype,
    prep_frontier: *std.ArrayList(Lv),
    target: []const Lv,
    ctx: anytype,
) Allocator.Error!void {
    if (std.mem.eql(Lv, prep_frontier.items, target)) return;
    var d = try history.diff(gpa, prep_frontier.items, target);
    defer d.deinit(gpa);
    // Retreat newest-first, advance oldest-first.
    var i = d.a_only.items.len;
    while (i > 0) {
        i -= 1;
        ctx.toggle(history, d.a_only.items[i], false);
    }
    for (d.b_only.items) |lv| ctx.toggle(history, lv, true);
    prep_frontier.clearRetainingCapacity();
    try prep_frontier.appendSlice(gpa, target);
}

/// `SeqWalker`'s `lv → arena` bookkeeping strategy — the only thing that
/// differs between instantiations; everything else about the
/// retreat/advance/apply discipline is identical.
///
/// - `.dense`: `TextDoc`'s shape. One `SeqWalker` covers the WHOLE
///   document, and every event in `TextDoc`'s graph is a sequence op —
///   the `Lv` space is dense and total, so `ArrayList(i32)` indexed
///   directly by `lv` is a perfect-fit, zero-waste O(1) slot per event
///   (and — since a merged ins/del array replaces the old two-array
///   `item_of`/`target_of` — actually leaner than before).
/// - `.sparse`: `ObjectDoc`'s shape. `Walker` keeps one `SeqWalker` PER
///   OBJECT over a `Lv` space shared with every other object (and every
///   map register) in the tree; a dense array here would cost memory
///   proportional to the WHOLE document, once per object, for slots that
///   are mostly never touched. `AutoHashMapUnmanaged(Lv, i32)`, sized to
///   that object's own event count, is the right fit.
///
/// Getting this wrong in the `.dense` direction is not just a style
/// choice: `benchCollab` measured a real ~10-12% regression on TextDoc's
/// cold full-replay paths (open-from-wire, `materializeAt`, `compact`,
/// large catch-up batches) when `.dense`'s array was replaced by a
/// hashmap to "unify" with `.sparse` — the hash lookup doesn't disappear
/// into the surrounding treap-traversal cost the way it does on the
/// warm, `merge_walk`-cached path.
pub const Storage = enum { dense, sparse };

/// One FugueMax sequence plus its retreat/advance/apply bookkeeping,
/// keyed by the global `Lv` of the events that touch it (see `Storage`
/// for how `item_of` is represented). Holds no reference to a graph or
/// an arena of its own beyond `Sequence`'s — every method takes what it
/// needs explicitly, so an instance is safe to keep as a long-lived
/// cache (`TextDoc.merge_walk`) or to create one per object (`ObjectDoc`'s
/// `Walker.seqs`) without dangling-pointer risk if the owning document
/// is copied or its graph is rebuilt.
pub fn SeqWalker(comptime storage: Storage) type {
    return struct {
        const Self = @This();

        s: Sequence = .empty,
        /// lv → arena of the item it inserted/deleted, for `toggle*` and
        /// (through `arenaOf`) any caller needing the raw arena (identity
        /// anchors). See `Storage`.
        item_of: switch (storage) {
            .dense => std.ArrayList(i32),
            .sparse => std.AutoHashMapUnmanaged(Lv, i32),
        } = .empty,

        pub const empty: Self = .{};

        pub fn deinit(self: *Self, gpa: Allocator) void {
            self.s.deinit(gpa);
            self.item_of.deinit(gpa);
        }

        /// Must precede any live item; see `Sequence.initBase`.
        pub fn initBase(self: *Self, gpa: Allocator, count: usize, placeholder_lv: Lv) Allocator.Error!void {
            try self.s.initBase(gpa, count, placeholder_lv);
        }

        fn recordArena(self: *Self, gpa: Allocator, lv: Lv, arena: i32) Allocator.Error!void {
            switch (storage) {
                .dense => {
                    if (lv >= self.item_of.items.len) {
                        try self.item_of.appendNTimes(gpa, Sequence.none, lv + 1 - self.item_of.items.len);
                    }
                    self.item_of.items[lv] = arena;
                },
                .sparse => try self.item_of.put(gpa, lv, arena),
            }
        }

        /// The arena an already-applied event `lv` inserted/deleted.
        /// Callers own the invariant that `lv` was already applied
        /// (`applyInsert`/`applyDelete`) — same contract `toggle*` relies
        /// on, exposed for callers that need the raw arena directly
        /// (`TextDoc.resolveAnchors`).
        pub fn arenaOf(self: *const Self, lv: Lv) i32 {
            return switch (storage) {
                .dense => self.item_of.items[lv],
                .sparse => self.item_of.get(lv).?,
            };
        }

        /// First application of an insert at prepare position `pos`,
        /// authored by event `lv`. `emit` doubles as
        /// `Sequence.applyInsert`'s `untrusted` flag: events being
        /// freshly integrated (from a remote batch) get error-checked
        /// positions; events replayed silently from already-trusted
        /// local history get asserts instead.
        pub fn applyInsert(
            self: *Self,
            gpa: Allocator,
            names: anytype,
            lv: Lv,
            pos: u64,
            emit: bool,
        ) Sequence.ApplyError!Sequence.Inserted {
            const res = try self.s.applyInsert(gpa, names, lv, pos, emit);
            try self.recordArena(gpa, lv, res.arena);
            return res;
        }

        /// First application of a delete at prepare position `pos`,
        /// authored by event `lv`. Same `emit`/`untrusted` double duty as
        /// `applyInsert`.
        pub fn applyDelete(
            self: *Self,
            gpa: Allocator,
            lv: Lv,
            pos: u64,
            emit: bool,
        ) (Allocator.Error || error{Corrupt})!Sequence.Deleted {
            const res = try self.s.applyDelete(pos, emit);
            try self.recordArena(gpa, lv, res.arena);
            return res;
        }

        /// Retreat/advance an already-applied insert's prepare state.
        pub fn toggleInsert(self: *Self, lv: Lv, on: bool) void {
            self.s.toggleInsert(self.arenaOf(lv), on);
        }

        /// Retreat/advance an already-applied delete's prepare state.
        pub fn toggleDelete(self: *Self, lv: Lv, on: bool) void {
            self.s.toggleDelete(self.arenaOf(lv), on);
        }
    };
}

// ── Identity anchors ────────────────────────────────────────────────
// Portable positions that survive *concurrent* edits: an anchor names the
// inserting event of a character (agent name + seq — globally stable,
// never a replica-local id) plus a side. Ported verbatim from
// `TextDoc.zig`'s original `anchorAt`/`resolveAnchors` (delta 3,
// `app/weft/doc/stemma-unification.md` §3 step 3) — the only things that
// moved are byte↔scalar conversion (stays with the caller's own rope) and
// the boundary short-circuit in `anchorAt` (stays with the caller, so a
// start/end anchor costs no replay at all, exactly as before).

pub const AnchorSide = enum { before, after };

pub const EventAnchor = struct {
    /// Inserting agent's name; empty = document/object boundary. Owned by
    /// the caller (`anchorAt` allocates it; free with `gpa.free`).
    agent: []const u8 = "",
    seq: u64 = 0,
    side: AnchorSide = .before,
};

pub const AnchorError = Allocator.Error || error{ Corrupt, MissingDependency, Compacted };

/// Sentinel `lv` `SeqWalker.initBase`'s placeholder items carry (pre-history
/// collapsed into a compacted base snapshot — see `TextDoc.compact`).
/// `anchorAt` reports these as `error.Compacted`. Reachable from `ObjectDoc`
/// too, as of delta 2 (`stemma-unification.md` §3 step 4,
/// `ObjectDoc.compact`): a compacted text object's per-object `SeqWalker`
/// gets `initBase`'d from `jw.TextBase` the same way TextDoc's does, so an
/// `objectAnchorAt`/`resolveObjectAnchors` call into compacted-away text
/// content hits this same branch and reports `error.Compacted` (tested in
/// `object_tests.zig`, "compact: identity anchors into compacted text
/// become error.Compacted") — no new error variant or signature change was
/// needed here, exactly as anticipated when this sentinel was added ahead
/// of delta 2 landing.
pub const base_placeholder_lv: Lv = std.math.maxInt(Lv);

/// Identity anchor for the item at scalar index `target_index` among a
/// fully-caught-up `sw`'s alive items (document/object order). Callers own
/// the boundary cases (`target_index` must be a valid alive-item index —
/// `0 <= target_index < alive count`): an anchor at the very start/end of
/// the content needs no replay at all (see `TextDoc.anchorAt`), so that
/// short-circuit stays in the caller, not here. `side` is stamped onto the
/// result unchanged (the caller's stickiness). `history` needs `idOf`/
/// `agentName` (any `causal.EventGraph(Op)`); `sw` is a `SeqWalker(_)`
/// already advanced to the state being anchored against. The returned
/// `agent` slice is gpa-owned.
pub fn anchorAt(
    gpa: Allocator,
    history: anytype,
    sw: anytype,
    target_index: u64,
    side: AnchorSide,
) AnchorError!EventAnchor {
    var it = sw.s.aliveIterator();
    var i: u64 = 0;
    while (it.next()) |alive| : (i += 1) {
        if (i == target_index) {
            if (alive.lv == base_placeholder_lv) return error.Compacted;
            const id = history.idOf(alive.lv);
            return .{
                .agent = try gpa.dupe(u8, history.agentName(id.agent)),
                .seq = id.seq,
                .side = side,
            };
        }
    }
    unreachable; // target_index < alive count by construction
}

/// Resolve identity anchors to current scalar positions within `sw`'s
/// object. Deleted targets collapse to their deletion point. One
/// O(sw's item count) pass amortized over the whole batch (plus one
/// `findAgent`/`lvOf` lookup per non-boundary anchor). `out[i]` is a
/// SCALAR position (document/object-local, not bytes) — convert with the
/// caller's own rope (`TextDoc`'s or one `ObjectDoc` text node's).
///
/// `ctx.isInsert(op) bool` decides whether `lv`'s op is this `SeqWalker`'s
/// own insert kind: `TextDoc`'s graph is entirely one sequence (`op ==
/// .ins` suffices); `ObjectDoc`'s is a tree sharing one graph across every
/// object, so its `ctx` also checks the op's `obj` matches `sw`'s own
/// object — rejecting a wrong-kind or wrong-object anchor as
/// `error.Corrupt` up front, instead of trusting `sw.arenaOf` (which
/// assumes its argument was already applied to *this* `sw` and does not
/// re-check).
pub fn resolveAnchors(
    gpa: Allocator,
    history: anytype,
    sw: anytype,
    ctx: anytype,
    anchors: []const EventAnchor,
    out: []usize,
) AnchorError!void {
    assert(anchors.len == out.len);

    // One pass: per-arena count of alive items strictly before it.
    const alive_before = try gpa.alloc(usize, sw.s.items.items.len);
    defer gpa.free(alive_before);
    var count: usize = 0;
    var order = sw.s.allIterator();
    while (order.next()) |arena| {
        alive_before[@intCast(arena)] = count;
        if (sw.s.items.items[@intCast(arena)].effect_visible) count += 1;
    }

    for (anchors, out) |a, *o| {
        if (a.agent.len == 0) {
            o.* = switch (a.side) {
                .before => 0,
                .after => count,
            };
            continue;
        }
        const agent = history.findAgent(a.agent) orelse return error.MissingDependency;
        const id: EventId = .{ .agent = agent, .seq = a.seq };
        const lv = history.lvOf(id) orelse {
            return if (history.isKnown(id)) error.Compacted else error.MissingDependency;
        };
        if (!ctx.isInsert(history.opOf(lv))) return error.Corrupt;
        // `lv` passed `ctx.isInsert`, so it was applied to `sw` — `arenaOf`
        // is guaranteed populated.
        const arena = sw.arenaOf(lv);
        const item = &sw.s.items.items[@intCast(arena)];
        var scalar = alive_before[@intCast(arena)];
        if (item.effect_visible and a.side == .after) scalar += 1;
        o.* = scalar;
    }
}

const testing = std.testing;

fn testNames(comptime Graph: type) type {
    return struct {
        g: *const Graph,
        pub fn agentNameOf(self: @This(), lv: Lv) []const u8 {
            return self.g.agentName(self.g.idOf(lv).agent);
        }
    };
}

const TestOp = union(enum) { ins: struct { pos: u64, ch: u21 }, del: u64 };
const TestGraph = causal.EventGraph(TestOp);
const TestNames = testNames(TestGraph);

/// A one-`SeqWalker` harness mirroring `TextDoc.Replay`'s shape, enough
/// to exercise `movePrepareTo` + `SeqWalker` together without pulling in
/// `TextDoc`'s rope/wire machinery. Parameterized by `Storage` so the
/// same battery of tests covers both instantiations.
fn Harness(comptime storage: Storage) type {
    return struct {
        const Self = @This();
        sw: SeqWalker(storage) = .empty,
        prep_frontier: std.ArrayList(Lv) = .empty,

        fn deinit(self: *Self, gpa: Allocator) void {
            self.sw.deinit(gpa);
            self.prep_frontier.deinit(gpa);
        }

        fn toggle(self: *Self, history: *const TestGraph, lv: Lv, on: bool) void {
            switch (history.opOf(lv)) {
                .ins => self.sw.toggleInsert(lv, on),
                .del => self.sw.toggleDelete(lv, on),
            }
        }

        fn apply(self: *Self, gpa: Allocator, history: *const TestGraph, lv: Lv, emit: bool) !void {
            try movePrepareTo(gpa, history, &self.prep_frontier, history.parentsOf(lv), self);
            switch (history.opOf(lv)) {
                .ins => |ins| _ = try self.sw.applyInsert(gpa, TestNames{ .g = history }, lv, ins.pos, emit),
                .del => |pos| _ = try self.sw.applyDelete(gpa, lv, pos, emit),
            }
            // lv now sits in the prepare frontier, replacing its parents
            // (mirrors TextDoc.Replay.replayAll / objects_state.Walker.replayAll).
            self.prep_frontier.clearRetainingCapacity();
            try self.prep_frontier.append(gpa, lv);
        }

        fn content(self: *const Self, history: *const TestGraph, gpa: Allocator) ![]u21 {
            var out: std.ArrayList(u21) = .empty;
            errdefer out.deinit(gpa);
            var it = self.sw.s.aliveIterator();
            while (it.next()) |alive| try out.append(gpa, history.opOf(alive.lv).ins.ch);
            return out.toOwnedSlice(gpa);
        }
    };
}

/// Count of items currently prepare-visible, across the whole sequence.
fn prepVisibleCount(s: *const Sequence) usize {
    var n: usize = 0;
    for (s.items.items) |it| {
        if (it.prepVisible()) n += 1;
    }
    return n;
}

fn testSequentialInserts(comptime storage: Storage) !void {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(storage) = .{};
    defer h.deinit(gpa);

    for ("abc", 0..) |ch, i| {
        const lv = try g.addLocal(gpa, a, .{ .ins = .{ .pos = i, .ch = ch } });
        try h.apply(gpa, &g, lv, true);
    }
    const got = try h.content(&g, gpa);
    defer gpa.free(got);
    try testing.expectEqualSlices(u21, &.{ 'a', 'b', 'c' }, got);
}

test "SeqWalker(.dense): sequential local inserts stay in order" {
    try testSequentialInserts(.dense);
}
test "SeqWalker(.sparse): sequential local inserts stay in order" {
    try testSequentialInserts(.sparse);
}

fn testDeleteRemovesTarget(comptime storage: Storage) !void {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(storage) = .{};
    defer h.deinit(gpa);

    for ("abc", 0..) |ch, i| {
        const lv = try g.addLocal(gpa, a, .{ .ins = .{ .pos = i, .ch = ch } });
        try h.apply(gpa, &g, lv, true);
    }
    const del_lv = try g.addLocal(gpa, a, .{ .del = 1 });
    try h.apply(gpa, &g, del_lv, true);

    const got = try h.content(&g, gpa);
    defer gpa.free(got);
    try testing.expectEqualSlices(u21, &.{ 'a', 'c' }, got);
}

test "SeqWalker(.dense): delete removes the targeted item" {
    try testDeleteRemovesTarget(.dense);
}
test "SeqWalker(.sparse): delete removes the targeted item" {
    try testDeleteRemovesTarget(.sparse);
}

fn testConcurrentInsertsTiebreak(comptime storage: Storage) !void {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");
    const b = try g.registerAgent(gpa, "b");

    var h: Harness(storage) = .{};
    defer h.deinit(gpa);

    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });
    try h.apply(gpa, &g, e0, true);

    // Both agents insert concurrently at position 1 (after 'x'), each
    // unaware of the other's event: agent "a" (event e1, 'y') and agent
    // "b" (event e2, 'z'), same origin_left/origin_right. FugueMax's
    // agent-name tiebreak (Sequence.integrate) orders concurrent inserts
    // at the same position by ascending agent name — "a" < "b" — so 'y'
    // must land before 'z' regardless of apply order.
    const e1 = try g.add(gpa, .{ .agent = a, .seq = 1 }, &.{e0}, .{ .ins = .{ .pos = 1, .ch = 'y' } });
    const e2 = try g.add(gpa, .{ .agent = b, .seq = 0 }, &.{e0}, .{ .ins = .{ .pos = 1, .ch = 'z' } });
    try h.apply(gpa, &g, e1, true);
    try h.apply(gpa, &g, e2, true);

    const got = try h.content(&g, gpa);
    defer gpa.free(got);
    try testing.expectEqualSlices(u21, &.{ 'x', 'y', 'z' }, got);
}

test "SeqWalker(.dense): concurrent inserts settle by the agent-name tiebreak" {
    try testConcurrentInsertsTiebreak(.dense);
}
test "SeqWalker(.sparse): concurrent inserts settle by the agent-name tiebreak" {
    try testConcurrentInsertsTiebreak(.sparse);
}

fn testToggleRetreatsAndReadvances(comptime storage: Storage) !void {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var sw: SeqWalker(storage) = .empty;
    defer sw.deinit(gpa);

    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });
    _ = try sw.applyInsert(gpa, TestNames{ .g = &g }, e0, 0, true);
    try testing.expectEqual(@as(u32, 1), sw.s.items.items[0].sub_prep);

    sw.toggleInsert(e0, false);
    try testing.expectEqual(@as(u32, 0), sw.s.items.items[0].sub_prep);

    sw.toggleInsert(e0, true);
    try testing.expectEqual(@as(u32, 1), sw.s.items.items[0].sub_prep);
}

test "SeqWalker(.dense): toggling retreats and re-advances a prior insert" {
    try testToggleRetreatsAndReadvances(.dense);
}
test "SeqWalker(.sparse): toggling retreats and re-advances a prior insert" {
    try testToggleRetreatsAndReadvances(.sparse);
}

test "SeqWalker: movePrepareTo no-ops when the frontier is unchanged" {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");
    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });

    var h: Harness(.dense) = .{};
    defer h.deinit(gpa);
    try h.apply(gpa, &g, e0, true);

    // Calling movePrepareTo again at the same target must not re-toggle
    // (toggling an already-inserted item off would desync sub_prep).
    try movePrepareTo(gpa, &g, &h.prep_frontier, &.{e0}, &h);
    try testing.expectEqual(@as(u32, 1), h.sw.s.items.items[0].sub_prep);
}

test "SeqWalker: movePrepareTo retreats and advances several events in one call" {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(.dense) = .{};
    defer h.deinit(gpa);

    var lvs: [4]Lv = undefined;
    for ("abcd", 0..) |ch, i| {
        lvs[i] = try g.addLocal(gpa, a, .{ .ins = .{ .pos = i, .ch = ch } });
        try h.apply(gpa, &g, lvs[i], true);
    }
    try testing.expectEqual(@as(usize, 4), prepVisibleCount(&h.sw.s));

    // Retreat two events (c, d) in ONE movePrepareTo call — exercises the
    // newest-first internal ordering across a multi-event diff, not just
    // a single toggle.
    try movePrepareTo(gpa, &g, &h.prep_frontier, &.{lvs[1]}, &h);
    try testing.expectEqual(@as(usize, 2), prepVisibleCount(&h.sw.s));

    // Advance the same two events back in ONE call — oldest-first.
    try movePrepareTo(gpa, &g, &h.prep_frontier, &.{lvs[3]}, &h);
    try testing.expectEqual(@as(usize, 4), prepVisibleCount(&h.sw.s));
}

// ── Identity anchors: generic-function-level coverage ──────────────
// `TextDoc`'s and `ObjectDoc`'s own test batteries (text_tests.zig,
// object_tests.zig) are the facade-level gate; these pin the SHARED
// `anchorAt`/`resolveAnchors` correctness directly against the harness,
// independent of either facade — same insurance role as the
// `movePrepareTo`/`SeqWalker` tests above.

const IsIns = struct {
    pub fn isInsert(_: @This(), op: TestOp) bool {
        return op == .ins;
    }
};

test "generic anchors: identity survives a concurrent insert landing before the target" {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(.dense) = .{};
    defer h.deinit(gpa);

    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });
    try h.apply(gpa, &g, e0, true);
    const e1 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 1, .ch = 'y' } });
    try h.apply(gpa, &g, e1, true);

    // Anchor names e1's insertion identity — the item currently at index 1.
    const anchor = try anchorAt(gpa, &g, &h.sw, 1, .before);
    defer gpa.free(anchor.agent);
    try testing.expectEqualStrings("a", anchor.agent);
    try testing.expectEqual(@as(u64, 1), anchor.seq);

    // A concurrent event from agent "AAA" (< "a" by the name tiebreak, so
    // it wins) branches from e0 at the SAME prepare position e1 saw — it
    // lands BEFORE 'y' in the final order, shifting 'y' from index 1 to 2.
    const other = try g.registerAgent(gpa, "AAA");
    const e2 = try g.add(gpa, .{ .agent = other, .seq = 0 }, &.{e0}, .{ .ins = .{ .pos = 1, .ch = 'w' } });
    try h.apply(gpa, &g, e2, true);
    const got = try h.content(&g, gpa);
    defer gpa.free(got);
    try testing.expectEqualSlices(u21, &.{ 'x', 'w', 'y' }, got);

    // The SAME anchor (agent+seq — nothing about it changed) now resolves
    // to index 2: it names the CHARACTER, not the offset it was minted at.
    var out: [1]usize = undefined;
    try resolveAnchors(gpa, &g, &h.sw, IsIns{}, &.{anchor}, &out);
    try testing.expectEqual(@as(usize, 2), out[0]);
}

test "generic anchors: resolveAnchors collapses a deleted target to its deletion point" {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(.dense) = .{};
    defer h.deinit(gpa);

    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });
    try h.apply(gpa, &g, e0, true);
    const e1 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 1, .ch = 'y' } });
    try h.apply(gpa, &g, e1, true);

    const anchor = try anchorAt(gpa, &g, &h.sw, 1, .before);
    defer gpa.free(anchor.agent);

    var before_off: [1]usize = undefined;
    try resolveAnchors(gpa, &g, &h.sw, IsIns{}, &.{anchor}, &before_off);

    const del_lv = try g.addLocal(gpa, a, .{ .del = 1 });
    try h.apply(gpa, &g, del_lv, true);

    var after_off: [1]usize = undefined;
    try resolveAnchors(gpa, &g, &h.sw, IsIns{}, &.{anchor}, &after_off);
    try testing.expectEqual(before_off[0], after_off[0]);
}

test "generic anchors: resolveAnchors rejects an anchor naming a delete event" {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(.dense) = .{};
    defer h.deinit(gpa);
    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });
    try h.apply(gpa, &g, e0, true);
    // del_lv is (agent "a", seq 1) — a delete event, never a valid anchor.
    const del_lv = try g.addLocal(gpa, a, .{ .del = 0 });
    try h.apply(gpa, &g, del_lv, true);

    const bad: EventAnchor = .{ .agent = "a", .seq = 1, .side = .before };
    var out: [1]usize = undefined;
    try testing.expectError(error.Corrupt, resolveAnchors(gpa, &g, &h.sw, IsIns{}, &.{bad}, &out));
}

test "generic anchors: resolveAnchors reports MissingDependency for an unknown agent" {
    const gpa = testing.allocator;
    var g: TestGraph = .empty;
    defer g.deinit(gpa);
    const a = try g.registerAgent(gpa, "a");

    var h: Harness(.dense) = .{};
    defer h.deinit(gpa);
    const e0 = try g.addLocal(gpa, a, .{ .ins = .{ .pos = 0, .ch = 'x' } });
    try h.apply(gpa, &g, e0, true);

    const bad: EventAnchor = .{ .agent = "ghost", .seq = 0, .side = .before };
    var out: [1]usize = undefined;
    try testing.expectError(error.MissingDependency, resolveAnchors(gpa, &g, &h.sw, IsIns{}, &.{bad}, &out));
}

test {
    std.testing.refAllDecls(@This());
}
