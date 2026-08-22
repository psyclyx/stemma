//! The eg-walker prepare/effect discipline for sequence-shaped ops,
//! shared by `TextDoc`'s single character sequence and every list/text
//! object inside `ObjectDoc`'s tree (see `Sequence.zig` for the FugueMax
//! ordering itself, `causal.zig` for the graph this walks).
//!
//! Two pieces, usable independently:
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
//!
//! Neither piece touches `Op`: the graph is passed in generically, and
//! callers already know (from their own op tag) which of a `SeqWalker`
//! instance's methods to call and at what position — that dispatch is
//! irreducibly caller-specific (TextDoc: always yes, the one sequence;
//! ObjectDoc: `Walker.resolveObj` + op tag).

const std = @import("std");
const Allocator = std.mem.Allocator;

const causal = @import("causal.zig");
const Lv = causal.Lv;
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

test {
    std.testing.refAllDecls(@This());
}
