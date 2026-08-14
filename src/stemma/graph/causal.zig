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
            /// seq → Lv (unit events arrive per-agent in seq order).
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
            return self.agents.items[@intFromEnum(agent)].lv_by_seq.items.len;
        }

        /// Resolve a global id to this replica's Lv, if known.
        pub fn lvOf(self: *const Self, id: EventId) ?Lv {
            const seqs = self.agents.items[@intFromEnum(id.agent)].lv_by_seq.items;
            if (id.seq >= seqs.len) return null;
            return seqs[id.seq];
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
    };
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
