//! The event-graph walker for text: replays the causal graph, reconstructing
//! transient CRDT state (YATA/Yjs-style sequence with tombstones) to
//! transform concurrent remote operations into edits that apply cleanly to
//! the local document. This is the eg-walker family (Gentle & Kleppmann,
//! EuroSys 2025): CRDT state exists only during a merge; the durable
//! artifacts are the event graph and the plain document.
//!
//! Two versions are tracked during replay:
//! - the *prepare* version: what the event's author could see when they made
//!   it (moved around the graph with retreat/advance);
//! - the *effect* state: what the merged output document contains so far.
//! Events already applied locally replay silently (state only); new remote
//! events additionally emit transformed scalar-space edits.
//!
//! v1 is correctness-first and deliberately naive where it doesn't change
//! the contract: replay starts from genesis (no LCA short-circuit or
//! placeholder runs yet), items are unit (per-scalar), and the sequence is a
//! linked list with O(n) position scans. Each is a benchmarked-optimization
//! ladder rung documented in BENCHMARKS.md — the emitted edits and
//! convergence semantics do not change as those land.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const Lv = causal.Lv;

/// Text operation payload, in **scalar** (Unicode codepoint) coordinates —
/// portable across replicas regardless of their byte encodings.
pub const TextOp = union(enum) {
    /// Insert scalar `ch` at scalar position `pos` (as seen by the author).
    ins: struct { pos: u64, ch: u21 },
    /// Delete the scalar at position `pos` (as seen by the author).
    del: u64,
};

pub const Graph = causal.EventGraph(TextOp);

/// A transformed operation in scalar space, valid against the document state
/// produced by all previously emitted edits.
pub const ScalarEdit = union(enum) {
    ins: struct { pos: u64, ch: u21 },
    del: u64,
};

const none: i32 = -1;

const Item = struct {
    lv: Lv,
    /// Arena index of the item this was inserted after (or `none` = start),
    /// as seen at the author's prepare version. Structural; never changes.
    origin_left: i32,
    /// Arena index of the next prepare-visible item at insert time (or
    /// `none` = end).
    origin_right: i32,
    /// Linked-list successor (arena index or `none`).
    next: i32,
    /// Prepare state: inserted at the current prepare version?
    prep_inserted: bool,
    /// Prepare state: number of deletes covering it at the prepare version.
    prep_deleted: u32,
    /// Effect state: present in the merged output document right now?
    effect_visible: bool,

    fn prepVisible(it: *const Item) bool {
        return it.prep_inserted and it.prep_deleted == 0;
    }
};

pub const Walker = struct {
    graph: *const Graph,
    items: std.ArrayList(Item) = .empty,
    head: i32 = none,
    /// lv → arena index of the item it inserted (`none` for deletes).
    item_of: std.ArrayList(i32) = .empty,
    /// lv → arena index of the item it deleted (`none` for inserts; filled
    /// on first application, reused by retreat/advance).
    target_of: std.ArrayList(i32) = .empty,
    prep_frontier: std.ArrayList(Lv) = .empty,

    pub fn init(graph: *const Graph) Walker {
        return .{ .graph = graph };
    }

    pub fn deinit(self: *Walker, gpa: Allocator) void {
        self.items.deinit(gpa);
        self.item_of.deinit(gpa);
        self.target_of.deinit(gpa);
        self.prep_frontier.deinit(gpa);
    }

    pub const ReplayError = Allocator.Error || error{Corrupt};

    /// Replay the whole graph in Lv order. Events with `lv < first_new`
    /// update state silently (their effects are already in the document);
    /// events at `lv >= first_new` append their transformed edits to `out`.
    /// Remote events with out-of-range positions (malicious or corrupt
    /// peers) yield `error.Corrupt` — never a crash. Trusted local history
    /// (`lv < first_new`) is asserted instead.
    pub fn replayAll(self: *Walker, gpa: Allocator, first_new: Lv, out: *std.ArrayList(ScalarEdit)) ReplayError!void {
        const n = self.graph.eventCount();
        try self.item_of.appendNTimes(gpa, none, n);
        try self.target_of.appendNTimes(gpa, none, n);
        for (0..n) |lv_usize| {
            const lv: Lv = @intCast(lv_usize);
            try self.movePrepareTo(gpa, self.graph.parentsOf(lv));
            const emitted = try self.apply(gpa, lv, lv >= first_new);
            if (emitted) |e| try out.append(gpa, e);
            self.prep_frontier.clearRetainingCapacity();
            try self.prep_frontier.append(gpa, lv);
        }
    }

    /// Move the prepare version to `target` via retreat/advance.
    fn movePrepareTo(self: *Walker, gpa: Allocator, target: []const Lv) Allocator.Error!void {
        if (std.mem.eql(Lv, self.prep_frontier.items, target)) return;
        var d = try self.graph.diff(gpa, self.prep_frontier.items, target);
        defer d.deinit(gpa);
        // Retreat newest-first, advance oldest-first.
        var i = d.a_only.items.len;
        while (i > 0) {
            i -= 1;
            self.retreat(d.a_only.items[i]);
        }
        for (d.b_only.items) |lv| self.advance(lv);
        self.prep_frontier.clearRetainingCapacity();
        try self.prep_frontier.appendSlice(gpa, target);
    }

    fn retreat(self: *Walker, lv: Lv) void {
        switch (self.graph.opOf(lv)) {
            .ins => {
                const idx = self.item_of.items[lv];
                assert(idx != none);
                self.items.items[@intCast(idx)].prep_inserted = false;
            },
            .del => {
                const idx = self.target_of.items[lv];
                assert(idx != none);
                self.items.items[@intCast(idx)].prep_deleted -= 1;
            },
        }
    }

    fn advance(self: *Walker, lv: Lv) void {
        switch (self.graph.opOf(lv)) {
            .ins => {
                const idx = self.item_of.items[lv];
                assert(idx != none);
                self.items.items[@intCast(idx)].prep_inserted = true;
            },
            .del => {
                const idx = self.target_of.items[lv];
                assert(idx != none);
                self.items.items[@intCast(idx)].prep_deleted += 1;
            },
        }
    }

    /// First application of event `lv` (prepare version == its parents).
    /// `emit` is true exactly for untrusted (new remote) events, so it also
    /// selects error-instead-of-assert on bad positions.
    fn apply(self: *Walker, gpa: Allocator, lv: Lv, emit: bool) ReplayError!?ScalarEdit {
        switch (self.graph.opOf(lv)) {
            .ins => |ins| {
                const anchors = try self.findInsertAnchors(ins.pos, emit);
                const idx = try self.integrate(gpa, lv, anchors.left, anchors.right);
                self.item_of.items[lv] = idx;
                const it = &self.items.items[@intCast(idx)];
                it.prep_inserted = true;
                it.effect_visible = true;
                if (!emit) return null;
                return .{ .ins = .{ .pos = self.effectPositionOf(idx), .ch = ins.ch } };
            },
            .del => |pos| {
                const idx = try self.findPrepVisibleAt(pos, emit);
                self.target_of.items[lv] = idx;
                const it = &self.items.items[@intCast(idx)];
                it.prep_deleted += 1;
                if (!it.effect_visible) return null; // concurrently deleted
                const effect_pos = self.effectPositionOf(idx);
                it.effect_visible = false;
                if (!emit) return null;
                return .{ .del = effect_pos };
            },
        }
    }

    /// (origin_left, origin_right) for an insert at prepare position `pos`:
    /// the prepare-visible item before the position, and the first
    /// prepare-visible item at/after it. An out-of-range position from an
    /// untrusted event is `error.Corrupt`; from local history it is a bug.
    fn findInsertAnchors(self: *const Walker, pos: u64, untrusted: bool) error{Corrupt}!struct { left: i32, right: i32 } {
        var left: i32 = none;
        var seen: u64 = 0;
        var cur = self.head;
        while (cur != none) {
            const it = &self.items.items[@intCast(cur)];
            if (it.prepVisible()) {
                if (seen == pos) return .{ .left = left, .right = cur };
                seen += 1;
                left = cur;
            }
            cur = it.next;
        }
        if (seen != pos) {
            if (untrusted) return error.Corrupt;
            unreachable; // local history references a position that must exist
        }
        return .{ .left = left, .right = none };
    }

    /// Arena index of the `pos`-th prepare-visible item.
    fn findPrepVisibleAt(self: *const Walker, pos: u64, untrusted: bool) error{Corrupt}!i32 {
        var seen: u64 = 0;
        var cur = self.head;
        while (cur != none) {
            const it = &self.items.items[@intCast(cur)];
            if (it.prepVisible()) {
                if (seen == pos) return cur;
                seen += 1;
            }
            cur = it.next;
        }
        if (untrusted) return error.Corrupt;
        unreachable; // local history deleted a position that must exist
    }

    /// Count of effect-visible items strictly before arena index `idx`.
    fn effectPositionOf(self: *const Walker, idx: i32) u64 {
        var n: u64 = 0;
        var cur = self.head;
        while (cur != idx) {
            const it = &self.items.items[@intCast(cur)];
            if (it.effect_visible) n += 1;
            cur = it.next;
        }
        return n;
    }

    fn nextOf(self: *const Walker, idx: i32) i32 {
        return if (idx == none) self.head else self.items.items[@intCast(idx)].next;
    }

    /// Total order on concurrent inserts: the Yjs/YATA integration rule,
    /// with the globally-stable agent *name* as the final tiebreak (local
    /// AgentId numbering differs between replicas).
    fn agentNameOf(self: *const Walker, lv: Lv) []const u8 {
        return self.graph.agentName(self.graph.idOf(lv).agent);
    }

    /// Insert a new item for `lv` between `origin_left` and `origin_right`,
    /// resolving concurrent siblings. Returns the arena index.
    fn integrate(self: *Walker, gpa: Allocator, lv: Lv, origin_left: i32, origin_right: i32) Allocator.Error!i32 {
        const idx: i32 = @intCast(self.items.items.len);
        try self.items.append(gpa, .{
            .lv = lv,
            .origin_left = origin_left,
            .origin_right = origin_right,
            .next = none,
            .prep_inserted = false,
            .prep_deleted = 0,
            .effect_visible = false,
        });

        // Scan the conflict window (Yjs Item.integrate, ported to indices).
        var before_origin: std.ArrayList(i32) = .empty;
        defer before_origin.deinit(gpa);
        var conflicting: std.ArrayList(i32) = .empty;
        defer conflicting.deinit(gpa);
        const contains = struct {
            fn contains(list: []const i32, v: i32) bool {
                return std.mem.indexOfScalar(i32, list, v) != null;
            }
        }.contains;

        var left = origin_left;
        var o = self.nextOf(origin_left);
        while (o != none and o != origin_right) {
            try before_origin.append(gpa, o);
            try conflicting.append(gpa, o);
            const oi = &self.items.items[@intCast(o)];
            if (oi.origin_left == origin_left) {
                // Same left origin: agent name decides. Equal agents can
                // only be the author's own sequential (non-concurrent)
                // items; they fall through to the right-origin rule.
                const cmp = std.mem.order(u8, self.agentNameOf(oi.lv), self.agentNameOf(lv));
                if (cmp == .lt) {
                    left = o;
                    conflicting.clearRetainingCapacity();
                } else if (oi.origin_right == origin_right) {
                    break;
                }
            } else if (oi.origin_left != none and contains(before_origin.items, oi.origin_left)) {
                if (!contains(conflicting.items, oi.origin_left)) {
                    left = o;
                    conflicting.clearRetainingCapacity();
                }
            } else {
                break;
            }
            o = self.items.items[@intCast(o)].next;
        }

        // Link after `left`.
        if (left == none) {
            self.items.items[@intCast(idx)].next = self.head;
            self.head = idx;
        } else {
            const l = &self.items.items[@intCast(left)];
            self.items.items[@intCast(idx)].next = l.next;
            l.next = idx;
        }
        return idx;
    }
};

test {
    std.testing.refAllDecls(@This());
}
