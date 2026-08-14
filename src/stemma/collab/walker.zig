//! The event-graph walker for text: replays the causal graph, reconstructing
//! transient CRDT state to transform concurrent remote operations into edits
//! that apply cleanly to the local document. This is the eg-walker family
//! (Gentle & Kleppmann, EuroSys 2025): CRDT state exists only during a
//! merge; the durable artifacts are the event graph and the plain document.
//!
//! Ordering: the YjsMod integration rule (josephg's reference eg-walker
//! implementation), which is order-equivalent to **FugueMax** (Weidner &
//! Kleppmann, "The Art of the Fugue") — maximally non-interleaving in both
//! directions. The final tiebreak is the globally-stable agent *name*
//! (replica-local AgentId numbering never decides order). This rule is part
//! of the wire contract: changing it after replicas exist would silently
//! diverge them.
//!
//! Two versions are tracked during replay:
//! - the *prepare* version: what the event's author could see when they made
//!   it (moved around the graph with retreat/advance);
//! - the *effect* state: what the merged output document contains so far.
//! Events already applied locally replay silently (state only); new remote
//! events additionally emit transformed scalar-space edits.
//!
//! Documents may carry a compacted *base*: placeholder items representing
//! frozen pre-history. Base items are prepare- and effect-visible from the
//! start and are never retreated (their events no longer exist); every live
//! event is causally after the base by the compaction rule.
//!
//! v1 remains correctness-first where it doesn't change the contract:
//! replay from genesis/base, unit items, array state with O(n) splices —
//! the benchmarked-optimization ladder lives in BENCHMARKS.md.

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
/// Sentinel `lv` for compacted base placeholder items.
pub const base_lv: Lv = std.math.maxInt(Lv);

const Item = struct {
    lv: Lv, // `base_lv` for base placeholders
    /// Arena index of the item this was inserted after (or `none` = start),
    /// as seen at the author's prepare version. Structural; never changes.
    origin_left: i32,
    /// Arena index of the item that was immediately right at insert time
    /// (or `none` = end).
    origin_right: i32,
    prep_inserted: bool,
    prep_deleted: u32,
    effect_visible: bool,

    fn prepVisible(it: *const Item) bool {
        return it.prep_inserted and it.prep_deleted == 0;
    }
};

pub const Walker = struct {
    graph: *const Graph,
    /// Append-only item arena; arena indices are stable identities.
    items: std.ArrayList(Item) = .empty,
    /// Document order: arena indices, including tombstones.
    seq: std.ArrayList(u32) = .empty,
    /// arena index → position in `seq` (maintained across splices).
    pos_of: std.ArrayList(u32) = .empty,
    /// lv → arena index of the item it inserted (`none` for deletes).
    item_of: std.ArrayList(i32) = .empty,
    /// lv → arena index of the item it deleted (`none` for inserts; filled
    /// on first application, reused by retreat/advance).
    target_of: std.ArrayList(i32) = .empty,
    prep_frontier: std.ArrayList(Lv) = .empty,
    base_count: usize = 0,

    pub fn init(graph: *const Graph) Walker {
        return .{ .graph = graph };
    }

    pub fn deinit(self: *Walker, gpa: Allocator) void {
        self.items.deinit(gpa);
        self.seq.deinit(gpa);
        self.pos_of.deinit(gpa);
        self.item_of.deinit(gpa);
        self.target_of.deinit(gpa);
        self.prep_frontier.deinit(gpa);
    }

    /// Install `count` compacted-base placeholder items (visible, chained
    /// origins). Must be called before `replayAll`.
    pub fn initBase(self: *Walker, gpa: Allocator, count: usize) Allocator.Error!void {
        assert(self.items.items.len == 0);
        self.base_count = count;
        try self.items.ensureTotalCapacity(gpa, count);
        try self.seq.ensureTotalCapacity(gpa, count);
        try self.pos_of.ensureTotalCapacity(gpa, count);
        for (0..count) |i| {
            self.items.appendAssumeCapacity(.{
                .lv = base_lv,
                .origin_left = if (i == 0) none else @intCast(i - 1),
                .origin_right = none,
                .prep_inserted = true,
                .prep_deleted = 0,
                .effect_visible = true,
            });
            self.seq.appendAssumeCapacity(@intCast(i));
            self.pos_of.appendAssumeCapacity(@intCast(i));
        }
    }

    pub const ReplayError = Allocator.Error || error{Corrupt};

    /// Replay the graph in Lv order. Events with `lv < first_new` update
    /// state silently (their effects are already in the document); events at
    /// `lv >= first_new` append their transformed edits to `out`. If
    /// `include` is given, events with `include[lv] == false` are skipped
    /// entirely (must be causally closed). Remote events with out-of-range
    /// positions yield `error.Corrupt` — never a crash.
    pub fn replayAll(
        self: *Walker,
        gpa: Allocator,
        first_new: Lv,
        include: ?[]const bool,
        out: *std.ArrayList(ScalarEdit),
    ) ReplayError!void {
        const n = self.graph.eventCount();
        try self.item_of.appendNTimes(gpa, none, n);
        try self.target_of.appendNTimes(gpa, none, n);
        for (0..n) |lv_usize| {
            const lv: Lv = @intCast(lv_usize);
            if (include) |inc| if (!inc[lv_usize]) continue;
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
    /// the prepare-visible item before the position, and the item
    /// immediately after that (visible or not, per YjsMod: the right origin
    /// is the next prepare-visible item; `none` = end). Out-of-range from an
    /// untrusted event is `error.Corrupt`; from local history it is a bug.
    fn findInsertAnchors(self: *const Walker, pos: u64, untrusted: bool) error{Corrupt}!struct { left: i32, right: i32 } {
        var left: i32 = none;
        var seen: u64 = 0;
        for (self.seq.items) |arena| {
            const it = &self.items.items[arena];
            if (it.prepVisible()) {
                if (seen == pos) return .{ .left = left, .right = @intCast(arena) };
                seen += 1;
                left = @intCast(arena);
            }
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
        for (self.seq.items) |arena| {
            const it = &self.items.items[arena];
            if (it.prepVisible()) {
                if (seen == pos) return @intCast(arena);
                seen += 1;
            }
        }
        if (untrusted) return error.Corrupt;
        unreachable; // local history deleted a position that must exist
    }

    /// Count of effect-visible items strictly before arena index `idx`.
    fn effectPositionOf(self: *const Walker, idx: i32) u64 {
        const end = self.pos_of.items[@intCast(idx)];
        var n: u64 = 0;
        for (self.seq.items[0..end]) |arena| {
            if (self.items.items[arena].effect_visible) n += 1;
        }
        return n;
    }

    /// Position of an origin in `seq` for order comparison: `none` (left
    /// end) sorts before everything; a right-origin `none` is the sequence
    /// length (right end).
    fn leftIdx(self: *const Walker, arena: i32) i64 {
        return if (arena == none) -1 else self.pos_of.items[@intCast(arena)];
    }

    fn rightIdx(self: *const Walker, arena: i32) i64 {
        return if (arena == none) @intCast(self.seq.items.len) else self.pos_of.items[@intCast(arena)];
    }

    /// Agent name for ordering ties. Base placeholders sort as the empty
    /// name: they are causally before everything live, and the rule only
    /// needs to be deterministic and identical across replicas.
    fn agentNameOf(self: *const Walker, item_lv: Lv) []const u8 {
        if (item_lv == base_lv) return "";
        return self.graph.agentName(self.graph.idOf(item_lv).agent);
    }

    /// The YjsMod integration loop (order-equivalent to FugueMax): find the
    /// insertion index for `lv` between its origins, resolving concurrent
    /// siblings; splice a new item there. Returns the arena index.
    fn integrate(self: *Walker, gpa: Allocator, lv: Lv, origin_left: i32, origin_right: i32) Allocator.Error!i32 {
        const left_pos = self.leftIdx(origin_left);
        const right_pos = self.rightIdx(origin_right);
        const seq_len: i64 = @intCast(self.seq.items.len);

        var dest: i64 = left_pos + 1;
        var scanning = false;
        var i: i64 = left_pos + 1;
        while (true) : (i += 1) {
            if (!scanning) dest = i;
            if (i == seq_len or i == right_pos) break;
            const o = &self.items.items[self.seq.items[@intCast(i)]];
            const o_left = self.leftIdx(o.origin_left);
            const o_right = self.rightIdx(o.origin_right);
            if (o_left < left_pos) break;
            if (o_left == left_pos) {
                if (o_right < right_pos) {
                    scanning = true;
                } else if (o_right == right_pos) {
                    // Deterministic global tiebreak: agent name.
                    const my_name = self.agentNameOf(lv);
                    const o_name = self.agentNameOf(o.lv);
                    if (std.mem.order(u8, my_name, o_name) == .lt) break;
                    scanning = false;
                } else {
                    scanning = false;
                }
            }
            // o_left > left_pos: keep walking without touching flags.
        }

        // Splice the new item at `dest`.
        const arena: u32 = @intCast(self.items.items.len);
        try self.items.append(gpa, .{
            .lv = lv,
            .origin_left = origin_left,
            .origin_right = origin_right,
            .prep_inserted = false,
            .prep_deleted = 0,
            .effect_visible = false,
        });
        errdefer _ = self.items.pop();
        const at: usize = @intCast(dest);
        try self.seq.insert(gpa, at, arena);
        try self.pos_of.append(gpa, @intCast(at));
        for (self.seq.items[at + 1 ..]) |shifted| {
            self.pos_of.items[shifted] += 1;
        }
        return @intCast(arena);
    }

    /// After a (fully silent) replay: iterate live content in document
    /// order, yielding (arena, lv) per visible item. Base placeholders have
    /// `lv == base_lv`.
    pub const AliveIterator = struct {
        walker: *const Walker,
        i: usize = 0,

        pub fn next(self: *AliveIterator) ?struct { arena: u32, lv: Lv } {
            while (self.i < self.walker.seq.items.len) {
                const arena = self.walker.seq.items[self.i];
                self.i += 1;
                const it = &self.walker.items.items[arena];
                if (it.effect_visible) return .{ .arena = arena, .lv = it.lv };
            }
            return null;
        }
    };

    pub fn aliveIterator(self: *const Walker) AliveIterator {
        return .{ .walker = self };
    }
};

test {
    std.testing.refAllDecls(@This());
}
