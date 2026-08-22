//! Transient replay state for ObjectDoc: the eg-walker prepare/effect
//! discipline generalized to a *tree* of objects — multi-value map
//! registers plus per-object sequences. All sequence ordering comes from
//! the shared FugueMax engine in sequence.zig; this file owns only what is
//! object-specific: per-object dispatch, the MV register semantics, and
//! kill-list bookkeeping for map writes.
//!
//! Map semantics are multi-value ("MV"): a set overwrites exactly the
//! values its author could see (prepare-visible); concurrent sets to the
//! same key all survive as siblings — the honest option in a clockless
//! system. Readers get a deterministic winner plus the full conflict set.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const Sequence = @import("Sequence.zig");
const seq_walker = @import("SeqWalker.zig");
// Each object's SeqWalker sits over a `Lv` space shared with every other
// object and every map register in the tree — a dense array would cost
// memory proportional to the WHOLE document, once per object (see
// `SeqWalker.Storage`). `pub`: `ObjectDoc`'s identity-anchor machinery
// (`objectAnchorAt`/`resolveObjectAnchors`) needs to name this exact type
// to pull one object's already-replayed `SeqWalker` out of `Walker.seqs`.
pub const SeqWalker = seq_walker.SeqWalker(.sparse);
const Lv = causal.Lv;
const EventId = causal.EventId;
const none = Sequence.none;

/// String reference into the owning ObjectDoc's append-only arena.
pub const Str = struct { start: u32, len: u32 };

/// Object identity: the event that created it. `null` obj refs = the root
/// map. DOC-LOCAL (embeds replica-local agent numbering); see
/// ObjectDoc.exportId for portable references.
pub const ObjId = EventId;

pub const ValPayload = union(enum) {
    null_,
    bool_: bool,
    int: i64,
    float: f64,
    str: Str,
    new_map,
    new_list,
    new_text,
};

pub const ObjectOp = union(enum) {
    map_set: struct { obj: ?ObjId, key: Str, val: ValPayload },
    map_del: struct { obj: ?ObjId, key: Str },
    list_ins: struct { obj: ObjId, pos: u64, val: ValPayload },
    list_del: struct { obj: ObjId, pos: u64 },
    text_ins: struct { obj: ObjId, pos: u64, ch: u21 },
    text_del: struct { obj: ObjId, pos: u64 },
};

pub const Graph = causal.EventGraph(ObjectOp);

/// Effects emitted for NEW events during replay, in application order.
/// Positions are indices/scalars valid against the state produced by all
/// previously emitted effects.
pub const Effect = union(enum) {
    map_add: struct { obj: ?Lv, key: Str, set_lv: Lv, val: ValPayload },
    map_remove: struct { obj: ?Lv, key: Str, set_lv: Lv },
    list_ins: struct { obj: Lv, index: u64, lv: Lv, val: ValPayload },
    list_del: struct { obj: Lv, index: u64 },
    text_ins: struct { obj: Lv, pos: u64, ch: u21 },
    text_del: struct { obj: Lv, pos: u64 },
};

// ── Per-(object, key) register state ────────────────────────────────────

const RegItem = struct {
    lv: Lv, // the map_set event
    prep_inserted: bool,
    prep_overwritten: u32,
    effect_active: bool,

    fn prepVisible(it: *const RegItem) bool {
        return it.prep_inserted and it.prep_overwritten == 0;
    }
};

const KeySlot = struct {
    key: Str,
    items: std.ArrayList(RegItem) = .empty,
};

const MapReg = struct {
    slots: std.ArrayList(KeySlot) = .empty,

    fn deinit(self: *MapReg, gpa: Allocator) void {
        for (self.slots.items) |*s| s.items.deinit(gpa);
        self.slots.deinit(gpa);
    }
};

// ── The replay state ────────────────────────────────────────────────────

pub const Walker = struct {
    graph: *const Graph,
    strings: []const u8,

    /// Object-lv (or `root_key`) → sequence / register state. Each
    /// `SeqWalker` owns its own object's retreat/advance/apply
    /// bookkeeping (see `SeqWalker.zig`) — the shared discipline `Walker`
    /// also drives, per object, alongside the map-register logic below.
    seqs: std.AutoHashMapUnmanaged(Lv, SeqWalker) = .empty,
    maps: std.AutoHashMapUnmanaged(Lv, MapReg) = .empty,
    /// map_set events: index of their own RegItem within the key slot.
    reg_pos_of: std.ArrayList(i32) = .empty,
    /// Kill lists for map_set/map_del: spans into `kills_pool` of RegItem
    /// indices (within the event's key slot) it overwrote.
    kills_start: std.ArrayList(u32) = .empty,
    kills_len: std.ArrayList(u32) = .empty,
    kills_pool: std.ArrayList(u32) = .empty,
    prep_frontier: std.ArrayList(Lv) = .empty,

    pub const root_key: Lv = std.math.maxInt(Lv);

    pub const ReplayError = Allocator.Error || error{Corrupt};

    const Names = struct {
        g: *const Graph,
        pub fn agentNameOf(self: @This(), lv: Lv) []const u8 {
            return self.g.agentName(self.g.idOf(lv).agent);
        }
    };

    pub fn init(graph: *const Graph, strings: []const u8) Walker {
        return .{ .graph = graph, .strings = strings };
    }

    pub fn deinit(self: *Walker, gpa: Allocator) void {
        var sit = self.seqs.valueIterator();
        while (sit.next()) |s| s.deinit(gpa);
        self.seqs.deinit(gpa);
        var mit = self.maps.valueIterator();
        while (mit.next()) |m| m.deinit(gpa);
        self.maps.deinit(gpa);
        self.reg_pos_of.deinit(gpa);
        self.kills_start.deinit(gpa);
        self.kills_len.deinit(gpa);
        self.kills_pool.deinit(gpa);
        self.prep_frontier.deinit(gpa);
    }

    fn str(self: *const Walker, s: Str) []const u8 {
        return self.strings[s.start..][0..s.len];
    }

    /// Resolve an op's object reference to its creation Lv (or root_key),
    /// checking existence, causal order, and kind. `expect`: 0 map, 1 list,
    /// 2 text.
    fn resolveObj(self: *const Walker, ref: ?ObjId, at_lv: Lv, comptime expect: u2, untrusted: bool) error{Corrupt}!Lv {
        const id = ref orelse {
            if (expect != 0) {
                if (untrusted) return error.Corrupt;
                unreachable;
            }
            return root_key;
        };
        const creation = self.graph.lvOf(id) orelse {
            if (untrusted) return error.Corrupt;
            unreachable;
        };
        if (creation >= at_lv) {
            if (untrusted) return error.Corrupt;
            unreachable;
        }
        const val: ?ValPayload = switch (self.graph.opOf(creation)) {
            .map_set => |m| m.val,
            .list_ins => |l| l.val,
            else => null,
        };
        const ok = if (val) |v| switch (v) {
            .new_map => expect == 0,
            .new_list => expect == 1,
            .new_text => expect == 2,
            else => false,
        } else false;
        if (!ok) {
            if (untrusted) return error.Corrupt;
            unreachable;
        }
        return creation;
    }

    /// Replay the whole graph in Lv order; events at `lv >= first_new`
    /// emit effects.
    pub fn replayAll(self: *Walker, gpa: Allocator, first_new: Lv, out: *std.ArrayList(Effect)) ReplayError!void {
        const n = self.graph.eventCount();
        try self.reg_pos_of.appendNTimes(gpa, none, n);
        try self.kills_start.appendNTimes(gpa, 0, n);
        try self.kills_len.appendNTimes(gpa, 0, n);
        for (0..n) |lv_usize| {
            const lv: Lv = @intCast(lv_usize);
            try seq_walker.movePrepareTo(gpa, self.graph, &self.prep_frontier, self.graph.parentsOf(lv), self);
            try self.apply(gpa, lv, lv >= first_new, out);
            self.prep_frontier.clearRetainingCapacity();
            try self.prep_frontier.append(gpa, lv);
        }
    }

    /// `SeqWalker.movePrepareTo`'s per-event callback: advance (`on =
    /// true`) or retreat an already-applied event's prepare effect.
    /// Sequence-shaped ops (list/text ins/del) delegate to the target
    /// object's own `SeqWalker`; map ops are Walker's own (registers
    /// aren't sequences, no `SeqWalker` involvement).
    pub fn toggle(self: *Walker, history: *const Graph, lv: Lv, on: bool) void {
        switch (history.opOf(lv)) {
            .list_ins, .text_ins => self.seqs.getPtr(self.appliedSeqObj(lv)).?.toggleInsert(lv, on),
            .list_del, .text_del => self.seqs.getPtr(self.appliedSeqObj(lv)).?.toggleDelete(lv, on),
            .map_set, .map_del => {
                const info = self.appliedMapSlot(lv);
                const slot = &self.maps.getPtr(info.obj).?.slots.items[info.slot];
                if (history.opOf(lv) == .map_set) {
                    const rp = self.reg_pos_of.items[lv];
                    assert(rp != none);
                    slot.items.items[@intCast(rp)].prep_inserted = on;
                }
                const ks = self.kills_start.items[lv];
                const kl = self.kills_len.items[lv];
                for (self.kills_pool.items[ks..][0..kl]) |victim| {
                    const v = &slot.items.items[victim];
                    if (on) v.prep_overwritten += 1 else v.prep_overwritten -= 1;
                }
            },
        }
    }

    fn appliedSeqObj(self: *const Walker, lv: Lv) Lv {
        const ref: ?ObjId = switch (self.graph.opOf(lv)) {
            .list_ins => |o| o.obj,
            .list_del => |o| o.obj,
            .text_ins => |o| o.obj,
            .text_del => |o| o.obj,
            else => unreachable,
        };
        return self.graph.lvOf(ref.?).?;
    }

    fn appliedMapSlot(self: *const Walker, lv: Lv) struct { obj: Lv, slot: usize } {
        const op = self.graph.opOf(lv);
        const ref: ?ObjId = switch (op) {
            .map_set => |m| m.obj,
            .map_del => |m| m.obj,
            else => unreachable,
        };
        const key: Str = switch (op) {
            .map_set => |m| m.key,
            .map_del => |m| m.key,
            else => unreachable,
        };
        const obj = if (ref) |id| self.graph.lvOf(id).? else root_key;
        const reg = self.maps.getPtr(obj).?;
        for (reg.slots.items, 0..) |s, i| {
            if (std.mem.eql(u8, self.str(s.key), self.str(key))) return .{ .obj = obj, .slot = i };
        }
        unreachable;
    }

    fn getSeqWalker(self: *Walker, gpa: Allocator, obj: Lv) Allocator.Error!*SeqWalker {
        const gop = try self.seqs.getOrPut(gpa, obj);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        return gop.value_ptr;
    }

    fn getSlot(self: *Walker, gpa: Allocator, obj: Lv, key: Str) Allocator.Error!*KeySlot {
        const gop = try self.maps.getOrPut(gpa, obj);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        const reg = gop.value_ptr;
        for (reg.slots.items) |*s| {
            if (std.mem.eql(u8, self.str(s.key), self.str(key))) return s;
        }
        try reg.slots.append(gpa, .{ .key = key });
        return &reg.slots.items[reg.slots.items.len - 1];
    }

    fn apply(self: *Walker, gpa: Allocator, lv: Lv, emit: bool, out: *std.ArrayList(Effect)) ReplayError!void {
        const names: Names = .{ .g = self.graph };
        switch (self.graph.opOf(lv)) {
            .map_set => |m| {
                const obj = try self.resolveObj(m.obj, lv, 0, emit);
                const slot = try self.getSlot(gpa, obj, m.key);
                try self.registerWrite(gpa, lv, obj, slot, m.key, emit, out);
                self.reg_pos_of.items[lv] = @intCast(slot.items.items.len);
                try slot.items.append(gpa, .{
                    .lv = lv,
                    .prep_inserted = true,
                    .prep_overwritten = 0,
                    .effect_active = true,
                });
                if (emit) {
                    const eobj: ?Lv = if (obj == root_key) null else obj;
                    try out.append(gpa, .{ .map_add = .{ .obj = eobj, .key = m.key, .set_lv = lv, .val = m.val } });
                }
            },
            .map_del => |m| {
                const obj = try self.resolveObj(m.obj, lv, 0, emit);
                const slot = try self.getSlot(gpa, obj, m.key);
                try self.registerWrite(gpa, lv, obj, slot, m.key, emit, out);
            },
            .list_ins => |l| {
                const obj = try self.resolveObj(l.obj, lv, 1, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyInsert(gpa, names, lv, l.pos, emit);
                if (emit) {
                    try out.append(gpa, .{ .list_ins = .{ .obj = obj, .index = res.effect_pos, .lv = lv, .val = l.val } });
                }
            },
            .list_del => |l| {
                const obj = try self.resolveObj(l.obj, lv, 1, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyDelete(gpa, lv, l.pos, emit);
                if (emit) {
                    if (res.effect_pos) |index| try out.append(gpa, .{ .list_del = .{ .obj = obj, .index = index } });
                }
            },
            .text_ins => |x| {
                const obj = try self.resolveObj(x.obj, lv, 2, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyInsert(gpa, names, lv, x.pos, emit);
                if (emit) {
                    try out.append(gpa, .{ .text_ins = .{ .obj = obj, .pos = res.effect_pos, .ch = x.ch } });
                }
            },
            .text_del => |x| {
                const obj = try self.resolveObj(x.obj, lv, 2, emit);
                const sw = try self.getSeqWalker(gpa, obj);
                const res = try sw.applyDelete(gpa, lv, x.pos, emit);
                if (emit) {
                    if (res.effect_pos) |pos| try out.append(gpa, .{ .text_del = .{ .obj = obj, .pos = pos } });
                }
            },
        }
    }

    /// Shared map_set/map_del front half: overwrite the prepare-visible
    /// values, recording the kill list and emitting removals.
    fn registerWrite(
        self: *Walker,
        gpa: Allocator,
        lv: Lv,
        obj: Lv,
        slot: *KeySlot,
        key: Str,
        emit: bool,
        out: *std.ArrayList(Effect),
    ) ReplayError!void {
        self.kills_start.items[lv] = @intCast(self.kills_pool.items.len);
        var killed: u32 = 0;
        for (slot.items.items, 0..) |*it, i| {
            if (!it.prepVisible()) continue;
            try self.kills_pool.append(gpa, @intCast(i));
            killed += 1;
            it.prep_overwritten += 1;
            if (it.effect_active) {
                it.effect_active = false;
                if (emit) {
                    const eobj: ?Lv = if (obj == root_key) null else obj;
                    try out.append(gpa, .{ .map_remove = .{ .obj = eobj, .key = key, .set_lv = it.lv } });
                }
            }
        }
        self.kills_len.items[lv] = killed;
    }
};

test {
    std.testing.refAllDecls(@This());
}
