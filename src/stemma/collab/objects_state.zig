//! Replay engine for JSON documents: the same eg-walker prepare/effect
//! discipline as the text walker, generalized to a *tree* of objects —
//! multi-value map registers plus per-object FugueMax sequences (lists and
//! text share the sequence machinery; a list element is just an item whose
//! payload is a value instead of a scalar).
//!
//! Map semantics are multi-value ("MV"): a set overwrites exactly the
//! values its author could see (prepare-visible); concurrent sets to the
//! same key all survive as siblings — the honest option in a clockless
//! system. Readers get a deterministic winner plus the full conflict set.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const Lv = causal.Lv;
const EventId = causal.EventId;

/// String reference into the owning JsonDoc's append-only arena.
pub const Str = struct { start: u32, len: u32 };

/// Object identity: the event that created it. `null` obj refs = the root map.
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

const none: i32 = -1;

// ── Per-object sequence state (lists and text) ──────────────────────────

const SeqItem = struct {
    lv: Lv,
    origin_left: i32,
    origin_right: i32,
    prep_inserted: bool,
    prep_deleted: u32,
    effect_visible: bool,

    fn prepVisible(it: *const SeqItem) bool {
        return it.prep_inserted and it.prep_deleted == 0;
    }
};

const Sequence = struct {
    items: std.ArrayList(SeqItem) = .empty,
    seq: std.ArrayList(u32) = .empty,
    pos_of: std.ArrayList(u32) = .empty,

    fn deinit(self: *Sequence, gpa: Allocator) void {
        self.items.deinit(gpa);
        self.seq.deinit(gpa);
        self.pos_of.deinit(gpa);
    }
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

// ── The walker ──────────────────────────────────────────────────────────

pub const Walker = struct {
    graph: *const Graph,
    strings: []const u8,

    /// Object-lv (or root sentinel) → state. Root map uses key `root_key`.
    seqs: std.AutoHashMapUnmanaged(Lv, Sequence) = .empty,
    maps: std.AutoHashMapUnmanaged(Lv, MapReg) = .empty,
    /// Per-event bookkeeping for retreat/advance.
    /// Inserts (seq): item arena index. Deletes (seq): target arena index.
    /// map_set: own RegItem index (hi 32) unused; see reg_pos.
    seq_slot_of: std.ArrayList(i32) = .empty,
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
        self.seq_slot_of.deinit(gpa);
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
    /// emit effects. See the text walker for the shared discipline.
    pub fn replayAll(self: *Walker, gpa: Allocator, first_new: Lv, out: *std.ArrayList(Effect)) ReplayError!void {
        const n = self.graph.eventCount();
        try self.seq_slot_of.appendNTimes(gpa, none, n);
        try self.reg_pos_of.appendNTimes(gpa, none, n);
        try self.kills_start.appendNTimes(gpa, 0, n);
        try self.kills_len.appendNTimes(gpa, 0, n);
        for (0..n) |lv_usize| {
            const lv: Lv = @intCast(lv_usize);
            try self.movePrepareTo(gpa, self.graph.parentsOf(lv));
            try self.apply(gpa, lv, lv >= first_new, out);
            self.prep_frontier.clearRetainingCapacity();
            try self.prep_frontier.append(gpa, lv);
        }
    }

    fn movePrepareTo(self: *Walker, gpa: Allocator, target: []const Lv) Allocator.Error!void {
        if (std.mem.eql(Lv, self.prep_frontier.items, target)) return;
        var d = try self.graph.diff(gpa, self.prep_frontier.items, target);
        defer d.deinit(gpa);
        var i = d.a_only.items.len;
        while (i > 0) {
            i -= 1;
            self.toggle(d.a_only.items[i], false);
        }
        for (d.b_only.items) |lv| self.toggle(lv, true);
        self.prep_frontier.clearRetainingCapacity();
        try self.prep_frontier.appendSlice(gpa, target);
    }

    /// Advance (`on = true`) or retreat an already-applied event's prepare
    /// effect.
    fn toggle(self: *Walker, lv: Lv, on: bool) void {
        switch (self.graph.opOf(lv)) {
            .list_ins, .text_ins => {
                const obj = self.appliedSeqObj(lv);
                const s = self.seqs.getPtr(obj).?;
                const idx = self.seq_slot_of.items[lv];
                assert(idx != none);
                s.items.items[@intCast(idx)].prep_inserted = on;
            },
            .list_del, .text_del => {
                const obj = self.appliedSeqObj(lv);
                const s = self.seqs.getPtr(obj).?;
                const idx = self.seq_slot_of.items[lv];
                assert(idx != none);
                const it = &s.items.items[@intCast(idx)];
                if (on) it.prep_deleted += 1 else it.prep_deleted -= 1;
            },
            .map_set, .map_del => {
                const info = self.appliedMapSlot(lv);
                const slot = &self.maps.getPtr(info.obj).?.slots.items[info.slot];
                if (self.graph.opOf(lv) == .map_set) {
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
        // Reconstructed cheaply: the op's object ref resolves without checks
        // for already-applied (trusted) events.
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

    fn getSeq(self: *Walker, gpa: Allocator, obj: Lv) Allocator.Error!*Sequence {
        const gop = try self.seqs.getOrPut(gpa, obj);
        if (!gop.found_existing) gop.value_ptr.* = .{};
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
        switch (self.graph.opOf(lv)) {
            .map_set => |m| {
                const obj = try self.resolveObj(m.obj, lv, 0, emit);
                const slot = try self.getSlot(gpa, obj, m.key);
                try self.registerWrite(gpa, lv, obj, slot, m.key, emit, out);
                // Add our own value item.
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
                const s = try self.getSeq(gpa, obj);
                const idx = try seqInsert(self, gpa, s, lv, l.pos, emit);
                if (emit) {
                    try out.append(gpa, .{ .list_ins = .{
                        .obj = obj,
                        .index = effectPosOf(s, idx),
                        .lv = lv,
                        .val = l.val,
                    } });
                }
                s.items.items[@intCast(idx)].effect_visible = true;
            },
            .list_del => |l| {
                const obj = try self.resolveObj(l.obj, lv, 1, emit);
                const s = try self.getSeq(gpa, obj);
                if (try self.seqDelete(s, lv, l.pos, emit)) |index| {
                    if (emit) try out.append(gpa, .{ .list_del = .{ .obj = obj, .index = index } });
                }
            },
            .text_ins => |x| {
                const obj = try self.resolveObj(x.obj, lv, 2, emit);
                const s = try self.getSeq(gpa, obj);
                const idx = try seqInsert(self, gpa, s, lv, x.pos, emit);
                if (emit) {
                    try out.append(gpa, .{ .text_ins = .{
                        .obj = obj,
                        .pos = effectPosOf(s, idx),
                        .ch = x.ch,
                    } });
                }
                s.items.items[@intCast(idx)].effect_visible = true;
            },
            .text_del => |x| {
                const obj = try self.resolveObj(x.obj, lv, 2, emit);
                const s = try self.getSeq(gpa, obj);
                if (try self.seqDelete(s, lv, x.pos, emit)) |pos| {
                    if (emit) try out.append(gpa, .{ .text_del = .{ .obj = obj, .pos = pos } });
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

    fn seqInsert(self: *Walker, gpa: Allocator, s: *Sequence, lv: Lv, pos: u64, untrusted: bool) ReplayError!i32 {
        // Anchors.
        var left: i32 = none;
        var right: i32 = none;
        {
            var seen: u64 = 0;
            var found = false;
            for (s.seq.items) |arena| {
                const it = &s.items.items[arena];
                if (it.prepVisible()) {
                    if (seen == pos) {
                        right = @intCast(arena);
                        found = true;
                        break;
                    }
                    seen += 1;
                    left = @intCast(arena);
                }
            }
            if (!found and seen != pos) {
                if (untrusted) return error.Corrupt;
                unreachable;
            }
        }
        return integrate(self, gpa, s, lv, left, right);
    }

    fn seqDelete(self: *Walker, s: *Sequence, lv: Lv, pos: u64, untrusted: bool) error{Corrupt}!?u64 {
        var seen: u64 = 0;
        for (s.seq.items) |arena| {
            const it = &s.items.items[arena];
            if (it.prepVisible()) {
                if (seen == pos) {
                    self.seq_slot_of.items[lv] = @intCast(arena);
                    it.prep_deleted += 1;
                    if (!it.effect_visible) return null; // concurrently deleted
                    const index = effectPosOf(s, @intCast(arena));
                    it.effect_visible = false;
                    return index;
                }
                seen += 1;
            }
        }
        if (untrusted) return error.Corrupt;
        unreachable;
    }

    fn effectPosOf(s: *const Sequence, idx: i32) u64 {
        const end = s.pos_of.items[@intCast(idx)];
        var n: u64 = 0;
        for (s.seq.items[0..end]) |arena| {
            if (s.items.items[arena].effect_visible) n += 1;
        }
        return n;
    }

    fn agentNameOf(self: *const Walker, lv: Lv) []const u8 {
        return self.graph.agentName(self.graph.idOf(lv).agent);
    }

    /// The YjsMod / FugueMax integration loop — identical discipline to the
    /// text walker's (see walker.zig; unification is a ledgered cleanup).
    fn integrate(self: *Walker, gpa: Allocator, s: *Sequence, lv: Lv, origin_left: i32, origin_right: i32) Allocator.Error!i32 {
        const left_pos: i64 = if (origin_left == none) -1 else s.pos_of.items[@intCast(origin_left)];
        const right_pos: i64 = if (origin_right == none) @intCast(s.seq.items.len) else s.pos_of.items[@intCast(origin_right)];
        const seq_len: i64 = @intCast(s.seq.items.len);

        var dest: i64 = left_pos + 1;
        var scanning = false;
        var i: i64 = left_pos + 1;
        while (true) : (i += 1) {
            if (!scanning) dest = i;
            if (i == seq_len or i == right_pos) break;
            const o = &s.items.items[s.seq.items[@intCast(i)]];
            const o_left: i64 = if (o.origin_left == none) -1 else s.pos_of.items[@intCast(o.origin_left)];
            const o_right: i64 = if (o.origin_right == none) @intCast(s.seq.items.len) else s.pos_of.items[@intCast(o.origin_right)];
            if (o_left < left_pos) break;
            if (o_left == left_pos) {
                if (o_right < right_pos) {
                    scanning = true;
                } else if (o_right == right_pos) {
                    if (std.mem.order(u8, self.agentNameOf(lv), self.agentNameOf(o.lv)) == .lt) break;
                    scanning = false;
                } else {
                    scanning = false;
                }
            }
        }

        const arena: u32 = @intCast(s.items.items.len);
        try s.items.append(gpa, .{
            .lv = lv,
            .origin_left = origin_left,
            .origin_right = origin_right,
            .prep_inserted = true,
            .prep_deleted = 0,
            .effect_visible = false,
        });
        errdefer _ = s.items.pop();
        const at: usize = @intCast(dest);
        try s.seq.insert(gpa, at, arena);
        try s.pos_of.append(gpa, @intCast(at));
        for (s.seq.items[at + 1 ..]) |shifted| {
            s.pos_of.items[shifted] += 1;
        }
        self.seq_slot_of.items[lv] = @intCast(arena);
        return @intCast(arena);
    }
};

test {
    std.testing.refAllDecls(@This());
}
