//! The one FugueMax sequence engine. Both collaborative sequence kinds —
//! TextDoc's character sequence and ObjectDoc's lists/text nodes — share
//! this state and integration rule; there is exactly one implementation of
//! the ordering that the wire contract depends on.
//!
//! The integration loop is YjsMod (josephg's reference eg-walker
//! implementation), order-equivalent to FugueMax (Weidner & Kleppmann,
//! "The Art of the Fugue"): maximally non-interleaving in both directions.
//! The final tiebreak comes from the caller-supplied `names` context
//! (`agentNameOf(lv)`), which must be globally stable across replicas.
//!
//! Prepare/effect discipline (the eg-walker split): `prep_*` fields track
//! visibility at the moving prepare version (retreat/advance via the
//! `toggle*` primitives); `effect_visible` tracks the merged output
//! document. Positions passed in are prepare-space; positions returned are
//! effect-space.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const causal = @import("causal.zig");
const Lv = causal.Lv;

pub const none: i32 = -1;

pub const Item = struct {
    lv: Lv,
    /// Arena index of the item this was inserted after (`none` = start), as
    /// seen at the author's prepare version. Structural; never changes.
    origin_left: i32,
    /// Arena index of the item immediately right at insert time (`none` =
    /// end).
    origin_right: i32,
    prep_inserted: bool,
    prep_deleted: u32,
    effect_visible: bool,

    pub fn prepVisible(it: *const Item) bool {
        return it.prep_inserted and it.prep_deleted == 0;
    }
};

const Sequence = @This();
/// Append-only arena; arena indices are stable identities.
items: std.ArrayList(Item) = .empty,
/// Document order (arena indices, including tombstones).
seq: std.ArrayList(u32) = .empty,
/// arena index → position in `seq` (maintained across splices).
pos_of: std.ArrayList(u32) = .empty,

pub const empty: Sequence = .{};

pub fn deinit(self: *Sequence, gpa: Allocator) void {
    self.items.deinit(gpa);
    self.seq.deinit(gpa);
    self.pos_of.deinit(gpa);
}

/// Pre-populate `count` visible placeholder items with chained origins
/// (compacted-base content). Must precede any live item.
pub fn initBase(self: *Sequence, gpa: Allocator, count: usize, placeholder_lv: Lv) Allocator.Error!void {
    assert(self.items.items.len == 0);
    try self.items.ensureTotalCapacity(gpa, count);
    try self.seq.ensureTotalCapacity(gpa, count);
    try self.pos_of.ensureTotalCapacity(gpa, count);
    for (0..count) |i| {
        self.items.appendAssumeCapacity(.{
            .lv = placeholder_lv,
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

pub const ApplyError = Allocator.Error || error{Corrupt};

pub const Inserted = struct { arena: i32, effect_pos: u64 };

/// First application of an insert at prepare position `pos`: find
/// anchors, integrate (FugueMax), mark prepare- and effect-visible.
/// The returned effect position is where the item surfaced in the
/// output document. Out-of-range positions: `error.Corrupt` when
/// `untrusted`, otherwise a bug in trusted history.
pub fn applyInsert(
    self: *Sequence,
    gpa: Allocator,
    names: anytype,
    lv: Lv,
    pos: u64,
    untrusted: bool,
) ApplyError!Inserted {
    var left: i32 = none;
    var right: i32 = none;
    {
        var seen: u64 = 0;
        var found = false;
        for (self.seq.items) |arena| {
            const it = &self.items.items[arena];
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
            unreachable; // trusted history references a position that must exist
        }
    }
    const arena = try self.integrate(gpa, names, lv, left, right);
    const effect_pos = self.effectPositionOf(arena);
    self.items.items[@intCast(arena)].effect_visible = true;
    return .{ .arena = arena, .effect_pos = effect_pos };
}

pub const Deleted = struct { arena: i32, effect_pos: ?u64 };

/// First application of a delete at prepare position `pos`. Marks the
/// target prepare-deleted; if it was effect-visible, hides it and
/// returns its effect position (null = concurrently deleted already).
pub fn applyDelete(self: *Sequence, pos: u64, untrusted: bool) error{Corrupt}!Deleted {
    var seen: u64 = 0;
    for (self.seq.items) |arena| {
        const it = &self.items.items[arena];
        if (it.prepVisible()) {
            if (seen == pos) {
                it.prep_deleted += 1;
                if (!it.effect_visible) return .{ .arena = @intCast(arena), .effect_pos = null };
                const index = self.effectPositionOf(@intCast(arena));
                it.effect_visible = false;
                return .{ .arena = @intCast(arena), .effect_pos = index };
            }
            seen += 1;
        }
    }
    if (untrusted) return error.Corrupt;
    unreachable; // trusted history deleted a position that must exist
}

/// Retreat/advance an already-applied insert's prepare state.
pub fn toggleInsert(self: *Sequence, arena: i32, on: bool) void {
    assert(arena != none);
    self.items.items[@intCast(arena)].prep_inserted = on;
}

/// Retreat/advance an already-applied delete's prepare state.
pub fn toggleDelete(self: *Sequence, arena: i32, on: bool) void {
    assert(arena != none);
    const it = &self.items.items[@intCast(arena)];
    if (on) it.prep_deleted += 1 else it.prep_deleted -= 1;
}

/// Count of effect-visible items strictly before arena index `idx`.
pub fn effectPositionOf(self: *const Sequence, idx: i32) u64 {
    const end = self.pos_of.items[@intCast(idx)];
    var n: u64 = 0;
    for (self.seq.items[0..end]) |arena| {
        if (self.items.items[arena].effect_visible) n += 1;
    }
    return n;
}

/// The YjsMod / FugueMax integration loop. `names.agentNameOf(lv)` is
/// the deterministic global tiebreak.
fn integrate(self: *Sequence, gpa: Allocator, names: anytype, lv: Lv, origin_left: i32, origin_right: i32) Allocator.Error!i32 {
    const left_pos: i64 = if (origin_left == none) -1 else self.pos_of.items[@intCast(origin_left)];
    const right_pos: i64 = if (origin_right == none) @intCast(self.seq.items.len) else self.pos_of.items[@intCast(origin_right)];
    const seq_len: i64 = @intCast(self.seq.items.len);

    var dest: i64 = left_pos + 1;
    var scanning = false;
    var i: i64 = left_pos + 1;
    while (true) : (i += 1) {
        if (!scanning) dest = i;
        if (i == seq_len or i == right_pos) break;
        const o = &self.items.items[self.seq.items[@intCast(i)]];
        const o_left: i64 = if (o.origin_left == none) -1 else self.pos_of.items[@intCast(o.origin_left)];
        const o_right: i64 = if (o.origin_right == none) @intCast(self.seq.items.len) else self.pos_of.items[@intCast(o.origin_right)];
        if (o_left < left_pos) break;
        if (o_left == left_pos) {
            if (o_right < right_pos) {
                scanning = true;
            } else if (o_right == right_pos) {
                if (std.mem.order(u8, names.agentNameOf(lv), names.agentNameOf(o.lv)) == .lt) break;
                scanning = false;
            } else {
                scanning = false;
            }
        }
        // o_left > left_pos: keep walking without touching flags.
    }

    const arena: u32 = @intCast(self.items.items.len);
    try self.items.append(gpa, .{
        .lv = lv,
        .origin_left = origin_left,
        .origin_right = origin_right,
        .prep_inserted = true,
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

/// Iterate live content in document order.
pub const AliveIterator = struct {
    s: *const Sequence,
    i: usize = 0,

    pub fn next(self: *AliveIterator) ?struct { arena: u32, lv: Lv } {
        while (self.i < self.s.seq.items.len) {
            const arena = self.s.seq.items[self.i];
            self.i += 1;
            const it = &self.s.items.items[arena];
            if (it.effect_visible) return .{ .arena = arena, .lv = it.lv };
        }
        return null;
    }
};

pub fn aliveIterator(self: *const Sequence) AliveIterator {
    return .{ .s = self };
}

test {
    std.testing.refAllDecls(@This());
}
