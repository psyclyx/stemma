//! AnchorSet — bulk edit-stable positions with stable handles.
//!
//! An editor holds thousands of anchors (diagnostics, marks, folds,
//! multi-cursor carets). Shifting each individually through every `Edit` is
//! O(A) per keystroke; this set keeps anchors sorted by (offset, bias) so a
//! shift is O(log A + affected): anchors before the edit are untouched,
//! anchors after it move by a uniform delta (order preserved), and only the
//! few inside a deleted span are repartitioned.
//!
//! Decomplected on purpose: this type never touches the rope — it consumes
//! the same `Edit` values the rope returns, so it works with any buffer that
//! speaks `Edit`, and the rope stays free of marker bookkeeping (the classic
//! source of O(n)-markers pathologies).
//!
//! Handles are stable across edits, insertions, and removals. Same allocator
//! discipline as the rest of stemma: unmanaged, explicit `gpa`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const geometry = @import("geometry.zig");
const Anchor = geometry.Anchor;
const Bias = geometry.Bias;
const Edit = geometry.Edit;
const Range = geometry.Range;

const AnchorSet = @This();
const Slot = struct { anchor: Anchor, alive: bool };

slots: std.ArrayList(Slot) = .empty,
/// Slot indices sorted by (offset, bias); may contain dead slots (their
/// anchors keep shifting so keys stay consistent) until compaction.
order: std.ArrayList(u32) = .empty,
/// Recycled slot indices. Capacity is pre-reserved in `add`, so `remove`
/// and compaction never allocate.
free: std.ArrayList(u32) = .empty,
dead: usize = 0,

pub const Handle = enum(u32) { _ };

pub const empty: AnchorSet = .{};

pub fn deinit(self: *AnchorSet, gpa: Allocator) void {
    self.slots.deinit(gpa);
    self.order.deinit(gpa);
    self.free.deinit(gpa);
    self.* = .{};
}

/// Number of live anchors.
pub fn count(self: *const AnchorSet) usize {
    return self.order.items.len - self.dead;
}

fn keyLessThan(a: Anchor, b: Anchor) bool {
    if (a.offset != b.offset) return a.offset < b.offset;
    return @intFromEnum(a.bias) < @intFromEnum(b.bias);
}

/// First index in `order` whose anchor is ≥ (offset, .left).
fn lowerBound(self: *const AnchorSet, offset: usize) usize {
    var lo: usize = 0;
    var hi: usize = self.order.items.len;
    const probe: Anchor = .{ .offset = offset, .bias = .left };
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const a = self.slots.items[self.order.items[mid]].anchor;
        if (keyLessThan(a, probe)) lo = mid + 1 else hi = mid;
    }
    return lo;
}

pub fn add(self: *AnchorSet, gpa: Allocator, anchor: Anchor) Allocator.Error!Handle {
    // Reserve free-list capacity up front so remove/compact never fail.
    try self.free.ensureTotalCapacity(gpa, self.slots.items.len + 1);
    const slot: u32 = if (self.free.pop()) |idx| idx else blk: {
        const idx: u32 = @intCast(self.slots.items.len);
        _ = try self.slots.addOne(gpa);
        break :blk idx;
    };
    self.slots.items[slot] = .{ .anchor = anchor, .alive = true };
    // Insert in sorted position (after equal keys, for stable behavior).
    var i = self.lowerBound(anchor.offset);
    while (i < self.order.items.len and
        !keyLessThan(anchor, self.slots.items[self.order.items[i]].anchor)) i += 1;
    self.order.insert(gpa, i, slot) catch |e| {
        self.slots.items[slot].alive = false;
        self.free.appendAssumeCapacity(slot);
        return e;
    };
    return @enumFromInt(slot);
}

pub fn get(self: *const AnchorSet, h: Handle) Anchor {
    const s = self.slots.items[@intFromEnum(h)];
    assert(s.alive);
    return s.anchor;
}

/// Move an existing anchor (e.g. a caret). O(A) worst case (reinsertion).
pub fn set(self: *AnchorSet, h: Handle, anchor: Anchor) void {
    const slot: u32 = @intFromEnum(h);
    assert(self.slots.items[slot].alive);
    // Remove from order, update, reinsert — all in place, no allocation.
    const old = self.slots.items[slot].anchor;
    var i = self.lowerBound(old.offset);
    while (self.order.items[i] != slot) i += 1;
    _ = self.order.orderedRemove(i);
    self.slots.items[slot].anchor = anchor;
    var j = self.lowerBound(anchor.offset);
    while (j < self.order.items.len and
        !keyLessThan(anchor, self.slots.items[self.order.items[j]].anchor)) j += 1;
    self.order.insertAssumeCapacity(j, slot);
}

pub fn remove(self: *AnchorSet, h: Handle) void {
    const slot: u32 = @intFromEnum(h);
    assert(self.slots.items[slot].alive);
    self.slots.items[slot].alive = false;
    self.dead += 1;
    if (self.dead * 2 > self.order.items.len) self.compact();
}

/// Drop dead entries from `order` and recycle their slots. Never
/// allocates (free capacity is pre-reserved).
fn compact(self: *AnchorSet) void {
    var w: usize = 0;
    for (self.order.items) |slot| {
        if (self.slots.items[slot].alive) {
            self.order.items[w] = slot;
            w += 1;
        } else {
            self.free.appendAssumeCapacity(slot);
        }
    }
    self.order.items.len = w;
    self.dead = 0;
}

/// Map every anchor across `edit`. O(log A + affected); anchors strictly
/// before the edit are never touched.
pub fn shift(self: *AnchorSet, edit: Edit) void {
    const start = self.lowerBound(edit.offset);
    const items = self.order.items;
    const removed_end = edit.offset + edit.removed;

    // Window of anchors whose original offset < removed_end: these
    // collapse onto (offset, .left) or (offset+inserted, .right) and need
    // repartitioning. Everything after moves by a uniform delta.
    var i = start;
    while (i < items.len and self.slots.items[items[i]].anchor.offset < removed_end) : (i += 1) {
        const s = &self.slots.items[items[i]];
        s.anchor = s.anchor.shift(edit);
    }
    var j = i;
    while (j < items.len) : (j += 1) {
        const s = &self.slots.items[items[j]];
        s.anchor = s.anchor.shift(edit);
    }
    // Repartition the collapse window: all lefts (== edit.offset) before
    // all rights (== edit.offset + inserted).
    if (i > start) {
        var w = start;
        var k = start;
        while (k < i) : (k += 1) {
            if (self.slots.items[items[k]].anchor.bias == .left) {
                const tmp = items[k];
                items[k] = items[w];
                items[w] = tmp;
                w += 1;
            }
        }
    }
}

/// Iterate live anchors with offsets in `r`, in order.
pub fn inRange(self: *const AnchorSet, r: Range) RangeIterator {
    return .{ .set = self, .i = self.lowerBound(r.start), .end_offset = r.end };
}

pub const Entry = struct { handle: Handle, anchor: Anchor };

pub const RangeIterator = struct {
    set: *const AnchorSet,
    i: usize,
    end_offset: usize,

    pub fn next(self: *RangeIterator) ?Entry {
        const s_ = self.set;
        while (self.i < s_.order.items.len) {
            const slot = s_.order.items[self.i];
            const s = s_.slots.items[slot];
            // `order` is sorted (dead entries keep consistent keys), so
            // the first out-of-range offset ends the walk.
            if (s.anchor.offset >= self.end_offset) return null;
            self.i += 1;
            if (s.alive) return .{ .handle = @enumFromInt(slot), .anchor = s.anchor };
        }
        return null;
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

const t = std.testing;

test "AnchorSet: sorted shift matches per-anchor shift" {
    const gpa = t.allocator;
    var aset: AnchorSet = .empty;
    defer aset.deinit(gpa);

    var prng = std.Random.DefaultPrng.init(0xa2c4);
    const random = prng.random();

    var naive: std.ArrayList(Anchor) = .empty;
    defer naive.deinit(gpa);
    var handles: std.ArrayList(AnchorSet.Handle) = .empty;
    defer handles.deinit(gpa);

    // Seed anchors across a 1000-byte document.
    for (0..200) |_| {
        const a: Anchor = .{
            .offset = random.uintLessThan(usize, 1001),
            .bias = if (random.boolean()) .left else .right,
        };
        try naive.append(gpa, a);
        try handles.append(gpa, try aset.add(gpa, a));
    }

    // Random edits; every anchor must match its individually-shifted twin.
    var doc_len: usize = 1000;
    for (0..300) |_| {
        const off = random.uintLessThan(usize, doc_len + 1);
        const removed = @min(random.uintLessThan(usize, 20), doc_len - off);
        const inserted = random.uintLessThan(usize, 20);
        const e: Edit = .{ .offset = off, .removed = removed, .inserted = inserted };
        doc_len = doc_len - removed + inserted;

        aset.shift(e);
        for (naive.items, handles.items) |*a, h| {
            a.* = a.shift(e);
            try t.expectEqual(a.*, aset.get(h));
        }
    }
}

test "AnchorSet: handles stable across add/remove/compact; range queries" {
    const gpa = t.allocator;
    var aset: AnchorSet = .empty;
    defer aset.deinit(gpa);

    const h10 = try aset.add(gpa, .{ .offset = 10 });
    const h20 = try aset.add(gpa, .{ .offset = 20 });
    const h30 = try aset.add(gpa, .{ .offset = 30, .bias = .left });
    const h40 = try aset.add(gpa, .{ .offset = 40 });
    try t.expectEqual(@as(usize, 4), aset.count());

    aset.remove(h20);
    aset.remove(h40); // triggers compaction (dead > half)
    try t.expectEqual(@as(usize, 2), aset.count());
    try t.expectEqual(@as(usize, 10), aset.get(h10).offset);
    try t.expectEqual(@as(usize, 30), aset.get(h30).offset);

    // Recycled slots don't disturb existing handles.
    const h15 = try aset.add(gpa, .{ .offset = 15 });
    try t.expectEqual(@as(usize, 30), aset.get(h30).offset);

    var it = aset.inRange(.{ .start = 12, .end = 31 });
    try t.expectEqual(@as(usize, 15), it.next().?.anchor.offset);
    try t.expectEqual(@as(usize, 30), it.next().?.anchor.offset);
    try t.expectEqual(@as(?AnchorSet.Entry, null), it.next());

    aset.set(h15, .{ .offset = 99 });
    var it2 = aset.inRange(.{ .start = 0, .end = 1000 });
    try t.expectEqual(@as(usize, 10), it2.next().?.anchor.offset);
    try t.expectEqual(@as(usize, 30), it2.next().?.anchor.offset);
    try t.expectEqual(@as(usize, 99), it2.next().?.anchor.offset);
}
