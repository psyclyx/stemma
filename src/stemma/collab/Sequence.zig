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
//!
//! ## Backing structure
//! Document order used to be a flat array (`seq`) plus an arena→position
//! map (`pos_of`), rebuilt/shifted on every insert — O(n) per insert and
//! per positional query. It's now an implicit treap keyed by in-order
//! position: each `Item` doubles as a tree node (`tleft`/`tright`/
//! `tparent`, a priority, and subtree aggregates for count/prep-visible/
//! effect-visible), so "insert at position k", "rank of item", and
//! "k-th item matching a predicate" are all O(log n). Node identity
//! (arena index) is still permanent and still what `origin_left`/
//! `origin_right`/`item_of`/`target_of` reference — only how positions
//! are derived from that identity changed. Priorities are a deterministic
//! hash of the arena index, not a stored RNG: balance only affects
//! performance, never the in-order sequence the tree represents (that's
//! fixed by the FugueMax integration decisions in `TextDoc.Replay`), so
//! there's nothing to keep in sync across replicas.

const std = @import("std");
const Allocator = std.mem.Allocator;

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

    /// Treap linkage, keyed by in-order (document) position. Structural,
    /// maintained only by `merge`/`split`/the ancestor-fixup helpers below
    /// — nothing outside this file touches these six fields.
    tleft: i32 = none,
    tright: i32 = none,
    tparent: i32 = none,
    priority: u32 = 0,
    /// Subtree aggregates, always exact for the subtree rooted here:
    /// total node count, count of `prepVisible()`, count of
    /// `effect_visible`.
    sub_size: u32 = 1,
    sub_prep: u32 = 0,
    sub_effect: u32 = 0,

    pub fn prepVisible(it: *const Item) bool {
        return it.prep_inserted and it.prep_deleted == 0;
    }
};

const Sequence = @This();
/// Append-only arena; arena indices are stable identities and double as
/// treap node handles.
items: std.ArrayList(Item) = .empty,
/// Treap root (arena index), or `none` for an empty sequence.
root: i32 = none,

pub const empty: Sequence = .{};

pub fn deinit(self: *Sequence, gpa: Allocator) void {
    self.items.deinit(gpa);
}

/// A deterministic pseudorandom priority from the arena index (murmur3
/// fmix32) — not a stored RNG. See the module doc comment: tree shape is
/// a performance concern only, so this needs no cross-replica agreement,
/// just a good distribution.
fn priorityOf(arena: i32) u32 {
    var x: u32 = @bitCast(arena);
    x ^= x >> 16;
    x *%= 0x85ebca6b;
    x ^= x >> 13;
    x *%= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

/// Recompute `idx`'s subtree aggregates from its own flags and its
/// children's (already-correct) aggregates. O(1).
fn fixup(self: *Sequence, idx: i32) void {
    if (idx == none) return;
    const it = &self.items.items[@intCast(idx)];
    var size: u32 = 1;
    var prep: u32 = if (it.prepVisible()) 1 else 0;
    var effect: u32 = if (it.effect_visible) 1 else 0;
    if (it.tleft != none) {
        const l = self.items.items[@intCast(it.tleft)];
        size += l.sub_size;
        prep += l.sub_prep;
        effect += l.sub_effect;
    }
    if (it.tright != none) {
        const r = self.items.items[@intCast(it.tright)];
        size += r.sub_size;
        prep += r.sub_prep;
        effect += r.sub_effect;
    }
    it.sub_size = size;
    it.sub_prep = prep;
    it.sub_effect = effect;
}

/// Standard treap merge: whichever root has higher priority stays on
/// top. Every node in `a` must precede every node in `b` in the intended
/// in-order result.
fn merge(self: *Sequence, a: i32, b: i32) i32 {
    if (a == none) return b;
    if (b == none) return a;
    if (self.items.items[@intCast(a)].priority >= self.items.items[@intCast(b)].priority) {
        const new_right = self.merge(self.items.items[@intCast(a)].tright, b);
        self.items.items[@intCast(a)].tright = new_right;
        if (new_right != none) self.items.items[@intCast(new_right)].tparent = a;
        self.fixup(a);
        return a;
    } else {
        const new_left = self.merge(a, self.items.items[@intCast(b)].tleft);
        self.items.items[@intCast(b)].tleft = new_left;
        if (new_left != none) self.items.items[@intCast(new_left)].tparent = b;
        self.fixup(b);
        return b;
    }
}

const Split = struct { left: i32, right: i32 };

/// Split `root` so the left result holds exactly the first `rank` nodes
/// (in-order) and the right result holds the rest. Both results are
/// independent subtree roots (`tparent == none`).
fn split(self: *Sequence, root: i32, rank: u32) Split {
    if (root == none) return .{ .left = none, .right = none };
    const left = self.items.items[@intCast(root)].tleft;
    const left_size: u32 = if (left == none) 0 else self.items.items[@intCast(left)].sub_size;
    if (rank <= left_size) {
        const s = self.split(left, rank);
        self.items.items[@intCast(root)].tleft = s.right;
        if (s.right != none) self.items.items[@intCast(s.right)].tparent = root;
        self.items.items[@intCast(root)].tparent = none;
        self.fixup(root);
        return .{ .left = s.left, .right = root };
    } else {
        const right = self.items.items[@intCast(root)].tright;
        const s = self.split(right, rank - left_size - 1);
        self.items.items[@intCast(root)].tright = s.left;
        if (s.left != none) self.items.items[@intCast(s.left)].tparent = root;
        self.items.items[@intCast(root)].tparent = none;
        self.fixup(root);
        return .{ .left = root, .right = s.right };
    }
}

/// Insert the fresh leaf `node` (already appended to `items`, own
/// aggregates already set) at in-order position `rank`. O(log n).
fn insertAtRank(self: *Sequence, root: i32, rank: u32, node: i32) i32 {
    const s = self.split(root, rank);
    return self.merge(self.merge(s.left, node), s.right);
}

/// In-order successor of `node` (`none` if `node` is last). Amortized
/// O(1) over a full traversal; O(log n) worst case for a single call.
fn successor(self: *const Sequence, node: i32) i32 {
    if (self.items.items[@intCast(node)].tright != none) {
        var cur = self.items.items[@intCast(node)].tright;
        while (self.items.items[@intCast(cur)].tleft != none) cur = self.items.items[@intCast(cur)].tleft;
        return cur;
    }
    var cur = node;
    while (self.items.items[@intCast(cur)].tparent != none) {
        const p = self.items.items[@intCast(cur)].tparent;
        if (self.items.items[@intCast(p)].tleft == cur) return p;
        cur = p;
    }
    return none;
}

/// First item in document order (`none` if empty).
fn firstInOrder(self: *const Sequence) i32 {
    if (self.root == none) return none;
    var cur = self.root;
    while (self.items.items[@intCast(cur)].tleft != none) cur = self.items.items[@intCast(cur)].tleft;
    return cur;
}

/// Count of items strictly before `node`, either unconditionally (`.all`
/// — document position) or among `effect_visible` items (`.effect` —
/// what `effectPositionOf` reports). O(log n): walk parent links,
/// crediting each ancestor's left subtree (and the ancestor itself) when
/// arriving from its right child.
fn rankOf(self: *const Sequence, comptime which: enum { all, effect }, node: i32) u64 {
    const of = struct {
        fn of(s: *const Sequence, w: @TypeOf(which), idx: i32) u32 {
            if (idx == none) return 0;
            const it = s.items.items[@intCast(idx)];
            return switch (w) {
                .all => it.sub_size,
                .effect => it.sub_effect,
            };
        }
    }.of;
    var count: u64 = of(self, which, self.items.items[@intCast(node)].tleft);
    var cur = node;
    while (self.items.items[@intCast(cur)].tparent != none) {
        const p = self.items.items[@intCast(cur)].tparent;
        if (self.items.items[@intCast(p)].tright == cur) {
            const pit = self.items.items[@intCast(p)];
            const p_self: u32 = switch (which) {
                .all => 1,
                .effect => if (pit.effect_visible) 1 else 0,
            };
            count += of(self, which, pit.tleft) + p_self;
        }
        cur = p;
    }
    return count;
}

/// The arena index of the `rank`-th (0-based) `prepVisible()` item, or
/// `none` if `rank` is out of range. O(log n).
fn selectPrepVisible(self: *const Sequence, rank: u32) i32 {
    var cur = self.root;
    var k = rank;
    while (cur != none) {
        const it = self.items.items[@intCast(cur)];
        const left_count: u32 = if (it.tleft == none) 0 else self.items.items[@intCast(it.tleft)].sub_prep;
        if (k < left_count) {
            cur = it.tleft;
            continue;
        }
        k -= left_count;
        if (it.prepVisible()) {
            if (k == 0) return cur;
            k -= 1;
        }
        cur = it.tright;
    }
    return none;
}

/// Adjust `arena`'s `prep_deleted` counter by `delta` and, if that flips
/// `prepVisible()`, propagate the change to every ancestor's `sub_prep`.
/// O(log n).
fn adjustPrepDeleted(self: *Sequence, arena: i32, delta: i32) void {
    const it = &self.items.items[@intCast(arena)];
    const was = it.prepVisible();
    it.prep_deleted = @intCast(@as(i32, @intCast(it.prep_deleted)) + delta);
    if (it.prepVisible() != was) self.bumpSubPrep(arena, if (was) -1 else 1);
}

fn setPrepInserted(self: *Sequence, arena: i32, value: bool) void {
    const it = &self.items.items[@intCast(arena)];
    const was = it.prepVisible();
    it.prep_inserted = value;
    if (it.prepVisible() != was) self.bumpSubPrep(arena, if (was) -1 else 1);
}

fn bumpSubPrep(self: *Sequence, arena: i32, delta: i32) void {
    var cur: i32 = arena;
    while (cur != none) {
        const it = &self.items.items[@intCast(cur)];
        it.sub_prep = @intCast(@as(i32, @intCast(it.sub_prep)) + delta);
        cur = it.tparent;
    }
}

fn setEffectVisible(self: *Sequence, arena: i32, value: bool) void {
    const it = &self.items.items[@intCast(arena)];
    if (it.effect_visible == value) return;
    it.effect_visible = value;
    const delta: i32 = if (value) 1 else -1;
    var cur: i32 = arena;
    while (cur != none) {
        const cit = &self.items.items[@intCast(cur)];
        cit.sub_effect = @intCast(@as(i32, @intCast(cit.sub_effect)) + delta);
        cur = cit.tparent;
    }
}

/// Pre-populate `count` visible placeholder items with chained origins
/// (compacted-base content). Must precede any live item. O(count log
/// count) (each appended at the current tail via a treap merge) — not
/// the O(count) a from-scratch Cartesian-tree build could reach, but a
/// one-time document-load cost, and simple enough to trust.
pub fn initBase(self: *Sequence, gpa: Allocator, count: usize, placeholder_lv: Lv) Allocator.Error!void {
    std.debug.assert(self.items.items.len == 0);
    try self.items.ensureTotalCapacity(gpa, count);
    for (0..count) |i| {
        const arena: i32 = @intCast(i);
        self.items.appendAssumeCapacity(.{
            .lv = placeholder_lv,
            .origin_left = if (i == 0) none else @intCast(i - 1),
            .origin_right = none,
            .prep_inserted = true,
            .prep_deleted = 0,
            .effect_visible = true,
            .priority = priorityOf(arena),
            .sub_size = 1,
            .sub_prep = 1,
            .sub_effect = 1,
        });
        self.root = self.merge(self.root, arena);
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
    const total_prep: u32 = if (self.root == none) 0 else self.items.items[@intCast(self.root)].sub_prep;
    if (pos > total_prep) {
        if (untrusted) return error.Corrupt;
        unreachable; // trusted history references a position that must exist
    }
    const right: i32 = if (pos == total_prep) none else self.selectPrepVisible(@intCast(pos));
    const left: i32 = if (pos == 0) none else self.selectPrepVisible(@intCast(pos - 1));

    const arena = try self.integrate(gpa, names, lv, left, right);
    const effect_pos = self.effectPositionOf(arena);
    self.setEffectVisible(arena, true);
    return .{ .arena = arena, .effect_pos = effect_pos };
}

pub const Deleted = struct { arena: i32, effect_pos: ?u64 };

/// First application of a delete at prepare position `pos`. Marks the
/// target prepare-deleted; if it was effect-visible, hides it and
/// returns its effect position (null = concurrently deleted already).
pub fn applyDelete(self: *Sequence, pos: u64, untrusted: bool) error{Corrupt}!Deleted {
    const total_prep: u32 = if (self.root == none) 0 else self.items.items[@intCast(self.root)].sub_prep;
    if (pos >= total_prep) {
        if (untrusted) return error.Corrupt;
        unreachable; // trusted history deleted a position that must exist
    }
    const arena = self.selectPrepVisible(@intCast(pos));
    self.adjustPrepDeleted(arena, 1);
    if (!self.items.items[@intCast(arena)].effect_visible) return .{ .arena = arena, .effect_pos = null };
    const index = self.effectPositionOf(arena);
    self.setEffectVisible(arena, false);
    return .{ .arena = arena, .effect_pos = index };
}

/// Retreat/advance an already-applied insert's prepare state.
pub fn toggleInsert(self: *Sequence, arena: i32, on: bool) void {
    std.debug.assert(arena != none);
    self.setPrepInserted(arena, on);
}

/// Retreat/advance an already-applied delete's prepare state.
pub fn toggleDelete(self: *Sequence, arena: i32, on: bool) void {
    std.debug.assert(arena != none);
    self.adjustPrepDeleted(arena, if (on) 1 else -1);
}

/// Count of effect-visible items strictly before arena index `idx`.
pub fn effectPositionOf(self: *const Sequence, idx: i32) u64 {
    return self.rankOf(.effect, idx);
}

/// The YjsMod / FugueMax integration loop. `names.agentNameOf(lv)` is
/// the deterministic global tiebreak.
fn integrate(self: *Sequence, gpa: Allocator, names: anytype, lv: Lv, origin_left: i32, origin_right: i32) Allocator.Error!i32 {
    const total_size: u32 = if (self.root == none) 0 else self.items.items[@intCast(self.root)].sub_size;
    const left_pos: i64 = if (origin_left == none) -1 else @intCast(self.rankOf(.all, origin_left));
    const right_pos: i64 = if (origin_right == none) @intCast(total_size) else @intCast(self.rankOf(.all, origin_right));

    var dest: i64 = left_pos + 1;
    var scanning = false;
    var i: i64 = left_pos + 1;
    var o: i32 = if (origin_left == none) self.firstInOrder() else self.successor(origin_left);
    while (true) : (i += 1) {
        if (!scanning) dest = i;
        if (o == none or i == right_pos) break;
        const oit = self.items.items[@intCast(o)];
        const o_left: i64 = if (oit.origin_left == none) -1 else @intCast(self.rankOf(.all, oit.origin_left));
        const o_right: i64 = if (oit.origin_right == none) @intCast(total_size) else @intCast(self.rankOf(.all, oit.origin_right));
        if (o_left < left_pos) break;
        if (o_left == left_pos) {
            if (o_right < right_pos) {
                scanning = true;
            } else if (o_right == right_pos) {
                if (std.mem.order(u8, names.agentNameOf(lv), names.agentNameOf(oit.lv)) == .lt) break;
                scanning = false;
            } else {
                scanning = false;
            }
        }
        // o_left > left_pos: keep walking without touching flags.
        o = self.successor(o);
    }

    const arena: i32 = @intCast(self.items.items.len);
    try self.items.append(gpa, .{
        .lv = lv,
        .origin_left = origin_left,
        .origin_right = origin_right,
        .prep_inserted = true,
        .prep_deleted = 0,
        .effect_visible = false,
        .priority = priorityOf(arena),
        .sub_size = 1,
        .sub_prep = 1,
        .sub_effect = 0,
    });
    errdefer _ = self.items.pop();
    self.root = self.insertAtRank(self.root, @intCast(dest), arena);
    return arena;
}

/// Iterate live content in document order.
pub const AliveIterator = struct {
    s: *const Sequence,
    cur: i32,

    pub fn next(self: *AliveIterator) ?struct { arena: u32, lv: Lv } {
        while (self.cur != none) {
            const arena = self.cur;
            const it = self.s.items.items[@intCast(arena)];
            self.cur = self.s.successor(arena);
            if (it.effect_visible) return .{ .arena = @intCast(arena), .lv = it.lv };
        }
        return null;
    }
};

pub fn aliveIterator(self: *const Sequence) AliveIterator {
    return .{ .s = self, .cur = self.firstInOrder() };
}

/// Every item in document order, alive or tombstoned (`resolveAnchors`
/// batches over this — see `TextDoc.resolveAnchors`).
pub const AllIterator = struct {
    s: *const Sequence,
    cur: i32,

    pub fn next(self: *AllIterator) ?i32 {
        const arena = self.cur;
        if (arena == none) return null;
        self.cur = self.s.successor(arena);
        return arena;
    }
};

pub fn allIterator(self: *const Sequence) AllIterator {
    return .{ .s = self, .cur = self.firstInOrder() };
}

test {
    std.testing.refAllDecls(@This());
}
