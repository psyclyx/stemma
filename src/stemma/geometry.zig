//! Pure value types shared across stemma: positions, ranges, edit deltas, and
//! the anchor-shifting primitive. No allocation, no tree — these compose with
//! any buffer state and are cheap to copy and store by the thousand.

const std = @import("std");
const assert = std.debug.assert;

/// A half-open byte range `[start, end)`. All ranges and offsets in this API
/// are **byte** offsets into the UTF-8 sequence and must land on scalar
/// boundaries.
pub const Range = struct {
    start: usize,
    end: usize,

    pub fn len(self: Range) usize {
        assert(self.end >= self.start);
        return self.end - self.start;
    }

    pub fn isEmpty(self: Range) bool {
        return self.end == self.start;
    }
};

/// A line/column position. `col` is a **byte** offset within the line — not a
/// grapheme or display column (those are the caller's job). Row and column are
/// 0-based; the newline terminating a row is not part of the row. Full-width
/// integers on purpose: a multi-gigabyte single-line file must not truncate.
pub const Point = struct {
    row: usize,
    col: usize,
};

/// Tie-breaking side for an `Anchor` sitting exactly at an edit boundary.
pub const Bias = enum { left, right };

/// The byte-offset delta produced by a mutation. Feed it to `Anchor.shift`
/// (or your own position mapping) to keep caller-held positions valid across
/// the edit. This is the decomplected substrate for cursors, selections, and
/// undo — the buffer itself tracks none of that state.
pub const Edit = struct {
    /// Byte offset at which the change began.
    offset: usize,
    /// Bytes removed at `offset`.
    removed: usize,
    /// Bytes inserted at `offset`.
    inserted: usize,

    /// The inverse delta, for mapping positions *backwards* (redo→undo). Note
    /// this carries no text: reconstruct removed bytes from a `snapshot()` you
    /// took before the edit, which is the intended O(1) undo path.
    pub fn invert(self: Edit) Edit {
        return .{ .offset = self.offset, .removed = self.inserted, .inserted = self.removed };
    }
};

/// A caller-held logical position with a tie-breaking bias. Anchors live in
/// the caller (cursors, marks, diagnostic ranges); the buffer does not track
/// them. Shift an anchor through each `Edit` you apply to keep it valid.
///
/// (The event-graph layer upgrades this to an identity-based anchor that also
/// survives *concurrent* remote edits; this single-user form only needs to
/// survive local edits, for which offset-shifting is exact.)
pub const Anchor = struct {
    offset: usize,
    bias: Bias = .right,

    /// Map this anchor across `edit`. An anchor strictly after the edit shifts
    /// by the net length change; one strictly before is unchanged; one exactly
    /// at the edit offset stays put if `left`-biased and moves past inserted
    /// text if `right`-biased; one inside a deleted span collapses to the edit
    /// offset.
    pub fn shift(self: Anchor, edit: Edit) Anchor {
        const removed_end = edit.offset + edit.removed;
        var off = self.offset;
        if (off < edit.offset) {
            // strictly before the edit — unaffected
        } else if (off == edit.offset) {
            if (self.bias == .right) off += edit.inserted;
        } else if (off >= removed_end) {
            // strictly after the removed span — shift by net delta
            off = off - edit.removed + edit.inserted;
        } else {
            // inside the removed span — collapse to the edit point
            off = edit.offset + if (self.bias == .right) edit.inserted else 0;
        }
        return .{ .offset = off, .bias = self.bias };
    }
};

test "Anchor.shift: insert before/at/after" {
    const t = std.testing;
    const e: Edit = .{ .offset = 5, .removed = 0, .inserted = 3 };
    // before the insert: unaffected
    try t.expectEqual(@as(usize, 2), (Anchor{ .offset = 2 }).shift(e).offset);
    // at the insert, right bias moves past; left bias stays
    try t.expectEqual(@as(usize, 8), (Anchor{ .offset = 5, .bias = .right }).shift(e).offset);
    try t.expectEqual(@as(usize, 5), (Anchor{ .offset = 5, .bias = .left }).shift(e).offset);
    // after the insert: shifted by inserted length
    try t.expectEqual(@as(usize, 13), (Anchor{ .offset = 10 }).shift(e).offset);
}

test "Anchor.shift: delete spanning the anchor collapses it" {
    const t = std.testing;
    const e: Edit = .{ .offset = 4, .removed = 6, .inserted = 0 }; // delete [4,10)
    try t.expectEqual(@as(usize, 3), (Anchor{ .offset = 3 }).shift(e).offset); // before
    try t.expectEqual(@as(usize, 4), (Anchor{ .offset = 7 }).shift(e).offset); // inside → edit offset
    try t.expectEqual(@as(usize, 6), (Anchor{ .offset = 12 }).shift(e).offset); // after → -6
}
