//! stemma — a library for the distributed, causal story of a text.
//!
//! *stemma (n., pl. stemmata): a diagram showing the descent and
//! relationships of the surviving manuscripts of a text.*
//!
//! The long-term shape is an event-graph CRDT library (eg-walker family):
//! a causal DAG of editing events, walked on demand to merge divergent
//! histories, generic over materialized types — with text as the first and
//! flagship materializer. See `graph` for the engine's contract-in-progress.
//!
//! What ships today is the flagship's foundation: `Rope`, a persistent,
//! snapshot-able UTF-8 text buffer (B-tree of chunks with per-node metric
//! summaries) built to be the fastest correct core for a text editor —
//! single-user editing pays zero collaboration tax.
//!
//! ## Contracts (library-wide)
//! - Byte offsets are the native coordinate; they must land on Unicode scalar
//!   boundaries. The atomic unit is the scalar value: graphemes, words, and
//!   display columns are the caller's concern.
//! - Content is always valid UTF-8; validity of *inputs* is an asserted
//!   precondition (checked in safe builds), not a returned error.
//! - Allocation is explicit and unmanaged: no type stores an allocator, and
//!   the same allocator must be used across a value and everything derived
//!   from it (snapshots, splits).
//! - Cursors/selections/undo *policy* live in the caller, composed from
//!   `Edit` deltas, `Anchor.shift`, and O(1) `snapshot()`s.

const std = @import("std");

const geometry = @import("geometry.zig");
const rope = @import("rope.zig");
const anchors = @import("anchors.zig");

pub const graph = @import("graph.zig");

// ── Value types ──
pub const Point = geometry.Point;
pub const PointUtf16 = geometry.PointUtf16;
pub const Range = geometry.Range;
pub const Bias = geometry.Bias;
pub const Edit = geometry.Edit;
pub const Anchor = geometry.Anchor;

// ── The text buffer ──
pub const Options = rope.Options;
pub const RopeWith = rope.RopeWith;
pub const Rope = rope.Rope;

// ── Bulk edit-stable positions (diagnostics, marks, multi-cursor) ──
pub const AnchorSet = anchors.AnchorSet;

test {
    _ = geometry;
    _ = rope;
    _ = anchors;
    _ = graph;
    _ = @import("rope_tests.zig");
    _ = @import("usage_tests.zig");
}
