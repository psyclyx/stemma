//! stemma — a library for the distributed, causal story of a text.
//!
//! *stemma (n., pl. stemmata): a diagram showing the descent and
//! relationships of the surviving manuscripts of a text.*
//!
//! Two layers, one flat surface:
//!
//! The **buffer**: `Rope`, a persistent, snapshot-able UTF-8 text buffer
//! (B-tree of chunks with per-node metric summaries) built to be the
//! fastest correct core for a text editor, plus `AnchorSet` for bulk
//! edit-stable positions.
//!
//! The **collaboration layer** (eg-walker family — Gentle & Kleppmann,
//! EuroSys 2025): the durable artifact is the causal event graph, the
//! *stemma* of the document; CRDT state is transient, rebuilt during
//! merges. `TextDoc` is a collaborative text document (lean events,
//! compaction, time travel, identity anchors); `ObjectDoc` is a
//! collaborative object tree (multi-value maps, lists, text nodes) for
//! JSON-blob-shaped applications. FugueMax ordering, opaque portable
//! version tokens, wire-ready sync. Replica-local identifiers never cross
//! the wire.
//!
//! ## Contracts (library-wide)
//! - Byte offsets are the native coordinate; they must land on Unicode
//!   scalar boundaries. The atomic unit is the scalar value: graphemes,
//!   words, and display columns are the caller's concern.
//! - Content is always valid UTF-8; validity of *inputs* is an asserted
//!   precondition (checked in safe builds), not a returned error. Hostile
//!   *wire* input is fully validated and can never crash a replica.
//! - Allocation is explicit and unmanaged: no type stores an allocator,
//!   and the same allocator must be used across a value and everything
//!   derived from it.
//! - Cursors/selections/undo *policy*, transport, presence payloads, and
//!   collaborative undo live in the caller, composed from `Edit` deltas,
//!   `Anchor.shift`, snapshots, and the merge change streams.

const std = @import("std");

const geometry = @import("geometry.zig");
const rope = @import("rope.zig");
const causal = @import("collab/causal.zig");
const seq_walker = @import("collab/SeqWalker.zig");

// ── Value types ──
pub const Point = geometry.Point;
pub const PointUtf16 = geometry.PointUtf16;
pub const Range = geometry.Range;
pub const Bias = geometry.Bias;
pub const Edit = geometry.Edit;
pub const Anchor = geometry.Anchor;

// ── The text buffer ──
pub const RopeOptions = rope.RopeOptions;
pub const RopeWith = rope.RopeWith;
pub const Rope = rope.Rope;
pub const AnchorSet = @import("AnchorSet.zig");

// ── The collaboration layer ──
pub const TextDoc = @import("collab/TextDoc.zig");
pub const ObjectDoc = @import("collab/ObjectDoc.zig");
pub const EventGraph = causal.EventGraph;
pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
pub const VersionOrder = causal.VersionOrder;
/// The portable identity-anchor shape (agent name + seq + side) shared by
/// `TextDoc` (whole-document `SeqWalker`) and `ObjectDoc` (one `SeqWalker`
/// per text object) — `SeqWalker.zig`'s `EventAnchor`/`AnchorSide`, named
/// here once so callers of either doc type (or code, like `Document`'s
/// facade, generic over which one backs it) can spell the shared type
/// without reaching through either doc's re-export.
pub const EventAnchor = seq_walker.EventAnchor;
pub const AnchorSide = seq_walker.AnchorSide;

test {
    _ = geometry;
    _ = rope;
    _ = AnchorSet;
    _ = causal;
    _ = TextDoc;
    _ = ObjectDoc;
    _ = @import("collab/core.zig");
    _ = @import("collab/Sequence.zig");
    _ = @import("collab/SeqWalker.zig");
    _ = @import("collab/objects_state.zig");
    _ = @import("collab/wire.zig");
    _ = @import("rope_tests.zig");
    _ = @import("usage_tests.zig");
    _ = @import("collab/text_tests.zig");
    _ = @import("collab/object_tests.zig");
}
