//! collab — the collaboration layer: event-sourced documents in the
//! eg-walker family (Gentle & Kleppmann, EuroSys 2025).
//!
//! The durable artifact is the causal event graph — the *stemma* of the
//! document: every replica's edits with their causal parents. CRDT state is
//! transient, reconstructed during merges and discarded; single-user editing
//! pays no collaboration tax.
//!
//! Two document surfaces over one shared machinery:
//! - `TextDoc` — a collaborative text document (a `Rope` plus its history).
//!   Lean events, compaction, time travel, identity anchors.
//! - `ObjectDoc` — a collaborative object tree (maps with honest multi-value
//!   conflicts, lists, inline scalars, and full text nodes) for
//!   JSON-blob-shaped applications. Dumps canonical JSON via `toJson`.
//!
//! Shared foundations: `causal.EventGraph` (the DAG, generic over the op
//! payload), FugueMax sequence ordering (maximally non-interleaving),
//! opaque portable version tokens, wire-ready sync (`eventsSince`/`merge`),
//! `compareVersions`. Replica-local identifiers (AgentId, ObjId, Lv) never
//! cross the wire — portable references are name-based tokens.
//!
//! Deferred to callers: transport, presence payloads, collaborative undo
//! policy. Ledgered: ObjectDoc compaction (object-id survival design),
//! Peritext-style rich-text marks, doc-core unification follow-through.

const std = @import("std");

pub const causal = @import("collab/causal.zig");
const core = @import("collab/core.zig");
const text = @import("collab/text.zig");
const objects = @import("collab/objects.zig");

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
pub const EventGraph = causal.EventGraph;
pub const VersionOrder = causal.VersionOrder;
pub const TextOp = text.TextOp;
pub const TextDoc = text.TextDoc;
pub const ObjectDoc = objects.ObjectDoc;
pub const ObjId = objects.ObjId;

test {
    _ = causal;
    _ = core;
    _ = @import("collab/sequence.zig");
    _ = @import("collab/objects_state.zig");
    _ = text;
    _ = objects;
    _ = @import("collab/wire.zig");
    _ = @import("collab/text_tests.zig");
    _ = @import("collab/object_tests.zig");
}
