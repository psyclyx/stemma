//! graph — the event-graph collaboration layer (eg-walker family).
//!
//! The durable artifact is the causal DAG of original editing events (Gentle
//! & Kleppmann, "Collaborative Text Editing with Eg-walker", EuroSys 2025);
//! CRDT state is transient, reconstructed during merges by walking the graph
//! and discarded afterwards. Single-user editing pays no collaboration tax —
//! the `Rope` never carries collab metadata; this layer wraps it.
//!
//! What consumers get (see `TextDoc`): local edit recording, `merge` that
//! returns the same byte-space `[]Edit` stream local edits produce (so
//! `AnchorSet`/cursors survive remote edits with zero extra machinery),
//! version frontiers + `eventsSince` for sync, and a versioned wire/disk
//! format. What stays theirs: transport, presence/awareness, collaborative
//! undo policy.
//!
//! Layering, decomplected: `causal` (the DAG, generic over the op payload —
//! it neither knows nor cares what an event means) / `walker` (the text
//! materializer: YATA-ordered transient state, retreat/advance replay) /
//! `TextDoc` (rope + graph + wire format). Additional materializers over the
//! same `causal.EventGraph` are the intended growth direction.
//!
//! v1 replay is correctness-first: from genesis, unit events, linked-list
//! state. The optimization ladder (run-RLE storage, LCA-bounded replay with
//! placeholder runs, B-tree state) changes none of the semantics and lands
//! rung by rung against benchmarks — see BENCHMARKS.md.

const std = @import("std");

pub const causal = @import("graph/causal.zig");
const walker = @import("graph/walker.zig");
const doc = @import("graph/doc.zig");

pub const AgentId = causal.AgentId;
pub const EventId = causal.EventId;
pub const EventGraph = causal.EventGraph;
pub const TextOp = walker.TextOp;
pub const TextDoc = doc.TextDoc;

test {
    _ = causal;
    _ = walker;
    _ = doc;
    _ = @import("graph/graph_tests.zig");
}
