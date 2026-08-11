//! graph — the event-graph engine (design stub).
//!
//! Target: the eg-walker family (Gentle & Kleppmann, "Collaborative Text
//! Editing with Eg-walker: Better, Faster, Smaller", EuroSys 2025): store the
//! causal DAG of *original editing events* (Git-like, no per-character CRDT
//! metadata or tombstones in steady state) and, on merge, replay only the
//! divergent slice of history against transient state. Works everywhere CRDTs
//! do — including peer-to-peer with no central server — while using ~an order
//! of magnitude less steady-state memory and loading orders of magnitude
//! faster than classic CRDTs (Yjs/Automerge/RGA).
//!
//! Design intent, fixed now so nothing above bakes in the wrong shape:
//! - **Type-generic from the first commit.** The DAG store, causal frontier /
//!   version machinery, and the walker are agnostic to what an event *means*;
//!   a comptime materializer parameter interprets events (apply, invert for
//!   retreat/advance). Text (`stemma.Rope`) is the first materializer;
//!   registers/maps/lists come later over the same engine.
//! - **The single-user path owes nothing to this module.** `Rope` never
//!   carries collab metadata; this layer wraps it.
//! - **History-dependent operations** (auto-indent, format-on-type) do not
//!   fit the merge model and belong above this layer.
//!
//! Deliberately unimplemented: the core rope lands and gets benchmarked
//! first; this engine is then designed against a finished foundation.
//! Eventual surface sketch: `Graph(Materializer)`, `EventId` (agent, seq),
//! `Version` (frontier of event ids), `applyLocal` / `mergeRemote`, and an
//! identity-based `Anchor` that survives concurrent remote edits.

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
