# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The doc-core unification: `TextDoc` and `ObjectDoc` now share the replay
engine one layer above the sequence CRDT, and the features that were built
once against `TextDoc`'s single sequence — identity anchors, compaction —
generalize to `ObjectDoc`'s tree of sequences.

### Added

- **Identity anchors on `ObjectDoc` text objects** — `objectAnchorAt` /
  `resolveObjectAnchors`: portable (agent + seq) positions inside a node's
  body that survive concurrent edits anywhere in the tree, collapse
  deterministically to the deletion point when their character dies, and
  are isolated per object (heavy edits to a sibling never move them).
  Anchors created on one replica resolve on another, including across
  divergent agent tables. List-object anchors are deferred (element-index
  shape, not byte offsets).
- **Text-object compaction for `ObjectDoc`** — `compact` collapses
  all-peers-stable text history into a frozen base. Scope is deliberate
  and enforced: only text events fold; map/list register history never
  compacts (conflict sets for late-arriving peers need it); a stable point
  with list content in its causal past refuses (`error.NotCompactable`) —
  list/map-content bases are future work. A retained write whose causal
  edge into the base routes through folded history also refuses: folding
  it would silently re-order register supersession on a fresh replica
  (caught by review with an empirical divergence probe; the probe ships as
  a regression test).
- **Wire v2 for compacted `ObjectDoc`s** (`stj\x02`: per-agent seq bases,
  a content-derived base-version token, text base sections). Never-
  compacted docs still emit byte-identical v1; old decoders refuse v2
  loudly rather than garbage-decoding. Syncing across a compaction
  boundary requires both replicas compacted to the identical stable point
  (`error.MissingDependency` otherwise, both directions — no silent
  divergence).
- `ObjectDoc.ref()` — the missing inverse of `ValueRef.objId`.
- `structure_sketch.zig` (test-only, not in the public module): validation
  of parent-register structural children + fractional order keys — 400
  seeded multi-replica schedules converge, stay acyclic, stay reachable;
  the groundwork for a future move op.

### Changed

- Internal: the prepare/effect retreat/advance replay driver, previously
  hand-written in both `TextDoc.Replay` and `ObjectDoc`'s walker, is one
  shared `SeqWalker` with comptime storage strategies (dense for
  `TextDoc`'s single total sequence, sparse for `ObjectDoc`'s per-object
  instances — chosen after benchmarks showed a hashmap on the dense path
  cost ~10% on cold full replays). Object identity is now keyed by
  creation `EventId` internally. No public API or wire change; benchmark
  parity verified.

## [0.1.0] — 2026-08-17

Initial release: the text rope and the collaboration layer.

### Rope

- Persistent, snapshot-able UTF-8 B-tree rope with per-node metric summaries
  (bytes / scalars / UTF-16 units / newlines) and O(log n) coordinate
  conversion between byte, point, UTF-16, and scalar domains.
- O(1) structural-sharing `snapshot()`; uniquely-owned edits mutate in place
  with zero allocation on the keystroke path.
- Borrowed backing (`fromBacking`) for mmap'd / caller-owned immutable spans.
- Lazy content: unrealized holes of known length (`fromUnrealized` /
  `realize`) with no I/O in the rope itself.
- Comptime metric dimensions (`RopeWith`), bidirectional chunk/scalar
  cursors, `split` / `append` / `eql`, a zero-copy `std.Io.Reader` adapter,
  search, line iteration, and `AnchorSet` for bulk edit-stable positions.

### Collaboration layer

- `TextDoc`: eg-walker collaborative text document over a causal event graph,
  with FugueMax ordering. `merge` returns a byte-space `[]Edit` stream.
- `ObjectDoc`: collaborative JSON-shaped object tree (maps, lists, inline
  scalars, text nodes) with honest multi-value conflict sets.
- Opaque portable versions, run-RLE wire encoding (`WireFormat.unit` for
  pre-RLE decoders), and `serialize` / `open` persistence.
- `materializeAt` (time travel), identity anchors (`anchorAt` /
  `resolveAnchors`), `compact` (frozen base for bounded history), and
  partial bases (`openPartial` / `realizeBase`).
- Persistent merge walker over an order-statistics sequence: sync cost is
  O(log n) per event since the last merge, not a replay from genesis.
- wasm32-wasi build target (`zig build wasm`).
- Hostile-input hardening (atomic batch rejection, fuzz-gated) and a
  multi-peer convergence oracle.
