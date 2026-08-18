# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
