# stemma

> **stemma** |ˈstɛmə| *n., pl.* **stemmata**
> 1. a family tree; a pedigree.
> 2. *(textual criticism)* a diagram showing the descent and relationships
>    of the surviving manuscripts of a text.

A library for the distributed, causal story of a text. The long-term shape is
an event-graph CRDT library in the eg-walker family — a causal DAG of editing
events, walked on demand to merge divergent histories, generic over
materialized types — with text as the first and flagship materializer. Before
CRDTs had the problem, scribes did: hand-copied manuscripts are distributed
replicas diverging edit by edit, and reconstructing their descent is exactly
the discipline this library is named for.

What ships today is the flagship's foundation: `Rope`, a persistent,
snapshot-able UTF-8 text buffer built to be the fastest correct core for a
text editor.

## The rope

A B-tree of UTF-8 chunks whose nodes cache aggregate summaries (bytes /
scalars / UTF-16 units / newlines), so metric queries and coordinate
conversions — byte ⇄ point ⇄ UTF-16 ⇄ scalar — are one O(log n) descent.
The design follows the structure shipping in Zed (SumTree), xi-rope, and
Ropey, with three tradeoffs those implementations had to pick a side on
dissolved instead:

- **In-place vs snapshots.** Nodes are refcounted: `snapshot()` is O(1)
  structural sharing, but an edit whose spine is uniquely owned mutates in
  place — zero allocation on the keystroke path. You pay copy-on-write only
  while a snapshot is actually alive.
- **Owned vs borrowed storage.** Leaves either own their chunk or borrow a
  span of caller-provided immutable backing (`fromBacking`, e.g. an mmap'd
  file). A multi-gigabyte file opens as a handful of borrowed spans — near
  zero resident memory — and edits splinter only the leaves they touch.
- **Metrics vs memory.** Summary dimensions are comptime options
  (`RopeWith(.{...})`): a build that doesn't speak LSP drops UTF-16 tracking
  and pays zero bytes and zero scan work for it. `Rope` is the
  all-dimensions default.
- **Eager vs lazy.** A rope over remote or unfetched data starts as an
  unrealized *hole* of known length (`fromUnrealized` — O(1), no reads);
  the caller fetches windows on its own transport and `realize`s them
  (offsets never shift — anchors are unaffected). Byte-domain operations
  work on holes; content access panics deterministically until realized
  (`isRealized`/`unrealized` are the fetch list); content metrics count
  realized text only and converge as fetching proceeds. Seeking the middle
  of a 10 GB network file is arithmetic plus one window fetch. The rope
  itself does no I/O, ever.

## Scope — decomplected on purpose

**Owns:** the sequence (always valid UTF-8), metrics, coordinate conversion,
O(1) snapshots, bidirectional chunk/scalar cursors, `split`/`append`, `eql`,
a zero-copy `std.Io.Reader` adapter, and the `Edit` + `Anchor.shift`
primitives that keep caller positions valid across edits.

**Leaves to the caller:** grapheme/word/display-column segmentation (the
atomic unit is the Unicode scalar), cursors and selections as editor state (a
slice of `Anchor`s you shift), undo/redo *policy* (compose it from snapshots
and edits), files/mmap lifecycle, and syntax. Collaboration lives in the
`graph` namespace — the event-graph engine, currently a design stub — so the
single-user path pays zero collaboration tax.

Allocation is explicit and unmanaged: no type stores an allocator; the same
allocator must serve a rope and everything derived from it, and must be
thread-safe if snapshots cross threads.

## The graph layer

`graph.TextDoc` is the collaboration surface: a rope plus the causal event
graph that explains it. Local edits record causally-stamped events (no CRDT
metadata on the document — the eg-walker model); `merge` integrates remote
batches and returns the same byte-space `[]Edit` stream local edits produce,
so `AnchorSet`s and cursors survive remote edits with zero extra machinery.
Versions are opaque portable tokens; `eventsSince` is wire-ready;
`serialize`/`open` persist the graph (the document of record).

Beyond the basics: **FugueMax ordering** (maximally non-interleaving in both
directions — locked by block-contiguity tests before any replica ever
shipped); **`materializeAt`** (time travel to any known version);
**identity anchors** (portable name+seq positions for remote cursors that
survive concurrent edits, batch-resolvable on any replica); **`compact`**
(collapse all-peers-stable history into a frozen base — graph growth
bounded by post-base activity, not document lifetime); and a **wasm32
target** (`zig build wasm`) so browser peers can speak the protocol.

Hostile input cannot crash a replica: malformed and malicious batches
(including out-of-range positions) are rejected atomically, fuzz-gated.
Convergence is oracle-tested: multi-peer seeded gossip, concurrent conflict
shapes, batch splitting, fuzzed sessions — all replicas byte-identical,
every merge's edit stream validated. Deferred to callers: transport,
presence payloads, collaborative undo policy.

## Status

Rope and graph layer implemented, tested, and benchmarked (see
[BENCHMARKS.md](BENCHMARKS.md)). Headlines: ~20 ns/keystroke bare, ~56 ns
through `TextDoc` (the collab tax is bookkeeping only), ~3 ns snapshots,
27 GB/s chunk scans. v1 merges replay from genesis — correctness-first by
design; the optimization ladder (run-RLE, LCA-bounded replay, B-tree walker
state) is documented against baseline numbers and lands rung by rung with
the convergence oracle as the gate.

## Develop

```sh
direnv allow           # or: nix-shell
zig build              # build the static library
zig build test         # run the test suite
```

The Zig toolchain (`zig_0_16`) comes from the npins-pinned nixpkgs, never the
ambient `PATH`.
