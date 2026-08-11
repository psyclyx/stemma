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

## Scope — decomplected on purpose

**Owns:** the sequence (always valid UTF-8), metrics, coordinate conversion,
O(1) snapshots, chunk/scalar cursors, `split`/`append`, and the `Edit` +
`Anchor.shift` primitives that keep caller positions valid across edits.

**Leaves to the caller:** grapheme/word/display-column segmentation (the
atomic unit is the Unicode scalar), cursors and selections as editor state (a
slice of `Anchor`s you shift), undo/redo *policy* (compose it from snapshots
and edits), files/mmap lifecycle, and syntax. Collaboration lives in the
`graph` namespace — the event-graph engine, currently a design stub — so the
single-user path pays zero collaboration tax.

Allocation is explicit and unmanaged: no type stores an allocator; the same
allocator must serve a rope and everything derived from it, and must be
thread-safe if snapshots cross threads.

## Status

Rope implemented, tested, and benchmarked. Tests: randomized oracle against a
reference implementation (contents, metrics, conversions, snapshot
immutability, leak checks) in Debug, ReleaseSafe, and ReleaseFast. Baseline
numbers and the tuning ledger live in [BENCHMARKS.md](BENCHMARKS.md) —
headlines: ~20 ns/keystroke (identical with a live snapshot), ~3 ns
snapshots, 27 GB/s chunk scans. The event-graph engine is next; it will be
type-generic from its first commit.

## Develop

```sh
direnv allow           # or: nix-shell
zig build              # build the static library
zig build test         # run the test suite
```

The Zig toolchain (`zig_0_16`) comes from the npins-pinned nixpkgs, never the
ambient `PATH`.
