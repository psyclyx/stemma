# stemma

> **stemma** |ˈstɛmə| *n., pl.* **stemmata**
> 1. a family tree; a pedigree.
> 2. *(textual criticism)* a diagram showing the descent and relationships
>    of the surviving manuscripts of a text.

An event-graph CRDT library in the eg-walker family (Gentle & Kleppmann,
EuroSys 2025), built around an editor-grade text rope. The durable
artifact is the causal event graph — the *stemma* of a document; the CRDT
state is transient, rebuilt only while merging divergent histories. Text
is the primary materializer, but the graph is generic over its operation
payload.

The name is the discipline. Before CRDTs had the problem, scribes did:
hand-copied manuscripts are distributed replicas diverging edit by edit,
and reconstructing their descent is what a stemma records.

Two layers, one flat API surface. `Rope` is usable on its own — the
single-user path carries no collaboration state — and `TextDoc` /
`ObjectDoc` add the causal graph on top of it.

## The rope

A B-tree of UTF-8 chunks whose nodes cache aggregate summaries (bytes /
scalars / UTF-16 units / newlines), so metric queries and coordinate
conversions — byte ⇄ point ⇄ UTF-16 ⇄ scalar — are one O(log n) descent.
The design follows the structure shipping in Zed (SumTree), xi-rope, and
Ropey, with four tradeoffs those implementations had to pick a side on
left open instead:

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

## Scope

**The rope owns:** the sequence (always valid UTF-8), metrics, coordinate
conversion, O(1) snapshots, bidirectional chunk/scalar cursors,
`split`/`append`, `eql`, a zero-copy `std.Io.Reader` adapter, and the
`Edit` + `Anchor.shift` primitives that keep caller positions valid across
edits.

**Left to the caller:** grapheme/word/display-column segmentation (the
atomic unit is the Unicode scalar), cursors and selections as editor state
(a slice of `Anchor`s you shift), undo/redo *policy* (compose it from
snapshots and edits), files/mmap lifecycle, syntax, transport, and
presence. Collaboration is a distinct layer, so the single-user path pays
no collaboration tax.

Allocation is explicit and unmanaged: no type stores an allocator; the
same allocator must serve a rope and everything derived from it, and must
be thread-safe if snapshots cross threads.

## The collaboration layer

Local edits record causally-stamped events; there is no CRDT metadata on
the document itself (the eg-walker model). `merge` integrates a remote
batch and returns the same byte-space `[]Edit` stream local edits produce,
so `AnchorSet`s and cursors survive remote edits with no extra machinery.
Two materializers share the graph, the wire format, and the sync protocol:

- **`TextDoc`** — a collaborative text document (a `Rope` plus its graph).
- **`ObjectDoc`** — a collaborative JSON-shaped object tree: maps, lists,
  inline scalars, and text nodes (each a full sequence CRDT with the same
  ordering). Map reads return a deterministic winner plus the honest
  multi-value conflict set — concurrent writes both survive, resolved by
  caller policy.

Both share:

- **FugueMax ordering** — maximally non-interleaving concurrent insertions
  in both directions, fixed by block-contiguity tests.
- **Opaque portable versions** — `version` / `eventsSince` are wire-ready
  and run-RLE encoded (a typing burst is one frame; `WireFormat.unit`
  serves pre-RLE decoders). Replica-local identifiers never cross the wire.
- **Persistence** — `serialize` / `open` store the event graph, which is
  the document of record; the materialized value is derived from it.

Both also carry **identity anchors** (portable positions that survive
concurrent edits — `anchorAt` / `resolveAnchors` on `TextDoc`;
`objectAnchorAt` / `resolveObjectAnchors` per text object on `ObjectDoc`,
isolated per object) and **`compact`** (collapse all-peers-stable history
into a frozen base, so graph growth tracks post-base activity, not
document lifetime — on `ObjectDoc`, text history folds while map/list
register history is deliberately retained, and a stable point with list
content in its causal past refuses rather than risk re-ordering register
supersession).

`TextDoc` additionally carries **`materializeAt`** (time travel to any
known version), **partial bases** (`openPartial` / `realizeBase` — a
replica of a huge document fetches only the base spans it touches; a
merge into an unrealized span rejects whole with `error.Unrealized`,
realize-then-retry), and a **wasm32 target** (`zig build wasm`) so
browser peers speak the same protocol.

Hostile input cannot crash a replica: malformed and malicious batches
(including out-of-range positions) are rejected atomically and leave the
document untouched, fuzz-gated. Convergence is oracle-tested — multi-peer
seeded gossip, concurrent conflict shapes, batch splitting, fuzzed
sessions — with all replicas byte-identical and every merge's edit stream
validated. Transport, presence payloads, and collaborative undo policy are
the caller's.

Not yet, and ledgered as such: an identity-preserving move/reparent op
(parent-register design validated in a test-only sketch), list-content
compaction bases and list-object anchors for `ObjectDoc`, `ObjectDoc`
partial checkout, Peritext-style rich-text marks, and incremental
persistence.

## Status

v0.1.0. The rope and both collaboration materializers are implemented,
tested, and benchmarked (see [BENCHMARKS.md](BENCHMARKS.md)). Headlines on
a Ryzen 9 5950X: ~20 ns/keystroke bare, ~60 ns through `TextDoc` (the
collab tax is event bookkeeping, no CRDT work on the local path), ~2 ns
snapshots, 27 GB/s chunk scans.

Merges resume a persistent walker over an order-statistics sequence, so a
sync costs O(log n) per event in the events since the last merge, not a
replay from genesis. `compact` bounds it further for long-lived documents.

## Develop

```sh
direnv allow           # or: nix-shell
zig build              # build the static library
zig build test         # run the test suite
zig build bench        # run the benchmarks (ReleaseFast)
zig build wasm         # build the wasm32-wasi library
```

The Zig toolchain (`zig_0_16`) comes from the npins-pinned nixpkgs, never
the ambient `PATH`.
