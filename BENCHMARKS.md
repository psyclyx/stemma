# Benchmarks

Harness: `zig build bench` (dev/bench, always ReleaseFast). Each line is the
median over n blocks of a deterministic PRNG workload, reported per work
unit; `best` is the fastest block. `zig build bench -- <filter>` additionally
sweeps knob instantiations (chunk capacity, branch factor, atomics).

Measured 2026-08-17 on an AMD Ryzen 9 5950X, Linux 7.0.14-zen1, Zig 0.16.0
(npins nixpkgs), rope defaults `chunk_capacity=512, branch_factor=16,
thread_safe=true`.

## Rope

```
typing ascii                       19 ns/op  (best       19)  n=11
typing unicode                     21 ns/op  (best       21)  n=11
typing ascii +snap                 19 ns/op  (best       18)  n=11
random-edit 1MiB                  333 ns/op  (best      327)  n=11
conv offsetToPoint                253 ns/op  (best      228)  n=11
conv pointToOffset                369 ns/op  (best      345)  n=11
conv offsetToUtf16                217 ns/op  (best      194)  n=11
load fromSlice 32MiB            1.823 GB/s   (best    1.937)  n=11
load fromBacking 32MiB          2.998 GB/s   (best    3.043)  n=11
load chunk scan 32MiB          22.702 GB/s   (best   24.813)  n=11
load snapshot+drop                  3 ns/op  (best        2)  n=11
```

- **typing** is the uniqueness fast path: in-place leaf mutation, zero
  allocation per keystroke. Holding a live snapshot (`+snap`, refreshed every
  1024 ops) is statistically identical — spine copy-on-write amortizes.
- **snapshot+drop at ~3 ns** is the O(1) refcount-bump claim, measured.
- **fromBacking vs fromSlice** (3.0 vs 1.8 GB/s): borrowed leaves skip the
  copy; what remains is the summary scan, the inherent floor for a
  metrics-carrying rope.

## Collaboration layer

```
collab doc-typing ascii            64 ns/op  (best       62)  n=11
collab merge linear 4k units      267 ns/op  (best      266)  n=5
collab wire 4k units             4119 B rle vs   44683 B unit (10x)
collab merge concurrent 1k+1k     827 ns/op  (best      808)  n=5
collab merge growing x256         380 ns/op  (best      367)  n=3
collab merge growing x2048        440 ns/op  (best      435)  n=3
```

- **doc-typing ~65 ns vs ~20 ns bare rope**: the local collab tax is event
  recording (frontier snapshot + graph append). No CRDT work happens on the
  local path — this is bookkeeping only.
- **Wire (run-RLE frames)**: a monotone typing or deletion burst encodes as
  one frame — ~10× smaller than per-unit on the 4k-unit typing history.
  `WireFormat.unit` still serves decoders that want the expanded form; runs
  decode back to unit events, so the graph model and replay are untouched.
- **merge growing** measures the shape a real session sees: many sequential
  single-event merges into a document that keeps growing. `merge` resumes a
  persistent walker over an order-statistics sequence (an implicit treap
  keyed by document position), so each merge replays only the events since
  the previous one, each in O(log n) — cost stays roughly flat as history
  grows (x256 → x2048 is 8× the history for ~1.2× the cost) rather than
  climbing with document lifetime. `compact` bounds it further by freezing
  all-peers-stable history into a base.

## ObjectDoc batch validation — O(n²) → O(n) — 2026-08-23

```
collab merge object 4096 scalars      388 ns/op  (best      379)  n=3
collab merge object 32768 scalars     422 ns/op  (best      415)  n=3
collab merge object 131072 scalars    450 ns/op  (best      443)  n=3
```

- **`ObjectDoc.Decoder.validate` had the same hidden O(n²) `seenEarlier`
  shape `TextDoc`'s validate was fixed for on 2026-08-15 (above) — that
  fix only landed on `TextDoc`; `ObjectDoc`'s copy stayed a per-event
  linear rescan of the batch prefix. A whole-file commit (one batch, one
  event per scalar — weft's non-bulk save path) hits it hardest: merging a
  single N-event batch into an empty doc went 3.2 µs/op (N=4096) → 33
  µs/op (N=32768) → 146 µs/op (N=131072) before the fix — visibly
  quadratic (8× N → ~10× per-op, ~80× total time). Replaced with the same
  incremental per-agent seen-range `TextDoc` already used (built in one
  pass alongside the existing contiguity/parent checks, same
  accept/reject decisions on every input — the atomic-rejection fuzz
  batteries pass unmodified): now flat at ~410–460 ns/op regardless of N —
  a 131072-scalar merge went 19.1s → 59ms, ~320× faster.
