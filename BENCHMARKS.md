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
typing unicode                     20 ns/op  (best       20)  n=11
typing ascii +snap                 18 ns/op  (best       17)  n=11
random-edit 1MiB                  360 ns/op  (best      343)  n=11
conv offsetToPoint                228 ns/op  (best      225)  n=11
conv pointToOffset                338 ns/op  (best      334)  n=11
conv offsetToUtf16                194 ns/op  (best      189)  n=11
load fromSlice 32MiB            1.957 GB/s   (best    2.031)  n=11
load fromBacking 32MiB          3.009 GB/s   (best    3.038)  n=11
load chunk scan 32MiB          27.652 GB/s   (best   28.566)  n=11
load snapshot+drop                  2 ns/op  (best        2)  n=11
```

- **typing** is the uniqueness fast path: in-place leaf mutation, zero
  allocation per keystroke. Holding a live snapshot (`+snap`, refreshed every
  1024 ops) is statistically identical — spine copy-on-write amortizes.
- **snapshot+drop at ~2 ns** is the O(1) refcount-bump claim, measured.
- **fromBacking vs fromSlice** (3.0 vs 2.0 GB/s): borrowed leaves skip the
  copy; what remains is the summary scan, the inherent floor for a
  metrics-carrying rope.

## Collaboration layer

```
collab doc-typing ascii            61 ns/op  (best       60)  n=11
collab merge linear 4k units      258 ns/op  (best      258)  n=5
collab wire 4k units             4119 B rle vs   44683 B unit (10x)
collab merge concurrent 1k+1k     792 ns/op  (best      774)  n=5
collab merge growing x256         379 ns/op  (best      373)  n=3
collab merge growing x2048        447 ns/op  (best      437)  n=3
```

- **doc-typing ~60 ns vs ~20 ns bare rope**: the local collab tax is event
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
