# Benchmarks

Harness: `zig build bench` (dev/bench, always ReleaseFast). Each line is the
median over n=11 blocks of a deterministic PRNG workload, reported per work
unit; `best` is the fastest block. `zig build bench -- <filter>` additionally
sweeps knob instantiations (chunk capacity, branch factor, atomics).

## Baseline — 2026-08-10

AMD Ryzen 9 5950X, Linux 7.0.14-zen1, Zig 0.16.0 (npins nixpkgs), defaults
`chunk_capacity=512, branch_factor=16, thread_safe=true`.

```
typing ascii                       20 ns/op  (best       18)  n=11
typing unicode                     21 ns/op  (best       20)  n=11
typing ascii +snap                 19 ns/op  (best       18)  n=11
random-edit 1MiB                  314 ns/op  (best      312)  n=11
conv offsetToPoint                229 ns/op  (best      223)  n=11
conv pointToOffset                342 ns/op  (best      333)  n=11
conv offsetToUtf16                189 ns/op  (best      189)  n=11
load fromSlice 32MiB            1.923 GB/s   (best    2.010)  n=11
load fromBacking 32MiB          3.142 GB/s   (best    3.174)  n=11
load chunk scan 32MiB          27.594 GB/s   (best   28.110)  n=11
load snapshot+drop                  3 ns/op  (best        2)  n=11
```

Reading the numbers:

- **typing** is the uniqueness fast path: in-place leaf mutation, zero
  allocation per keystroke. Holding a live snapshot (`+snap`, refreshed every
  1024 ops) is statistically identical — spine copy-on-write amortizes.
- **snapshot+drop at ~3 ns** is the O(1) refcount-bump claim, measured.
- **fromBacking vs fromSlice** (3.1 vs 1.9 GB/s): borrowed leaves skip the
  copy; what remains is the summary scan, which is the inherent floor for a
  metrics-carrying rope.

## Graph layer baseline — 2026-08-14

```
graph doc-typing ascii             56 ns/op  (best       54)  n=11
graph merge linear 4k units      7304 ns/op  (best     7249)  n=5
graph merge concurrent 1k+1k     3485 ns/op  (best     3424)  n=5
```

- **doc-typing 56 ns vs 20 ns bare rope**: the local collab tax is event
  recording (~36 ns/keystroke: frontier snapshot + graph append). No CRDT
  work happens on the local path — this is bookkeeping only.
- **Merges are the deliberate v1 baseline**: replay-from-genesis over unit
  events in a linked list, O(n·m)-ish by design (correctness first). The
  optimization ladder, each rung to be landed against these numbers with
  convergence tests green:
  1. run-RLE event storage (typing runs collapse ~64×);
  2. LCA-bounded replay with placeholder runs (eg-walker's actual
     contribution — merge cost proportional to divergence, not history);
  3. B-tree walker state (positional scans O(log n)).

## Tuning ledger

- **2026-08-10 — chunk 256→512, branch 8→16** (sweep over c128/c256/c512/
  c1024b16 × b4/b8/b16 × atomics): random-edit −17%, chunk scan +176%,
  fromSlice +23%, typing neutral; point conversions +12–20% (absolute cost
  still sub-µs; ~25 µs per 100 rendered lines). Adopted.
- **2026-08-10 — thread_safe=false** is neutral on every workload on x86-64
  (uncontended atomics; snapshot 1 ns vs 3 ns). Not worth a non-default;
  kept as an option for platforms where it matters.
- **2026-08-10 — c1024b16**: best scan (65 GB/s) and load (2.2 GB/s) but
  conversions +40–60% and typing +20% — rejected as default; viable
  specialization for scan-dominated read-mostly workloads via `RopeWith`.
- **2026-08-11 — consumer-surface additions** (find/lineIterator/AnchorSet/
  PointUtf16): random-edit median moved 322→334 ns in a same-session
  stash A/B despite the new code never executing on that path — attributed
  to code-layout/alignment effects of a larger compilation unit (best-case
  312-316 ns unchanged, all other workloads flat). Accepted; revisit only
  if it compounds.
