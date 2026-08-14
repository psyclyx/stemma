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

## Graph layer baseline — 2026-08-14 (FugueMax walker)

```
graph doc-typing ascii             64 ns/op  (best       60)  n=11
graph merge linear 4k units      3619 ns/op  (best     3590)  n=5
graph merge concurrent 1k+1k     2140 ns/op  (best     2079)  n=5
```

- **doc-typing ~60 ns vs 20 ns bare rope**: the local collab tax is event
  recording (frontier snapshot + graph append). No CRDT work happens on
  the local path — this is bookkeeping only. (Initial linked-list walker
  measured 56 ns; the delta is compilation-unit layout scale, not a path
  change — local typing never touches the walker.)
- **The FugueMax swap halved merge cost for free**: the YjsMod integrate
  loop wanted array-order comparisons, and the resulting seq-array walker
  state (vs the original linked list) merged 4k linear units at 3.6 µs/unit
  vs 7.3, concurrent at 2.1 vs 3.5 — cache-friendly scans beating pointer
  chasing before any deliberate optimization.
- **Merges remain the deliberate v1 baseline**: replay-from-genesis/base
  over unit events, O(n·m)-ish by design (correctness first). The
  optimization ladder, each rung to be landed against these numbers with
  convergence tests green:
  1. run-RLE event storage (typing runs collapse ~64×);
  2. LCA-bounded replay with placeholder runs (eg-walker's actual
     contribution — merge cost proportional to divergence, not history;
     the placeholder machinery already exists for compaction bases);
  3. B-tree walker state (positional scans O(log n)).
  `compact()` is the fourth lever in practice: a compacted document's
  replay cost is proportional to post-base history, not lifetime history.

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
- **2026-08-14 — unrealized-content (holes) feature**: guarding the delete
  fast path with an isRealized walk cost random-edit ~+8% (351 vs ~320 ns).
  Fixed with a sticky `may_have_holes` flag (set by fromUnrealized,
  inherited by snapshot/split/append, never cleared): ropes that never
  touched lazy content skip every hole check. Re-measured 320 ns — back in
  the noise band. Lesson repeated: any check added to a hot path needs a
  "does this rope even need it" gate.
- **2026-08-11 — consumer-surface additions** (find/lineIterator/AnchorSet/
  PointUtf16): random-edit median moved 322→334 ns in a same-session
  stash A/B despite the new code never executing on that path — attributed
  to code-layout/alignment effects of a larger compilation unit (best-case
  312-316 ns unchanged, all other workloads flat). Accepted; revisit only
  if it compounds.
