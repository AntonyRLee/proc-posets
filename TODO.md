# TODO — procposets

Dev tasks. Newest at top.

## Split Miner anomaly (parked from the geometry paper, 2026-07-18)

**Context:** In the geometry paper's Synthea fleet exhibit (demo/12_synthea_fleet), Split Miner is the
one miner that does NOT launder the concurrency-vs-balanced-choice pair C3/C4 to zero — it returns a
nonzero C3–C4 distance (1.78) and erratic clustering (ARI 0.54±0.22), where the other four miners
(Alpha, Heuristics, Inductive, ILP) all correctly collapse to 0.000. Since the flattened C3 and C4 logs
have **identical trace distributions**, any separation is provably a finite-sample fitting artefact, not
real signal. In the paper this is now handled honestly (a neutral footnote states it is an
implementation artefact under separate investigation; the red proofing note was removed) — the
root-cause diagnosis is parked HERE.

**Task:** determine why pm4py's Split Miner fabricates per-site structure between identically-distributed
logs.
- [ ] Reproduce: run Split Miner via pm4py on the C3 and C4 per-site logs (demo/12) and confirm the
      nonzero C3–C4 + ARI spread; pin the pm4py version (the paper's suspicion was a defect in pm4py's
      recent Split-Miner integration — verify or refute).
- [ ] Isolate: is it (a) a genuine pm4py Split-Miner integration bug, (b) Split Miner fitting per-site
      sampling noise into spurious structure (concurrency vs sequence flips on tiny frequency
      differences), or (c) the leg-bundle concurrency read on the BPMN output being invalid on
      concurrency-rich/small logs? (Prior journal note flagged Split Miner as "erratic on
      concurrency-rich/small logs — fabricates structure between identical sites".)
- [ ] Decide the fix: pin/patch the pm4py call, exclude Split Miner from the fleet harness, or add a
      guard. Feed the verdict back to the paper's footnote if it changes the honest framing.

## Release blockers (geometry paper ships procposets alongside — see that repo's docs/TODO.md)
- [ ] **Fleet-wide constant |X| in `distance.py smd_rows`** — currently builds a per-pair union of the
      two models' states (breaks the triangle inequality). Must take the fleet-wide state space as an
      argument. LOCKED decision, do not revert to per-pair. (Geometry paper §VI now states this;
      metric-hood depends on it.)
- [ ] Supplement reproduction commands must run against the released package (3 of 4 crashed after the
      geometry repo's demo purge).
- [ ] Repo URL + Zenodo DOI to pin in the paper's Code Availability paragraph at camera-ready.
