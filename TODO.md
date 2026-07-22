# TODO — procposets

Dev tasks. Newest at top.

## ✅ DONE — Output-sensitive `extract_signature` (kill the |B|×|F| product) (2026-07-20)

**Context:** `extract_signature` (engine.py:182) emits one generator per `(P,S)`
firing-choice context = `|B|×|F|` per activity. On a typed-merge OCPN a hub shared
across `k` object types is **exponential in `k`** (Bundestag `Beratung` ⇒ ≈1.4×10³⁰
generators; stage B never returns). Those all collapse onto the distinct-`CanonKey` count.

**Shipped:** `cospan/engine_fast.py :: extract_signature_fast` — coupled-component
decomposition (union-find on shared reachable type) + collapse to the CanonKey-relevant
`(is_gamma2, type)`-multiset per component, then Cartesian product over type-disjoint
components. Output-sensitive; byte-exact with the slow engine on the CanonKey set.

- [x] `cospan/engine_fast.py` (`extract_signature_fast`), exported from `cospan/__init__.py`.
- [x] `tests/regression/test_cpm_extract_fast.py` — fast CanonKeys == slow CanonKeys on
      running-example / mixed-graph / surface-termini / typed-hub / OCPN-wrapper fixtures
      (both `surface_termini`). 8 tests, green.
- [x] Full suite: **335 passed, 20 skipped** (`extract_signature` untouched, no regressions).
- [x] Opt-in wire: `from_ocpn.signature_from_ocpn(ocpn, *, canonical=…)` (default
      `canonical=False` → slow/full; `True` → fast). Wrapper lives at the ocpn-dict layer
      (OCEL→ocpn discovery stays in the consumer), not the plan's non-existent `discover.py`.
- ⚠️ **The plan's "ready-to-drop-in" code was NOT byte-exact** — the cross-check caught two
      bugs (no `surface=False` terminus strip; false type-independence under an *untyped*
      choice node). Corrected algorithm shipped + documented. See the plan's Correction note.
- **Plan + correction + evidence:** [`docs/2026-07-20-fast-signature-extraction.md`](docs/2026-07-20-fast-signature-extraction.md).
- Still open there: 8 further optimisations (adjacency index, `_traverse` memo,
  arity-range collapse, faster `without_silent`, OCPN reduction, …) — **not** done here.

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
