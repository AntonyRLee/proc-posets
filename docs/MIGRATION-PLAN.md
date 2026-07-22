# procposets — migration & self-containment plan

Goal: procposets is a **standalone, independently-releasable** library (poset / cospan /
string-diagram calculation, simulation, estimation, comparison). Finish extracting all reusable
code out of the three consumer repos, delete every duplicate, and remove all consumer/pipeline
detail so the package stands alone. **procposets is NOT any paper's artifact** — it releases on its
own track; paper-specific coupling is leakage to remove, not release-gating.

Drafted 2026-07-22 from a 5-agent read-only audit of procposets + the three consumers.

---

## Status at a glance (verified against live repos 2026-07-22)

- **Internal 8-phase refactor: essentially COMPLETE** (Phases 0–7 landed) — but on the **unmerged
  branch `output-sensitive-extract-signature`** (tip `bc9e7c0`, 362 passed / 13 skipped). `main` is
  still **pre-refactor** at `3c4c503`. A release cannot be cut until this lands on `main`.
- **PMN** (`poset-mixture-npmle`): cut over. One 105-line shim, zero library logic. Aliases migrated
  on unmerged branch `arl/alias-cutover` (1 commit ahead of PMN `main` `6a6cd2b`).
- **SPM** (`stochastic_process_mining/experiments`): cut over. 43-line `spm/` shim, zero logic.
  5 demo/`paper_figures` scripts + 10 tests still import `spm.*`.
- **cpm** (`string-diagram-process-mining/sim`): **NOT ported.** Full ~9,475 LoC copy; ~211
  `from cpm…` sites in 62 files; the **last independent byte-exact oracle** for 13 goldens.

## The one hard sequencing constraint

`sim/cpm` is the last live oracle for the `procposets == cpm` cross-checks (`tests/test_golden_*`
+ parts of `tests/regression/test_cpm_*`). So: **move/verify every un-reproduced cpm module INTO
procposets (with a golden) while `cpm/` is still on disk → flip all imports → delete `cpm/` last →
then delete the scaffolds that pointed at it.** Nothing that deletes `cpm/` or a `*_matches_cpm`
scaffold may run before its replacement golden is green.

Risk tags (from the refactor plan): `none` (docstring/comment) · `value-preserving` · `changes-values`
(gate on full golden suite) · `api-breaking` (needs alias/coordinated bump) · `blocked` (on another step).

---

## Workstream 1 — Port the cpm consumer (only remaining *code* migration)

- [x] **1.1 Wire the dependency** — DONE (sim WS 1.1): `procposets[pm4py,viz]` + `[tool.uv.sources]`
  editable path added to `sim/pyproject.toml`; `uv sync` no longer wipes the editable install. `none`
- [x] **1.2 Move `cpm/simulate.py` (cospan→OCEL) into procposets, with a golden** — DONE 2026-07-22
  (`2f94c21`): landed as `procposets/cospan/simulate.py` behind `[pm4py]` (avoids the top-level
  `simulate.py` sampler clash); `test_golden_ocel` cross-check green. `changes-values`
- [x] **1.3 Move `cpm/faithful_simulate.py` + `cpm/discover.py` into procposets, with goldens** —
  DONE 2026-07-22 (`2f94c21`): `faithful_simulate`/`discover` ported behind `[pm4py]` with OCEL
  goldens; `splice_representation_from_signature` exposed as a library fn. `changes-values`
- [x] **1.4 Close 4 golden gaps for already-reproduced modules** — DONE 2026-07-22 (`e6cf885`,
  `tests/test_golden_analysis.py`, suite 370/13): `signature_diff`, `typebalance`, full
  `class_extraction` structural equality, `gamma_normalize`, and silent-elimination (the engine's
  `without_silent`/`remove_silent`, distinct from `degenerate_filtered`). All byte-exact ports, so the
  goldens lock procposets == live cpm before cpm is deleted. `none`
- [x] **1.5 Flip `from cpm.X` → `from procposets.X`** — DONE 2026-07-22 (sim `9d9aff9`, **42 files**,
  sim suite **209 passed on procposets**). Flipped the transitive-closed pure-library-consumer set;
  the viz split/`unroll` split/`cpm.simulate`→`cospan.simulate` collision all handled; adapted 3 api
  evolutions (`canonical_key`→`label_multiset_key`, viz `LayoutStyle` threading). **20 files kept on cpm**
  (glue-entangled + the build_running_example fixture cluster that shares a fixture with the glue-bound
  test_explain_split) → they flip in 1.7 when the glue relocates. `api-breaking`
- [ ] **1.6 Repoint the manuscript figure chain** — `master_signature.py`, `trend_signature.py`,
  `ground_signature.py`, `liss_faithful_signature.py`, the `render_ed_chest_pain_*.py` scripts still
  build on `cpm.*`. This is what makes the paper's artifact depend on cpm today. `api-breaking`
- [ ] **1.7 Relocate cpm pipeline/CLI glue OUT to `sim/demos/` (NOT into the library)** —
  `pipeline/generate/spec/artifacts/report/conversion_consistency/binding_probe/signature_cli/splice_cli`
  carry ED/Liss pipeline specifics. Keep as thin glue importing procposets. `api-breaking`
- [ ] **1.8 Retire the oracle suite** — convert `sim/tests` `*_matches_cpm` cross-checks to
  self-consistency (or delete) once cpm is gone. `blocked` (on 1.2–1.7)
- [ ] **1.9 Delete `sim/cpm/`** (~9,475 LoC) + drop the ad-hoc procposets install from `sim/.venv`.
  Verify sim's suite green on procposets-only. `blocked` (final teardown)

## Workstream 2 — Self-containment of procposets (in progress on branch `self-containment-sweep`)

- [x] **2.1 Strip consumer-doc provenance from source docstrings** — DONE 2026-07-22 (`2cb9e11`):
  ~40 consumer-doc refs, the `npmle.py` reverse-leak, and the `spm_viz.py` `cpm/vis.py` provenance
  all removed. Verified: the shipping tree (package source minus tests) now greps clean of every
  leak string — locked by the WS 4.3 release-gate test. `none`
- [ ] **2.2 Golden-test decision (self-containment crux)** — KEEP `tests/regression/test_{cpm,np,spm}_*`
  as the library's behaviour contract (already self-contained); only **de-name** prefixes + docstrings
  (`test_cpm_*`→cospan-suite, etc.). DELETE the `tests/test_golden_*` cross-check scaffolds (hardcode
  `/home/arl/Research`, import old consumer code) — PMN/SPM-side once their old modules are gone,
  cpm-side `blocked` on 1.9. `value-preserving` / `blocked`
- [x] **2.3 Rename `viz/spm_viz.py`** — DONE 2026-07-22 (`7df83a9`): renamed to
  `viz/signature_diagram.py`; consumer imports migrated in lockstep (sim flipped in WS 1.5). `api-breaking`
- [ ] **2.4 Scope the `occn/` miner** — `fhm.py` = the Liss et al. FHM discovery miner. Keep as a
  clearly-labelled `[pm4py]` **inbound adapter** (already out of top-level `__all__`), or push back to
  the cpm consumer if the library should carry no miner. `changes-values`
- [ ] **2.5 Promote SPM-reached privates to public** — `discrete._alphabet`, `matrix._block_sequence`,
  `equivalence._activity_pomsets` (SPM imports these). `value-preserving`
- [ ] **2.6 Remove leftover compat shims** — root `equivalence.py` shim + remaining Phase-4 aliases
  (`is_boundary_label`, …), once consumers stop using the old paths. Keep the intentional public aliases
  (`count_linear_extensions`, `Model`). `api-breaking` / `blocked`

## Workstream 3 — Finish PMN & SPM (small)

- [ ] **3.1 PMN: merge `arl/alias-cutover`** (1 commit ahead of PMN `main`); fix stale `CLAUDE.md`
  "107 tests" → 201. `value-preserving`
- [ ] **3.2 SPM: re-point 5 demo/`paper_figures` scripts + 10 tests** off `spm.*` → `procposets.*`
  (1:1 names); delete the 8 tests procposets already mirrors (or keep as smoke). `value-preserving`
- [ ] **3.3 SPM: delete dead `spm/viz.py`, then the `spm/` shim** + its wheel-packaging; update the
  stale `experiments/README.md` (still calls `spm/` "the reusable library"). `none` / `blocked`

## Workstream 4 — Independent-release hygiene

- [x] **4.1 Add a LICENSE** — DONE 2026-07-22: MIT `LICENSE` (© Antony Lee) + `pyproject` metadata
  (`license = "MIT"`, `license-files`, `authors`, `[project.urls]`, classifiers). Wheel builds; METADATA
  carries `License-Expression: MIT` + `licenses/LICENSE`. `none`
- [x] **4.2 Ship `procposets/py.typed`** — DONE 2026-07-22: PEP 561 marker added + `Typing :: Typed`
  classifier; verified present in the built wheel (`procposets/py.typed`). `none`
- [x] **4.3 Add a release-gate grep** — DONE 2026-07-22 (`procposets/tests/test_release_gate.py`):
  scans the shipping tree (package source minus `tests/`) and fails on `/home/arl`, `sim/cpm`,
  the three consumer repo names, and the consumer-doc filenames. Currently green. `none`
- [ ] **4.4 Merge `output-sensitive-extract-signature` → `main`** — everything above assumes the
  refactor is mainline; it currently isn't. (Owner decision.) `none`
- [ ] **4.5 (open Phase-0 latents)** triage 3 correctness smells (compose LoopBox dedup bypass,
  morphism_schema unweighted Counter, simulate `trees[0]` shared-alphabet) + 2 import-time side effects
  (`outbound` `filterwarnings`, `spm_viz` `matplotlib.use`). `changes-values` / `value-preserving`

### NOT procposets' concern (push to the geometry-paper repo)
Split-Miner anomaly investigation · supplement reproduction commands · the Zenodo DOI "to pin in the
paper." These are consumer/paper items. (The one real library item in that TODO section — fleet-wide
`|X|` in `distance.py` — is already done on `main` at `3a62e18`.)

### What stays OUT of the library (consumer drivers, keep in place)
cpm pipeline/CLI glue (→ `sim/demos/`) · SPM exhibit/figure scripts + Split-Miner baseline · PMN
benchmark drivers · all three repos' demos. The library exposes the primitives these call.

---

## Recommended order
2.1 (in progress) → 1.1 + 2.5 + 3.1/3.2 (safe, unblock) → 1.2–1.4 (move cpm logic in, with goldens) →
1.5–1.7 (flip imports, relocate glue) → 1.8–1.9 (delete cpm) → 2.2/2.6 scaffold+shim removal →
4.x release hygiene → 4.4 merge to `main`.
