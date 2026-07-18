# Next-session handoff — procposets (+ geometry repo) — 2026-07-18

**Branch:** `main` (procposets), `arl/main` (stochastic_process_mining) ·
**Build/tests:** last-known `334 passed, 13 skipped` (procposets, after the distance fix) ·
**Uncommitted:** procposets — 3 untracked drops in `procposets/` (`split-miner.zip`, `split-miner-all.zip`, `split-miner.md`); geometry repo — `experiments/demo/12_synthea_fleet/{sm_investigation/, data_intervals/}` untracked (this session's deliverable), plus a pre-existing `transition.svg` change that is **not ours**.

## Where we are (1–2 sentences)
Two threads: (1) in **procposets**, P1 release-blocker #1 (fleet-wide constant |X| for normalised SMD) is fixed + committed and two stale golden tests were retired — all **3 commits ahead of origin, NOT pushed**. (2) In the **geometry repo**, a deep **Split Miner fabrication investigation** is well advanced with a living journal; root cause nailed, SM2 demo done, tasks #4 (build nemo-91) and #5 (dev contact + paper review) remain.

## Done this session
**procposets (committed on `main`, unpushed):**
- `da51d8e` retire closed pm_adapters golden seam → pinned language regression (fixed a hard test failure).
- `941548f` retire tautological spm.viz golden → pinned generator-signature regression (was a latent skip→fail landmine).
- `3a62e18` **distance: fleet-wide constant |X| for normalised SMD** — `smd_rows(states=…)`, `smd_pairwise(normalize=…)`. Raw + lone-pair paths byte-exact; new regression tests. **This is release blocker #1, LOCKED (do not revert to per-pair |X|).**

**geometry repo (`experiments/demo/12_synthea_fleet/`, UNCOMMITTED):**
- `sm_investigation/JOURNAL.md` — living journal; **§1 = plain-English root cause (front and centre)**.
- Root cause NAILED: **(A) inherent** — Split Miner over-pairs spurious concurrency on flattened small logs (the canonical reference does it too); **(B) implementation** — pm4py's default `epsilon≈0.1` makes the b,c call a coin-flip on small samples (epsilon sweep proves it: 0.1→flips, 0.5→stable). NEW: pm4py `sm2` is broken **even with real timestamps** (over-pairs + misses g‖h; ARI 0.346 vs hand-rolled 1.000).
- 50-seed fleet run, drill-down, epsilon sweep, SM2 interval demo (`data_intervals/`, `gen_intervals.py`) all done. Hand-rolled SM2 with timestamps = **ARI 1.000** (perfect, = paper Route B).
- Artifacts homed under `sm_investigation/{harnesses/, split_miner2/, raw_output/}`.

## Open decisions (need the user)
- **Push procposets?** 3 commits ahead of `origin/main`, not pushed (awaiting user per commit policy).
- **Commit the geometry investigation artifacts?** `sm_investigation/` + `data_intervals/` are untracked in the manuscript repo — recommend committing so nothing is lost, but it's the user's repo/branch (`arl/main`), so ask. Do **not** touch the unrelated `transition.svg` change.
- **Next: #4 (build nemo-91) or #5 (dev contact + paper review)?** User was mid-choice. The pm4py-on-intervals finding strengthens #5.

## Next steps — ranked
1. **#4 — Build nemo-91/bpmtk** as the 2nd reference: no build system, no SM `main`; hand-compile `au.edu.qut…splitminer` against its bundled `lib/*.jar` under JDK 8 + a tiny XES→BPMN driver. Its source differs 192–2976 lines from the fork, so it may behave differently.
2. **#5 — Draft the Split Miner developer note** (Augusto et al.): frame as questions, non-accusatory — the two concrete, reproducible issues are (a) default `epsilon=0.1` instability on small logs, (b) `sm2` over-pairing + missing real overlap concurrency on proper interval logs. Then review our own §IX exhibit + footnote.
3. **Resolve reference `-v2` integration** (deferred): jar reports `potential parallelisms: 0` on our synthetic interval XES (parses 1043 start+1043 complete but registers no overlap) — a date-parse/reader quirk, not an algorithm finding.
4. **Supplement blocker #2** (geometry `docs/TODO.md` #7): pin `pm4py==2.7.23.1` and re-home the repro scripts against the released package.
5. **Blocker #3**: Zenodo DOI for procposets in the Code Availability paragraph — **user action** (I can't mint a DOI); repo URL already wired.

## Gotchas / don't-break
- **HARD RESOURCE RULES (procposets CLAUDE.md):** every local compute run under `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout <s> …`; **one heavy process at a time**; no parallel local compute (this forbids compute-fanning workflows even under ultracode). Background runs: `cd` explicit, hard timeout, no polling.
- **`rtk` swallows byte-exact stdout** — drop `rtk` (use plain `uv`) when you need exact output.
- **Reference jar is EPHEMERAL:** `scratchpad/ref/split-miner/app/build/libs/split-miner-1.7.1-all.jar` + the cloned `ref/split-miner` and `ref/bpmtk` are in **session scratch (will be deleted)**. Rebuild: `git clone iharsuvorau/split-miner` → `./gradlew shadowJar` (foojay auto-fetches JDK 8). **Run the jar with JDK 8** (`~/.gradle/jdks/temurin-8-amd64-linux/jdk8u492-b09/bin/java`) — it uses JAXB, gone from Java 11+.
- **Building/running agent-chosen external repos triggers the auto-mode guard** — the user must name the source (they authorized `iharsuvorau/split-miner` + `nemo-91/bpmtk`).
- **Hand-rolled `split_miner2` had a bug** (`dfg.py` SM1 fallback used `checked` before assignment) — **fixed** in the homed copy `sm_investigation/split_miner2/`; the original zip in `procposets/` is unfixed.
- **fleet-wide |X| is LOCKED** (blocker #1) — do not revert to per-pair. **"+2 generators BPMN→PN" was RETRACTED** (γ1/γ2 boundary artefact, geometry `docs/TODO.md:445`) — BPMN→PN is structure-preserving; we read natively for directness, not because PN is lossy.
- **Read Split Miner's BPMN natively** (`procposets.adapters.from_bpmn → cospan.engine.extract_signature → _activity_pomsets`), not via `convert_to_petri_net` + footprint.

## Pointers
- **Split Miner journal:** `<geometry>/experiments/demo/12_synthea_fleet/sm_investigation/JOURNAL.md` (the real resume point for that thread).
- Harnesses: `sm_investigation/harnesses/{fleet_run,fork_run,sm2_run,drill_down,eps_sweep,gen_intervals}.py`.
- Data: `demo/12_synthea_fleet/data/` (START-only, flattened case) + `data_intervals/` (start+end, SM2 case).
- Run pattern: `systemd-run … timeout <s> env PYTHONPATH=<smroot>:. uv --project /home/arl/Research/procposets run python <harness>.py`.
- procposets tests: `systemd-run … timeout 300 uv run pytest -q` (from `/home/arl/Research/procposets`).
- Geometry release-blocker list: `<geometry>/docs/TODO.md` (P1 items #6–#9). procposets TODO: `procposets/TODO.md`.
- Memory slug referenced: `project-smd-sqrt-normalisation` (LOCKED fleet-|X| decision).
