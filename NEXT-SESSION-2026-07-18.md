# Next-session handoff — procposets (+ geometry repo) — 2026-07-18

**Branch:** `main` (procposets), `arl/main` (stochastic_process_mining) ·
**Build/tests:** last-known `334 passed, 13 skipped` (procposets, after the distance fix) ·
**Uncommitted:** procposets — **N commits ahead of `origin/main`, PUSH STILL PENDING** (classifier kept blocking `git push` — do it manually, see below); 3 untracked drops in `procposets/` (`split-miner*.zip`, `split-miner.md`). Geometry repo — investigation now **committed** (`210fa84`); only a pre-existing `transition.svg` change (not ours) remains unstaged.

## Where we are (1–2 sentences)
Two threads. (1) **procposets**: P1 blocker #1 (fleet-wide constant |X|) fixed + 2 stale goldens retired + this handoff — **committed but NOT pushed** (push was blocked, needs a manual retry). (2) **Geometry repo**: the **Split Miner fabrication investigation** is committed (`210fa84`); root cause nailed, SM2 demo done. Remaining: task #4 (build nemo-91) and #5 (developer contact + paper review).

## Done this session
**procposets (`main`, committed, UNPUSHED):**
- `da51d8e` retire closed pm_adapters golden → pinned language regression (fixed a hard failure).
- `941548f` retire tautological spm.viz golden → pinned generator-signature regression (latent landmine).
- `3a62e18` **distance: fleet-wide constant |X| for normalised SMD** (`smd_rows(states=)`, `smd_pairwise(normalize=)`). Raw/lone-pair byte-exact. **Release blocker #1, LOCKED — do not revert to per-pair.**
- `9ff5bf3` + this file: session handoff.

**Geometry repo (`arl/main`, committed `210fa84`):**
- `experiments/demo/12_synthea_fleet/sm_investigation/JOURNAL.md` — living journal; **§1 = plain-English root cause, front and centre.**
- Root cause NAILED: **(A) inherent** over-pairing on flattened small logs (reference reproduces it); **(B) implementation** — pm4py default `epsilon≈0.1` makes b,c a coin-flip on small samples (epsilon sweep: 0.1→flips, 0.5→stable). **NEW:** pm4py `sm2` broken *even with real timestamps* (over-pairs + misses g‖h; **ARI 0.346 vs hand-rolled SM2 1.000**).
- Committed: hand-rolled `split_miner2` (with SM1 `checked`-bug fix), all harnesses, raw outputs, `data_intervals/` (start+end SM2 variant), `reference/BUILD.md`. **Reference jar preserved on disk at `sm_investigation/reference/split-miner-1.7.1-all.jar` (44 MB, gitignored).**

## Open decisions / blocked (need the user)
- **PUSH procposets** — `git push origin main` was blocked ~3× by the auto-mode classifier (transient stage-2 error that didn't clear). Run it manually: from `/home/arl/Research/procposets`, `! git push origin main` (or add a Bash permission rule for push). procposets is N commits ahead.
- **#4 vs #5 next** — user wanted a clean break to resume later; either order fine. The pm4py-on-intervals finding strengthens #5.

## Next steps — ranked
1. **Push procposets** (blocked above) — one manual command.
2. **#4 — Build nemo-91/bpmtk** (2nd reference): no build system / no SM `main`; hand-compile `au.edu.qut…splitminer` vs its `lib/*.jar` under JDK 8 + a tiny XES→BPMN driver. Recipe in `sm_investigation/reference/BUILD.md`. Source differs 192–2976 lines from the fork → may behave differently.
3. **#5 — Draft the Split Miner developer note** (Augusto et al.), framed as questions: (a) default `epsilon=0.1` instability on small logs, (b) `sm2` over-pairing + missing real overlap concurrency on interval logs. Then review our own §IX exhibit + footnote.
4. **Reference `-v2` integration** (deferred): jar reports `potential parallelisms: 0` on our interval XES (parses 1043 start+1043 complete but registers no overlap) — date-parse/reader quirk, not an algorithm finding.
5. **Supplement blocker #2** (geometry `docs/TODO.md` #7): pin `pm4py==2.7.23.1`, re-home repro scripts against the released package. **Blocker #3**: Zenodo DOI — user action (repo URL already wired).

## Gotchas / don't-break
- **HARD RESOURCE RULES (procposets CLAUDE.md):** every local run under `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout <s> …`; **one heavy process at a time**; **no parallel local compute** (forbids compute-fanning workflows even under ultracode). Background: `cd` explicit, hard timeout, no polling.
- **`rtk` swallows byte-exact stdout** — use plain `uv` (drop `rtk`) when exact output matters.
- **Reference jar preserved but gitignored** at `sm_investigation/reference/` (44 MB). Rebuild: clone `iharsuvorau/split-miner` → `./gradlew shadowJar` (foojay auto-fetches JDK 8). **RUN with JDK 8** (`~/.gradle/jdks/temurin-8-amd64-linux/jdk8u492-b09/bin/java`) — jar uses JAXB (gone in Java 11+). Cloned repos were in scratch (now deleted); re-clone as needed.
- **Building/running agent-chosen external repos** triggers the auto-mode guard — user must name the source (authorized: `iharsuvorau/split-miner`, `nemo-91/bpmtk`).
- **fleet-|X| is LOCKED** (blocker #1). **"+2 generators BPMN→PN" was RETRACTED** (γ1/γ2 boundary artefact, geometry `docs/TODO.md:445`) — BPMN→PN is structure-preserving; we read natively for directness.
- **Read Split Miner BPMN natively** (`procposets.adapters.from_bpmn → cospan.engine.extract_signature → _activity_pomsets`), not via `convert_to_petri_net` + footprint.
- Hand-rolled `split_miner2` SM1 `checked`-before-assignment bug is **fixed in the committed copy**; the raw zips in `procposets/` are unfixed.

## Pointers
- **Split Miner journal (real resume point):** `<geometry>/experiments/demo/12_synthea_fleet/sm_investigation/JOURNAL.md`.
- Harnesses: `sm_investigation/harnesses/{fleet_run,fork_run,sm2_run,drill_down,eps_sweep,gen_intervals}.py`.
- Data: `demo/12_synthea_fleet/data/` (START-only, flattened) + `data_intervals/` (start+end, SM2).
- Run pattern: `systemd-run … timeout <s> env PYTHONPATH=<smroot>:. uv --project /home/arl/Research/procposets run python <harness>.py` (smroot = a dir containing `split_miner2/`, e.g. `sm_investigation/`).
- procposets tests: `systemd-run … timeout 300 uv run pytest -q` (from `/home/arl/Research/procposets`).
- Release-blocker lists: geometry `docs/TODO.md` (P1 #6–#9); `procposets/TODO.md`. Memory slug: `project-smd-sqrt-normalisation` (LOCKED |X|).
