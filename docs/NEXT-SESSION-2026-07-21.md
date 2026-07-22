# Next-session handoff — procposets — 2026-07-21

**Branch:** `output-sensitive-extract-signature` (NOT pushed; base branch is `main`) ·
**Build/tests:** last-known **343 passed, 20 skipped** (`uv run pytest -q`, green after every commit this session) ·
**Uncommitted:** working tree clean except one pre-existing untracked file `NEXT-SESSION-2026-07-20.md` at repo root (from a prior session, NOT this one — left as-is). `sandbox/` is gitignored.

## Where we are (1–2 sentences)
Ran a 14-agent static review of the whole package and executed an 8-phase refactoring plan through
**Phase 3** — Phases 0→3 are done and committed (20 commits), each byte-exact-gated, suite green throughout.
The full plan + a **live execution log** is the source of truth: **`docs/2026-07-20-refactoring-plan.md`**.

## Done this session (20 commits, `3980623`..`3ef559a`)
- **Phase 0** `3980623` — P0 safety fix (`extract_dp.extract_classes` now enforces the documented
  `max_pomsets_per_state` cap; was unimplemented → no OOM valve) + `engine_fast` determinism + hygiene
  (stray zips → `sandbox/`, wheel excludes) + landed the review plan.
- **Phase 1** `5d40bfa` boundary-predicate hazard (`is_gamma_or_marker` vs `_is_start_end_marker`);
  `b081eb3` single-home `Model` alias.
- **Phase 2 (dedup)** `c779f7c` Bhattacharyya-angle kernel (6→3 helpers); `2667dfe` + `85a6b74` `preds`
  routing (traces/simulate/likelihood); `90d9e90` feasibility `_enumerate`; `5733672` rel `_canonical_key`
  + `_ideals`/`_filters`→`_closed_sets`; `bceb94c` shared `procposets/_unionfind.py`.
- **Phase 3 (maintainability/docs/types)** `a0b55f6` dead code; **5 god-function splits** — `92344ac`
  Oracle.__init__, `3f6acd2` npmle.fit, `9094a2f` discrete._build_refined, `f7388f4` to_event_dag,
  `b464805` splice.from_extraction_result; `c6ef417`+`fb2b1c2` full `cpm.`→`procposets.` doc sweep +
  public-API docstrings + `Literal` enum types.

## Open decisions (need the user)
- **Which phase next?** User was asked and has not answered. Options below. Their **stated end goal**
  (from the very first message) is Phase 7 — "systematically stress test for efficiency/scalability".

## Next steps — ranked
1. **Phase 7 — efficiency stress test** (the user's stated goal). A 13-row ranked hotspot map is prepared
   in the plan doc §7. Build a **serial, cgroup-capped** harness against it (transitive-closure O(n³),
   ideal-DP, feasibility enumeration, `smd_rows` density, `swap_kernel`, …). Probe *at* the load-bearing
   caps, don't raise them silently. Every speedup must stay value-preserving (golden-locked).
2. **Phase 6 — module moves + the two file-splits** (value-preserving, mostly gate-able here): relocate
   `equivalence.py`→`cospan/` (re-export shim); split `viz/string_diagram.py` (1172 LoC) into
   `viz/_layout.py` — this is where `string_diagram.render` should be split (deferred from Phase 3);
   split `rel.py`; `from_petri` placement.
3. **Phase 5 — layer guard test** (small): extend `test_lazy_numpy.py` to assert the eager core is
   networkx- and matplotlib-free (currently only numpy is guarded; `equivalence.py` imports networkx eagerly).
4. **Phase 4 — api-breaking name unifications** — do **LAST**, behind deprecation aliases, coordinated with
   the consumer repos being present (e(P) spellings, `then/par`→`series/parallel`, `noise_kernel`, the 3
   `canonical_key`/`canon_key` symbols, viz `color_map`, diagnostics report family).
- Also open: the remaining Phase-2 dedups (OCCN `_lift`, adapter wrappers, `cospan/_boundary.py`) — all
  [graph]/[pm4py]-gated, so **can't be byte-verified standalone**; do them with consumer repos checked out.

## Gotchas / don't-break
- **HARD RESOURCE RULES (project CLAUDE.md):** every local run under
  `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout <s> …`; **one heavy process at
  a time**; **no parallel local compute** (a review Workflow with only text agents is fine; a compute
  fan-out is not). Background: `cd` explicit, hard timeout, no polling.
- **Gating reality:** the `[graph]`/`[pm4py]` byte-exact goldens **SKIP in this checkout** (they cross-check
  the original `sim/cpm`, `poset-mixture-npmle`, `stochastic_process_mining` repos, not on disk — 20 skips,
  all "…not checked out"). networkx IS installed, so those modules RUN; only their cpm cross-check skips.
  For byte-exact gating of float/structure-sensitive changes I used a **fixed-`PYTHONHASHSEED=0` before/after
  `repr()` capture + diff** (scratchpad, now gone — recreate the technique). Use **plain `uv`, not `rtk`**,
  for exact stdout (`rtk` swallows it — it reported "ok" for a full pytest run).
- **LOCKED decisions (do not revert):** fleet-wide constant `|X|` in `distance.smd_rows` (metric-hood);
  the two renderers (`->`/`||` SPTree vs `;`/`*` moddecomp) stay SEPARATE; `Poset` (id+label) vs `Rel`
  (distinct-label) split; byte-exact-on-VALUES discipline (rename/refactor freely, never change a number).
- **Deferred with rationale (don't force):** `extract_classes` left un-split (delicate `_mint`/
  `closing_pomsets` closures over shared mutable state — net-negative to extract); kept public
  `moddecomp .atomic` (no in-repo reads but a consumer may use it).
- `sandbox/split-miner/` holds the moved SM zips/md (gitignored) — the **parked Split-Miner
  investigation** (separate task, see `TODO.md` + the geometry repo's JOURNAL).

## Pointers
- **Plan + live execution log:** `docs/2026-07-20-refactoring-plan.md` (read FIRST — has every phase, risk
  tag, what's landed vs deferred, and the §7 stress-test hotspot table).
- Memory slug: `refactoring-plan-progress`.
- Test (exact counts): `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout 480 uv run pytest -q` (from repo root).
- Prior handoffs: `NEXT-SESSION-2026-07-18.md` (Split-Miner + distance fix), untracked `NEXT-SESSION-2026-07-20.md`.
