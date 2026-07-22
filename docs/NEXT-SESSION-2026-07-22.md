# Next-session handoff — procposets — 2026-07-22

**Branch:** procposets `self-containment-sweep` (tip `e6cf885`) · sim `arl/sosym` (tip `9d9aff9`) ·
**Build/tests (last-known):** procposets **370 passed / 13 skipped**; sim **209 passed on procposets** ·
**Uncommitted (procposets):** `docs/MIGRATION-PLAN.md` (untracked working plan — I keep editing it; **decide whether to commit**), `NEXT-SESSION-2026-07-20.md` (untracked, leave).

## Where we are (1–2 sentences)
Mid **migration/self-containment** workstream (plan = `docs/MIGRATION-PLAN.md`). This session landed **WS 1.4** (analysis-layer goldens, procposets) and **WS 1.5** (flipped sim's pure-library imports `cpm.*`→`procposets.*`, sim repo). The remaining sim migration is **WS 1.7** (relocate the cpm glue, which unblocks the 20 files still on cpm), then teardown 1.8/1.9.

## Done this session
- `e6cf885` (procposets) **WS 1.4** — new `procposets/tests/test_golden_analysis.py`: cpm cross-check goldens for `signature_diff`, `typebalance`, full `class_extraction` structural equality, `discovery_cleanup` (gamma_normalize/degenerate_filtered/forget_provenance), + silent-elimination. Suite 366→370.
- `9d9aff9` (sim) **WS 1.5** — flipped the **transitive-closed pure-library-consumer set = 42 files** to procposets. **20 files kept on cpm** (glue-entangled + the build_running_example fixture cluster). Adapted 3 api evolutions. sim suite **209 passed on procposets** (same count ⇒ cross-validates procposets==cpm across the library surface). Driven by discovery workflow `wf_f017fd7c-44d` + an AST rewriter; verified all 293 procposets imports resolve, zero residual cpm in flipped files.

## Open decisions (need the user)
- **Continue to WS 1.7?** (relocate cpm glue `splice_cli/generate/report/pipeline/spec/binding_probe/conversion_consistency/artifacts` + `loop_family` to `sim/demos/`, flip their internals to procposets, then flip the 20 kept-cpm files). This is the next big consumer step. Or pick a smaller thread (WS 3 consumer cleanups, WS 4 release hygiene).
- **Commit `docs/MIGRATION-PLAN.md`?** It's been untracked but is the authoritative plan; I keep updating it.
- **Merge to `main` (WS 4.4)?** The whole internal refactor + this work is still on unmerged feature branches; `main` is pre-refactor. A release can't cut until then. Owner call.

## Next steps — ranked
1. **WS 1.7** — relocate the cpm glue to `sim/demos/` (NOT into the library), flip the glue's own internals to procposets, then flip the **20 kept-cpm files** (glue-entangled tests/modules + the 3-file `build_running_example` fixture cluster). Gate: sim suite stays 209.
2. **WS 1.6 wrap** — the signature/render modules already flipped in 1.5, but the 3 `render_ed_chest_pain_morphism_classes_*.py` scripts are **not run by pytest** — do a visual/end-to-end figure check that they render on procposets (imports verified, private-viz calls fixed).
3. **WS 1.8 / 1.9** — retire the `*_matches_cpm` oracle suite, then delete `sim/cpm/` (~9,475 LoC) LAST.
4. **WS 3** (small): PMN merge `arl/alias-cutover`; SPM re-point 5 scripts + 10 tests. **WS 4**: LICENSE, py.typed, release-gate grep, merge to main.

## Gotchas / don't-break
- **HARD RESOURCE RULES** (project CLAUDE.md): every local run under `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout <s> …`; ONE heavy process at a time; NO parallel local compute (read-only text agents OK); no `python - <<EOF` heredocs (write a scratchpad script). A parallel fan-out OOM-crashed the machine before.
- **sim env**: run sim's suite with **`uv run python -m pytest`** (bare `pytest` mis-resolves), Py3.12, ~73s. procposets is now a proper editable dep in sim's pyproject (WS 1.1), so `uv sync` no longer wipes it.
- **`sim/cpm/` is the LAST byte-exact ORACLE — keep it pristine, do NOT edit or flip anything inside it.** It backs procposets' own goldens + sim's `*_matches_cpm`. Deleted only in WS 1.9, after replacements are green.
- **WS 1.5 type-mixing landmines (locked decisions):** (a) a test-fixture and ALL its importers must live on the SAME side — the `build_running_example` cluster (test_engine_running_example + test_typebalance + test_compose_running_example) stays on cpm because glue-bound `test_explain_split` shares it. (b) `cpm.cospan.loop_family` is a worked-EXAMPLE, do NOT port into the library — keep cpm, relocate to demos. (c) collision trap: `cpm.simulate`→`procposets.cospan.simulate`, NEVER top-level `procposets.simulate` (NPMLE sampler).
- **procposets API the consumer must use** (renamed from cpm): `CompositeDiagram.label_multiset_key()` (was `canonical_key`); viz layout fns take a `LayoutStyle` — `_sub(style)`/`_finish(sub,style)`/`_draw_wires(...,style)`, thread `DEFAULT_STYLE.layout`/`.draw` (mirror `render()`).
- **The 20 kept-cpm files are correct-as-is** — do NOT force-flip them before WS 1.7 relocates their glue, or you reintroduce cpm/procposets type mixing.
- Both feature branches are **local/unpushed**.

## Pointers
- **Plan:** `docs/MIGRATION-PLAN.md` (4 workstreams; 1.1/1.2/1.3/1.4/1.5 + 2.1/2.3/2.5 ticked). Memory slug: `refactoring-plan-progress` (full live status, updated through WS 1.5).
- **Discovery artifact:** WS 1.5 workflow `wf_f017fd7c-44d` (10-agent verified import map + per-file plan) — transcript under `…/subagents/workflows/wf_f017fd7c-44d/`.
- **Test procposets:** `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout 480 uv run pytest -q`.
- **Test sim:** `cd ~/Research/string-diagram-process-mining/sim && systemd-run … timeout 480 uv run python -m pytest -q`.
- **Test consumers:** PMN `cd ~/Research/poset-mixture-npmle && uv run pytest -q`; SPM `cd ~/Research/stochastic_process_mining/experiments && uv run pytest -q --ignore=tests/test_pm4py_adapters.py --ignore=tests/test_signature_lift.py`.
- Prior handoff (stale, pre-migration): `docs/NEXT-SESSION-2026-07-21-2024.md`.
