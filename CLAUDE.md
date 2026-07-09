# procposets — project instructions

Reusable poset / cospan / string-diagram calculation, simulation, estimation
and comparison core, factored out of three research repos
(`poset-mixture-npmle`, `-DIAGRAM-.../sim`, `stochastic_process_mining`) so
they stop `sys.path`-shimming each other. Env is uv: `uv sync`,
`uv run pytest`. Layered so the numpy-only core installs at Python 3.10;
cospan-graph / pm4py / viz sit behind extras (`.[graph]`, `.[pm4py]`,
`.[viz]`). Design + migration plan: the consumer repos' `.claude/`
(REFACTOR_DESIGN.md, REFACTOR_CRITIQUE.md).

## Hard resource rules — never violate

A parallel fan-out of local compute OOM-crashed the dev machine on 2026-07-08
and cost a full session. The machine has 14 GiB RAM / 8 cores. These rules
exist so that never happens again:

1. **Every local compute process runs under a cgroup memory cap and a hard
   timeout.** Canonical wrapper:
   `systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 timeout <secs> rtk uv run ...`
   No exceptions for "quick" runs. Write `rtk` explicitly (the auto-rewrite
   hook does not reach inside `systemd-run`/`timeout`); use `rtk proxy uv run`
   or drop `rtk` when capturing byte-exact output. No `python - <<EOF`
   heredocs — write a scratchpad script and `rtk uv run python <file>`. No
   `time` prefixes.
2. **One heavy local process at a time.** pytest runs serially, never
   concurrently. Multi-agent workflows must not run local compute in parallel:
   restrict agents to reading/writing text, or precompute serially and feed
   the numbers in as prompt data.
3. **Background runs:** `cd` explicitly (cwd resets), hard timeout on every
   process, no polling loops — launch, end the turn, act on completion.
4. **Timing budgets are findings, not knobs.** pytest ≤ 10 min. A run that
   hits its timeout is investigated, not re-run with a bigger cap.

## Migration discipline (this package is mid-extraction)

- The package is being built **before** any consumer repo is cut over. Until a
  repo's cut-over phase, that repo still runs its own copy of the code, so
  `procposets` must reproduce the originals **byte-for-byte on values** — every
  reconciliation seam ships a golden cross-check test (new == old on a fixed
  corpus) before the old code is deleted.
- Do **not** unify repo-specific presentation (the `->`/`||` SPTree renderer vs
  the `;`/`*` moddecomp renderer are both kept — unifying them breaks a
  byte-exact regression downstream). Keep equality/canonical forms label-based.
- Canonical poset type = id+label base (repeated-label-capable); `Rel =
  frozenset[(str,str)]` is the certified distinct-label view, never a lossy
  cast (`to_rel` asserts label distinctness).
