# Next-session handoff — procposets — 2026-07-20 (fast `extract_signature`)

**Branch:** `main` · **Build/tests:** last-known `334 passed, 13 skipped`
(`uv run pytest`, unchanged this session — no code touched) · **Uncommitted:** this
handoff + `docs/2026-07-20-fast-signature-extraction.md` + a `TODO.md` top entry
(all docs, written from the `ocel-data-audit` session; **no code edits made here** by
design). Pre-existing untracked `split-miner*.zip` / `split-miner.md` left as-is.

## Where we are (1–2 sentences)

While running the SoSyM Bundestag OCPN exhibit from `ocel-data-audit`, we root-caused
why `extract_signature` "is intractable at full scale": it is a **materialised-DNF
product blow-up in our own engine** (`|B|×|F|` per activity, exponential in object-type
count — hub `Beratung` ⇒ ≈1.4×10³⁰ generators), **not** pm4py (6.2 s) or log size. A
direct, output-sensitive extractor has been **designed and validated** (exact vs the
slow engine for k=1,2,4,6; full 44-type net = 35,009 CanonKeys in 4.25 s). Per the
workflow, the code is **not** yet applied — this session only logs the plan.

## The task for this session — implement it

Follow **`docs/2026-07-20-fast-signature-extraction.md`** (full derivation, validation
table, ready-to-drop-in `cospan/engine_fast.py`, test plan). In order:

1. Add `procposets/cospan/engine_fast.py` (code is in the plan, copy verbatim);
   export `extract_signature_fast` from `cospan/__init__.py`.
2. Add `tests/test_extract_fast.py`: assert `canonical_generators(extract_signature(g,
   **kw)).keys() == canonical_generators(extract_signature_fast(g, **kw)).keys()` on the
   running-example / object-centric / gamma-boundary / surface-termini fixtures (both
   `surface_termini` values). Reuse the graph builders in the existing tests.
3. `uv run pytest` under the resource wrapper (§ CLAUDE.md) — expect 334 + new green;
   `extract_signature` is untouched so no regressions.
4. Opt-in wire: `signature_from_ocpn(ocel, *, canonical=False)` in `discover.py`
   dispatching to the fast path; keep `extract_signature` the default (splice/behavioural
   users unaffected). Decide the default with the user.

## Evidence / provenance (read-only, in the ocel-data-audit repo)

- `research/audits/2026-07-20-ocpn-extract-rootcause.md` — the root cause + figures.
- `scripts/audits/fast_extract_prototype.py` — the validated prototype (exact vs slow).
- `scripts/audits/sosym_ocpn_extract_diagnose.py` — the |B|×|F| product diagnostic.
- `research/audits/2026-07-20-sosym-strip-and-ocpn-frontier.md` — the stage split.

## Caveats / decisions to confirm

- Fast path returns **one representative Generator per CanonKey** (OCPN carries no
  bindings) — correct for `compare`/type-level views; **not** a substitute for the full
  per-context set used by splice (`extract_dp`/`occurrence`). Keep both.
- The 35,009 is inflated by same-type multi-arc **arity ranges** (inductive-miner
  variable arcs). Optimisation #4 (arity-range collapse) is a *semantic* choice for the
  paper, not part of this drop-in.

## After this — other optimisations logged in the plan (ranked)

adjacency index in `LMGraph` (#2), memoise `_traverse` (#3), arity-range collapse (#4),
faster `without_silent` (#5), OCPN structural reduction (#6), per-type-then-merge (#7),
CanonKey-native compare (#8), occurrence/unroll audit (#9). See plan §"Other optimisations".
