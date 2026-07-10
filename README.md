# procposets

Reusable **poset / cospan / string-diagram** calculation, simulation,
estimation and comparison core for process-mining research — the shared
library behind three research repos that previously duplicated and
`sys.path`-shimmed each other's code.

## Defaults and conventions (state space & the SMD)

These are the conventions every comparison in this package uses, stated once
(they mirror "the construction, end to end" in the geometry paper, §IV, and
its Remark V.1):

- **State space (canonical, no choice).** A model — a weighted list of
  labelled posets (its variants) — is tiled by the total modular
  decomposition, which is *unique*: the block alphabet (leaves, parallel
  blocks, primes; concurrency stays one atomic token) is canonical. States
  are the block symbols joined into order-`context_depth` windows
  (`matrix.build`; depth 1 = memoryless default — the paper's faithful
  default is depth maximal for the signature, which consumers pass
  explicitly), plus the bare sentinels `START` (γ₁) and `END` (γ₂).
- **P matrices.** Each variant's block word traces a path through the states;
  variant weights are spread along the paths, summed, and each row
  normalised. Closure (`distance.NORMALISATION = "sink"`, the default): a
  state with no outgoing mass routes to `END`, and `END` resets to `START`,
  so every model is a genuine γ₁→γ₂ generative chain on any common state
  space; `"selfloop"` is the distance-paper alternative.
- **The SMD.** `distance.smd`: rows are compared by the Bhattacharyya angle
  on the union state space, `d = 2·sqrt(Σ_rows arccos² BC)`. The raw value is
  extensive (it grows with the number of differing rows); `normalize=True`
  applies the Result-4 `1/sqrt(|X|)` factor (root-mean-square row angle,
  bounded by π) — use it whenever the two objects can differ in state-space
  size.
- **Refinement (off everywhere by default).** `matrix.build`, `distance.smd`
  and `discrete.block_angle` are always atomic — the paper's default
  comparison object. `discrete.disc_angle(refine=...)` is the refined
  family's entry point (paper, Remark V.1): primes fan out over their
  labelled covering-relation atoms `"x<y"`, parallel blocks over typed
  element atoms `"sym||"`, with uniform (maximum-entropy) splits, typed
  intermediate states, and the SMD formula unchanged. `refine=True` enables
  both instantiations; `refine={"prime"}` / `{"parallel"}` selects one;
  `refine=False` is atomic. Refined states carry the same memory windows as
  atomic ones (`context_depth`: an atom is typed by its preceding block
  context — global at depth 1), and **exactness is enforced by default**: a
  state recurring within one variant raises (`strict=True`) rather than
  silently merging rows into a chain with spurious trajectories; raise the
  depth for the faithful chain, or pass `strict=False` to accept the merge
  as the paper's declared robustness relaxation. Isolated same-kind block
  pairs obey, over atom multiplicities m, the closed
  form `2·arccos(Σ sqrt(m·m')/sqrt(|A||A'|))` (the count form
  `|A∩A'|/sqrt(|A||A'|)` when shared multiplicities are equal); between
  totally parallel models the
  refined SMD coincides with the Bhattacharyya angle on activity sets (the
  activity-marginal comparison). Pinned in `tests/test_refinement.py`.

## Layers (install what you need)

The core is **dependency-free** (Python ≥3.10, no third-party packages);
numpy / networkx / pm4py / matplotlib are opt-in extras. It is consumed today
as an editable uv path-dependency (it is not published to PyPI), so a
consumer's `pyproject.toml` declares:

```toml
dependencies = ["procposets"]                 # dependency-free core
# or "procposets[estimate]" for the numpy NPMLE estimator, etc.
[tool.uv.sources]
procposets = { path = "../procposets", editable = true }
```

then `uv sync`. To add an extra to an existing uv project:

```
uv add "procposets[estimate]"   # + numpy: the NPMLE estimator + M9 initialiser
uv add "procposets[graph]"      # + networkx: occurrence nets, splice, DP class-extraction
uv add "procposets[pm4py]"      # + pm4py model adapters (inbound/outbound)
uv add "procposets[viz]"        # + matplotlib/graphviz renderers
uv add "procposets[all]"
```

- **A0 — pure stdlib:** canonical labelled `Poset` and its algebra (meet,
  refine, transitive closure/reduction, budget-guarded linear-extension
  counter/sampler); total modular (SP / Gallai) decomposition; the
  uniform-over-extensions trace law; grouping.
- **A1 — pure stdlib:** stochastic-matrix distance (`smd`), known-law
  EM/counting over weights, loop unrolling, simulation.
- **A1 `[estimate]` — numpy:** the certified NPMLE mixture estimator
  (Frank–Wolfe with duality-gap certificates + the M9 moment initialiser)
  and its diagnostics. Loaded **lazily**, so `import procposets` stays
  dependency-free until you reach for `fit`/`GroupedLog`/`Oracle`/….
- **B0 — pure stdlib cospan algebra:** `Signature`, `LMGraph`, engine
  extraction, composition, constraints/feasibility, signature comparison/diff,
  occurrence-net pomsets.
- **B1 `[graph]`** — networkx-backed occurrence/splice/trace-language/DP
  extraction. **B2 `[pm4py]`** — model adapters. **C `[viz]`** — renderers
  (matplotlib/graphviz; also pulls numpy).

## Canonical poset representation

The core `Poset` carries integer elements + a label side-table + a
transitively-closed order, so it can represent **repeated labels**.
`Rel = frozenset[(label, label)]` is the certified **distinct-label view**
(`to_rel` asserts distinctness; it is never a lossy cast).

## Status

Under extraction from `poset-mixture-npmle`, `-DIAGRAM-.../sim` (`cpm`), and
`stochastic_process_mining` (`spm`). Each reconciliation seam is proven
value-equivalent to the original before the original is deleted. See the
consumer repos' `.claude/REFACTOR_DESIGN.md`.
