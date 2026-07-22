# proc-posets

Reusable **poset / cospan / string-diagram** calculation, simulation,
estimation and comparison core for process-mining research — the shared
library behind three research repos that previously duplicated and
`sys.path`-shimmed each other's code.

## Quickstart

The core is pure standard library, so this runs on a bare `import procposets`
(no extras):

```python
from procposets import leaf, then, par, modular_decompose, smd

# A "model" is a weighted list of labelled posets (its variants).
seq  = [(then(leaf("a"), leaf("b"), leaf("c")), 1.0)]        # a < b < c
conc = [(then(leaf("a"), par(leaf("b"), leaf("c"))), 1.0)]   # a < (b ∥ c)

# Canonical block tiling (the unique total modular decomposition):
print(modular_decompose(then(leaf("a"), par(leaf("b"), leaf("c")))).canonical())
# -> (a ; (b * c))

# Stochastic-matrix distance between the two models (Result 3), and the
# per-state angle breakdown:
dist, per_state = smd(seq, conc, normalize=True)
print(round(dist, 4))
```

Reach for `fit` / `GroupedLog` / `Oracle` and you're in the numpy-backed NPMLE
estimator (the `[estimate]` extra), loaded lazily; the cospan / discovery /
visualisation layers sit behind `[graph]` / `[pm4py]` / `[viz]` (see **Layers**
below).

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
dependencies = ["proc-posets"]                # dependency-free core
# or "proc-posets[estimate]" for the numpy NPMLE estimator, etc.
[tool.uv.sources]
proc-posets = { path = "../proc-posets", editable = true }
```

then `uv sync`. To add an extra to an existing uv project:

```
uv add "proc-posets[estimate]"  # + numpy: the NPMLE estimator + M9 initialiser
uv add "proc-posets[graph]"     # + networkx: occurrence nets, splice, DP class-extraction
uv add "proc-posets[pm4py]"     # + pm4py model adapters (inbound/outbound)
uv add "proc-posets[viz]"       # + matplotlib/graphviz renderers
uv add "proc-posets[all]"
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

Standalone and stabilized. The reusable core was extracted from three research
codebases (poset-mixture NPMLE, a string-diagram process-mining engine, and a
stochastic-process-mining comparison sandbox), each seam reproduced
**value-for-value** before the original was retired; that behaviour is now
locked by the in-repo regression suite (`procposets/tests/regression/`, run
with `uv run pytest`), which is self-contained (no sibling repo required).

Released on its own track and not yet on PyPI — consumed today as an editable
uv path dependency (see **Layers**). Versioning follows semantic versioning
from the current `0.1.0`; see `CHANGELOG.md`.
