# procposets

Reusable **poset / cospan / string-diagram** calculation, simulation,
estimation and comparison core for process-mining research — the shared
library behind three research repos that previously duplicated and
`sys.path`-shimmed each other's code.

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
