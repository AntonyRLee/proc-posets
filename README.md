# procposets

Reusable **poset / cospan / string-diagram** calculation, simulation,
estimation and comparison core for process-mining research — the shared
library behind three research repos that previously duplicated and
`sys.path`-shimmed each other's code.

## Layers (install what you need)

```
pip install procposets            # numpy-only core, Python >=3.10
pip install procposets[graph]     # + networkx: occurrence nets, splice, DP class-extraction
pip install procposets[pm4py]     # + pm4py model adapters (inbound/outbound)
pip install procposets[viz]       # + matplotlib/graphviz renderers
pip install procposets[all]
```

- **A0 — pure stdlib:** canonical labelled `Poset` and its algebra (meet,
  refine, transitive closure/reduction, budget-guarded linear-extension
  counter/sampler); total modular (SP / Gallai) decomposition; the
  uniform-over-extensions trace law; grouping.
- **A1 — numpy:** the mixture estimators (NPMLE Frank–Wolfe with duality-gap
  certificates + the M9 moment initialiser; known-law EM/counting over
  weights); stochastic-matrix distance; simulation; diagnostics.
- **B0 — pure stdlib cospan algebra:** `Signature`, `LMGraph`, engine
  extraction, composition, constraints/feasibility, signature comparison/diff,
  occurrence-net pomsets.
- **B1 `[graph]`** — networkx-backed occurrence/splice/trace-language/DP
  extraction. **B2 `[pm4py]`** — model adapters. **C `[viz]`** — renderers.

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
