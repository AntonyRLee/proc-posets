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

Reach for the MCMC extension sampler and you're in the numpy layer (the
`[numeric]` extra), loaded lazily; the cospan / discovery /
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
  activity-marginal comparison). 

## Layers (install what you need)

The core is **dependency-free** (Python ≥3.10, no third-party packages);
numpy / networkx / pm4py / matplotlib are opt-in extras. It is consumed today
as an editable uv path-dependency (it is not published to PyPI), so a
consumer's `pyproject.toml` declares:

```toml
dependencies = ["proc-posets"]                # dependency-free core
# or "proc-posets[numeric]" for the numpy layer, etc.
[tool.uv.sources]
proc-posets = { path = "../proc-posets", editable = true }
```

then `uv sync`. To add an extra to an existing uv project:

```
uv add "proc-posets[numeric]"   # + numpy: MCMC extension sampling, numeric helpers
uv add "proc-posets[graph]"     # + networkx: occurrence nets, splice, DP class-extraction
uv add "proc-posets[pm4py]"     # + pm4py model adapters (inbound/outbound)
uv add "proc-posets[viz]"       # + matplotlib/graphviz renderers
uv add "proc-posets[all]"
```

- **A0 — pure stdlib:** canonical labelled `Poset` and its algebra (meet,
  refine, transitive closure/reduction, budget-guarded linear-extension
  counter/sampler); total modular (SP / Gallai) decomposition; the
  uniform-over-extensions trace law.
- **A1 — pure stdlib:** stochastic-matrix distance (`smd`), known-law
  EM/counting over weights, loop unrolling, simulation.
- **A1 `[numeric]` — numpy:** MCMC linear-extension sampling with its count
  estimates, for the posets whose exact count is out of reach. Loaded
  **lazily**, so `import procposets` stays dependency-free until you reach for
  `approx_count_extensions`/`sample_extensions_mcmc`.
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

## Screening a log before you use it

Every construction here reads a case as a *partial order*. Most event logs
cannot supply one — they record a total order, or ties that are one actor
writing several fields at a single instant — and running the machinery on such
a log still returns a definite answer, computed from structure the log never
contained. `procposets.logscreen` (stdlib only, streaming) is the input-side
falsification report that says when to stop:

```
python -m procposets.logscreen LOG.xes.gz --report ties       # simultaneity as timestamp equality
python -m procposets.logscreen LOG.xes.gz --report intervals  # simultaneity as overlapping lifecycle intervals
```

A log can pass one reading and fail the other, so both are reported. The tie
reading is gated on whether the clock can represent simultaneity at all: at a
one-second ceiling a tie may be simultaneity or same-second batching, and that
is reported as *undecidable*, never as a pass.

Applying it across the public corpus produced
[`docs/log-screening-candidates.md`](docs/log-screening-candidates.md) — 59
distinct candidate logs (20 measured on disk), each with the decisive reason it
was set aside and the smallest change that would make it usable, with
[a machine-readable copy](docs/data/log-screening-candidates.json). No public
log in it carries within-case concurrency that survives inspection.

## Status

Standalone and stabilized. The reusable core was extracted from three research
codebases, each seam reproduced
**value-for-value** before the original was retired; that behaviour is now
locked by the in-repo regression suite (`procposets/tests/regression/`, run
with `uv run pytest`), which is self-contained (no sibling repo required).

Released on its own track and not yet on PyPI — consumed today as an editable
uv path dependency (see **Layers**). Versioning follows semantic versioning
from the current `0.1.0`; see `CHANGELOG.md`.

## Third-party software, licences and attribution

Nothing third-party is vendored: every file in this repository is first-party
and MIT-licensed. The dependencies below are *declared*, never redistributed --
they are installed from PyPI by the user, and the base install
(`pip install proc-posets`) pulls **none** of them.

| extra | package | licence | note |
|---|---|---|---|
| `[numeric]` | numpy | BSD-3-Clause | |
| `[graph]` | networkx | BSD-3-Clause | |
| `[pm4py]` | pm4py | **AGPL-3.0** | see below |
| `[pm4py]` | pandas | BSD-3-Clause | |
| `[viz]` | matplotlib | matplotlib licence (BSD-compatible) | |
| `[viz]` | graphviz (Python bindings) | MIT | the `dot` binary itself is EPL-1.0 and is not bundled |

**pm4py is AGPL-3.0, and `[pm4py]` and `[all]` pull it in.** That does not
affect this package's own licence -- no pm4py code is included here, and
MIT is compatible with the AGPL in the one direction that matters. It does
affect *you*: installing the extra creates a combined work on your machine, and
if you convey that combination, or offer it to users over a network, AGPL
section 13 obligations attach to it. The base install and every other extra are
free of that, and nothing outside `procposets.discover`,
`procposets.adapters.outbound` and `procposets.viz.occn_vis` imports pm4py at
all. If the AGPL is a problem for you, use the library without the `[pm4py]`
extra and feed it logs through the stdlib or `[graph]` paths.

### Algorithms implemented from published work

`procposets.occn` implements object-centric causal net discovery from
**Liss, Mensing and van der Aalst, "Object-Centric Causal Nets", CAiSE 2025**
(doi:10.1007/978-3-031-94571-7_6). It is written from the paper's published
pseudocode (Listing 1.1) and deliberately diverges from the authors' reference
implementation, which enumerates marker keys by set partition -- so marker *key
labels* here differ from the reference even where marker *structure* agrees.
No code from that implementation is included or redistributed. Cite the paper
for the algorithm and this package for the implementation.

## Citing this software

Cite the archived release rather than the repository URL: a URL moves, a DOI
does not. `CITATION.cff` carries the machine-readable record and GitHub renders
a *Cite this repository* button from it.

<!-- DOI: recorded here once the release is archived on Zenodo.  The concept
     DOI (the one that always resolves to the latest version) belongs in this
     paragraph and in CITATION.cff; the version DOI belongs in whatever text
     cites a specific release. -->
