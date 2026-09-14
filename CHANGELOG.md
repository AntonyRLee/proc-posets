# Changelog

All notable changes to **proc-posets** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-14

### Added

- **Poset core (stdlib).** Labelled posets over an id+label base, so repeated
  labels are representable; meet, join, refinement, transitive closure and
  reduction; a budget-guarded linear-extension counter and sampler, with a
  closed form for disjoint unions of chains taken before the ideal-lattice DP.
- **Total modular decomposition.** The unique SP / Gallai block tiling of a
  labelled poset, and the SP-tree view over it.
- **Stochastic matrix distance.** `smd` and its row form, the Bhattacharyya
  angle, the discrete correspondences (Kemeny on total orders, Hamming on
  deterministic block models), and the known-law counting and EM estimators
  over mixture weights.
- **Loops.** Unrolling, the loop model and its geometric limit, and the
  empirical loop model read off a log.
- **Cospan algebra (stdlib).** `Signature` and `LMGraph`, engine extraction,
  composition, constraints and feasibility, signature comparison and diff, and
  occurrence-net pomsets.
- **Object-centric causal net discovery** (`procposets.occn`), implementing the
  algorithm of Liss, Mensing and van der Aalst (CAiSE 2025) from its published
  pseudocode. See the attribution note in `README.md`.
- **Log screening.** Does a log carry within-case concurrency at all, and does
  it carry loops — asked before a model is fitted rather than after.
- **Adapters and renderers** behind extras: networkx extraction (`[graph]`),
  pm4py model adapters (`[pm4py]`), matplotlib/graphviz diagrams (`[viz]`),
  and MCMC extension sampling (`[numeric]`).

### Notes

- The core is **dependency-free**: `import procposets` and the whole stdlib
  layer pull nothing third-party. numpy, networkx, pm4py and matplotlib are
  opt-in extras, resolved lazily through a PEP 562 `__getattr__`.
- Python 3.10 and later. MIT licensed; ships `py.typed`.
- `[pm4py]` and `[all]` install AGPL-3.0 software. Nothing of it is
  redistributed here and this package stays MIT — see **Third-party software,
  licences and attribution** in `README.md` for what that does and does not
  mean for you.
