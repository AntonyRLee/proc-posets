# Changelog

All notable changes to **procposets** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `TYPE_CHECKING` re-exports so type-checkers resolve the lazily-loaded numpy
  (`[estimate]`) API to real signatures; the public type aliases `Model`,
  `Mode`, `Trace`, `Law` and the `sample_extension` sampler are now exported.
- Self-contained regression coverage for the type-balance (`⋈`) admissibility
  filter.
- `CHANGELOG.md`, `CONTRIBUTING.md`, a CI matrix (Python 3.10–3.13 × extras,
  plus a numpy-free base job guarding the lazy-import contract), and a runnable
  Quickstart in the README.

### Changed
- Single-sourced four localized code duplications (`from_heuristics` LM-graph
  wrappers, the two signature extractors' shared preamble/tail, the OCEL
  simulators' column vocabulary + frame assembly, the OCPN discover→extract
  pipeline) — byte-exact.
- De-named the regression suite off the retired consumer-repo prefixes
  (`test_cpm_*`/`test_np_*`/`test_spm_*` → subject-based `test_cospan_*` /
  `test_npmle_*` / `test_poset_core` / `test_comparison_*`); optional-extra
  tests are now gated by capability (`find_spec`).
- `adapters.outbound` no longer mutates the process-wide warnings filter on
  import.

### Removed
- Dead internals (`_fmt_group`, `_Status`, `_check_ideal_budget`, unused
  imports, an unused palette entry) and the last leftover compatibility shims
  (the root `equivalence` module and the `is_boundary_label` alias).

### Fixed
- Three latent correctness issues, each byte-exact on current inputs: the
  `compose` LoopBox branch now honours interleaving-dedup; `morphism_schema`
  weights its leg tallies by object count (§38 grounding); the timed sampler
  derives its alphabet per mixture component.

## [0.1.0]

### Added
- Initial standalone release: the reusable poset / cospan / string-diagram
  calculation, simulation, estimation and comparison core, extracted
  value-for-value from three research codebases and locked by a self-contained
  regression suite. Numpy-free stdlib core with opt-in `[estimate]`, `[graph]`,
  `[pm4py]`, and `[viz]` extras; MIT-licensed; ships `py.typed`.
