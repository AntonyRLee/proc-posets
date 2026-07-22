# Contributing to procposets

## Development setup

procposets uses [uv](https://docs.astral.sh/uv/). The dev environment pulls
every layer (numpy + matplotlib) so the whole suite runs:

```bash
uv sync            # base + dev group
uv run pytest -q   # the regression suite
```

## The one hard rule: byte-exact behaviour

procposets was extracted value-for-value from three research codebases, and its
**self-contained regression suite** (`procposets/tests/regression/`) pins the
library's numeric and structural output. There is no external oracle anymore, so
that suite *is* the behaviour contract:

- Any change that could alter a pinned value must be treated as **breaking** and
  land with the regression baselines updated deliberately (and called out in
  `CHANGELOG.md`), never silently.
- Refactors, dedups, and cleanups must keep every test green with **no**
  baseline change — state "byte-exact" in the PR when that is the intent.

Deliberately-kept design points (do **not** "simplify" them away):

- The two distinct string renderers (the `->`/`||` SP-tree renderer vs the
  `;`/`*` modular-decomposition renderer) are both kept on purpose.
- Equality / canonical forms are label-based; the canonical `Poset` is an
  id+label base (repeated-label-capable) and `Rel = frozenset[(str, str)]` is
  the certified distinct-label view (`to_rel` asserts distinctness).
- The numpy-free import contract: `import procposets` must stay third-party-free.
  numpy / networkx / pm4py / matplotlib load only via the extras, the
  NPMLE layer lazily (PEP 562). This is guarded by `test_lazy_numpy.py`.

## Style & checks

- `ruff` (lint + format) and `mypy` back the shipped `py.typed` marker; run
  `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy
  procposets` before opening a PR. CI runs these plus the suite across
  Python 3.10–3.13 and each extra.
- Keep new code in the layer its dependencies belong to (numpy-free core →
  root; `[graph]`/`[pm4py]` adapters → `cospan`/`adapters`; renderers → `viz`).

## Pull requests

Small, single-purpose PRs. Say whether the change is byte-exact or
value-changing, and add a `CHANGELOG.md` entry under `[Unreleased]`.
