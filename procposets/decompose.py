"""Decomposition facade — both decomposition views under one import.

Two decompositions coexist by design (do NOT unify their renderers — the
consumer repos bake each canonical string into byte-exact regressions):

- The **SP-tree** view (`SPTree`, `decompose(elements, rel) -> SPTree|None`),
  from the NPMLE engine: series/parallel only, returns ``None`` on a non-SP
  order, renders with ``->`` / ``||``, and carries the VTL extension
  count/sampler.  This is what the estimator's fast path uses.
- The **total modular** view (`Leaf`/`Series`/`Parallel`/`Prime`,
  `modular_decompose(Poset)`), from the SPME engine: defined on *every*
  poset (a non-SP order surfaces a ``Prime`` node), renders with ``;`` /
  ``*``.

`is_sp(P)` holds iff the modular decomposition contains no ``Prime`` — the
two views agree on series-parallelness (a golden cross-check pins this).
"""

from __future__ import annotations

# SP-tree view (NPMLE) — series/parallel, None-on-non-SP, ->/|| renderer
from .rel import (
    SPTree,
    decompose,
    enumerate_sp,
    extension_count,
    is_sp,
    parallel,
    sample_extension,
    sample_extension_tree,
    series,
    tree_relations,
)

# Total modular view (SPME) — Leaf/Series/Parallel/Prime, ;/* renderer
from .moddecomp import (
    Leaf,
    Parallel,
    Prime,
    Series,
)
from .moddecomp import decompose as modular_decompose
from .moddecomp import tiling

__all__ = [
    "SPTree", "decompose", "is_sp", "extension_count",
    "sample_extension", "sample_extension_tree",
    "tree_relations", "series", "parallel", "enumerate_sp",
    "Leaf", "Series", "Parallel", "Prime", "modular_decompose", "tiling",
]
