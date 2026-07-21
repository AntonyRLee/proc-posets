"""Shared LM-graph assembly for the inbound model adapters.

The build-and-validate wrappers and the object-type namespacing prefix used by all
three inbound adapters: ``cospan.from_petri`` (B0, stdlib-only) and the ``[pm4py]``
``adapters.from_bpmn`` / ``adapters.from_process_tree``. Lives under ``cospan`` (not
``adapters``) so ``from_petri`` can share it without importing the pm4py-bearing
``adapters`` package -- ``import procposets.adapters`` runs its ``__init__`` and pulls
pm4py, which would break from_petri's numpy-only-core guarantee.
"""
from __future__ import annotations

from collections.abc import Callable

from .lmgraph import LMGraph


def _type_prefix(otype: str | None) -> str:
    """Object-type namespace prefix for adapter-internal node ids. The emitted string
    is load-bearing for cross-notation signature comparison -- keep it exact."""
    return f"{otype}__" if otype is not None else ""


def _assemble(add: Callable, models_by_type: dict) -> LMGraph:
    """Overlay each ``{otype: model}`` onto one LM-graph via the adapter's ``add``
    routine, then validate -- the typed object-centric build shared by all three
    ``lmgraph_from_*`` plural wrappers."""
    g = LMGraph()
    for otype, model in models_by_type.items():
        add(g, model, otype)
    g.validate()
    return g


def _assemble_single(add: Callable, model, otype: str | None) -> LMGraph:
    """Overlay a single (optionally untyped) model via ``add`` and validate."""
    g = LMGraph()
    add(g, model, otype)
    g.validate()
    return g
