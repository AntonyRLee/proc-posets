"""The certified bridge between the two poset views.

The canonical object is the id+label `Poset` (`poset.py`), which can carry
*repeated* labels.  `Rel = frozenset[(label, label)]` (`rel.py`) is the
estimation vocabulary and the **distinct-label view** of a poset.  `to_rel`
is a *certified projection*, never a lossy cast: it asserts labels are
distinct and raises `LabelCollision` otherwise, so no code silently collapses
two same-labelled elements into one `Rel` pair.
"""

from __future__ import annotations

import itertools
from typing import Iterable

from .poset import Poset
from .rel import Rel


class LabelCollision(ValueError):
    """Raised when a poset with repeated labels is projected to a `Rel`,
    which cannot represent them (label == element there)."""


def to_rel(p: Poset) -> Rel:
    """Distinct-label projection of a poset to a `Rel`.  Raises
    `LabelCollision` if any label repeats (the projection would be lossy)."""
    labels = [p.labels[e] for e in p.elements]
    if len(set(labels)) != len(labels):
        dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
        raise LabelCollision(
            f"poset has repeated labels {dupes}; a Rel (label == element) "
            f"cannot represent them -- keep the Poset, or relabel"
        )
    return frozenset((p.labels[u], p.labels[v]) for (u, v) in p.less)


def rel_elements(p: Poset) -> "frozenset[str]":
    """The label set of a distinct-label poset (the `elements` companion a
    `Rel` needs)."""
    to_rel(p)  # assert distinctness
    return frozenset(p.labels.values())


def from_rel(elements: Iterable[str], rel: Rel) -> Poset:
    """Build a canonical `Poset` from a distinct-label `Rel` plus its element
    (label) set.  `rel` is assumed transitively closed (as every `Rel` in the
    estimation code is); the labels become the elements one-for-one."""
    labels = sorted(set(elements) | {a for pair in rel for a in pair})
    ids = {lab: i for i, lab in enumerate(labels)}
    less = {(ids[u], ids[v]) for (u, v) in rel}
    return Poset(list(ids.values()), {i: lab for lab, i in ids.items()}, less)
