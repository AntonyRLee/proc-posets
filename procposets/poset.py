"""Labelled finite posets = the causal partial order of a process diagram.

A diagram is projected to its causal poset (def:causal-order); everything downstream
(modular decomposition, the block matrix) operates here.

Constructors mirror the two compositions of the string-diagram calculus:
    leaf(label)      an atomic activity
    then(A, B, ...)  sequential ; composition  (all of A before all of B before ...)
    par(A, B, ...)   parallel  (x) composition  (incomparable)
    n_poset()        the smallest non-series-parallel ("N") poset
Elements are opaque integers; only their activity labels matter for comparison.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_ids = itertools.count()


@dataclass
class Poset:
    """A labelled finite poset: opaque integer ``elements``, their activity
    ``labels`` side-table (so labels may REPEAT), and the strict order ``less`` as
    a transitively-closed set of ``(u, v)`` = "u < v" pairs.  Only labels matter
    for comparison; the ids keep repeated-label elements distinct."""

    elements: list[int]
    labels: dict[int, str]
    less: set[tuple[int, int]] = field(default_factory=set)  # (u, v) means u < v, transitively closed

    def comparable(self, u: int, v: int) -> bool:
        """Are elements ``u`` and ``v`` ordered either way (not incomparable)?"""
        return (u, v) in self.less or (v, u) in self.less

    def restrict(self, sub: list[int]) -> "Poset":
        """The induced sub-poset on the element ids ``sub`` (order restricted)."""
        s = set(sub)
        return Poset(
            list(sub),
            {e: self.labels[e] for e in sub},
            {(u, v) for (u, v) in self.less if u in s and v in s},
        )

    def __len__(self) -> int:
        return len(self.elements)


# A *model* is a weighted set of labelled posets (its variants): the object the
# distance / estimation / trace / loop layers all operate on.  Single home for the
# alias that four of those modules each used to redefine (and ``traces`` weakened
# to a bare ``list``).  Annotation-only -- never constructed at runtime.
Model = list[tuple[Poset, float]]


def _transitive_closure(less: set) -> set:
    """Close a set of ``(u, v)`` = "u < v" pairs under transitivity, in place.

    Single home for what ``from_dag`` and ``from_edges`` used to inline
    byte-identically (an O(n^3) fixpoint loop)."""
    changed = True
    while changed:
        changed = False
        for a, b in list(less):
            for c, d in list(less):
                if b == c and (a, d) not in less:
                    less.add((a, d))
                    changed = True
    return less


def _fresh(p: Poset) -> Poset:
    """A copy of p with fresh element ids (so compositions never alias)."""
    m = {e: next(_ids) for e in p.elements}
    return Poset(
        [m[e] for e in p.elements],
        {m[e]: p.labels[e] for e in p.elements},
        {(m[u], m[v]) for (u, v) in p.less},
    )


def leaf(label: str) -> Poset:
    """A single-element ``Poset`` carrying ``label`` -- the atom of the
    ``then`` / ``par`` algebra."""
    e = next(_ids)
    return Poset([e], {e: label}, set())


def then(*parts: Poset) -> Poset:
    """Sequential composition of ``Poset`` objects: part_0 < part_1 < ... (ordinal sum).

    The ``Poset``-object (id+label, repeated-label-capable) series combinator,
    rendered via the moddecomp ``;``/``*`` view. Intentionally named ``then``/``par``
    per this type -- the distinct-label ``SPTree`` combinators are
    :func:`procposets.rel_sp.series` / :func:`procposets.rel_sp.parallel` (and the
    package-root ``series``/``parallel`` bind THOSE), so the two SP-composition
    vocabularies are NOT interchangeable.
    """
    parts = [_fresh(p) for p in parts]
    elements: list[int] = []
    labels: dict[int, str] = {}
    less: set[tuple[int, int]] = set()
    for p in parts:
        elements += p.elements
        labels.update(p.labels)
        less |= p.less
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            for u in parts[i].elements:
                for v in parts[j].elements:
                    less.add((u, v))
    return Poset(elements, labels, less)


def par(*parts: Poset) -> Poset:
    """Parallel composition of ``Poset`` objects: disjoint union, no cross relations
    (antichain of blocks). The ``Poset``-object parallel combinator; named ``par``
    (not ``parallel``) because the ``parallel`` name is the ``SPTree`` constructor
    :func:`procposets.rel_sp.parallel` -- see :func:`then`."""
    parts = [_fresh(p) for p in parts]
    elements: list[int] = []
    labels: dict[int, str] = {}
    less: set[tuple[int, int]] = set()
    for p in parts:
        elements += p.elements
        labels.update(p.labels)
        less |= p.less
    return Poset(elements, labels, less)


def n_poset(a: str = "a", b: str = "b", c: str = "c", d: str = "d") -> Poset:
    """The canonical non-SP poset: a<c, b<c, b<d (a,b minimal; c,d incomparable)."""
    va, vb, vc, vd = (next(_ids) for _ in range(4))
    labels = {va: a, vb: b, vc: c, vd: d}
    less = {(va, vc), (vb, vc), (vb, vd)}
    return Poset([va, vb, vc, vd], labels, less)


def from_edges(nodes: dict, edges) -> Poset:
    """Build a labelled poset from explicit node KEYS, so activity labels may REPEAT.
    `nodes` maps an arbitrary node key -> its activity label; `edges` are (key_u, key_v) pairs meaning
    u < v. Result is the transitive closure. Use this for non-series-parallel shapes with a repeated
    label, which `from_dag` cannot express (it keys nodes by label). E.g. an N with two 'a's:
        from_edges({"a1":"a","b":"b","a2":"a","d":"d"}, [("a1","a2"),("b","a2"),("b","d")])."""
    ids = {key: next(_ids) for key in nodes}
    less = {(ids[u], ids[v]) for u, v in edges}
    _transitive_closure(less)                        # transitive closure
    return Poset(list(ids.values()), {ids[k]: lab for k, lab in nodes.items()}, less)


def from_dag(edges, nodes=()) -> Poset:
    """Build a labelled poset from immediate precedence edges (label_u, label_v) meaning u < v,
    plus any isolated `nodes`. Labels must be distinct (one element each); the result is the
    transitive closure. Use this for non-series-parallel shapes (e.g. the N: from_dag([("a","c"),
    ("b","c"),("b","d")])). For a non-SP shape with a REPEATED label, use `from_edges` instead."""
    order = []
    for u, v in edges:
        order += [u, v]
    order += list(nodes)
    uniq = list(dict.fromkeys(order))
    ids = {lab: next(_ids) for lab in uniq}
    less = {(ids[u], ids[v]) for u, v in edges}
    _transitive_closure(less)                        # transitive closure
    return Poset(list(ids.values()), {i: lab for lab, i in ids.items()}, less)


# ---------------------------------------------------------------------------
# Guarded linear-extension count / sample on the canonical Poset.
#
# Same engine as the Rel view (procposets._extensions): the canonical id-keyed
# Poset gains the budget-guarded counter/sampler it lacked, so counting no
# longer means enumerating every word (traces.linear_extensions) and cannot
# hang on a wide poset (it raises IdealBudgetExceeded instead).  Works for
# posets with REPEATED labels too (elements are ids, not labels).
# ---------------------------------------------------------------------------

from ._extensions import IdealBudgetExceeded  # noqa: E402,F401  (re-export)
from ._extensions import count_extensions as _count
from ._extensions import sample_extension_poset as _sample


def count_extensions(P: "Poset") -> int:
    """e(P): the number of linear extensions, via the guarded ideal-lattice
    DP.  Exact for every poset; raises ``IdealBudgetExceeded`` on a poset too
    wide for the declared budget rather than hanging."""
    return _count(P.elements, P.less)


def sample_extension(P: "Poset", rng) -> tuple:
    """A uniform random linear extension of ``P`` as a tuple of element ids
    (map through ``P.labels`` for a label word).  Guarded like
    ``count_extensions``."""
    return _sample(P.elements, P.less, rng)
