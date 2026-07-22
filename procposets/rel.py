"""Partial orders: general posets (the default hypothesis class) and SP posets.

Representation
--------------
A partial order on a finite alphabet is represented *extensionally* as a
``frozenset`` of ordered pairs ``(a, b)`` meaning "a strictly precedes b",
always transitively closed.  This is the natural representation for the
estimation problem because the two operations the pipeline lives on are

* **meet** (intersection of orders)  -- set intersection of pair sets, and
* **containment** (P is refined by Q) -- subset test of pair sets,

both of which are trivial extensionally.  Crucially the meet of *any* two
partial orders is a partial order, so the general class is closed under the
one lattice operation the oracle needs -- unlike SP, where meets can leave
the class and had to be skipped.

General posets (default class)
------------------------------
Extension counting for general posets is #P-complete in the worst case
(Brightwell-Winkler), but the standard dynamic program over order ideals,

    e(I) = sum over minimal x of I of e(I \\ {x}),      e(empty) = 1,

is exponential only in the *width*: its state space is the ideal lattice,
at most 2^m states, trivially small at the alphabet sizes where exhaustive
class enumeration is feasible anyway (labeled posets: 219 on 4 elements,
4231 on 5, 130023 on 6 -- OEIS A001035).  The same DP samples uniform
extensions.  So at small m nothing forces the SP restriction, and the
hardness argument only justifies *guarding* the DP at large m, not shrinking
the declared hypothesis class.  SP remains available as an opt-in class
(``poset_class="sp"``), where the Valdes-Tarjan-Lawler recursion

    e(P . Q)  = e(P) e(Q)                       (series)
    e(P || Q) = C(|P|+|Q|, |P|) e(P) e(Q)       (parallel)

evaluates e(P) on the decomposition tree in linear time.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List, Sequence, Tuple

Pair = Tuple[str, str]
Rel = FrozenSet[Pair]


# ---------------------------------------------------------------------------
# Basic relation-set operations
# ---------------------------------------------------------------------------

def rel_from_trace(trace: Sequence[str]) -> Rel:
    """The total order induced by a trace: all pairs (earlier, later)."""
    return frozenset(
        (trace[i], trace[j]) for i in range(len(trace)) for j in range(i + 1, len(trace))
    )


def meet(*rels: Rel) -> Rel:
    """Intersection of partial orders.

    The intersection of transitive relations is transitive, so no re-closure
    is needed.  This is the lattice meet on the poset-of-posets and the
    workhorse of group compression: Q_g = meet(traces of group g).
    """
    it = iter(rels)
    out = set(next(it))
    for r in it:
        out &= r
    return frozenset(out)


def respects(trace: Sequence[str], rel: Rel) -> bool:
    """Is the trace a linear extension of the order? (all pairs respected)"""
    pos = {a: i for i, a in enumerate(trace)}
    return all(pos[a] < pos[b] for (a, b) in rel)


def refines(p: Rel, q: Rel) -> bool:
    """P <= Q in the refinement order: every relation of P holds in Q."""
    return p <= q


def is_partial_order(elements: FrozenSet[str], rel: Rel) -> bool:
    """Is the relation set an irreflexive, antisymmetric, transitive order?"""
    if any(a == b or (b, a) in rel for (a, b) in rel):
        return False
    if any(a not in elements or b not in elements for (a, b) in rel):
        return False
    return all(
        (a, c) in rel for (a, b) in rel for (b2, c) in rel if b2 == b
    )


def transitive_reduction(rel: Rel) -> Rel:
    """Cover relations of a (transitively closed) order: its Hasse diagram."""
    return frozenset(
        (a, b) for (a, b) in rel
        if not any((a, c) in rel and (c, b) in rel for c in {x for _, x in rel})
    )


# ---------------------------------------------------------------------------
# General posets: extension counting / sampling by DP over order ideals
# ---------------------------------------------------------------------------

# The guarded ideal-lattice engine (count / sample / budget guard) now lives
# once in procposets._extensions and is shared with the canonical Poset
# (poset.py); the element type is irrelevant, only the order pairs matter.
# The thin wrappers below read this module's MAX_IDEAL_STATES *at call time*
# and pass it in, so monkeypatching rel.MAX_IDEAL_STATES (the historical
# knob, e.g. in the oracle-downgrade test) still steers the guard.
from ._extensions import (  # noqa: E402
    IdealBudgetExceeded,  # noqa: F401  re-export: `procposets.rel.IdealBudgetExceeded` is a documented consumer surface (PMN)
    MAX_IDEAL_STATES,
)
from . import _extensions as _ext  # noqa: E402


def count_linear_extensions(elements, rel) -> int:
    """Rel-view spelling of e(P) (the number of linear extensions).

    Same guarded ideal-lattice DP engine (``_extensions.count_extensions``) as the
    canonical Poset-object spelling :func:`procposets.count_extensions`; kept under
    its historical NPMLE name as a documented alias. See
    :func:`procposets.rel_sp.extension_count` for the SP closed-form spelling.
    """
    return _ext.count_extensions(elements, rel, max_states=MAX_IDEAL_STATES)


def sample_linear_extension(elements, rel, rng):
    return _ext.sample_extension_poset(elements, rel, rng, max_states=MAX_IDEAL_STATES)


def enumerate_posets(elements: Iterable[str]) -> List[Rel]:
    """All partial orders on the given labeled elements, as relation sets.

    Recursion: a poset on e_1..e_k restricts to a poset R on e_1..e_{k-1};
    conversely e_k can be inserted into R by choosing its strict down-set D
    (an order ideal of R) and strict up-set U (a filter), disjoint, with
    D x U inside R -- exactly the closure condition, so the result is
    transitively closed by construction and every labeled poset is produced
    exactly once.  Counts match OEIS A001035: 3, 19, 219, 4231, 130023 for
    m = 2..6.  m = 6 takes a few seconds and a few hundred MB; beyond that
    use the meet-closure oracle instead of class enumeration.
    """
    els = sorted(elements)
    posets: List[Rel] = [frozenset()]
    for k, ek in enumerate(els):
        sub = els[:k]
        nxt: List[Rel] = []
        for R in posets:
            ideals = _ideals(sub, R)
            filters = _filters(sub, R)
            for D in ideals:
                for U in filters:
                    if D & U:
                        continue
                    if all((d, u) in R for d in D for u in U):
                        nxt.append(
                            R
                            | frozenset((d, ek) for d in D)
                            | frozenset((ek, u) for u in U)
                        )
        posets = nxt
    return sorted(posets, key=_canonical_key)


def _canonical_key(r: Rel):
    """Canonical ordering of a relation set: ``(size, sorted pairs)``.  Single
    source for the sort key that ``enumerate_posets`` / ``meet_closure`` /
    ``enumerate_sp`` each spelled inline."""
    return (len(r), sorted(r))


def _closed_sets(elements: Sequence[str], rel: Rel, *, down: bool) -> List[FrozenSet[str]]:
    """Order ideals (``down=True``: down-closed sets) or filters (``down=False``:
    up-closed sets) of a poset, by brute 2^n subset check (elements here never
    exceeds the enumeration bound, so 2^m is small).  ``_ideals``/``_filters`` are
    the same scan differing only in the neighbour direction."""
    els = list(elements)
    n = len(els)
    if down:
        nbr = {e: {a for (a, b) in rel if b == e} for e in els}
    else:
        nbr = {e: {b for (a, b) in rel if a == e} for e in els}
    out = []
    for mask in range(2 ** n):
        s = {els[i] for i in range(n) if mask >> i & 1}
        if all(nbr[e] <= s for e in s):
            out.append(frozenset(s))
    return out


def _ideals(elements: Sequence[str], rel: Rel) -> List[FrozenSet[str]]:
    """All order ideals (down-closed sets) of a poset."""
    return _closed_sets(elements, rel, down=True)


def _filters(elements: Sequence[str], rel: Rel) -> List[FrozenSet[str]]:
    """All filters (up-closed sets) of a poset."""
    return _closed_sets(elements, rel, down=False)


def meet_closure(rels: Iterable[Rel], cap: int = 200_000) -> Tuple[List[Rel], bool]:
    """Closure of a set of orders under pairwise meet (hence all finite meets).

    Returns (closure sorted, hit_cap).  The closure of the observed chains is
    an *exact* pricing domain for the general class under the uniform noise
    kernel (see oracle.py for the reduction theorem); ``hit_cap`` marks the
    rare data sets where the closure is truncated and the certificate reverts
    to lattice-restricted status.
    """
    closed = set(rels)
    # frontier and base iterate in canonical sorted order so that a cap hit
    # truncates deterministically -- hash order made the kept subset vary
    # with PYTHONHASHSEED, making the downgraded regime irreproducible.
    frontier = sorted(closed, key=_canonical_key)
    hit_cap = False
    while frontier and not hit_cap:
        base = sorted(closed, key=_canonical_key)
        fresh = set()
        for r1 in frontier:
            for r2 in base:
                q = r1 & r2
                if q not in closed and q not in fresh:
                    fresh.add(q)
                    if len(closed) + len(fresh) > cap:
                        hit_cap = True
                        break
            if hit_cap:
                break
        closed |= fresh
        frontier = sorted(fresh, key=_canonical_key)
    return sorted(closed, key=_canonical_key), hit_cap


# ---------------------------------------------------------------------------
# The SP-tree view and the hypothesis classes were split into companion
# modules for size; re-export them so ``from procposets.rel import ...`` and
# the package ``__init__`` keep working unchanged.
# ---------------------------------------------------------------------------
from .rel_sp import (  # noqa: E402
    SPTree,
    decompose,
    describe,
    enumerate_sp,
    extension_count,
    is_sp,
    parallel,
    sample_extension_tree,
    series,
    tree_relations,
)
from .rel_classes import (  # noqa: E402
    GENERAL,
    SP,
    GeneralPosets,
    PosetClass,
    SPPosets,
    get_poset_class,
)
