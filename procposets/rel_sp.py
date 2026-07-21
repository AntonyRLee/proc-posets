"""SP-tree view of a partial order: SP decomposition, extension counting
and uniform sampling, exhaustive SP enumeration, and the convenience
constructors.

Split out of :mod:`procposets.rel` for size; ``rel`` re-exports every name
here so the public import path (``procposets.rel`` and the package root) is
unchanged.  numpy-free, like the rest of the poset core.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from .rel import Rel, _canonical_key, transitive_reduction


# ---------------------------------------------------------------------------
# SP decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SPTree:
    """SP decomposition tree.  kind in {'leaf', 'series', 'parallel'}."""

    kind: str
    label: Optional[str] = None
    children: Tuple["SPTree", ...] = ()

    def elements(self) -> Tuple[str, ...]:
        if self.kind == "leaf":
            return (self.label,)
        out: List[str] = []
        for c in self.children:
            out.extend(c.elements())
        return tuple(out)

    def __str__(self) -> str:  # human-readable, e.g. (a -> (b || c) -> d)
        if self.kind == "leaf":
            return self.label
        sep = " -> " if self.kind == "series" else " || "
        return "(" + sep.join(str(c) for c in self.children) + ")"


def decompose(elements: FrozenSet[str], rel: Rel) -> Optional[SPTree]:
    """SP decomposition of a partial order, or None if the order is not SP.

    Algorithm: at each level, first try a parallel split (connected
    components of the comparability graph), then a series split (a
    down-closed block S with S x (V \\ S) fully related; any such block is a
    prefix of the ancestor-count ordering, which makes the search linear in
    the number of candidate cut points).
    """
    els = sorted(elements)
    if len(els) == 1:
        return SPTree("leaf", label=els[0])
    r = {(a, b) for (a, b) in rel if a in elements and b in elements}

    # --- parallel: components of the comparability graph
    adj: Dict[str, set] = {e: set() for e in els}
    for a, b in r:
        adj[a].add(b)
        adj[b].add(a)
    comps: List[set] = []
    seen: set = set()
    for e in els:
        if e in seen:
            continue
        comp, stack = set(), [e]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        comps.append(comp)
        seen |= comp
    if len(comps) > 1:
        kids = []
        for comp in comps:
            t = decompose(frozenset(comp), frozenset(r))
            if t is None:
                return None
            kids.append(t)
        return SPTree("parallel", children=tuple(sorted(kids, key=str)))

    # --- series: smallest proper prefix S of the ancestor-count order with
    #     full S x (V\S) cross relations
    anc = {e: sum(1 for (a, b) in r if b == e) for e in els}
    order = sorted(els, key=lambda e: (anc[e], e))
    n = len(order)
    for k in range(1, n):
        if anc[order[k - 1]] == anc[order[k]]:
            continue  # a valid cut boundary cannot split an ancestor-count tie
        S, T = order[:k], order[k:]
        if all((s, t) in r for s in S for t in T):
            left = decompose(frozenset(S), frozenset(r))
            right = decompose(frozenset(T), frozenset(r))
            if left is None or right is None:
                return None
            # flatten nested series for canonical trees
            lk = left.children if left.kind == "series" else (left,)
            rk = right.children if right.kind == "series" else (right,)
            return SPTree("series", children=lk + rk)
    return None  # connected, no series cut, >1 element: contains an N -> not SP


def is_sp(elements: FrozenSet[str], rel: Rel) -> bool:
    return decompose(elements, rel) is not None


# ---------------------------------------------------------------------------
# Extension counting and uniform sampling (VTL recursion)
# ---------------------------------------------------------------------------

def extension_count(tree: SPTree) -> int:
    """SP closed-form spelling of e(P) for a series-parallel tree.

    Agrees with :func:`procposets.count_extensions` /
    :func:`procposets.count_linear_extensions` on SP posets, but is the ungated
    linear-time VTL recursion (series: product of children; parallel: multinomial
    interleavings) -- a SEPARATE function by design; do not collapse it into the
    guarded ideal-lattice DP engine.
    """
    if tree.kind == "leaf":
        return 1
    if tree.kind == "series":
        out = 1
        for c in tree.children:
            out *= extension_count(c)
        return out
    # parallel: multinomial interleavings of the children's extensions
    sizes = [len(c.elements()) for c in tree.children]
    total = sum(sizes)
    out = factorial(total)
    for s in sizes:
        out //= factorial(s)
    for c in tree.children:
        out *= extension_count(c)
    return out


def tree_relations(tree: SPTree) -> Rel:
    """Relation set of an SP tree (series adds all cross pairs, in order)."""
    if tree.kind == "leaf":
        return frozenset()
    rel: set = set()
    for c in tree.children:
        rel |= tree_relations(c)
    if tree.kind == "series":
        blocks = [c.elements() for c in tree.children]
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                rel |= {(a, b) for a in blocks[i] for b in blocks[j]}
    return frozenset(rel)


def sample_extension_tree(tree: SPTree, rng) -> Tuple[str, ...]:
    """Uniform linear extension of an SP poset (the input-typed SPTree sampler).

    Correctness: extensions of a parallel composition biject with (extension
    of each child, interleaving pattern), all combinations equally likely, so
    sampling each factor uniformly and a uniform shuffle of block labels
    yields the uniform distribution.
    """
    if tree.kind == "leaf":
        return (tree.label,)
    parts = [list(sample_extension_tree(c, rng)) for c in tree.children]
    if tree.kind == "series":
        out: List[str] = []
        for p in parts:
            out.extend(p)
        return tuple(out)
    labels: List[int] = []
    for i, p in enumerate(parts):
        labels.extend([i] * len(p))
    rng.shuffle(labels)
    its = [iter(p) for p in parts]
    return tuple(next(its[i]) for i in labels)


# ---------------------------------------------------------------------------
# Exhaustive enumeration of SP posets on a labeled alphabet
# ---------------------------------------------------------------------------

def enumerate_sp(elements: Iterable[str]) -> List[Rel]:
    """All SP partial orders on the given labeled elements, as relation sets.

    This makes the pricing oracle *exact* for small alphabets (m <= ~6),
    which is what turns the Frank-Wolfe duality gap into a genuine
    optimality certificate.  Beyond that size, replace with the
    lattice-restricted oracle (see oracle.py) -- the single approximate
    flank of the pipeline.

    Recursion: an SP poset on S is a leaf, or a series composition whose
    first block is a proper subset A (all of A before all of S\\A), or a
    parallel composition one of whose blocks A contains a fixed anchor
    element (to avoid double counting unordered blocks).  Duplicates from
    flattened compositions are removed by keying on the relation set.
    """
    els = frozenset(elements)

    @lru_cache(maxsize=None)
    def rec(s: FrozenSet[str]) -> FrozenSet[Rel]:
        s_sorted = sorted(s)
        if len(s_sorted) == 1:
            return frozenset({frozenset()})
        out: set = set()
        members = list(s_sorted)
        anchor = members[0]
        n = len(members)
        for mask in range(1, 2 ** n - 1):
            A = frozenset(members[i] for i in range(n) if mask >> i & 1)
            B = s - A
            cross = frozenset((a, b) for a in A for b in B)
            for ra in rec(A):
                for rb in rec(B):
                    out.add(ra | rb | cross)          # series: A before B
                    if anchor in A:                    # parallel (anchor breaks symmetry)
                        out.add(ra | rb)
        return frozenset(out)

    return sorted(rec(els), key=_canonical_key)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def series(*parts) -> SPTree:
    """Series composition of ``SPTree`` nodes (the distinct-label ``->``/``||`` view).

    The ``SPTree`` constructor bound as the package-root ``procposets.series``.
    Distinct from the ``Poset``-object combinator :func:`procposets.poset.then` --
    same SP concept, different type/renderer; the two are intentionally NOT unified.
    """
    kids = tuple(p if isinstance(p, SPTree) else SPTree("leaf", label=p) for p in parts)
    flat: List[SPTree] = []
    for k in kids:
        flat.extend(k.children if k.kind == "series" else (k,))
    return SPTree("series", children=tuple(flat))


def parallel(*parts) -> SPTree:
    """Parallel composition of ``SPTree`` nodes (children sorted for canonicity).

    The ``SPTree`` constructor bound as the package-root ``procposets.parallel``; the
    ``Poset``-object counterpart is :func:`procposets.poset.par` (see :func:`series`).
    """
    kids = tuple(p if isinstance(p, SPTree) else SPTree("leaf", label=p) for p in parts)
    return SPTree("parallel", children=tuple(sorted(kids, key=str)))


def describe(elements: FrozenSet[str], rel: Rel) -> str:
    """Readable string for a relation set: its SP tree when the order is SP
    (the compact, familiar form), otherwise its cover relations (Hasse
    diagram), which determine the order without the transitive clutter."""
    t = decompose(elements, rel)
    if t is not None:
        return str(t)
    covers = transitive_reduction(rel)
    return "{" + ", ".join(f"{a}<{b}" for a, b in sorted(covers)) + "}"
