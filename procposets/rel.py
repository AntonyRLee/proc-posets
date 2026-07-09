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

from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

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

def _preds(elements: Iterable[str], rel: Rel) -> Dict[str, FrozenSet[str]]:
    p: Dict[str, set] = {e: set() for e in elements}
    for a, b in rel:
        p[b].add(a)
    return {e: frozenset(s) for e, s in p.items()}


# Hard ceiling on the ideal-lattice DP's state count.  The DP is exponential
# only in poset width; the meet-closure oracle runs at *any* m, where meets of
# a few long chains are typically wide -- without a guard this is the one
# place the library can hang or exhaust memory with no printed verdict
# (DESIGN_REVIEW W12.1).  The bound below is certified before recursing.
MAX_IDEAL_STATES = 1_000_000


class IdealBudgetExceeded(ValueError):
    """The ideal-lattice DP would exceed MAX_IDEAL_STATES states.

    Raised *before* any exponential work happens.  The oracle catches this,
    skips the offending candidate, and downgrades the certificate loudly."""


def _ideal_state_bound(elements: FrozenSet[str], rel: Rel) -> int:
    """Upper bound on the number of order ideals, via a greedy chain cover.

    Every ideal meets each chain of a chain cover in a prefix, so
    #ideals <= prod_i (|c_i| + 1) for ANY chain cover -- a sound (if not
    minimal) budget check.  Chains are peeled greedily longest-first by DP
    on the DAG; the whole check is polynomial.
    """
    remaining = set(elements)
    bound = 1
    while remaining:
        succs = {e: [b for (a, b) in rel if a == e and b in remaining]
                 for e in remaining}
        best: Dict[str, Tuple[int, Optional[str]]] = {}

        def longest(x: str) -> int:
            got = best.get(x)
            if got is not None:
                return got[0]
            ln, nxt = 1, None
            for y in succs[x]:
                cand = 1 + longest(y)
                if cand > ln:
                    ln, nxt = cand, y
            best[x] = (ln, nxt)
            return ln

        top = max(remaining, key=longest)
        chain = [top]
        while best[chain[-1]][1] is not None:
            chain.append(best[chain[-1]][1])
        bound *= len(chain) + 1
        remaining -= set(chain)
        if bound > MAX_IDEAL_STATES:
            return bound
    return bound


def _check_ideal_budget(elements: FrozenSet[str], rel: Rel) -> None:
    bound = _ideal_state_bound(elements, rel)
    if bound > MAX_IDEAL_STATES:
        raise IdealBudgetExceeded(
            f"ideal-lattice DP refused: chain-cover bound of {bound:.2e} "
            f"states exceeds MAX_IDEAL_STATES = {MAX_IDEAL_STATES:.0e} "
            f"(poset too wide for exact e(P) on this budget)"
        )


def count_linear_extensions(elements: FrozenSet[str], rel: Rel) -> int:
    """e(P) for an arbitrary partial order, by DP over the ideal lattice.

    e(remaining) = sum over minimal x of remaining of e(remaining - x).
    The memo has one entry per order ideal -- at most 2^m, and typically far
    fewer; this is exact for every poset, with cost exponential only in the
    width.  A chain-cover bound on the ideal count is certified against
    MAX_IDEAL_STATES before recursing; a too-wide poset raises
    IdealBudgetExceeded instead of hanging (DESIGN_REVIEW W12.1).
    """
    _check_ideal_budget(elements, rel)
    preds = _preds(elements, rel)
    memo: Dict[FrozenSet[str], int] = {frozenset(): 1}

    def rec(rem: FrozenSet[str]) -> int:
        got = memo.get(rem)
        if got is not None:
            return got
        out = sum(rec(rem - {x}) for x in rem if not (preds[x] & rem))
        memo[rem] = out
        return out

    return rec(frozenset(elements))


def sample_linear_extension(elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
    """Uniform linear extension of an arbitrary partial order.

    Sequential sampling with the ideal-lattice DP as the exact proposal:
    the first element is x (minimal) with probability e(P - x) / e(P), which
    telescopes to 1/e(P) for every completed extension.  Guarded by the same
    ideal-state budget as count_linear_extensions.
    """
    _check_ideal_budget(elements, rel)
    preds = _preds(elements, rel)
    memo: Dict[FrozenSet[str], int] = {frozenset(): 1}

    def count(rem: FrozenSet[str]) -> int:
        got = memo.get(rem)
        if got is not None:
            return got
        out = sum(count(rem - {x}) for x in rem if not (preds[x] & rem))
        memo[rem] = out
        return out

    out: List[str] = []
    rem = frozenset(elements)
    while rem:
        mins = sorted(x for x in rem if not (preds[x] & rem))
        weights = [count(rem - {x}) for x in mins]
        total = sum(weights)
        r = rng.random() * total
        acc = 0
        for x, w in zip(mins, weights):
            acc += w
            if r < acc:
                break
        out.append(x)
        rem = rem - {x}
    return tuple(out)


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
    return sorted(posets, key=lambda r: (len(r), sorted(r)))


def _ideals(elements: Sequence[str], rel: Rel) -> List[FrozenSet[str]]:
    """All order ideals (down-closed sets) of a poset, by brute subset check
    (elements here never exceeds the enumeration bound, so 2^m is small)."""
    els = list(elements)
    n = len(els)
    down = {e: {a for (a, b) in rel if b == e} for e in els}
    out = []
    for mask in range(2 ** n):
        s = {els[i] for i in range(n) if mask >> i & 1}
        if all(down[e] <= s for e in s):
            out.append(frozenset(s))
    return out


def _filters(elements: Sequence[str], rel: Rel) -> List[FrozenSet[str]]:
    """All filters (up-closed sets) of a poset."""
    els = list(elements)
    n = len(els)
    up = {e: {b for (a, b) in rel if a == e} for e in els}
    out = []
    for mask in range(2 ** n):
        s = {els[i] for i in range(n) if mask >> i & 1}
        if all(up[e] <= s for e in s):
            out.append(frozenset(s))
    return out


def meet_closure(rels: Iterable[Rel], cap: int = 200_000) -> Tuple[List[Rel], bool]:
    """Closure of a set of orders under pairwise meet (hence all finite meets).

    Returns (closure sorted, hit_cap).  The closure of the observed chains is
    an *exact* pricing domain for the general class under the uniform noise
    kernel (see oracle.py for the reduction theorem); ``hit_cap`` marks the
    rare data sets where the closure is truncated and the certificate reverts
    to lattice-restricted status.
    """
    _key = lambda r: (len(r), sorted(r))  # noqa: E731 -- canonical order
    closed = set(rels)
    # frontier and base iterate in canonical sorted order so that a cap hit
    # truncates deterministically -- hash order made the kept subset vary
    # with PYTHONHASHSEED, making the downgraded regime irreproducible
    # (DESIGN_REVIEW W12.4)
    frontier = sorted(closed, key=_key)
    hit_cap = False
    while frontier and not hit_cap:
        base = sorted(closed, key=_key)
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
        frontier = sorted(fresh, key=_key)
    return sorted(closed, key=_key), hit_cap


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


def sample_extension(tree: SPTree, rng) -> Tuple[str, ...]:
    """Uniform linear extension of an SP poset.

    Correctness: extensions of a parallel composition biject with (extension
    of each child, interleaving pattern), all combinations equally likely, so
    sampling each factor uniformly and a uniform shuffle of block labels
    yields the uniform distribution.
    """
    if tree.kind == "leaf":
        return (tree.label,)
    parts = [list(sample_extension(c, rng)) for c in tree.children]
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

    return sorted(rec(els), key=lambda r: (len(r), sorted(r)))


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def series(*parts) -> SPTree:
    kids = tuple(p if isinstance(p, SPTree) else SPTree("leaf", label=p) for p in parts)
    flat: List[SPTree] = []
    for k in kids:
        flat.extend(k.children if k.kind == "series" else (k,))
    return SPTree("series", children=tuple(flat))


def parallel(*parts) -> SPTree:
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


# ---------------------------------------------------------------------------
# Hypothesis classes: the declared search space of the estimator (choice C)
# ---------------------------------------------------------------------------

class PosetClass:
    """Protocol for a hypothesis class of partial orders (README choice C).

    A class declares two capability flags the oracle dispatches on -- these
    document exactly what the meet-closure reduction theorem needs, instead
    of encoding it as pointer identity (DESIGN_REVIEW W18):

    * ``contains_all_posets`` -- every partial order on the alphabet is in
      the class (so a sup over the class is a sup over all posets);
    * ``closed_under_meet``   -- meets of class members stay in the class
      (so the closure of the observed chains is inside the class).

    The exact meet-closure oracle regime requires BOTH (plus the uniform
    kernel, untimed); a class missing either falls to the heuristic regime.
    Required methods: ``contains``, ``extension_count``, ``enumerate``,
    ``sample_extension``; required attributes: ``name`` and the two flags.
    """

    name: str
    contains_all_posets: bool = False
    closed_under_meet: bool = False

    def contains(self, elements: FrozenSet[str], rel: Rel) -> bool:
        raise NotImplementedError

    def extension_count(self, elements: FrozenSet[str], rel: Rel) -> int:
        raise NotImplementedError

    def enumerate(self, elements: Iterable[str]) -> List[Rel]:
        raise NotImplementedError

    def sample_extension(self, elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
        raise NotImplementedError


class GeneralPosets(PosetClass):
    """All partial orders on the alphabet -- the default hypothesis class.

    Closed under meet (so the oracle's lattice moves never leave the class),
    exhaustively enumerable to m = 6, with exact e(P) by the ideal DP.
    """

    name = "general"
    contains_all_posets = True
    closed_under_meet = True

    def contains(self, elements: FrozenSet[str], rel: Rel) -> bool:
        return True  # rel is maintained transitively closed by construction

    def extension_count(self, elements: FrozenSet[str], rel: Rel) -> int:
        return count_linear_extensions(elements, rel)

    def enumerate(self, elements: Iterable[str]) -> List[Rel]:
        return enumerate_posets(elements)

    def sample_extension(self, elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
        return sample_linear_extension(elements, rel, rng)


class SPPosets(PosetClass):
    """Series-parallel orders only -- the original, smaller class.

    Kept for (i) speed at m = 6 (fewer atoms, linear-time e(P)) and (ii) as
    the declared class when the domain genuinely excludes N-shaped
    dependencies.  Meets can leave the class; the oracle skips those points.
    """

    name = "sp"
    contains_all_posets = False
    closed_under_meet = False

    def contains(self, elements: FrozenSet[str], rel: Rel) -> bool:
        return decompose(elements, rel) is not None

    def extension_count(self, elements: FrozenSet[str], rel: Rel) -> int:
        tree = decompose(elements, rel)
        if tree is None:
            raise ValueError("relation set is not series-parallel")
        return extension_count(tree)

    def enumerate(self, elements: Iterable[str]) -> List[Rel]:
        return enumerate_sp(elements)

    def sample_extension(self, elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
        tree = decompose(elements, rel)
        if tree is None:
            raise ValueError("relation set is not series-parallel")
        return sample_extension(tree, rng)


GENERAL = GeneralPosets()
SP = SPPosets()


def get_poset_class(spec):
    """Resolve a class spec: "general" | "sp" | a PosetClass instance.

    Arbitrary objects are rejected with the missing protocol members named
    (DESIGN_REVIEW W21): a class that reaches the oracle unvalidated would
    fail deep inside atom construction, or worse, silently mis-dispatch.
    """
    if isinstance(spec, str):
        try:
            return {"general": GENERAL, "sp": SP}[spec]
        except KeyError:
            raise ValueError(f"unknown poset class {spec!r}; use 'general' or 'sp'")
    missing = [
        member
        for member in ("name", "contains_all_posets", "closed_under_meet",
                       "contains", "extension_count", "enumerate",
                       "sample_extension")
        if not hasattr(spec, member)
    ]
    if missing:
        raise TypeError(
            f"poset class {spec!r} does not implement the PosetClass "
            f"protocol: missing {missing}"
        )
    return spec
