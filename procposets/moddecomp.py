"""Modular (substitution) decomposition of a causal poset = the canonical tiling.

Gallai's theorem on the comparability graph gives a unique decomposition tree:
    - comparability graph disconnected  -> PARALLEL node (disjoint union / (x))
    - complement disconnected           -> SERIES node   (ordinal sum / ;)
    - otherwise                         -> PRIME node    (irreducibly non-SP; e.g. the N-poset)

Series-parallel posets are exactly the prime-free ones. Prime blocks are kept ATOMIC
(compared all-or-nothing), matching docs/DESIGN-comparison-object.md.

`.canonical()` gives a label that identifies a block across models (same activity structure
=> same string). Prime canonicalisation: distinct labels use the sorted label-edge list (fast path);
REPEATED labels within a prime use a label-preserving-iso canonical form (`_prime_canonical_iso`,
brute-force over element orderings) so iso primes match and non-iso primes with the same label
multiset are kept apart.

The recursion, precisely
------------------------
`decompose` runs the classical Gallai recursion. At each call, on the CURRENT fragment only:
(1) if the comparability graph is disconnected, the node is PARALLEL and each component gets a
fresh call from the top; (2) else if the incomparability graph is disconnected, the node is
SERIES, the co-components (which are totally ordered as blocks) are sorted low->high, and each
gets a fresh call from the top; (3) else the fragment is one atomic PRIME tile -- no recursion
inside it. Two facts shape the tree. First, the tests are EXCLUSIVE, not ordered by preference:
a graph and its complement cannot both be disconnected, so at most one split fires per node.
Second, the "go back to (1)" is per CHILD, never a loop on the same node: a parallel child is
comparability-connected by construction, so its call falls straight through to the series test,
and a series child is co-connected, so its call can only split parallel (or be prime/leaf) --
the tree therefore alternates series/parallel levels automatically, which is also why it is
canonical (a parallel child of a parallel node would have merged into its parent).

Simple vs linear-time
---------------------
This implementation is the naive recursion: each level recomputes connectivity on the induced
fragment, so the worst case is roughly O(n * m) over the whole tree. The linear-time O(n + m)
algorithms in the literature (the bound cited by the paper's canonical-tiling proposition) do NOT
run this recursion top-down; they work in two phases: (i) compute a FACTORIZING PERMUTATION -- an
ordering of the elements in which every strong module is consecutive -- by partition refinement,
splitting classes against pivot vertices (below / above / incomparable) under a charging scheme
that lets each edge pay for O(1) splits amortised; (ii) recover the tree from that permutation in
one stack-based bracketing scan, then label each node series/parallel/prime from one child pair.
References: McConnell & Spinrad 1999 (with transitive orientation; the paper's citation) and
Cournier & Habib 1994 for the first linear-time constructions; Tedder, Corneil, Habib & Paul 2008
for a substantially simpler linear-time algorithm; for posets one may also run the undirected
algorithm on the comparability graph (poset modules = comparability-graph modules) and orient the
series nodes afterwards; the series-parallel special case has a direct linear-time algorithm
(Valdes, Tarjan & Lawler 1982). Caveats: "linear" means O(n + m) in the CLOSED comparability
relation (this package's `Poset` stores the transitive closure, so linear-in-input is the honest
reading; from a Hasse diagram the closure must be built first), and the repeated-label prime
canonical form is a separate cost (factorial in the prime's size; primes are small). The naive
recursion is deliberate here: block-scale posets are tiny, the code is transparently checkable,
and the asymptotic claim in the paper rests on the literature algorithms, not on this file.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .poset import Poset


def _prime_canonical_iso(P: Poset) -> str:
    """Canonical string of a prime with (possibly) REPEATED labels, invariant under label-preserving
    isomorphism. Brute-force over element orderings: for each ordering we read off (labels-by-position,
    order-pairs-on-positions) and keep the lexicographic minimum. Two primes are the SAME block iff
    their minima agree -- a complete invariant (a shared minimum encoding IS a label-preserving iso).
    Primes are small (typically 4-6 elements), so n! is cheap."""
    elems = list(P.elements)
    n = len(elems)
    best = None
    for perm in itertools.permutations(range(n)):
        pos = {elems[i]: perm[i] for i in range(n)}
        labelseq = tuple(lab for _, lab in sorted((pos[e], P.labels[e]) for e in elems))
        edges = tuple(sorted((pos[u], pos[v]) for (u, v) in P.less))
        enc = (labelseq, edges)
        if best is None or enc < best:
            best = enc
    labelseq, edges = best
    body = ",".join(labelseq) + ";" + "".join(f"({u}<{v})" for u, v in edges)
    return "N[" + body + "]"


# --- decomposition tree nodes -------------------------------------------------

@dataclass
class Leaf:
    label: str

    def canonical(self) -> str:
        return self.label

    @property
    def atomic(self) -> bool:
        return True

    @property
    def children(self):
        return []


@dataclass
class Series:
    parts: list  # ordered

    def canonical(self) -> str:
        return "(" + " ; ".join(c.canonical() for c in self.parts) + ")"

    @property
    def atomic(self) -> bool:
        return False

    @property
    def children(self):
        return self.parts


@dataclass
class Parallel:
    parts: list  # unordered (commutative)

    def canonical(self) -> str:
        return "(" + " * ".join(sorted(c.canonical() for c in self.parts)) + ")"

    @property
    def atomic(self) -> bool:  # concurrency is an atomic block
        return True

    @property
    def children(self):
        return self.parts


@dataclass
class Prime:
    poset: Poset  # the indecomposable sub-poset (kept atomic)

    def canonical(self) -> str:
        labels = list(self.poset.labels.values())
        if len(set(labels)) == len(labels):
            # distinct labels: the sorted label-edge list is already a complete invariant (fast path)
            edges = sorted(
                (self.poset.labels[u], self.poset.labels[v]) for (u, v) in self.poset.less
            )
            return "N{" + ", ".join(f"{a}<{b}" for a, b in edges) + "}"
        return _prime_canonical_iso(self.poset)  # v2: repeated labels -> label-preserving iso

    @property
    def atomic(self) -> bool:
        return True

    @property
    def children(self):
        return []


# --- helpers ------------------------------------------------------------------

def _components(nodes: list[int], edge) -> list[list[int]]:
    """Connected components of the graph on `nodes` with predicate edge(u, v)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if edge(u, v):
                parent[find(u)] = find(v)
    groups: dict[int, list[int]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


# --- decomposition ------------------------------------------------------------

def decompose(P: Poset):
    V = P.elements
    if len(V) == 1:
        return Leaf(P.labels[V[0]])

    comps = _components(V, lambda u, v: P.comparable(u, v))
    if len(comps) > 1:
        return Parallel([decompose(P.restrict(c)) for c in comps])

    cocomps = _components(V, lambda u, v: not P.comparable(u, v))
    if len(cocomps) > 1:
        # SERIES: the co-components are totally ordered blocks; order them low -> high.
        def rank(block):
            # number of elements below this block (blocks are totally ordered, so this is a key)
            u = block[0]
            return sum(1 for other in cocomps for w in other if (w, u) in P.less)
        ordered = sorted(cocomps, key=rank)
        return Series([decompose(P.restrict(c)) for c in ordered])

    return Prime(P)  # atomic


def tiling(P: Poset) -> str:
    """Canonical string of the whole tiling tree (a normal-form identifier)."""
    return decompose(P).canonical()
