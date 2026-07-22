"""Cospan equivalence checks over generator signatures ``Sigma``.

Primary (label-aware, exact): two signatures are equivalent iff their generator
frozensets coincide.  Because discovered ED models share the activity
vocabulary, this is the practical comparison and is fully deterministic --
exactly ``thm:canonical-presentation`` read as ``Sigma_1 = Sigma_2``.

Secondary (label-blind, structural): isomorphism of the typed
port/generator incidence graph via networkx VF2, for the general case where
activity names need not match (e.g. comparing across renamings).

Tertiary (behavioural cross-check): a bounded sample of the partial-order
trace language ``Gamma(F(Sigma))`` obtained by composing generators
start-to-end; corroborates the structural verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from networkx.algorithms.isomorphism import categorical_node_match

from .signature import Generator, Signature


# --- label-aware exact ------------------------------------------------------
def equal(s1: Signature, s2: Signature) -> bool:
    return set(s1.generators) == set(s2.generators)


def diff(s1: Signature, s2: Signature) -> tuple[set[Generator], set[Generator]]:
    """Generators in ``s1`` only, and in ``s2`` only."""
    a, b = set(s1.generators), set(s2.generators)
    return a - b, b - a


def jaccard(s1: Signature, s2: Signature) -> float:
    a, b = set(s1.generators), set(s2.generators)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def drop_boundary_activities(sig: Signature, is_boundary) -> Signature:
    """Drop generators whose label matches ``is_boundary`` and strip dangling
    port references to them from every other generator.

    ``is_boundary`` is a predicate ``str -> bool`` (a plain ``set`` also works,
    via ``__contains__``). For comparing notations with different boundary
    vocabularies (e.g. BPMN's explicit start/end events against PN/PT/CN,
    which have none): dropping a pure-boundary label's own generator and its
    references on neighbours reproduces the signature as if that activity
    were never there, enabling a like-for-like comparison.
    """
    pred = is_boundary if callable(is_boundary) else is_boundary.__contains__
    gens = set()
    for g in sig.generators:
        if pred(g.label):
            continue
        left = frozenset(p for p in g.left if not pred(p.src))
        right = frozenset(p for p in g.right if not pred(p.tgt))
        gens.add(Generator(g.label, left, right))
    return Signature(frozenset(gens))


# --- label-blind structural isomorphism -------------------------------------
def _incidence_graph(sig: Signature) -> nx.Graph:
    """Bipartite typed incidence graph: generator-nodes (typed by arity +
    label-blind degree) and port-nodes (typed by object type).  Activity labels
    are deliberately discarded so VF2 tests *structure* up to relabelling."""
    g = nx.Graph()
    for i, gen in enumerate(sorted(sig.generators, key=str)):
        gid = ("gen", i)
        g.add_node(gid, kind="gen", sig=(len(gen.left), len(gen.right)))
        for p in gen.left:
            pid = ("port", p.src, p.typ, p.tgt)
            g.add_node(pid, kind="port", typ=p.typ)
            g.add_edge(gid, pid, side="L")
        for p in gen.right:
            pid = ("port", p.src, p.typ, p.tgt)
            g.add_node(pid, kind="port", typ=p.typ)
            g.add_edge(gid, pid, side="R")
    return g


def isomorphic(s1: Signature, s2: Signature) -> bool:
    """Label-blind structural isomorphism of the two signatures."""
    if len(s1) != len(s2):
        return False
    g1, g2 = _incidence_graph(s1), _incidence_graph(s2)
    nm = categorical_node_match(["kind", "sig", "typ"], [None, None, None])
    return nx.is_isomorphic(g1, g2, node_match=nm)


# --- behavioural cross-check (bounded trace-language sample) -----------------
@dataclass
class TraceCheck:
    same_language: bool
    only_in_1: frozenset
    only_in_2: frozenset


def activity_pomsets(sig: Signature, max_traces: int = 5000) -> frozenset:
    """Bounded sample of the partial-order language: enumerate composable
    generator chains start-to-end and record the resulting activity-precedence
    relation (a Hasse-style frozenset of label pairs).  Approximate by design.

    A chain composes when each successor generator's left ports are produced as
    right ports earlier in the chain; we track the frontier of open ports.
    """
    gens = list(sig.generators)
    starts = [g for g in gens if not g.left]
    results: set = set()
    stack: list[tuple[frozenset, frozenset]] = []
    for s in starts:
        stack.append((frozenset(s.right), frozenset()))
    while stack and len(results) < max_traces:
        frontier, edges = stack.pop()
        if not frontier:  # all ports consumed -> a complete scenario
            results.add(edges)
            continue
        for g in gens:
            if g.left and g.left <= frontier:
                new_front = (frontier - g.left) | g.right
                new_edges = edges | frozenset(
                    (p.src, g.label) for p in g.left
                ) | frozenset((g.label, p.tgt) for p in g.right)
                stack.append((new_front, new_edges))
    return frozenset(results)


def trace_language_check(s1: Signature, s2: Signature) -> TraceCheck:
    p1, p2 = activity_pomsets(s1), activity_pomsets(s2)
    return TraceCheck(p1 == p2, p1 - p2, p2 - p1)
