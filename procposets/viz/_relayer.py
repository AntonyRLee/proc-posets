"""Faithful re-layering of a linearised pomset for crossing reduction.

The term-path lowering (:func:`._layout.lower_term`) draws a :class:`Diagram`
whose box *columns* come straight from the order the generators were composed in
(``step0 >> step1 >> …``).  For a class diagram that order is one arbitrary linear
extension of the class poset -- the miner's -- and it fixes which long edges span
which columns, hence which lane arcs interleave and cross.

But every linear extension of the same partial order draws the **same morphism**:
the generators and their ports are identical, only the x-columns differ.  So the
column assignment is free to be chosen for readability.  This module rebuilds the
producer→consumer partial order from port identity and offers alternative faithful
layerings -- **ASAP** (each box as early as its inputs allow) and **ALAP** (as late
as its outputs allow) -- as candidate :class:`Diagram` terms.  Lowering all of
{as-composed, ASAP, ALAP} and keeping the fewest-crossing drawing
(:func:`._layout.lower_best`) cuts the residual lane-arc crossings the local
port/side/level passes cannot touch (they are invariant under those, but not under
re-columning).

Same-layer generators are pairwise incomparable by construction (an edge always
strictly increases the layer), so grouping a layer into an ``⊗`` block is valid --
*provided the relation is acyclic*, which is a precondition, not a theorem: ports
are keyed by label, so a body whose generators mutually produce each other's ports
(a loop body drawn as an endomorphism on its anchor, or the same firing unrolled
twice) induces a cycle.  :func:`relayered_terms` therefore checks acyclicity and
declines to offer candidates when it fails -- see :func:`_has_cycle`.
Deterministic throughout (sorted iteration only)."""
from __future__ import annotations

from collections import defaultdict
from functools import reduce

from ..cospan.signature import Generator
from ._layout import Diagram, _box_sub


def _box(g: Generator) -> Diagram:
    return Diagram(lambda st: _box_sub(g, st))


def _production_dag(gens: list[Generator]):
    """Partial order over generator indices: an edge ``i -> j`` whenever some
    port produced by ``gens[i]`` (in its ``right``) is consumed by ``gens[j]``
    (in its ``left``).  Every producer of a shared port is linked to every
    consumer, so a linear extension of this DAG keeps *every* drawn wire pointing
    forward (a column always precedes the columns that consume its outputs)."""
    producers: dict = defaultdict(list)
    consumers: dict = defaultdict(list)
    for i, g in enumerate(gens):
        for p in g.right:
            producers[p].append(i)
        for p in g.left:
            consumers[p].append(i)
    preds: dict[int, set] = {i: set() for i in range(len(gens))}
    succs: dict[int, set] = {i: set() for i in range(len(gens))}
    for p, prods in producers.items():
        for i in prods:
            for j in consumers.get(p, ()):
                if i != j:
                    preds[j].add(i)
                    succs[i].add(j)
    return preds, succs


def _has_cycle(n: int, succs: dict[int, set]) -> bool:
    """``True`` when the production relation is cyclic (a Kahn peel leaves nodes).

    :func:`_toposort` cannot answer this: it marks a node ``seen`` *before*
    recursing, so a cycle yields a silently NON-topological order instead of an
    error.  Layers built from such an order no longer strictly increase along
    every edge, which breaks the antichain invariant :func:`_layered_term` relies
    on -- ``_par`` would stack comparable boxes into one column and the drawing
    would be re-wired (an endomorphism's anchor boundary flips; an unrolled body
    sheds an internal wire into two dangling stubs).  Both are silent: the figure
    renders, it is simply no longer the morphism it claims to be."""
    indeg = [0] * n
    for i in range(n):
        for j in succs[i]:
            indeg[j] += 1
    ready = [i for i in range(n) if not indeg[i]]
    peeled = 0
    while ready:
        for j in sorted(succs[ready.pop()]):
            indeg[j] -= 1
            if not indeg[j]:
                ready.append(j)
        peeled += 1
    return peeled != n


def _toposort(n: int, preds: dict[int, set]) -> list[int]:
    """Predecessors-before-node order (deterministic: ascending index tie-break)."""
    seen: set = set()
    out: list[int] = []

    def visit(i: int) -> None:
        if i in seen:
            return
        seen.add(i)
        for p in sorted(preds[i]):
            visit(p)
        out.append(i)

    for i in range(n):
        visit(i)
    return out


def _asap_layers(n: int, preds: dict[int, set]) -> list[int]:
    """Earliest feasible layer for each node: one past its latest predecessor."""
    layer = [0] * n
    for i in _toposort(n, preds):
        layer[i] = max((layer[p] + 1 for p in preds[i]), default=0)
    return layer


def _alap_layers(n: int, preds: dict[int, set], succs: dict[int, set]) -> list[int]:
    """Latest feasible layer: one before its earliest successor; sinks pinned to
    the deepest ASAP layer so the drawing keeps the same overall width."""
    sink_layer = max(_asap_layers(n, preds), default=0)
    layer = [sink_layer] * n
    for i in reversed(_toposort(n, preds)):  # sinks first
        if succs[i]:
            layer[i] = min(layer[s] - 1 for s in succs[i])
    return layer


def _layered_term(gens: list[Generator], layer: list[int]) -> Diagram:
    """``>>``-chain of ``⊗``-grouped columns for the given layer assignment."""
    cols: dict[int, list[Generator]] = defaultdict(list)
    for i, g in enumerate(gens):
        cols[layer[i]].append(g)
    steps = [
        reduce(lambda a, b: a @ b, (_box(g) for g in sorted(cols[c], key=str)))
        for c in sorted(cols)
    ]
    return reduce(lambda a, b: a >> b, steps)


def relayered_terms(gens: list[Generator]) -> list[Diagram]:
    """Alternative faithful column-layerings of the generators ``gens`` (a
    linearised pomset) as :class:`Diagram` terms -- the ASAP and ALAP extensions
    of the port-production partial order.  Connectivity is identical to any other
    layering of the same generators; only the box columns differ.  Returns ``[]``
    for a trivial (0/1-generator) pomset, where re-columning cannot change
    anything, and -- fail-safe -- for a **cyclic** production relation, where no
    faithful layering exists at all (:func:`_has_cycle`).  An empty result simply
    leaves the caller with its as-composed term, i.e. the drawing it would have
    had before re-layering existed."""
    n = len(gens)
    if n < 2:
        return []
    preds, succs = _production_dag(gens)
    if _has_cycle(n, succs):
        return []
    asap = _asap_layers(n, preds)
    alap = _alap_layers(n, preds, succs)
    terms = [_layered_term(gens, asap)]
    if alap != asap:
        terms.append(_layered_term(gens, alap))
    return terms
