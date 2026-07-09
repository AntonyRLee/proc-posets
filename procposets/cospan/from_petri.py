"""Adapter: pm4py Petri net(s) -> logic-mediated multipartite LM-graph.

A Petri net is already a flat notation (``rem:hierarchical-compilation``), so the
mapping is direct and mirrors the running-example construction
(``RUNNING_EXAMPLE.md`` section 3):

  place               -> XOR mediator  (choice over consumers / producers)
  silent transition   -> SEQ mediator, marked silent (eliminated by default;
                         see LMGraph.without_silent / engine.extract_signature)
  labelled transition -> absorbing activity (node id = label)

Because process trees, BPMN and heuristics nets all convert to Petri nets in
pm4py, every discovered model class is funnelled through this one adapter.

Object-centricity is recovered by *typed merge*: each object type ``theta`` is
flattened and discovered independently, its net is stamped with type ``theta``
and its places/silents are namespaced, while labelled activities are shared by
label across types.  An activity touching several types (e.g. ``assess``) thus
acquires typed in/out ports from each type's flow, and the engine's AND over an
activity's ports synchronises them -- exactly the object-centric generators of
the paper.
"""

from __future__ import annotations

from .lmgraph import Kind, LMGraph


def add_petri(g: LMGraph, net, otype: str | None) -> None:
    """Overlay a pm4py ``PetriNet`` onto ``g``, typing every edge with ``otype``.

    Places and silent transitions are namespaced by ``otype`` (kept distinct
    across types); labelled transitions are shared by label.
    """
    pre = f"{otype}__" if otype is not None else ""

    def place_id(p) -> str:
        return f"{pre}P_{p.name}"

    def silent_id(t) -> str:
        return f"{pre}S_{t.name}"

    # register nodes
    for p in net.places:
        g.add_mediator(place_id(p), Kind.XOR)
    node: dict = {}  # transition object -> LM-graph node id
    for t in net.transitions:
        if t.label is None:
            node[t] = g.add_mediator(silent_id(t), Kind.SEQ, silent=True)
        else:
            node[t] = g.add_activity(t.label)

    # register arcs (place<->transition only; structural rule holds by design)
    for arc in net.arcs:
        src, tgt = arc.source, arc.target
        if src in net.places:  # place -> transition
            g.add_edge(place_id(src), node[tgt], otype)
        else:  # transition -> place
            g.add_edge(node[src], place_id(tgt), otype)


def lmgraph_from_petri_nets(nets_by_type: dict) -> LMGraph:
    """Build one typed object-centric LM-graph from ``{otype: PetriNet}``."""
    g = LMGraph()
    for otype, net in nets_by_type.items():
        add_petri(g, net, otype)
    g.validate()
    return g


def lmgraph_from_petri(net, otype: str | None = None) -> LMGraph:
    """Single (optionally untyped) Petri net -> LM-graph."""
    g = LMGraph()
    add_petri(g, net, otype)
    g.validate()
    return g
