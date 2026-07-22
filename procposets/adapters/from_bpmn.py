"""Adapter: pm4py BPMNDiagram -> logic-mediated multipartite LM-graph.

Direct, no detour through Petri nets. Per ``def:bpmn``: every
flow object is either an activity ($\\ell$-value ``activity``) or a gateway
($\\ell$-value ``AND``/``XOR``/``OR``); there is no third "transparent" class.
Start/end events and tasks are therefore all absorbing activities -- BPMN's
vocabulary genuinely differs from a Petri net's silent transitions here, and
any resulting signature mismatch against PN/PT/CN is a real finding, not an
adapter bug.

  Task (and other Activity subclasses) -> absorbing activity, labelled by name
  Event (Start/End/Intermediate/...)     -> absorbing activity, labelled
                                             synthetically (``__Event_<id>__``)
  ExclusiveGateway          -> XOR mediator
  ParallelGateway            -> AND mediator
  InclusiveGateway            -> OR mediator (XOR-of-AND-subsets, eq:bpmn-or-explosion)
  EventBasedGateway / other  -> XOR mediator (fallback; not emitted by pm4py's
                                 inductive-miner-derived BPMN)

Events are deliberately *not* labelled by ``get_name()``: pm4py's PT->BPMN
converter names every start/end event "start"/"end" regardless of the
underlying process, which would silently collide with a genuine activity
named "start" or "end" in the discovered vocabulary (this happened with the
ED pathway's own "end" activity) and corrupt cross-notation comparison. The
synthetic, id-qualified label keeps every event distinct and never collides
with a real activity label.

A direct activity->activity flow (structurally disallowed, ``def:lm-graph``)
gets a trivial pass-through SEQ mediator inserted, per the digest's stated
normalisation.
"""

from __future__ import annotations

from pm4py.objects.bpmn.obj import BPMN

from ..cospan._lmgraph_build import _assemble, _assemble_single, _type_prefix
from ..cospan.lmgraph import Kind, LMGraph

_GATEWAY_KIND = {
    BPMN.ExclusiveGateway: Kind.XOR,
    BPMN.ParallelGateway: Kind.AND,
    BPMN.InclusiveGateway: Kind.OR,
}

EVENT_LABEL_PREFIX = "__Event_"


def add_bpmn(g: LMGraph, bpmn, otype: str | None) -> None:
    """Overlay a pm4py ``BPMNDiagram`` onto ``g``, typing every edge with
    ``otype``. Gateways are namespaced by ``otype``; activities (tasks and
    events) are shared by label, as in :mod:`from_petri`.
    """
    pre = _type_prefix(otype)

    def gateway_kind(n) -> Kind | None:
        for cls, kind in _GATEWAY_KIND.items():
            if isinstance(n, cls):
                return kind
        if isinstance(n, BPMN.Gateway):
            return Kind.XOR  # fallback for EventBasedGateway / generic gateway
        return None

    node: dict = {}  # pm4py BPMN node -> LM-graph node id
    is_gateway: dict = {}
    for n in bpmn.get_nodes():
        kind = gateway_kind(n)
        if kind is not None:
            node[n] = g.add_mediator(f"{pre}G_{n.get_id()}", kind)
            is_gateway[n] = True
        elif isinstance(n, BPMN.Event):
            node[n] = g.add_activity(f"{EVENT_LABEL_PREFIX}{n.get_id()}__")
            is_gateway[n] = False
        else:
            node[n] = g.add_activity(n.get_name() or n.get_id())
            is_gateway[n] = False

    seq_count = 0
    for f in bpmn.get_flows():
        src, tgt = f.get_source(), f.get_target()
        if not is_gateway[src] and not is_gateway[tgt]:
            seq_count += 1
            mid = g.add_mediator(f"{pre}SEQ_{seq_count}", Kind.SEQ)
            g.add_edge(node[src], mid, otype)
            g.add_edge(mid, node[tgt], otype)
        else:
            g.add_edge(node[src], node[tgt], otype)


def lmgraph_from_bpmn_diagrams(diagrams_by_type: dict) -> LMGraph:
    """Build one typed object-centric LM-graph from ``{otype: BPMNDiagram}``."""
    return _assemble(add_bpmn, diagrams_by_type)


def lmgraph_from_bpmn(bpmn, otype: str | None = None) -> LMGraph:
    """Single (optionally untyped) BPMNDiagram -> LM-graph."""
    return _assemble_single(add_bpmn, bpmn, otype)
