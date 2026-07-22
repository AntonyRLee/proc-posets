"""Adapter: pm4py HeuristicsNet -> logic-mediated multipartite LM-graph.

A ``HeuristicsNet``'s base structure (``Node.output_connections`` /
``input_connections``) is a plain weighted DFG -- no AND/XOR/OR is visible
there. The causal-net structure is encoded separately, per node, as a
*pairwise statistic*: ``Node.and_measures_out`` / ``and_measures_in``, a
sparse dict giving an AND-score between two of a node's successors (resp.
predecessors), already thresholded at discovery time (pm4py default
``and_measure_thresh = 0.65``). A successor pair absent from the dict is
implicitly exclusive (XOR).

This is exactly the structure pm4py's own ``convert_to_petri_net`` decodes
to insert invisible AND-fork/AND-join transitions (verified by inspecting
``pm4py.objects.conversion.heuristics_net.variants.to_petri_net.find_bindings``,
which takes maximal cliques of the AND-relation graph as the AND-bundles,
leaving the rest as competing XOR/OR alternatives). We reconstruct the same
OR = XOR-of-AND-bundles mediator pattern directly into the LM-graph instead of
routing through pm4py's Petri-net conversion:

  activity                       -> absorbing activity
  per-node successor/predecessor
    AND-bundle (maximal clique
    of the AND-measure graph)    -> AND mediator
  per-node XOR-of-bundles        -> XOR mediator (one per node, per direction)

Known limitation: if a node's successors form *overlapping* maximal cliques
(b AND c, b AND d, but c not AND d), the single underlying DFG edge x->b is
routed through only one of the two AND mediators (last one wins). This does
not arise for the structured running example or the synthetic ED logs.
"""

from __future__ import annotations

import networkx as nx

from ._lmgraph_build import _assemble, _assemble_single, _type_prefix
from .lmgraph import Kind, LMGraph


def _cliques(and_measures: dict) -> list[list[str]]:
    """Maximal AND-bundles (size > 1) from a sparse pairwise AND-measure dict.

    Mirrors pm4py's own ``find_bindings``: build the undirected AND-relation
    graph (the dict is stored asymmetrically, one direction per pair) and take
    its maximal cliques.
    """
    graph = nx.Graph()
    for n1, rels in and_measures.items():
        graph.add_node(n1)
        for n2 in rels:
            graph.add_node(n2)
            graph.add_edge(n1, n2)
    return [c for c in nx.find_cliques(graph) if len(c) > 1]


def add_heuristics(g: LMGraph, hn, otype: str | None) -> None:
    """Overlay a pm4py ``HeuristicsNet`` onto ``g``, typing every edge with
    ``otype``. Mediators are namespaced by ``otype``; activities are shared
    by label, as in :mod:`from_petri`.
    """
    pre = _type_prefix(otype)
    nodes = hn.nodes

    def out_id(name: str) -> str:
        return f"{pre}OUT_{name}"

    def in_id(name: str) -> str:
        return f"{pre}IN_{name}"

    for name in nodes:
        g.add_activity(name)

    out_clique_of: dict[str, dict[str, str]] = {}
    for name, node in nodes.items():
        if not node.output_connections:
            continue
        g.add_mediator(out_id(name), Kind.XOR)
        for i, clique in enumerate(_cliques(node.and_measures_out)):
            mid = g.add_mediator(f"{pre}ANDOUT_{name}_{i}", Kind.AND)
            g.add_edge(out_id(name), mid, otype)
            for member in clique:
                out_clique_of.setdefault(name, {})[member] = mid

    in_clique_of: dict[str, dict[str, str]] = {}
    for name, node in nodes.items():
        if not node.input_connections:
            continue
        g.add_mediator(in_id(name), Kind.XOR)
        for i, clique in enumerate(_cliques(node.and_measures_in)):
            mid = g.add_mediator(f"{pre}ANDIN_{name}_{i}", Kind.AND)
            g.add_edge(mid, in_id(name), otype)
            for member in clique:
                in_clique_of.setdefault(name, {})[member] = mid

    # each activity sits between its own IN/OUT mediators
    for name, node in nodes.items():
        if node.input_connections:
            g.add_edge(in_id(name), name, otype)
        if node.output_connections:
            g.add_edge(name, out_id(name), otype)

    # wire every DFG edge x->y through the correct out/in mediator endpoints
    for name, node in nodes.items():
        for succ in node.output_connections:
            sname = succ.node_name
            src = out_clique_of.get(name, {}).get(sname, out_id(name))
            dst = in_clique_of.get(sname, {}).get(name, in_id(sname))
            g.add_edge(src, dst, otype)


def lmgraph_from_heuristics_nets(nets_by_type: dict) -> LMGraph:
    """Build one typed object-centric LM-graph from ``{otype: HeuristicsNet}``."""
    return _assemble(add_heuristics, nets_by_type)


def lmgraph_from_heuristics(hn, otype: str | None = None) -> LMGraph:
    """Single (optionally untyped) HeuristicsNet -> LM-graph."""
    return _assemble_single(add_heuristics, hn, otype)
