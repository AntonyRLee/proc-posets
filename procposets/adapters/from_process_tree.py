"""Adapter: pm4py ProcessTree -> logic-mediated multipartite LM-graph.

Direct, no detour through Petri nets. A process tree is hierarchical, so this
adapter recursively compiles each node into a *flow fragment* with one entry
point and one exit point (an activity, for a leaf; a mediator, for every
composite node), then stitches fragments together exactly as their operator
prescribes:

  leaf (visible)        -> absorbing activity                     (entry = exit = activity)
  leaf (silent / tau)    -> transparent SEQ mediator                (rem:hierarchical-compilation)
  SEQUENCE(c1..cn)       -> chain ci's fragments via SEQ mediators
  XOR(c1..cn)            -> XOR split mediator / XOR join mediator
  PARALLEL(c1..cn)       -> AND split mediator / AND join mediator
  OR(c1..cn)             -> OR split mediator / OR join mediator (XOR-of-AND-subsets)
  LOOP(do, redo1..redok) -> do; (XOR split: exit, or one redo_i; redo_i; back-edge to do)*

The LOOP back-edge makes the LM-graph cyclic; the engine's memoised
cycle-cut (``engine._traverse``, ``lem:gf-termination``) already terminates
on this without extra work -- this is the documented "loop handling" limit
in the README (cycle-cut, not full loop-language realisation).

Because two children that are both bare leaves cannot be wired directly
(``def:lm-graph``'s structural rule forbids an edge between two absorbing
nodes), SEQUENCE always inserts an explicit pass-through mediator between
consecutive children, even when both sides are activities.
"""

from __future__ import annotations

from itertools import count

from pm4py.objects.process_tree.obj import Operator

from ..cospan._lmgraph_build import _assemble, _assemble_single, _type_prefix
from ..cospan.lmgraph import Kind, LMGraph


def add_process_tree(g: LMGraph, tree, otype: str | None) -> tuple[str, str]:
    """Overlay a pm4py ``ProcessTree`` onto ``g``, typing every edge with
    ``otype``. Mediators are namespaced by ``otype``; activities are shared
    by label, as in :mod:`from_petri`. Returns the (entry, exit) node ids of
    the whole tree's flow fragment (rarely needed by callers; the boundary is
    otherwise inferred structurally, as in :mod:`from_petri`).
    """
    pre = _type_prefix(otype)
    counter = count()

    def fresh(kind: Kind, tag: str) -> str:
        return g.add_mediator(f"{pre}{tag}_{next(counter)}", kind)

    def edge(src: str, tgt: str) -> None:
        g.add_edge(src, tgt, otype)

    def seq_chain(frags: list[tuple[str, str]]) -> tuple[str, str]:
        if len(frags) == 1:
            return frags[0]
        entry = frags[0][0]
        prev_exit = frags[0][1]
        for nxt_entry, nxt_exit in frags[1:]:
            mid = fresh(Kind.SEQ, "SEQ")
            edge(prev_exit, mid)
            edge(mid, nxt_entry)
            prev_exit = nxt_exit
        return entry, prev_exit

    def split_join(frags: list[tuple[str, str]], kind: Kind) -> tuple[str, str]:
        split = fresh(kind, f"{kind.value}SPLIT")
        join = fresh(kind, f"{kind.value}JOIN")
        for c_entry, c_exit in frags:
            edge(split, c_entry)
            edge(c_exit, join)
        return split, join

    def loop(frags: list[tuple[str, str]]) -> tuple[str, str]:
        do_entry, do_exit = frags[0]
        if len(frags) == 1:
            return do_entry, do_exit
        split = fresh(Kind.XOR, "LOOPSPLIT")
        back = fresh(Kind.XOR, "LOOPBACK")
        edge(do_exit, split)
        for r_entry, r_exit in frags[1:]:
            edge(split, r_entry)
            edge(r_exit, back)
        edge(back, do_entry)
        return do_entry, split

    def compile_node(t) -> tuple[str, str]:
        if t.operator is None:
            if t.label is None:
                m = fresh(Kind.SEQ, "TAU")
                g.silent.add(m)
                return m, m
            act = g.add_activity(t.label)
            return act, act
        frags = [compile_node(c) for c in t.children]
        if t.operator is Operator.SEQUENCE:
            return seq_chain(frags)
        if t.operator is Operator.XOR:
            return split_join(frags, Kind.XOR)
        if t.operator is Operator.PARALLEL:
            return split_join(frags, Kind.AND)
        if t.operator is Operator.OR:
            return split_join(frags, Kind.OR)
        if t.operator is Operator.LOOP:
            return loop(frags)
        raise ValueError(f"unsupported process-tree operator: {t.operator}")

    return compile_node(tree)


def lmgraph_from_process_trees(trees_by_type: dict) -> LMGraph:
    """Build one typed object-centric LM-graph from ``{otype: ProcessTree}``."""
    return _assemble(add_process_tree, trees_by_type)


def lmgraph_from_process_tree(tree, otype: str | None = None) -> LMGraph:
    """Single (optionally untyped) ProcessTree -> LM-graph."""
    return _assemble_single(add_process_tree, tree, otype)
