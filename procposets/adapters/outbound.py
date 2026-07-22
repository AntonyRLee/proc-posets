"""Adapters from the sandbox's objects to pm4py's -- the bridge to the standard PM stack.

The sandbox model (list of (Poset, weight)) maps onto pm4py three ways, one per technique
family:
    to_process_tree / to_petri_net   an XOR over the variants' SP trees (series -> SEQUENCE,
                                     parallel -> PARALLEL, leaf -> activity), then pm4py's
                                     tree->net conversion. Weights are dropped: classical
                                     conformance (alignments, token replay) is support-only.
    to_language                      the model's analytic trace law as a pm4py stochastic
                                     language {trace tuple: prob} -- the input to the Earth
                                     Mover's Distance (EMSC-style, normalised-Levenshtein
                                     ground distance).
    to_dataframe                     an event log (list of trace tuples) as the pm4py event
                                     table (case:concept:name / concept:name / time:timestamp).

Only series-parallel posets convert to process trees (a prime block has no tree operator);
every benchmark scenario is SP, matching the scope of the classical baselines.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import pm4py
from pm4py.algo.evaluation.earth_mover_distance import algorithm as _emd

from ..moddecomp import Leaf, Parallel, Series, decompose
from ..poset import Model
from ..traces import trace_distribution


def _tree_of(node):
    from pm4py.objects.process_tree.obj import Operator, ProcessTree
    if isinstance(node, Leaf):
        return ProcessTree(label=node.label)
    if isinstance(node, Series):
        op = Operator.SEQUENCE
        parts = node.parts
    elif isinstance(node, Parallel):
        op = Operator.PARALLEL
        parts = node.parts
    else:
        raise ValueError(f"prime block has no process-tree operator: {node.canonical()}")
    t = ProcessTree(operator=op)
    t.children = [_tree_of(p) for p in parts]
    for c in t.children:
        c.parent = t
    return t


def to_process_tree(model: Model):
    """XOR over the variants' SP trees (weight-free: the classical-conformance view)."""
    from pm4py.objects.process_tree.obj import Operator, ProcessTree
    trees = [_tree_of(decompose(P)) for P, _ in model]
    if len(trees) == 1:
        return trees[0]
    root = ProcessTree(operator=Operator.XOR)
    root.children = trees
    for c in trees:
        c.parent = root
    return root


def to_petri_net(model: Model):
    """(net, initial marking, final marking) for alignments / token replay."""
    return pm4py.convert_to_petri_net(to_process_tree(model))


def to_language(model: Model) -> dict[tuple, float]:
    """The model's analytic trace law as a pm4py stochastic language."""
    return dict(trace_distribution(model))


def log_language(traces: list[tuple]) -> dict[tuple, float]:
    """The empirical stochastic language of an event log."""
    out: dict[tuple, float] = {}
    for t in traces:
        out[t] = out.get(t, 0.0) + 1.0
    n = len(traces)
    return {t: c / n for t, c in out.items()}


def emd(lang1: dict[tuple, float], lang2: dict[tuple, float]) -> float:
    """Earth Mover's Distance between stochastic languages (pm4py; normalised-Levenshtein
    ground distance) -- the EMSC-style stochastic conformance measure."""
    return _emd.apply(lang1, lang2)


def pairwise_emd(langs: list[dict[tuple, float]]) -> list[list[float]]:
    n = len(langs)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = emd(langs[i], langs[j])
    return D


def to_dataframe(traces: list[tuple]) -> "pd.DataFrame":
    """An event log as the pm4py event table (synthetic timestamps in trace order)."""
    rows = []
    t0 = pd.Timestamp("2026-01-01")
    for i, tr in enumerate(traces):
        for j, act in enumerate(tr):
            rows.append({"case:concept:name": f"c{i}", "concept:name": act,
                         "time:timestamp": t0 + pd.Timedelta(seconds=i * 1000 + j)})
    return pd.DataFrame(rows)


def alignment_fitness(traces: list[tuple], model: Model) -> float:
    """pm4py alignment-based log fitness of the log against the (weight-free) model net."""
    net, im, fm = to_petri_net(model)
    return pm4py.fitness_alignments(to_dataframe(traces), net, im, fm)["log_fitness"]


def token_fitness(traces: list[tuple], model: Model) -> float:
    """pm4py token-based-replay log fitness (the cheaper classical measure)."""
    net, im, fm = to_petri_net(model)
    return pm4py.fitness_token_based_replay(to_dataframe(traces), net, im, fm)["log_fitness"]
