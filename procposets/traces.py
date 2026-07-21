"""Trace-level view of a model -- the baseline the structural SMD is compared against.

trace_distribution enumerates linear extensions (uniform, per Assumption 3) and maps them to
label words; trace_bhattacharyya is the Result-1 angle between two models' trace distributions.
This is exactly what a trace/behaviour-based comparison sees -- and it is blind to the difference
between genuine concurrency and a coin-flip between orders.
"""
from __future__ import annotations

from collections import defaultdict

from .distance import _bhattacharyya_angle
from .poset import Model, Poset


def linear_extensions(P: Poset) -> list[tuple[str, ...]]:
    """All linear extensions of P as label words (distinct-label posets: one word each)."""
    preds: dict[int, set[int]] = {e: set() for e in P.elements}
    for (u, v) in P.less:
        preds[v].add(u)
    out: list[tuple[str, ...]] = []

    def rec(remaining: set[int], placed: set[int], acc: list[int]):
        if not remaining:
            out.append(tuple(P.labels[e] for e in acc))
            return
        for e in list(remaining):
            if preds[e] <= placed:
                rec(remaining - {e}, placed | {e}, acc + [e])

    rec(set(P.elements), set(), [])
    return out


def trace_distribution(model: Model) -> dict[tuple[str, ...], float]:
    dist: dict[tuple[str, ...], float] = defaultdict(float)
    tot = 0.0
    for P, w in model:
        les = linear_extensions(P)
        for le in les:
            dist[le] += w / len(les)
        tot += w
    return {k: v / tot for k, v in dist.items()}


def trace_bhattacharyya(model1: Model, model2: Model) -> float:
    return _bhattacharyya_angle(trace_distribution(model1), trace_distribution(model2))
