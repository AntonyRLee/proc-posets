"""Trace-level view of a model -- the baseline the structural SMD is compared against.

trace_distribution enumerates linear extensions (uniform, per Assumption 3) and maps them to
label words; trace_bhattacharyya is the Result-1 angle between two models' trace distributions.
This is exactly what a trace/behaviour-based comparison sees -- and it is blind to the difference
between genuine concurrency and a coin-flip between orders.
"""
from __future__ import annotations

from collections import defaultdict

from ._extensions import preds as _preds
from .distance import _bhattacharyya_angle
from .poset import IdealBudgetExceeded, Model, Poset, count_extensions

# Materialisation budget for the trace view: refuse to build a list of more than
# this many linear-extension words.  ``count_extensions`` (the cheap guarded
# ideal-lattice DP) already bounds the IDEAL count (~2^width) via
# ``_extensions.MAX_IDEAL_STATES``, but a wide antichain of width < 20 passes that
# guard yet has e(P) = n! words -- materialising them exhausts memory.  So guard the
# MATERIALISED list size directly, using the cheap exact e(P) count (which the golden
# pins equal to ``len(linear_extensions(P))``).  Read at call time so a caller may
# raise it deliberately for a known small-word corpus.
MAX_LINEAR_EXTENSIONS = 1_000_000


def linear_extensions(P: Poset) -> list[tuple[str, ...]]:
    """All linear extensions of P as label words (distinct-label posets: one word each).

    Refuses with :class:`~procposets.IdealBudgetExceeded` rather than exhausting
    memory when e(P) exceeds :data:`MAX_LINEAR_EXTENSIONS`.  The sibling counter and
    sampler (:func:`procposets.count_extensions`, ``sample_extension``) are already
    budget-guarded; this materialising view now is too -- a wide antichain passes the
    ideal-lattice budget (~2^width) but has n! words, so the guard is on the list size
    via the cheap e(P) count."""
    n_ext = count_extensions(P)  # cheap ideal-DP; raises IdealBudgetExceeded if 2^width too big
    if n_ext > MAX_LINEAR_EXTENSIONS:
        raise IdealBudgetExceeded(
            f"e(P) = {n_ext:,} linear extensions exceeds the materialisation budget "
            f"MAX_LINEAR_EXTENSIONS = {MAX_LINEAR_EXTENSIONS:,}; the trace view cannot "
            f"enumerate a factorial-sized word list (a wide antichain passes the "
            f"ideal-lattice budget but has n! extensions)."
        )
    preds = _preds(P.elements, P.less)  # {e: frozenset(predecessors)}
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
    """The model's distribution over label words: each variant spreads its weight
    uniformly over its linear extensions (Assumption 3), summed and normalised."""
    dist: dict[tuple[str, ...], float] = defaultdict(float)
    tot = 0.0
    for P, w in model:
        les = linear_extensions(P)
        for le in les:
            dist[le] += w / len(les)
        tot += w
    return {k: v / tot for k, v in dist.items()}


def trace_bhattacharyya(model1: Model, model2: Model) -> float:
    """Result-1 Bhattacharyya angle between two models' trace distributions -- the
    trace/behaviour-based comparison (blind to genuine concurrency vs a coin-flip
    between orders)."""
    return _bhattacharyya_angle(trace_distribution(model1), trace_distribution(model2))
