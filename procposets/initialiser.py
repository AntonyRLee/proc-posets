"""Higher-order moment initialiser for the NPMLE.

**This changes only the warm start, never what is identifiable or fitted.**
The NPMLE consumes the per-group trace densities directly (`GroupedLog`); its
fitted mixing measure is the unique optimum of a convex program over the
declared candidate class, independent of where column generation is seeded.
The initialiser projects each candidate's clean trace law onto a low-order
precedence tensor

    phi^(k)(sigma)_{(x_1..x_k)} = 1[x_1 prec x_2 prec ... prec x_k in sigma]

(dimension ~ m^k) and seeds column generation from the candidate whose moment
best matches the data -- a cheaper-than-likelihood proxy that matters in the
large-m regime where evaluating every candidate's likelihood is
the cost.  Climbing ``k`` (up to ``k = m``, the full order = the trace law
itself) is a *compute/quality dial*, never an identifiability parameter:
`fit(..., init_order=k)` reaches the *same optimum* for every ``k``.

**A refinement of §8, measured here.**  §8 motivated the ladder by claiming
distinct general posets can share every pairwise (`phi^(2)`) margin, so a
`phi^(2)` seed lands on a margin-equivalence *class*.  That is *false for the
clean uniform-extension moment*: a comparable pair ``a < b`` has
``P(a prec b) = 1`` exactly (no extension inverts a strict relation), so the
`phi^(2)` moment already pins the relation -- ``phi^(2)`` is **injective on
single posets** (proven at any ``m`` for the subsequence feature; verified
exhaustively through ``m = 6`` for the adjacency feature too).  So genuine
margin-equivalent *poset pairs* do not exist under this moment.  Where higher
order genuinely buys resolution is the **mixture** moment tensor (Kruskal,
§8.2) when the component count exceeds the low-order feature space, and the
**finite-sample / noisy** margin where the exact-arithmetic argument no
longer bites.  `find_margin_equivalences` is kept as the honest detector
(it fires on near-ties, and reports exact collisions -- of which there are
none for clean single posets -- so escalation is driven by *ranking*
ambiguity, `moment_seed`).

Everything here is numpy-light and exact for small ``m`` (it enumerates the
linear extensions); for larger ``m`` it draws uniform extensions with the
library's existing ideal-lattice sampler.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .rel import Rel, count_linear_extensions, respects, sample_linear_extension

Tuple_k = Tuple[str, ...]

# Exact enumeration of L(P) is m! in the worst case; above this, moments are
# estimated by Monte-Carlo over uniform linear extensions instead.
MAX_EXACT_MOMENT_M = 8


# ---------------------------------------------------------------------------
# Feature coordinates and the trace feature map
# ---------------------------------------------------------------------------

def ktuple_index(alphabet: Sequence[str], order: int) -> List[Tuple_k]:
    """The phi^(order) coordinates: ordered tuples of `order` *distinct*
    labels (the possible strict-precedence chains of length `order`)."""
    if order < 2:
        raise ValueError("moment order must be >= 2")
    return [t for t in permutations(sorted(alphabet), order)]


def _positions(trace: Sequence[str]) -> Dict[str, int]:
    return {a: i for i, a in enumerate(trace)}


def trace_phi(trace: Sequence[str], keys: Sequence[Tuple_k]) -> np.ndarray:
    """phi^(order)(trace): 1 where the key labels occur in that strict order
    as a subsequence of `trace`.  (For complete duplicate-free traces --
    the likelihood's domain -- a length-k key is a subsequence iff its
    labels appear in that relative order, i.e. their positions increase.)"""
    pos = _positions(trace)
    out = np.empty(len(keys))
    for i, key in enumerate(keys):
        p = [pos.get(x) for x in key]
        out[i] = 1.0 if (None not in p and all(
            p[j] < p[j + 1] for j in range(len(p) - 1))) else 0.0
    return out


def empirical_moment(traces: Sequence[Sequence[str]],
                     keys: Sequence[Tuple_k]) -> np.ndarray:
    """Mean phi^(order) over a bag of traces (the data's moment vector)."""
    if not traces:
        return np.zeros(len(keys))
    return np.mean([trace_phi(t, keys) for t in traces], axis=0)


# ---------------------------------------------------------------------------
# A poset's moment signature: E over its uniform linear-extension law
# ---------------------------------------------------------------------------

def poset_moment(elements: FrozenSet[str], rel: Rel,
                 keys: Sequence[Tuple_k], *, mc_samples: int = 4000,
                 seed: int = 0) -> np.ndarray:
    """E_{sigma ~ Uniform(L(P))} [ phi^(order)(sigma) ] as a vector over
    `keys` -- the *exact* moment the estimator would see from a pure-P,
    noise-free, arbitrarily-large sample.  Exact for small m (enumerate the
    respecting permutations), Monte-Carlo otherwise."""
    els = sorted(elements)
    m = len(els)
    if rel == frozenset():
        # antichain: every permutation is an extension -- uniform over S_m
        acc = np.zeros(len(keys))
        if m <= MAX_EXACT_MOMENT_M:
            perms = list(permutations(els))
            for p in perms:
                acc += trace_phi(p, keys)
            return acc / len(perms)
    if m <= MAX_EXACT_MOMENT_M:
        exts = [p for p in permutations(els) if respects(p, rel)]
        acc = np.zeros(len(keys))
        for p in exts:
            acc += trace_phi(p, keys)
        return acc / len(exts)
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(keys))
    for _ in range(mc_samples):
        acc += trace_phi(sample_linear_extension(elements, rel, rng), keys)
    return acc / mc_samples


# ---------------------------------------------------------------------------
# Margin-equivalence: distinct posets that a low-order tensor conflates
# ---------------------------------------------------------------------------

def margin_equivalent(elements: FrozenSet[str], rel_p: Rel, rel_q: Rel,
                      order: int = 2, tol: float = 1e-9) -> bool:
    """True iff P and Q are *distinct posets* (a poset equals the
    intersection of its extensions -- Dushnik-Miller -- so distinct relations
    means distinct extension sets) with *identical* phi^(order) moments.
    Such a pair is invisible to an order-`order` moment initialiser."""
    if rel_p == rel_q:
        return False
    keys = ktuple_index(sorted(elements), order)
    mp = poset_moment(elements, rel_p, keys)
    mq = poset_moment(elements, rel_q, keys)
    return bool(np.max(np.abs(mp - mq)) <= tol)


def find_margin_equivalences(elements: FrozenSet[str],
                             rels: Sequence[Rel], order: int = 2,
                             tol: float = 1e-9) -> List[Tuple[int, int]]:
    """Index pairs (i, j) among `rels` that are order-`order`
    margin-equivalent -- distinct relations with the same phi^(order)
    moment.  The detector that decides whether to escalate the tensor
    order."""
    keys = ktuple_index(sorted(elements), order)
    moments = [poset_moment(elements, r, keys) for r in rels]
    hits = []
    for i in range(len(rels)):
        for j in range(i + 1, len(rels)):
            if rels[i] == rels[j]:
                continue
            if np.max(np.abs(moments[i] - moments[j])) <= tol:
                hits.append((i, j))
    return hits


# ---------------------------------------------------------------------------
# The initialiser: a best-first seed ordering over candidate posets
# ---------------------------------------------------------------------------

class SeedRanking:
    """Result of the moment initialiser: candidate posets ranked by how well
    their moment signature aligns with the data's, the order actually used,
    the winning-margin (top-2 alignment gap), and whether auto-escalation
    fired."""

    def __init__(self, order: int, ranking: List[int], dist: np.ndarray,
                 escalated_from: Optional[int]):
        self.order = order                    # phi order actually used
        self.ranking = ranking                # candidate indices, best first
        self.dist = dist                      # alignment distance per candidate
        self.escalated_from = escalated_from  # start order if we escalated

    @property
    def top(self) -> int:
        return self.ranking[0]

    @property
    def margin(self) -> float:
        """Alignment gap between the best and second-best candidate; a small
        margin is the ranking ambiguity that auto-escalation responds to."""
        if len(self.ranking) < 2:
            return float("inf")
        return float(self.dist[self.ranking[1]] - self.dist[self.ranking[0]])


def moment_seed(elements: FrozenSet[str], candidate_rels: Sequence[Rel],
                traces: Sequence[Sequence[str]], *, order="auto",
                max_order: Optional[int] = None,
                tie_tol: float = 1e-3) -> SeedRanking:
    """Rank candidate posets best-first by moment alignment to the data.

    ``order``: an int fixes the tensor order; ``"auto"`` starts at phi^(2)
    and climbs (up to ``max_order``, default m) while the top-two candidates
    are within ``tie_tol`` in alignment -- a seed ambiguity a larger tensor
    can break.  The ranking is a *warm start* only: alignment is a cheap
    likelihood proxy and the NPMLE converges to the same optimum from any
    seed, so escalation trades compute for a sharper seed, never for a
    different answer.
    """
    els = sorted(elements)
    m = len(els)
    cap = max_order if max_order is not None else m
    start = 2 if order == "auto" else int(order)
    escalated_from = None

    def rank_at(k: int) -> Tuple[List[int], np.ndarray]:
        keys = ktuple_index(els, k)
        emp = empirical_moment(traces, keys)
        moments = [poset_moment(elements, r, keys) for r in candidate_rels]
        d = np.array([np.linalg.norm(mu - emp) for mu in moments])
        return list(np.argsort(d, kind="stable")), d

    k = start
    ranking, dist = rank_at(k)
    if order == "auto":
        while k < cap and len(candidate_rels) > 1:
            gap = dist[ranking[1]] - dist[ranking[0]]
            if gap > tie_tol:
                break
            escalated_from = start
            k += 1
            ranking, dist = rank_at(k)
    return SeedRanking(k, ranking, dist, escalated_from)
