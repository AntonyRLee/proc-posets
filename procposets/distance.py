"""Distances.

smd  -- Result 3 of the distance paper: the stochastic-matrix distance on the block-transition
        matrices. Compares Markov *transitions*. Concurrency-preserving and graded.
bhattacharyya_angle -- Result 1: the Bhattacharyya angle on the flat normal-form distribution.
        Compares Markov *states* (a vector). Brittle: any distinct normal form is maximally far.

The contrast between the two on the same pair of models is the point of the sandbox.
"""
from __future__ import annotations

import math

from .matrix import END, START, build, normal_form_distribution
from .poset import Poset

Model = list[tuple[Poset, float]]

# Matrix normalisation on the common state space (docs/DESIGN-comparison-object.md, TODO §8):
#   "selfloop" (option 1) -- unused state s -> s; END -> END (absorbing). Distance-paper convention.
#   "sink"     (option 2) -- unused state s -> END (=gamma2); END -> START (=gamma1) reset, so the
#                            matrix is a genuine gamma1->gamma2 generative chain with a steady state.
NORMALISATION = "sink"


def _augment(matrix, states, mode=None):
    mode = mode or NORMALISATION
    out = {}
    for s in states:
        row = matrix.get(s)
        if row:                                  # state used, with real transitions
            out[s] = row
        elif mode == "sink":
            out[s] = {START: 1.0} if s == END else {END: 1.0}   # END resets to gamma1; unused -> gamma2
        else:                                    # selfloop
            out[s] = {s: 1.0}
    return out


def _bc(row1: dict[str, float], row2: dict[str, float], states: list[str]) -> float:
    return sum(math.sqrt(row1.get(s, 0.0) * row2.get(s, 0.0)) for s in states)


def smd_rows(built1, built2, mode=None, normalize=False) -> tuple[float, dict[str, float]]:
    """SMD between two already-BUILT (matrix, states) pairs, on their union state space --
    for reusing builds, or for comparing against a hand-built chain that no finite variant
    set produces (e.g. the cyclic loop limit of `spm.loops.loop_limit`).

    `normalize` divides by sqrt(|states|) -- the distance paper's Result-4 factor (Eq. D4):
    d/sqrt(|X|) is the ROOT-MEAN-SQUARE Bhattacharyya angle over the rows, bounded in [0, pi]
    and comparable across DIFFERENT state-space sizes (the raw SMD is extensive: it grows with
    the number of differing rows). Use this whenever the two objects may have different |X|
    -- e.g. fleet sites with different block counts, or an empirical object as N grows."""
    m1, s1 = built1
    m2, s2 = built2
    states = sorted(set(s1) | set(s2))
    a1, a2 = _augment(m1, states, mode), _augment(m2, states, mode)
    total = 0.0
    per_block: dict[str, float] = {}
    for s in states:
        bc = min(1.0, max(0.0, _bc(a1[s], a2[s], states)))
        ang = math.acos(bc)
        per_block[s] = ang
        total += ang * ang
    scale = 1.0 / math.sqrt(len(states)) if normalize else 1.0
    return 2.0 * math.sqrt(total) * scale, per_block


def smd(model1: Model, model2: Model, mode=None, context_depth: int = 1,
        normalize=False) -> tuple[float, dict[str, float]]:
    """Stochastic-matrix distance (Result 3) + the per-state (per-block) angle breakdown
    (the diagnostic: which block drives the divergence). `mode` selects the normalisation
    ("sink" default, or "selfloop"); see NORMALISATION above. `context_depth` is the VLMC dial
    passed to `build` (1 = memoryless blocks; higher = sharper, less row-sharing). `normalize`
    applies the Result-4 sqrt(|X|) factor for cross-state-space-size comparability (see
    `smd_rows`)."""
    return smd_rows(build(model1, context_depth), build(model2, context_depth), mode, normalize)


def _pairwise_rows(rowmaps, states) -> list[list[float]]:
    """Pairwise matrix-angle distances over pre-augmented row maps on a common state space."""
    n = len(rowmaps)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            total = 0.0
            for s in states:
                r1, r2 = rowmaps[i][s], rowmaps[j][s]
                if r1 == r2:
                    continue
                if len(r2) < len(r1):
                    r1, r2 = r2, r1
                bc = min(1.0, max(0.0, sum(math.sqrt(v * r2.get(t, 0.0)) for t, v in r1.items())))
                ang = math.acos(bc)
                total += ang * ang
            D[i][j] = D[j][i] = 2.0 * math.sqrt(total)
    return D


def smd_pairwise(models: list[Model], mode=None, context_depth: int = 1) -> list[list[float]]:
    """Pairwise SMD over a fleet of models, building each block matrix ONCE on the fleet-wide
    common state space. Agrees with smd() on every pair: a state unused by both models of a pair
    augments to identical rows (angle 0) in either normalisation, and extra target states carry
    zero Bhattacharyya mass, so widening the common state space beyond the pair's own union
    changes nothing -- it only saves the n-1 rebuilds per model."""
    built = [build(m, context_depth) for m in models]
    states = sorted(set().union(*(s for _, s in built)))
    aug = [_augment(m, states, mode) for m, _ in built]
    return _pairwise_rows(aug, states)


def bhattacharyya_angle(model1: Model, model2: Model) -> float:
    """Result 1: Bhattacharyya angle between the flat normal-form distributions."""
    p = normal_form_distribution(model1)
    q = normal_form_distribution(model2)
    support = set(p) | set(q)
    bc = min(1.0, max(0.0, sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in support)))
    return 2.0 * math.acos(bc)
