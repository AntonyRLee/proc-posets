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
from .poset import Model

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


def _bc(row1: dict[str, float], row2: dict[str, float], keys) -> float:
    """Bhattacharyya coefficient of two rows over `keys` (in `keys` order)."""
    return sum(math.sqrt(row1.get(k, 0.0) * row2.get(k, 0.0)) for k in keys)


def _clamp01(x: float) -> float:
    """Clamp a Bhattacharyya coefficient into [0, 1] (guards ``acos`` against fp drift)."""
    return min(1.0, max(0.0, x))


def _row_angle(row1: dict[str, float], row2: dict[str, float], keys) -> float:
    """Per-row Bhattacharyya angle ``arccos(BC)`` -- the per-state SMD term. BC is
    summed in `keys` order, so passing the same key sequence reproduces the exact
    float accumulation of the open-coded form it replaces."""
    return math.acos(_clamp01(_bc(row1, row2, keys)))


def _bhattacharyya_angle(p: dict[str, float], q: dict[str, float]) -> float:
    """The single-vector Bhattacharyya angle ``2*arccos(BC)`` over ``set(p)|set(q)``
    -- the Result-1 form shared by :func:`bhattacharyya_angle` and
    :func:`traces.trace_bhattacharyya`."""
    return 2.0 * math.acos(_clamp01(_bc(p, q, set(p) | set(q))))


def smd_rows(built1, built2, mode=None, normalize=False, states=None) -> tuple[float, dict[str, float]]:
    """SMD between two already-BUILT (matrix, states) pairs -- for reusing builds, or for
    comparing against a hand-built chain that no finite variant set produces (e.g. the cyclic
    loop limit of `spm.loops.loop_limit`).

    `normalize` divides by sqrt(|X|) -- the distance paper's Result-4 factor (Eq. D4):
    d/sqrt(|X|) is the ROOT-MEAN-SQUARE Bhattacharyya angle over the rows, bounded in [0, pi]
    and comparable across DIFFERENT state-space sizes (the raw SMD is extensive: it grows with
    the number of differing rows). Use this whenever the two objects may have different |X|
    -- e.g. fleet sites with different block counts, or an empirical object as N grows.

    `states` fixes the space X the distance is evaluated on. Pass the FLEET-WIDE union (states
    over ALL models under comparison, e.g. what `smd_pairwise` builds) so that with `normalize`
    the 1/sqrt(|X|) factor is one global constant and the distance is a genuine metric. This is
    LOCKED: |X| is fleet-wide, NOT per-pair -- a per-pair |X| gives each pair its own denominator
    and breaks the triangle inequality (see the geometry paper §VI.A). `states` must be a
    superset of both builds' own states; widening beyond a pair's own union leaves the RAW
    distance unchanged (the extra states route to the sink on both models -> angle 0) and only
    rescales the normalised one. Defaults to the pair's own union -- correct for a lone pair,
    where the fleet IS the pair, so plain `smd(..., normalize=True)` is already metric-correct."""
    m1, s1 = built1
    m2, s2 = built2
    if states is None:
        states = sorted(set(s1) | set(s2))
    a1, a2 = _augment(m1, states, mode), _augment(m2, states, mode)
    total = 0.0
    per_block: dict[str, float] = {}
    for s in states:
        ang = _row_angle(a1[s], a2[s], states)
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
    `smd_rows`); for a lone pair the pair IS the fleet, so this is metric-correct, but to
    normalise ACROSS a fleet use `smd_pairwise(models, normalize=True)` (one fleet-wide |X|)."""
    return smd_rows(build(model1, context_depth), build(model2, context_depth), mode, normalize)


def _pairwise_rows(rowmaps, states, normalize=False) -> list[list[float]]:
    """Pairwise matrix-angle distances over pre-augmented row maps on a common state space.
    `normalize` applies the single fleet-wide 1/sqrt(|states|) scale to every entry."""
    n = len(rowmaps)
    D = [[0.0] * n for _ in range(n)]
    scale = 1.0 / math.sqrt(len(states)) if normalize and states else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            total = 0.0
            for s in states:
                r1, r2 = rowmaps[i][s], rowmaps[j][s]
                if r1 == r2:
                    continue
                if len(r2) < len(r1):
                    r1, r2 = r2, r1
                bc = _clamp01(sum(math.sqrt(v * r2.get(t, 0.0)) for t, v in r1.items()))
                ang = math.acos(bc)
                total += ang * ang
            D[i][j] = D[j][i] = 2.0 * math.sqrt(total) * scale
    return D


def smd_pairwise(models: list[Model], mode=None, context_depth: int = 1,
                 normalize=False) -> list[list[float]]:
    """Pairwise SMD over a fleet of models, building each block matrix ONCE on the fleet-wide
    common state space X (the union over ALL models).

    Raw (normalize=False) agrees with smd() on every pair: a state unused by both models of a
    pair augments to identical rows (angle 0), and extra target states carry zero Bhattacharyya
    mass, so widening the common space beyond the pair's own union changes nothing -- it only
    saves the n-1 rebuilds per model.

    normalize=True divides EVERY entry by sqrt(|X|) on that one fleet-wide space, so the
    1/sqrt(|X|) factor is a single global constant and the result is a genuine metric (LOCKED:
    fleet-wide constant |X|, per the geometry paper §VI.A). This is the correct normalised fleet
    distance -- do NOT normalise pair-by-pair via `smd(a, b, normalize=True)` across a fleet with
    differing state-space sizes: that gives each pair its own denominator and breaks the triangle
    inequality."""
    built = [build(m, context_depth) for m in models]
    states = sorted(set().union(*(s for _, s in built)))
    aug = [_augment(m, states, mode) for m, _ in built]
    return _pairwise_rows(aug, states, normalize)


def bhattacharyya_angle(model1: Model, model2: Model) -> float:
    """Result 1: Bhattacharyya angle between the flat normal-form distributions."""
    return _bhattacharyya_angle(
        normal_form_distribution(model1), normal_form_distribution(model2)
    )
