"""smd_pairwise must agree with smd() on every pair, whatever the fleet-wide state space."""
import math

from procposets.distance import smd, smd_pairwise
from procposets.poset import leaf, par, then

MODELS = [
    [(then(leaf("a"), par(leaf("b"), leaf("c"))), 1.0)],
    [(then(leaf("a"), leaf("b"), leaf("c")), 0.5), (then(leaf("a"), leaf("c"), leaf("b")), 0.5)],
    [(then(leaf("x"), leaf("y")), 1.0)],                                     # disjoint alphabet
    [(then(leaf("a"), par(leaf("b"), leaf("c"))), 0.7), (then(leaf("a"), leaf("d")), 0.3)],
]


def test_pairwise_matches_smd_sink():
    D = smd_pairwise(MODELS)
    for i in range(len(MODELS)):
        assert D[i][i] == 0.0
        for j in range(len(MODELS)):
            assert math.isclose(D[i][j], smd(MODELS[i], MODELS[j])[0], abs_tol=1e-12)


def test_pairwise_matches_smd_selfloop():
    D = smd_pairwise(MODELS, mode="selfloop")
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            assert math.isclose(D[i][j], smd(MODELS[i], MODELS[j], mode="selfloop")[0], abs_tol=1e-12)


def test_pairwise_matches_smd_at_depth():
    # repeated blocks, where deeper memory changes the matrices
    deep = [[(then(leaf("a"), leaf("b"), leaf("a"), leaf("c"), leaf("a")), 1.0)],
            [(then(leaf("a"), leaf("c"), leaf("a"), leaf("b"), leaf("a")), 1.0)]]
    for k in (1, 2):
        D = smd_pairwise(deep, context_depth=k)
        assert math.isclose(D[0][1], smd(deep[0], deep[1], context_depth=k)[0], abs_tol=1e-12)
