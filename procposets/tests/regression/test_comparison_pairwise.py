"""smd_pairwise must agree with smd() on every pair, whatever the fleet-wide state space."""
import math

from procposets.distance import smd, smd_pairwise, smd_rows
from procposets.matrix import build
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


# ---------------------------------------------------------------------------
# Fleet-wide constant |X| normalisation (LOCKED, geometry paper §VI.A): d/sqrt(|X|)
# with X the union over ALL fleet models -- one global scale, so d_SMD is a genuine
# metric. A per-pair |X| gives each pair its own denominator and breaks the triangle
# inequality; that is exactly what these tests forbid.
# ---------------------------------------------------------------------------

def _fleet_states(models, context_depth=1):
    return sorted(set().union(*(s for _, s in (build(m, context_depth) for m in models))))


def test_pairwise_normalize_single_global_scale():
    """normalize=True is the RAW matrix divided by ONE fleet-wide sqrt(|X|), every entry."""
    raw = smd_pairwise(MODELS)
    norm = smd_pairwise(MODELS, normalize=True)
    scale = math.sqrt(len(_fleet_states(MODELS)))          # |X| = 9 -> sqrt = 3.0
    assert len(_fleet_states(MODELS)) == 9
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            assert math.isclose(norm[i][j] * scale, raw[i][j], abs_tol=1e-12)


def test_pairwise_normalize_pinned():
    """Regression pin of the fleet-normalised matrix on the MODELS fixture."""
    expected = [
        [0.0,          1.2825498302, 1.8137993642, 0.3864264936],
        [1.2825498302, 0.0,          1.9591272264, 1.2825498302],
        [1.8137993642, 1.9591272264, 0.0,          1.8137993642],
        [0.3864264936, 1.2825498302, 1.8137993642, 0.0],
    ]
    norm = smd_pairwise(MODELS, normalize=True)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            assert math.isclose(norm[i][j], expected[i][j], abs_tol=1e-9)


def test_pairwise_normalize_beats_per_pair():
    """The fix must CHANGE the number: per-pair smd(normalize=True) uses the pair's own |X|,
    so it differs from the fleet-wide value on every pair whose union is smaller than the fleet."""
    norm = smd_pairwise(MODELS, normalize=True)
    differs = False
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            per_pair = smd(MODELS[i], MODELS[j], normalize=True)[0]
            if not math.isclose(per_pair, norm[i][j], abs_tol=1e-9):
                differs = True
    assert differs, "fleet-wide normalisation collapsed to per-pair -- the metric fix is a no-op"


def test_pairwise_normalize_is_metric():
    """A single fleet-wide |X| is a constant rescale of the raw row-Fisher-Rao metric, so the
    normalised matrix obeys the triangle inequality across every triple."""
    norm = smd_pairwise(MODELS, normalize=True)
    n = len(MODELS)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert norm[i][j] <= norm[i][k] + norm[k][j] + 1e-12


def test_smd_rows_accepts_fleet_states():
    """smd_rows on the fleet-wide state space reproduces the pairwise entry, and widening the
    space leaves the RAW distance unchanged (extra states route to the sink -> angle 0)."""
    states = _fleet_states(MODELS)
    builts = [build(m, 1) for m in MODELS]
    norm = smd_pairwise(MODELS, normalize=True)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            r = smd_rows(builts[i], builts[j], normalize=True, states=states)[0]
            assert math.isclose(r, norm[i][j], abs_tol=1e-12)
    # raw is space-invariant: pair-union vs fleet-wide give the same unnormalised distance
    raw_pair = smd_rows(builts[0], builts[2])[0]
    raw_fleet = smd_rows(builts[0], builts[2], states=states)[0]
    assert math.isclose(raw_pair, raw_fleet, abs_tol=1e-12)
