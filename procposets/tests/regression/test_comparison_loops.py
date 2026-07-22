"""The loop machinery: truncated unrollings, the cyclic limit chain, and their convergence."""
import random

from procposets.distance import smd, smd_rows
from procposets.loops import (empirical_loop_model, loop_limit, loop_model, sample_repeats,
                       truncated_geometric, unrolling)
from procposets.matrix import END, build
from procposets.moddecomp import decompose
from procposets.poset import leaf, par, then
from procposets.traces import trace_distribution

PRE, POST = (leaf("a"),), (leaf("d"),)
BODY = par(leaf("b"), leaf("c"))
BNAME = decompose(BODY).canonical()


def test_truncated_geometric_conventions():
    for tail in ("renorm", "lump"):
        w = truncated_geometric(0.5, 6, tail)
        assert abs(sum(w) - 1.0) < 1e-12
    lump = truncated_geometric(0.5, 6, "lump")
    assert all(abs(lump[n - 1] - 0.5 ** n) < 1e-12 for n in range(1, 6))   # exact below K
    assert abs(lump[-1] - 0.5 ** 5) < 1e-12                                 # tail mass on K


def test_truncated_matrix_body_row_is_the_expectation_ratio():
    # P_K(body -> body) = E[n-1]/E[n] under the truncated weights (Eq. analytic-block).
    q, K = 0.6, 8
    w = truncated_geometric(q, K)
    expected = sum(wn * (n - 1) for n, wn in enumerate(w, 1)) / sum(wn * n for n, wn in enumerate(w, 1))
    m, _ = build(loop_model(BODY, q, K, PRE, POST))
    assert abs(m[BNAME][BNAME] - expected) < 1e-12
    assert abs(m[BNAME]["d"] - (1.0 - expected)) < 1e-12


def test_loop_limit_is_the_cyclic_chain():
    q = 0.35
    m, states = loop_limit(BODY, q, PRE, POST)
    assert m[BNAME] == {BNAME: q, "d": 1.0 - q}
    assert m["a"] == {BNAME: 1.0} and m["d"] == {END: 1.0}
    assert states == {"START", "END", "a", BNAME, "d"}
    # multi-block body: the CLOSING block loops back to the OPENING one
    m2, _ = loop_limit(then(leaf("b"), leaf("c")), q, PRE, POST)
    assert m2["c"] == {"b": q, "d": 1.0 - q} and m2["b"] == {"c": 1.0}


def test_truncation_converges_geometrically_to_the_limit():
    q = 0.5
    lim = loop_limit(BODY, q, PRE, POST)
    d = [smd_rows(build(loop_model(BODY, q, K, PRE, POST)), lim)[0] for K in (2, 4, 8, 12, 16)]
    assert all(a > b for a, b in zip(d, d[1:]))          # monotone in the truncation depth
    assert d[-1] < 1e-3                                  # essentially converged by K = 16
    assert d[-1] / d[-2] < (q + 0.1) ** 4                # ~ geometric rate q per unit K
    # both tail conventions converge to the SAME limit
    d_lump = smd_rows(build(loop_model(BODY, q, 16, PRE, POST, tail="lump")), lim)[0]
    assert d_lump < 1e-3


def test_smd_rows_agrees_with_smd():
    m1 = loop_model(BODY, 0.3, 5, PRE, POST)
    m2 = loop_model(BODY, 0.7, 5, PRE, POST)
    total, per_block = smd(m1, m2)
    total_r, per_block_r = smd_rows(build(m1), build(m2))
    assert abs(total - total_r) < 1e-12 and per_block == per_block_r


def test_trace_identical_loop_twin():
    # loop over b(x)c vs loop over a fair coin-flip of the two orders: identical trace law
    # at every truncation depth, separated by the block distance at every truncation depth.
    q, K = 0.5, 4
    conc = loop_model(BODY, q, K, PRE, POST)
    w = truncated_geometric(q, K)
    twin = []
    for n in range(1, K + 1):
        for bits in range(2 ** n):
            parts = [leaf("a")]
            for i in range(n):
                parts += [leaf("b"), leaf("c")] if (bits >> i) & 1 else [leaf("c"), leaf("b")]
            twin.append((then(*parts, leaf("d")), w[n - 1] / 2 ** n))
    p, t = trace_distribution(conc), trace_distribution(twin)
    assert set(p) == set(t) and all(abs(p[k] - t[k]) < 1e-12 for k in p)
    assert smd(conc, twin)[0] > 0.5


def test_empirical_loop_model_is_regime1_counting():
    rng = random.Random(0)
    reps = sample_repeats(0.5, 2000, rng)
    assert min(reps) >= 1
    e = empirical_loop_model(BODY, reps, PRE, POST)
    assert abs(sum(w for _, w in e) - 1.0) < 1e-12
    # the empirical matrix approaches the limit chain (consistency through the truncation)
    d = smd_rows(build(e), loop_limit(BODY, 0.5, PRE, POST))[0]
    assert d < 0.1
    # degenerate log: every case ran the body once -- no loop-back evidence at all
    e1 = empirical_loop_model(BODY, [1, 1, 1], PRE, POST)
    m, _ = build(e1)
    assert m[BNAME] == {"d": 1.0}


def test_unrolling_validation():
    try:
        unrolling(BODY, 0)
        raise AssertionError("n = 0 should be rejected (the body runs at least once)")
    except ValueError:
        pass
    try:
        loop_limit(BODY, 0.5, pre=(par(leaf("b"), leaf("c")),))   # pre collides with body
        raise AssertionError("colliding block names should be rejected at depth 1")
    except ValueError:
        pass
