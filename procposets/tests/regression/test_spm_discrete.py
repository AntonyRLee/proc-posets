"""Discrete-comparison identities: the two metrics twin as pi*sqrt(combinatorial) on their
deterministic loci, both see concurrency, and the block metric has reach Kemeny lacks."""
import math

import pytest

from procposets.discrete import block_angle, disc_angle, kemeny, order_angle
from procposets.poset import from_dag, leaf, par, then


def one(P):
    return [(P, 1.0)]


ABC = one(then(leaf("a"), leaf("b"), leaf("c")))
ACB = one(then(leaf("a"), leaf("c"), leaf("b")))
CBA = one(then(leaf("c"), leaf("b"), leaf("a")))
CONC = one(then(leaf("a"), par(leaf("b"), leaf("c"))))
ORDER = [(then(leaf("a"), leaf("b"), leaf("c")), 0.5), (then(leaf("a"), leaf("c"), leaf("b")), 0.5)]


def test_order_angle_is_pi_root_half_kemeny_on_total_orders():
    for m, n in [(ABC, ACB), (ABC, CBA), (ACB, CBA)]:
        K = kemeny(m[0][0], n[0][0])
        assert math.isclose(order_angle(m, n)[0], math.pi * math.sqrt(K / 2), rel_tol=1e-9)


def test_block_angle_is_pi_root_hamming_on_deterministic():
    # block distance between two deterministic block models is pi*sqrt(H), H an integer
    d = block_angle(ABC, ACB, "uniform_variant", context_depth=3)[0]
    H = (d / math.pi) ** 2
    assert math.isclose(H, round(H), abs_tol=1e-9) and H > 0


def test_both_families_see_concurrency():
    # concurrency vs interleaving is nonzero for BOTH discrete metrics (a set phenomenon)
    assert order_angle(CONC, ORDER)[0] > 0
    assert block_angle(CONC, ORDER, "uniform_variant", 3)[0] > 0


def test_metrics_cross_on_primes_vs_concurrency():
    # block FINER on concurrency (conc-order > conc-seq) but order_angle ties them
    seq = ABC
    assert block_angle(CONC, ORDER, "uniform_variant", 3)[0] > block_angle(CONC, seq, "uniform_variant", 3)[0]
    assert math.isclose(order_angle(CONC, ORDER)[0], order_angle(CONC, seq)[0])
    # block COARSER inside a prime: two primes sharing relations are closer under order_angle
    n1 = one(from_dag([("a", "c"), ("b", "c"), ("b", "d")]))
    n2 = one(from_dag([("a", "c"), ("b", "c"), ("a", "d")]))
    assert block_angle(n1, n2, "uniform_variant", 3)[0] < order_angle(n1, n2)[0]


def test_block_has_reach_across_alphabets():
    m1 = one(then(leaf("a"), leaf("b"), leaf("c")))
    m2 = one(then(leaf("a"), leaf("b"), leaf("d")))
    assert block_angle(m1, m2, "uniform_variant", 3)[0] > 0   # defined
    with pytest.raises(ValueError):
        order_angle(m1, m2)                                    # undefined: no common alphabet


# --- prime-gradation hybrid (fan-out) ---
N_BASE = one(from_dag([("a", "c"), ("b", "c"), ("b", "d")]))
N_SHARE2 = one(from_dag([("a", "c"), ("b", "c"), ("a", "d")]))       # shares a<c, b<c
N_DISJOINT = one(from_dag([("d", "b"), ("d", "a"), ("c", "a")]))     # shares no Hasse edge


def test_hybrid_grades_primes_atomic_does_not():
    # atomic pins any two distinct primes at pi; the hybrid grades sharers below pi
    assert math.isclose(disc_angle(N_BASE, N_SHARE2, refine=False)[0], math.pi)
    assert disc_angle(N_BASE, N_SHARE2, refine=True)[0] < math.pi - 1e-6


def test_hybrid_monotone_in_overlap():
    # more shared Hasse edges -> smaller hybrid distance; disjoint primes stay at pi (= atomic)
    assert disc_angle(N_BASE, N_SHARE2, refine=True)[0] < disc_angle(N_BASE, N_DISJOINT, refine=True)[0]
    assert math.isclose(disc_angle(N_BASE, N_DISJOINT, refine=True)[0], math.pi)
    assert disc_angle(N_BASE, N_BASE, refine=True)[0] < 1e-12          # identical primes -> 0


def test_prime_only_refinement_leaves_sp_unchanged():
    # refine={"prime"} touches only primes; on prime-free (SP) models it equals atomic exactly.
    # (refine=True is the FULL refined family of the paper's Remark V.1 and additionally fans
    # parallel blocks out over typed element atoms -- pinned in test_refinement.py.)
    conc = one(then(leaf("a"), par(leaf("b"), leaf("c"))))
    seq = one(then(leaf("a"), leaf("b"), leaf("c")))
    assert math.isclose(disc_angle(conc, seq, refine={"prime"})[0],
                        disc_angle(conc, seq, refine=False)[0])
