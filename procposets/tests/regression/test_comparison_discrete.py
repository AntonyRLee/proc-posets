"""Discrete-comparison identities: the two metrics twin as pi*sqrt(combinatorial) on their
deterministic loci, both see concurrency, and the block metric has reach Kemeny lacks.

The pairwise family splits in two.  ``footprint_angle`` reads the POSET (alpha-relations, 'par'
= genuine concurrency); ``trace_precedence_angle`` is the paper's r_Pi = Prec o r_T and reads the
TRACE distribution.  They agree on total orders and part company on a concurrency."""
import math

import pytest

from procposets.discrete import (block_angle, disc_angle, footprint, footprint_angle, kemeny,
                                 order_angle, precedence, trace_precedence,
                                 trace_precedence_angle)
from procposets.poset import Poset, from_dag, leaf, par, then


def one(P):
    return [(P, 1.0)]


ABC = one(then(leaf("a"), leaf("b"), leaf("c")))
ACB = one(then(leaf("a"), leaf("c"), leaf("b")))
CBA = one(then(leaf("c"), leaf("b"), leaf("a")))
CONC = one(then(leaf("a"), par(leaf("b"), leaf("c"))))
ORDER = [(then(leaf("a"), leaf("b"), leaf("c")), 0.5), (then(leaf("a"), leaf("c"), leaf("b")), 0.5)]


def test_order_angle_is_pi_root_half_kemeny_on_total_orders():
    # both pairwise charts coincide on total orders, where neither 'par' nor 'ilv' can occur
    for m, n in [(ABC, ACB), (ABC, CBA), (ACB, CBA)]:
        K = kemeny(m[0][0], n[0][0])
        target = math.pi * math.sqrt(K / 2)
        assert math.isclose(footprint_angle(m, n)[0], target, rel_tol=1e-9)
        assert math.isclose(trace_precedence_angle(m, n)[0], target, rel_tol=1e-9)


def test_block_angle_is_pi_root_hamming_on_deterministic():
    # block distance between two deterministic block models is pi*sqrt(H), H an integer
    d = block_angle(ABC, ACB, "uniform_variant", context_depth=3)[0]
    H = (d / math.pi) ** 2
    assert math.isclose(H, round(H), abs_tol=1e-9) and H > 0


def test_both_families_see_concurrency():
    # concurrency vs interleaving is nonzero for BOTH discrete metrics (a set phenomenon)
    assert footprint_angle(CONC, ORDER)[0] > 0
    assert block_angle(CONC, ORDER, "uniform_variant", 3)[0] > 0


def test_trace_precedence_ties_concurrency_to_the_coin_flip():
    # r_Pi = Prec o r_T factors through the trace distribution, which the two share: the
    # paper's impossibility, and the one place the two pairwise charts must disagree.
    assert trace_precedence_angle(CONC, ORDER)[0] == 0.0
    assert footprint_angle(CONC, ORDER)[0] > 0
    assert trace_precedence(CONC)[("b", "c")] == {"lt": 0.5, "gt": 0.5, "ilv": 0.0, "excl": 0.0}
    assert footprint(CONC)[("b", "c")] == {"lt": 0.0, "gt": 0.0, "par": 1.0, "excl": 0.0}


def test_exclusion_is_its_own_outcome_not_a_zero():
    # a pair that never co-occurs must not read as an invariable order
    excl = [(then(leaf("a"), leaf("b")), 0.5), (then(leaf("a"), leaf("c")), 0.5)]
    order = [(then(leaf("a"), leaf("b"), leaf("c")), 1.0)]
    for chart, angle in ((trace_precedence, trace_precedence_angle), (footprint, footprint_angle)):
        assert chart(excl)[("b", "c")]["excl"] == 1.0
        assert angle(excl, order)[0] > 0


def test_repeated_label_is_interleaving_in_the_trace_chart():
    rep = [(Poset(elements=[0, 1, 2], labels={0: "a", 1: "b", 2: "a"},
                  less={(0, 1), (1, 2), (0, 2)}), 1.0)]           # a ; b ; a
    assert trace_precedence(rep)[("a", "b")] == {"lt": 0.0, "gt": 0.0, "ilv": 1.0, "excl": 0.0}
    with pytest.raises(ValueError):
        footprint(rep)      # a repeated label has no single pairwise relation on the poset


def test_released_names_still_mean_the_footprint():
    assert precedence is footprint and order_angle is footprint_angle


def test_metrics_cross_on_primes_vs_concurrency():
    # block FINER on concurrency (conc-order > conc-seq) but the footprint ties them
    seq = ABC
    assert block_angle(CONC, ORDER, "uniform_variant", 3)[0] > block_angle(CONC, seq, "uniform_variant", 3)[0]
    assert math.isclose(footprint_angle(CONC, ORDER)[0], footprint_angle(CONC, seq)[0])
    # block COARSER inside a prime: two primes sharing relations are closer under the footprint
    n1 = one(from_dag([("a", "c"), ("b", "c"), ("b", "d")]))
    n2 = one(from_dag([("a", "c"), ("b", "c"), ("a", "d")]))
    assert block_angle(n1, n2, "uniform_variant", 3)[0] < footprint_angle(n1, n2)[0]


def test_block_has_reach_across_alphabets():
    m1 = one(then(leaf("a"), leaf("b"), leaf("c")))
    m2 = one(then(leaf("a"), leaf("b"), leaf("d")))
    assert block_angle(m1, m2, "uniform_variant", 3)[0] > 0   # defined
    with pytest.raises(ValueError):
        footprint_angle(m1, m2)                                # undefined: no common alphabet


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
