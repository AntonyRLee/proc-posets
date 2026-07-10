"""Refined-family (fan-out) pins: the paper's Remark V.1 conventions, closed form, defaults.

Prime blocks fan out over labelled covering-relation atoms "x<y"; parallel blocks over typed
element atoms "sym||"; splits are uniform (maximum entropy); the SMD row formula, sink-and-reset
closure, and union state space are unchanged. Isolated same-kind block pairs obey the closed form
d = 2*arccos(|A & A'| / sqrt(|A| |A'|)). refine={"prime"} reproduces the historical prime-only
behaviour exactly; refine=False is atomic; refine=True is the full family.
"""
import math

import pytest

from procposets.discrete import disc_angle
from procposets.poset import from_dag, leaf, par, then


def one(P):
    return [(P, 1.0)]


def closed(h, n1, n2):
    return 2.0 * math.acos(h / math.sqrt(n1 * n2))


def anti(labels):
    return one(par(*(leaf(x) for x in labels)))


N_BASE = one(from_dag([("a", "c"), ("b", "c"), ("b", "d")]))
N_SHARE2 = one(from_dag([("a", "c"), ("b", "c"), ("a", "d")]))       # shares a<c, b<c
N_DISJOINT = one(from_dag([("d", "b"), ("d", "a"), ("c", "a")]))     # shares no Hasse edge


def test_prime_pairs_match_closed_form():
    assert disc_angle(N_BASE, N_BASE)[0] < 1e-12
    assert math.isclose(disc_angle(N_BASE, N_SHARE2)[0], closed(2, 3, 3), rel_tol=1e-9)
    assert math.isclose(disc_angle(N_BASE, N_DISJOINT)[0], math.pi, rel_tol=1e-9)


def test_parallel_ladder_matches_closed_form():
    B = ["a", "b", "c", "d"]
    for h in (4, 3, 2, 1, 0):
        other = anti(B[:h] + [f"u{i}" for i in range(4 - h)])
        assert math.isclose(disc_angle(anti(B), other)[0], closed(h, 4, 4), rel_tol=1e-9)


def test_parallel_large_and_size_mismatch():
    big = [f"t{i}" for i in range(100)]
    d99 = disc_angle(anti(big), anti(big[:99] + ["u"]))[0]
    assert math.isclose(d99, closed(99, 100, 100), rel_tol=1e-9)      # ~0.2831
    B = ["a", "b", "c", "d"]
    assert math.isclose(disc_angle(anti(B), anti(B[:2]))[0], math.pi / 2, rel_tol=1e-9)


def test_totally_parallel_equals_activity_marginal():
    # between totally parallel models the refined SMD IS the Bhattacharyya angle on the uniform
    # activity distributions (the activity-marginal / bag comparison), Remark V.1's limiting case
    B1, B2 = ["a", "b", "c", "d"], ["a", "b", "c", "x"]
    bag = 2.0 * math.acos(sum(math.sqrt(0.25 * 0.25) for _ in ("a", "b", "c")))
    assert math.isclose(disc_angle(anti(B1), anti(B2))[0], bag, rel_tol=1e-9)


def test_typed_atoms_preserve_concurrency_separation():
    # element atoms are typed ("b||" is not the block "b"), so refining a parallel block cannot
    # merge a concurrency with its interleaving mixture
    conc = one(then(leaf("a"), par(leaf("b"), leaf("c"))))
    order = [(then(leaf("a"), leaf("b"), leaf("c")), 0.5),
             (then(leaf("a"), leaf("c"), leaf("b")), 0.5)]
    assert disc_angle(conc, order, refine=True)[0] > 1.5
    assert disc_angle(conc, order, refine=False)[0] > 1.5


def test_refine_kind_selection_and_backcompat():
    conc = one(then(leaf("a"), par(leaf("b"), leaf("c"))))
    seq = one(then(leaf("a"), leaf("b"), leaf("c")))
    # prime-only refinement equals atomic on prime-free models, exactly (the historical contract)
    assert math.isclose(disc_angle(conc, seq, refine={"prime"})[0],
                        disc_angle(conc, seq, refine=False)[0], rel_tol=1e-12)
    # parallel-only refinement leaves primes atomic
    assert math.isclose(disc_angle(N_BASE, N_SHARE2, refine={"parallel"})[0], math.pi, rel_tol=1e-9)
    # full family separates prime-free models from their sequences too
    assert disc_angle(conc, seq, refine=True)[0] > 1.0


def test_refine_rejects_unknown_kind():
    with pytest.raises(ValueError):
        disc_angle(N_BASE, N_BASE, refine={"loop"})
