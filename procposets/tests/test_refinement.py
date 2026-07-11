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


# ---- occurrence collisions, exactness (strict), and the faithful depth --------------------

N2 = from_dag([("e", "g"), ("f", "g"), ("e", "h")])


def _DB():
    return one(then(leaf("a"), par(leaf("b"), leaf("d")), N2, leaf("z")))


def _DBp():
    # D_B' = a ; (b x d) ; N2 ; (b x b x d) ; z : atom names b||, d|| recur across two blocks
    return one(then(leaf("a"), par(leaf("b"), leaf("d")), N2,
                    par(leaf("b"), leaf("b"), leaf("d")), leaf("z")))


def _DE():
    # same skeleton, final block (b x d): graded against _DBp at the faithful depth
    return one(then(leaf("a"), par(leaf("b"), leaf("d")), N2,
                    par(leaf("b"), leaf("d")), leaf("z")))


def test_strict_default_refuses_occurrence_collision():
    with pytest.raises(ValueError):
        disc_angle(_DBp(), _DBp())                       # depth 1: b||/d|| recur -> exactness lost


def test_relaxed_merge_is_the_declared_fallback():
    from procposets.discrete import _build_refined
    rm, _ = _build_refined(_DBp(), {"prime", "parallel"}, context_depth=1, strict=False)
    assert math.isclose(rm["b||"]["z"], 4 / 7, rel_tol=1e-12)      # mixed-context row
    assert math.isclose(rm["d||"]["z"], 2 / 5, rel_tol=1e-12)
    assert disc_angle(_DBp(), _DBp(), strict=False)[0] < 1e-12     # identical models still 0


def test_faithful_depth_two_is_exact_and_acyclic():
    from procposets.discrete import _build_refined
    rm, _ = _build_refined(_DBp(), {"prime", "parallel"}, context_depth=2)   # no raise
    assert math.isclose(rm["a|b||"]["(b * d)|e<g"], 1 / 3, rel_tol=1e-12)
    par2 = rm["(b * d)|e<g"]                                       # fan into the final block
    assert math.isclose(par2["N{e<g, e<h, f<g}|b||"], 2 / 3, rel_tol=1e-12)
    assert math.isclose(par2["N{e<g, e<h, f<g}|d||"], 1 / 3, rel_tol=1e-12)
    # acyclic apart from the reset: no state reachable from itself through the body
    assert disc_angle(_DBp(), _DBp(), context_depth=2)[0] < 1e-12


def test_graded_comparison_at_faithful_depth():
    # final blocks (b,b,d) vs (b,d): the three fan rows carry the multiplicity coefficient,
    # the two second-occurrence atom rows differ maximally (different following window)
    bc = math.sqrt(1 / 3) + math.sqrt(1 / 6)
    expect = 2 * math.sqrt(3 * math.acos(bc) ** 2 + 2 * (math.pi / 2) ** 2)
    assert math.isclose(disc_angle(_DBp(), _DE(), context_depth=2)[0], expect, rel_tol=1e-9)


def test_insertion_at_faithful_depth():
    # D_B vs D_B': one inserted block; refined pi*sqrt(5), atomic (build, depth 2) pi*sqrt(2)
    assert math.isclose(disc_angle(_DB(), _DBp(), context_depth=2)[0],
                        math.pi * math.sqrt(5), rel_tol=1e-9)
    from procposets.distance import smd_rows
    from procposets.matrix import build
    d_atomic = smd_rows(build(_DB(), 2), build(_DBp(), 2))[0]
    assert math.isclose(d_atomic, math.pi * math.sqrt(2), rel_tol=1e-9)


# ---- E1 recursive fan-out (paper App C "Outlook"; pre-adoption pins) ----------------------


def test_recursive_flat_invariance():
    # any block whose children are all leaves has no disclosure: recursive=True is bit-identical
    # to recursive=False on flat primes, flat parallels, and whole flat models
    for m1, m2 in [(N_BASE, N_SHARE2), (N_BASE, N_DISJOINT),
                   (anti(["a", "b", "c", "d"]), anti(["a", "b", "u", "v"]))]:
        assert disc_angle(m1, m2, recursive=True)[0] == disc_angle(m1, m2, recursive=False)[0]
    assert disc_angle(_DB(), _DBp(), context_depth=2, recursive=True)[0] == \
        disc_angle(_DB(), _DBp(), context_depth=2, recursive=False)[0]


def test_recursive_nested_series_ladder():
    # (a;b) x (c;d;e): level-1 atoms {(a;b)||, (c;d;e)||} + disclosures {a<b, c<d, d<e} -> 5 atoms
    base = one(par(then(leaf("a"), leaf("b")), then(leaf("c"), leaf("d"), leaf("e"))))
    ladder = [
        (par(then(leaf("a"), leaf("b")), then(leaf("c"), leaf("d"), leaf("e"))), 5),  # identical
        (par(then(leaf("a"), leaf("b")), then(leaf("c"), leaf("d"), leaf("f"))), 3),  # share c<d
        (par(then(leaf("a"), leaf("b")), then(leaf("c"), leaf("f"), leaf("g"))), 2),  # branch only
        (par(then(leaf("f"), leaf("g")), then(leaf("x"), leaf("y"), leaf("z"))), 0),  # disjoint
    ]
    prev = -1.0
    for P, h in ladder:
        d = disc_angle(base, one(P), recursive=True)[0]
        assert math.isclose(d, closed(h, 5, 5), rel_tol=1e-9)
        assert d >= prev
        prev = d
    # the flat family cannot credit the shared nested cover: it sees 1 shared atom of 2
    flat = disc_angle(base, one(ladder[1][0]), recursive=False)[0]
    assert math.isclose(flat, closed(1, 2, 2), rel_tol=1e-9)
    assert disc_angle(base, one(ladder[1][0]), recursive=True)[0] < flat


def test_recursive_nested_parallel_membership():
    # ((a x b); c) x d vs ((a x x); c) x d: the shared member a|| of the nested parallel earns
    # credit through the recursion; atoms are {((a*b);c)||, d||, (a*b)<c, a||, b||} etc.
    m1 = one(par(then(par(leaf("a"), leaf("b")), leaf("c")), leaf("d")))
    m2 = one(par(then(par(leaf("a"), leaf("x")), leaf("c")), leaf("d")))
    assert math.isclose(disc_angle(m1, m2, recursive=True)[0], closed(2, 5, 5), rel_tol=1e-9)
    assert math.isclose(disc_angle(m1, m2, recursive=False)[0], closed(1, 2, 2), rel_tol=1e-9)


def test_recursive_identical_nested_is_zero():
    m = one(par(then(leaf("a"), leaf("b")), then(leaf("c"), leaf("d"), leaf("e"))))
    assert disc_angle(m, m, recursive=True)[0] < 1e-12


def test_repeated_label_multiplicity_closed_form():
    # isolated (b x b x d) vs (b x d): BC = sum sqrt(m m') / sqrt(|A||A'|) = (sqrt2+1)/sqrt6,
    # NOT the multiset-count form 2/sqrt6 (which overstates the distance)
    d = disc_angle(one(par(leaf("b"), leaf("b"), leaf("d"))),
                   one(par(leaf("b"), leaf("d"))))[0]
    assert math.isclose(d, 2 * math.acos((math.sqrt(2) + 1) / math.sqrt(6)), rel_tol=1e-9)
    assert not math.isclose(d, 2 * math.acos(2 / math.sqrt(6)), rel_tol=1e-3)
