import math

from procposets.distance import bhattacharyya_angle, smd
from procposets.poset import leaf, par, then


def one(P):
    return [(P, 1.0)]


M_conc = one(par(leaf("a"), leaf("b")))
M_seq = one(then(leaf("a"), leaf("b")))
M_mix = [(then(leaf("a"), leaf("b")), 0.5), (then(leaf("b"), leaf("a")), 0.5)]
M1 = one(then(leaf("a"), par(leaf("b"), leaf("c")), leaf("d")))
M2 = one(then(leaf("a"), par(leaf("b"), leaf("c")), leaf("e")))


def test_self_distance_zero():
    assert smd(M1, M1)[0] < 1e-12


def test_symmetry():
    assert math.isclose(smd(M_conc, M_seq)[0], smd(M_seq, M_conc)[0])


def test_concurrency_vs_sequence_distinguished():
    assert smd(M_conc, M_seq)[0] > 0.0


def test_concurrency_vs_mixture_distinguished():
    # the crux: a(x)b and 1/2 a;b + 1/2 b;a are identical at the trace level but must differ here
    assert smd(M_conc, M_mix)[0] > 1.0


def test_smd_grades_by_shared_prefix():
    # M1, M2 share the 'a ; (b(x)c)' prefix -> closer than two wholly different models
    assert smd(M1, M2)[0] < smd(M_conc, M_seq)[0]


def test_result1_is_brittle():
    # Result 1 cannot see the shared prefix: both pairs have disjoint normal forms
    assert math.isclose(bhattacharyya_angle(M1, M2), bhattacharyya_angle(M_conc, M_seq))
    assert math.isclose(bhattacharyya_angle(M1, M2), math.pi)
