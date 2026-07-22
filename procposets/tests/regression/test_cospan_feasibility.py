"""§32 backend 1: bounded-enumeration feasibility for leg constraints."""
from __future__ import annotations

import pytest

from procposets.cospan.constraints import at_least, at_most, exactly, interval, partition
from procposets.cospan.feasibility import (
    FeasibilityTooLarge,
    feasible,
    ranges,
    solve,
)
from procposets.cospan.signature import Port

S = Port("s", "order", "r")   # total orders into r
I = Port("r", "order", "i")   # investigated
N = Port("r", "order", "n")   # not investigated


def test_default_bound_pins_everything_to_one():
    # an unconstrained-but-mentioned var: only value 1 is in the default box
    assert solve([at_least(I, 1)]) == {I: 1}
    assert feasible([exactly(I, 1)]) is True
    assert feasible([exactly(I, 2)]) is False  # 2 not in [1,1]


def test_interval_needs_a_big_enough_bound():
    cons = interval(S, 2, 5)
    assert feasible(cons, bound=1) is False  # box [1,1], but >=2 required
    assert solve(cons, bound=3) == {S: 2}    # smallest feasible
    assert ranges(cons, bound=10) == {S: (2, 5)}


def test_partition_distribution_like_liss_r():
    # i + n == S, i == 1, n in [1,4], S in [1,5]
    cons = [partition(S, [I, N]), exactly(I, 1), *interval(N, 1, 4), *interval(S, 1, 5)]
    rng = ranges(cons, bound=5)
    assert rng[I] == (1, 1)
    assert rng[N] == (1, 4)
    assert rng[S] == (2, 5)  # S = i + n = 1 + [1,4] = [2,5]


def test_pinned_total_solves_the_split():
    # fix the upstream total S = 4 -> exactly one i, the rest (3) n
    cons = [partition(S, [I, N]), exactly(I, 1), *interval(N, 1, 4)]
    assign = solve(cons, bound=5, pinned={S: 4})
    assert assign == {S: 4, I: 1, N: 3}


def test_infeasible_returns_none_empty():
    cons = [exactly(I, 1), at_least(I, 2)]  # 1 and >=2 conflict
    assert solve(cons, bound=5) is None
    assert ranges(cons, bound=5) == {}


def test_too_large_box_raises():
    many = [at_most(Port("x", "t", str(i)), 50) for i in range(8)]
    with pytest.raises(FeasibilityTooLarge):
        solve(many, bound=50, max_assignments=1000)
