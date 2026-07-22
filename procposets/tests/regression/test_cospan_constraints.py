"""§32: leg-multiplicity linear constraints + OCCN binding surfacing."""
from __future__ import annotations

import pytest

from procposets.cospan.constraints import (
    at_least,
    at_most,
    constraint,
    cset,
    exactly,
    interval,
    partition,
)
from procposets.cospan.signature import Generator, LinearConstraint, Port
from procposets.occn.markers import Marker
from procposets.occn.to_signature import _in_port, _leg_constraints, _out_port

P = Port("a", "order", "b")
Q = Port("b", "order", "c")


def test_generator_backward_compatible_no_constraints():
    g = Generator("a", frozenset({P}), frozenset({Q}))
    assert g.constraints == frozenset()
    assert g.constrained_ports() == set()


def test_linear_constraint_validation_and_accessors():
    c = constraint({P: 1, Q: -1}, "==", 0)
    assert c.rel == "=="
    assert c.coeffs() == {P: 1, Q: -1}
    assert c.ports() == {P, Q}
    with pytest.raises(ValueError):
        LinearConstraint(frozenset({(P, 1)}), "<", 2)


def test_builders():
    assert at_least(P, 2) == constraint({P: 1}, ">=", 2)
    assert at_most(P, 3) == constraint({P: 1}, "<=", 3)
    assert exactly(P, 1) == constraint({P: 1}, "==", 1)
    assert interval(P, 1) == [at_least(P, 1)]  # cmax None -> no upper bound
    assert interval(P, 2, 5) == [at_least(P, 2), at_most(P, 5)]


def test_partition_is_sum_equals_total():
    i, n = Port("r", "order", "i"), Port("r", "order", "n")
    c = partition(P, [i, n])  # i + n - P == 0
    assert c.rel == "=="
    assert c.rhs == 0
    assert c.coeffs() == {i: 1, n: 1, P: -1}


def test_cset_flattens_constraints_and_lists():
    s = cset(interval(P, 2, 5), exactly(Q, 1))
    assert isinstance(s, frozenset)
    assert len(s) == 3  # >=2, <=5, ==1


def test_occn_leg_constraints_surface_cardinality_and_partition():
    # r: 1..5 orders in; split into exactly-one i and 1..4 n sharing a key.
    ig = frozenset({Marker("s", "order", 1, 5, 0)})
    og = frozenset({Marker("i", "order", 1, 1, 7), Marker("n", "order", 1, 4, 7)})
    cons = _leg_constraints("r", ig, og)

    s_in = _in_port("r", Marker("s", "order", 1, 5, 0))
    i_out = _out_port("r", Marker("i", "order", 1, 1, 7))
    n_out = _out_port("r", Marker("n", "order", 1, 4, 7))

    # the shared-key partition: i + n - s == 0
    part = constraint({i_out: 1, n_out: 1, s_in: -1}, "==", 0)
    assert part in cons
    # the input cardinality is on r's input leg ...
    assert at_most(s_in, 5) in cons and at_least(s_in, 1) in cons
    # ... and (§42 blueprint) the OUTPUT marker cardinalities are now surfaced on the
    # output legs too -- each generator carries its full per-leg parameterisation; the
    # producer/consumer reconciliation happens at composition (`union` = the pushout),
    # not by dropping the output side here.
    assert at_most(n_out, 4) in cons and at_least(n_out, 1) in cons
    assert at_least(i_out, 1) in cons  # i's exactly-1 output (==1 interval) is present


def test_unique_key_outputs_get_no_partition():
    # two outputs of the SAME type but DISTINCT keys -> independent, no partition
    ig = frozenset({Marker("s", "order", 1, 1, 0)})
    og = frozenset({Marker("i", "order", 1, 1, 1), Marker("n", "order", 1, 1, 2)})
    cons = _leg_constraints("r", ig, og)
    assert not any(c.rel == "==" and len(c.terms) > 1 for c in cons)


def test_union_collects_generator_constraints():
    from procposets.cospan.constraints import union
    from procposets.cospan.signature import Generator

    g1 = Generator("b", frozenset(), frozenset({Q}), cset(at_least(Q, 2)))
    g2 = Generator("s", frozenset({Q}), frozenset(), cset(at_most(Q, 5)))
    assert union([g1, g2]) == frozenset({at_least(Q, 2), at_most(Q, 5)})


def test_forget_provenance_carries_and_remaps_constraints():
    from procposets.cospan.discovery_cleanup import forget_provenance
    from procposets.cospan.signature import Generator, Signature

    p = Port("b", "order", "s")
    g = Generator("s", frozenset({p}), frozenset(), cset(interval(p, 2, 5)))
    out = forget_provenance(Signature(frozenset({g})))
    (gq,) = out.generators
    qp = Port("*", "order", "s")  # src forgotten
    assert gq.constraints == cset(interval(qp, 2, 5))
