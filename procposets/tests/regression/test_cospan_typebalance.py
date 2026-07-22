"""Regression: type-balance admissibility (the ``⋈`` filter). Self-contained.

Covers ``procposets/cospan/typebalance.py``, which post-migration had no direct
procposets test of its own -- it was previously exercised only through the
deleted cpm cross-check golden. Pins the current ``admissible`` /
``generator_violation`` / ``type_balance`` behaviour on a hand-built signature;
no pm4py needed (type-balance operates on a ``Signature``).
"""
from procposets.cospan.signature import Generator, Port, Signature
from procposets.cospan.typebalance import (
    Profile,
    admissible,
    generator_violation,
    type_balance,
)

# activity "a" consumes an "order"-typed wire and mints an "item"-typed one
_IN = Port("pre", "order", "a")
_OUT = Port("a", "item", "post")
_A = Generator("a", frozenset({_IN}), frozenset({_OUT}))
_SIG = Signature(frozenset({_A}))

_P = {("pre", "order")}   # backward bundle: (source, type)
_S = {("post", "item")}   # forward bundle:  (target, type)


def test_admissible_respects_the_license():
    licensed = {"a": Profile(creates=frozenset({"item"}), consumes=frozenset({"order"}))}
    assert admissible("a", _P, _S, licensed) is True
    # withdraw the creation licence -> minting "item" is now inadmissible
    no_create = {"a": Profile(creates=frozenset(), consumes=frozenset({"order"}))}
    assert admissible("a", _P, _S, no_create) is False


def test_unconstrained_activity_is_permissive():
    assert admissible("a", _P, _S, {}) is True          # absent from kappa
    assert generator_violation(_A, {}) is None
    assert type_balance(_SIG, {}) == []


def test_generator_violation_reports_unlicensed_types():
    kappa = {"a": Profile(creates=frozenset(), consumes=frozenset({"order"}))}
    v = generator_violation(_A, kappa)
    assert v is not None
    assert v.bad_creates == frozenset({"item"})   # minted "item" with no licence
    assert v.bad_consumes == frozenset()          # consuming "order" is licensed
    assert type_balance(_SIG, kappa) == [v]


def test_balanced_signature_has_no_violations():
    kappa = {"a": Profile(creates=frozenset({"item"}), consumes=frozenset({"order"}))}
    assert type_balance(_SIG, kappa) == []
