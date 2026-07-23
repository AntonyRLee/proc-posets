"""The marginal ``compare`` key (opt-in): the joint per-type-arity CanonKey
decomposed into per-``(activity, side, type)`` :class:`ArityFact` rows.

Pins (a) the decomposition itself (optionality-as-0, empty sides contributing
no facts, the documented cross-type-XOR losiness), (b) the localisation win --
one per-type arity change is ONE diff row where the joint view scatters it
over ``2^(k-1)`` absent/novel key pairs, (c) the default gate: ``compare``
without ``key`` is byte-identical to ``key="joint"`` (the 0-drift oracles keep
pinning joint numbers), and (d) agreement of the marginal view across the slow
and fast extractors (they share the CanonKey set, so they must share its
marginals).
"""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from procposets.cospan.engine import extract_signature
from procposets.cospan.engine_fast import extract_signature_fast
from procposets.cospan.signature import Generator, Port, Signature
from procposets.cospan.signature_compare import (
    ArityFact,
    MarginalKey,
    compare,
    marginal_facts,
    report_text,
    report_to_dict,
)

from .test_cospan_extract_fast import _typed_hub


def _gen(label, left, right):
    """left/right as [(neighbour, typ), ...] -- distinct neighbours give arity."""
    L = frozenset(Port(p, t, label) for p, t in left)
    R = frozenset(Port(label, t, s) for s, t in right)
    return Generator(label, L, R)


def _hub_sig(t2_arity: int = 1) -> Signature:
    """``H``: fixed one-``t0``-leg out side; in side = every subset of t1..t4
    (16 contexts, each type independently optional), ``t2`` legs carried at
    ``t2_arity``.  The miniature of a wide factoring OCPN hub."""
    types = ["t1", "t2", "t3", "t4"]
    gens = set()
    for r in range(len(types) + 1):
        for combo in combinations(types, r):
            left = []
            for t in combo:
                n = t2_arity if t == "t2" else 1
                left += [(f"P{t}{j}", t) for j in range(n)]
            gens.add(_gen("H", left, [("S", "t0")]))
    return Signature(frozenset(gens))


def test_marginal_facts_optionality_and_empty_sides():
    facts = marginal_facts(_hub_sig())
    # each optional in-type: {0,1}; the fixed out leg: {1}
    for t in ("t1", "t2", "t3", "t4"):
        assert facts[MarginalKey("H", "in", t)] == ArityFact((0, 1))
    assert facts[MarginalKey("H", "out", "t0")] == ArityFact((1,))
    # the empty in-context contributes 0s, not facts of its own; no other rows
    assert len(facts) == 5


def test_marginal_facts_erase_cross_type_xor():
    """The documented losiness: mode-refined ``s`` routes con XOR box; the
    marginals record both as optional and forget the coupling (the joint keys
    keep it -- which is exactly why the joint view stays the default)."""
    s_c = _gen("s", [("b", "ord"), ("c", "con")], [("r", "ord")])
    s_b = _gen("s", [("b", "ord"), ("d", "box")], [("r", "ord")])
    facts = marginal_facts(Signature(frozenset({s_c, s_b})))
    assert facts[MarginalKey("s", "in", "ord")] == ArityFact((1,))
    assert facts[MarginalKey("s", "in", "con")] == ArityFact((0, 1))
    assert facts[MarginalKey("s", "in", "box")] == ArityFact((0, 1))


def test_marginal_localises_one_arity_change_to_one_row():
    """sig_b doubles only t2's arity.  Joint view: the 8 t2-carrying keys all
    mismatch (8 absent + 8 novel over 24 rows).  Marginal view: 5 rows, exactly
    one ``diff`` -- the (H, in, t2) fact {0,1} vs {0,2}."""
    named = {"a": _hub_sig(1), "b": _hub_sig(2)}
    joint = compare(named).per_notation()["b"]
    assert joint == {"match": 8, "diff": 0, "absent": 8, "novel": 8}
    marg = compare(named, key="marginal")
    assert marg.per_notation()["b"] == {"match": 4, "diff": 1, "absent": 0, "novel": 0}
    (diff_row,) = [r for r in marg.rows if r.verdict() == "param-diff"]
    assert diff_row.key == MarginalKey("H", "in", "t2")
    cells = dict(diff_row.cells)
    assert cells["a"].profile == ArityFact((0, 1))
    assert cells["b"].profile == ArityFact((0, 2))


def test_default_key_is_joint_and_unknown_key_raises():
    named = {"a": _hub_sig(1), "b": _hub_sig(2)}
    assert compare(named) == compare(named, key="joint")
    with pytest.raises(ValueError, match="joint|marginal"):
        compare(named, key="typo")


def test_marginal_agrees_across_slow_and_fast_extractors():
    """Same CanonKey set (the extract_fast contract) => same marginal facts;
    the marginal compare of the two engines is all-match."""
    g = _typed_hub()
    slow = extract_signature(g)
    fast = extract_signature_fast(g)
    assert marginal_facts(slow) == marginal_facts(fast)
    rep = compare({"slow": slow, "fast": fast}, key="marginal")
    assert rep.per_notation()["fast"]["match"] == len(rep.rows) > 0
    assert rep.summary()["match"] == len(rep.rows)


def test_legless_generator_stays_visible():
    """A zero-left zero-right generator (a per-type source->a->sink net under
    the default terminus strip) must not vanish from the marginal view: its
    always-empty sides are recorded as ``(label, side, None) {0}`` presence
    facts, so whole-generator existence stays comparable (a discovery dropping
    the activity shows as absent, not as perfect rederivation)."""
    q = _gen("q", [], [])
    h = _gen("H", [("g", "t1")], [])
    a = Signature(frozenset({q, h}))
    b = Signature(frozenset({h}))
    fa = marginal_facts(a)
    assert fa[MarginalKey("q", "in", None)] == ArityFact((0,))
    assert fa[MarginalKey("q", "out", None)] == ArityFact((0,))
    rep = compare({"a": a, "b": b}, key="marginal")
    assert rep.per_notation()["b"]["absent"] == 2  # q's two presence facts
    assert rep.summary()["partial"] == 2


def test_untyped_vs_literal_none_type_rows_are_distinct_and_ordered():
    """``typ=None`` (an untyped leg) and an object type literally named
    ``"None"`` are distinct facts with a deterministic row order (None-typed
    first) -- ``str(typ)`` would tie them and leave the order to hash-seed
    set-iteration luck."""
    g1 = _gen("H", [("p", None), ("q", "None")], [("s", "t0")])
    g2 = _gen("H", [("p", None)], [("s", "t0")])
    sig = Signature(frozenset({g1, g2}))
    facts = marginal_facts(sig)
    assert facts[MarginalKey("H", "in", None)] == ArityFact((1,))
    assert facts[MarginalKey("H", "in", "None")] == ArityFact((0, 1))
    rep = compare({"a": sig, "b": sig}, key="marginal")
    in_typs = [r.key.typ for r in rep.rows if r.key.side == "in"]
    assert in_typs == [None, "None"]


def test_marginal_report_renders_and_serialises():
    """report_text renders marginal rows through the same duck-typed surface
    (arity_str / render); report_to_dict emits the marginal row shape and
    round-trips through JSON byte-stably; boundary facts sort last."""
    g1 = _gen("gamma1", [], [("H", "t1")])
    h = _gen("H", [("gamma1", "t1")], [])
    named = {
        "a": Signature(frozenset({g1, h})),
        "b": Signature(frozenset({g1, h})),
    }
    rep = compare(named, key="marginal")
    txt = report_text(rep)
    assert "↓t1" in txt and "{1}" in txt
    d = report_to_dict(rep)
    assert json.loads(json.dumps(d, sort_keys=True)) == d
    assert all({"label", "side", "type", "boundary", "verdict", "cells"} <= set(r)
               for r in d["generators"])
    # boundary (gamma1) facts after the interior H facts
    labels = [r["label"] for r in d["generators"]]
    assert labels.index("H") < labels.index("gamma1")
