"""M9 higher-order moment initialiser (procposets.initialiser).

The load-bearing property is that the initialiser is a *warm start only*: it
must never change the fitted optimum, only the seed.  Also pinned here is the
§8 refinement measured during implementation -- the phi^(2) precedence moment
is injective on single posets, so genuine margin-equivalent poset pairs do
not exist for the clean uniform-extension law.
"""

import numpy as np
import pytest

from procposets import (
    GroupedLog, enumerate_posets, fit, find_margin_equivalences,
    margin_equivalent, moment_seed, parallel, poset_moment, sample_grouped_log,
    series, tree_relations,
)
from procposets.initialiser import empirical_moment, ktuple_index, trace_phi

ALPHA = frozenset("abcd")
CHAIN = tree_relations(series("a", "b", "c", "d"))
DIAMOND = tree_relations(series("a", parallel("b", "c"), "d"))
ANTICHAIN = frozenset()


def _mom(rel, order=2):
    keys = ktuple_index(sorted(ALPHA), order)
    return dict(zip(keys, poset_moment(ALPHA, rel, keys)))


def test_trace_phi_subsequence_indicator():
    keys = ktuple_index(("a", "b", "c"), 2)
    v = dict(zip(keys, trace_phi(("a", "b", "c"), keys)))
    assert v[("a", "b")] == 1.0 and v[("a", "c")] == 1.0 and v[("b", "c")] == 1.0
    assert v[("b", "a")] == 0.0 and v[("c", "a")] == 0.0


def test_poset_moment_known_values():
    mc = _mom(CHAIN)
    assert mc[("a", "b")] == pytest.approx(1.0)   # comparable -> exactly 1
    assert mc[("b", "a")] == pytest.approx(0.0)
    ma = _mom(ANTICHAIN)
    assert ma[("a", "b")] == pytest.approx(0.5)   # incomparable -> 1/2
    md = _mom(DIAMOND)
    assert md[("a", "d")] == pytest.approx(1.0)   # a below everything
    assert md[("b", "c")] == pytest.approx(0.5)   # b, c concurrent


def test_phi2_is_injective_on_single_posets():
    """The §8 refinement: a comparable pair has precedence-probability exactly
    1, so the phi^(2) moment pins the relation -- no two distinct m=4 posets
    share it.  (Stronger than §8's premise, which this disproves for the
    clean law.)"""
    rels = enumerate_posets(ALPHA)
    assert find_margin_equivalences(ALPHA, rels, order=2) == []


def test_margin_equivalent_predicate():
    assert not margin_equivalent(ALPHA, CHAIN, CHAIN)      # identical rel
    assert not margin_equivalent(ALPHA, CHAIN, DIAMOND)    # distinct moments
    # with an absurdly loose tolerance every distinct pair is "equivalent"
    assert margin_equivalent(ALPHA, CHAIN, DIAMOND, tol=10.0)


def _truth_log(seed=7, G=120, n_g=4):
    from procposets import TrueMixture
    truth = TrueMixture(trees=[series("a", parallel("b", "c"), "d"),
                               series("a", "c", "b", "d")],
                        weights=[0.6, 0.4])
    g, _ = sample_grouped_log(truth, G=G, n_g=n_g, seed=seed)
    return GroupedLog(g), [t for gg in g for t in gg]


def test_init_order_reaches_the_same_optimum():
    """The compute-dial invariant: default seed and every moment order fit to
    the identical convex optimum (loglik and poset marginal)."""
    log, _ = _truth_log()
    base = fit(log)
    for init_order in (2, 3, "auto"):
        r = fit(log, init_order=init_order)
        assert r.loglik == pytest.approx(base.loglik, abs=1e-6)
        mb = {rel: w for rel, _, w, _, _, _ in base.poset_marginal()}
        mr = {rel: w for rel, _, w, _, _, _ in r.poset_marginal()}
        for rel in set(mb) | set(mr):
            assert mr.get(rel, 0.0) == pytest.approx(mb.get(rel, 0.0), abs=1e-5)


def test_moment_seed_ranking_and_auto_escalation():
    log, traces = _truth_log()
    rels = enumerate_posets(ALPHA)
    rk2 = moment_seed(ALPHA, rels, traces, order=2)
    assert rk2.order == 2
    assert 0 <= rk2.top < len(rels)
    assert rk2.margin >= 0.0
    # auto may climb when the top-two candidates tie in alignment; it never
    # exceeds the alphabet size and always returns a valid seed
    rk_auto = moment_seed(ALPHA, rels, traces, order="auto")
    assert 2 <= rk_auto.order <= len(ALPHA)
    if rk_auto.order > 2:
        assert rk_auto.escalated_from == 2


def test_empirical_moment_matches_a_pure_poset_sample():
    """Empirical phi^(2) of a large clean sample from one poset approaches
    that poset's moment signature."""
    from procposets import TrueMixture
    truth = TrueMixture(trees=[series("a", parallel("b", "c"), "d")],
                        weights=[1.0])
    g, _ = sample_grouped_log(truth, G=400, n_g=5, seed=3)
    traces = [t for gg in g for t in gg]
    keys = ktuple_index(sorted(ALPHA), 2)
    emp = empirical_moment(traces, keys)
    theo = poset_moment(ALPHA, DIAMOND, keys)
    assert np.max(np.abs(emp - theo)) < 0.05
