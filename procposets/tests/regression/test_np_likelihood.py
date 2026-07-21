"""Tests for procposets/likelihood.py.

Oracle style: every density/kernel quantity is cross-checked against an
independent brute-force computation over itertools.permutations, or against a
hand-derived analytic value.  All randomness is seeded; the whole file runs in
a few seconds.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections import Counter

import numpy as np
import pytest

from procposets.likelihood import Atom, GroupedLog, TimedGroupedLog, make_atom
from procposets.rel import (
    GENERAL,
    SP,
    count_linear_extensions,
    enumerate_posets,
    is_partial_order,
    parallel,
    rel_from_trace,
    series,
    tree_relations,
)
from procposets.simulate import TrueMixture, sample_timed_grouped_log

ABC = frozenset("abc")
ABCD = frozenset("abcd")
CHAIN3 = rel_from_trace(("a", "b", "c"))
CHAIN4 = rel_from_trace(("a", "b", "c", "d"))
AB = frozenset({("a", "b")})  # single relation a<b (already transitively closed)
# the N poset on {a,b,c,d}: a<c, a<d, b<d -- the canonical non-SP order
N_REL = frozenset({("a", "c"), ("a", "d"), ("b", "d")})


# ---------------------------------------------------------------------------
# independent brute-force oracles
# ---------------------------------------------------------------------------

def brute_respects(trace, rel):
    """Independent linear-extension test via list.index."""
    t = list(trace)
    return all(t.index(a) < t.index(b) for (a, b) in rel)


def brute_extensions(alphabet, rel):
    return [
        p for p in itertools.permutations(sorted(alphabet)) if brute_respects(p, rel)
    ]


def brute_n1(alphabet, rel):
    """N1(rel) by full enumeration: non-extensions one adjacent swap from L."""
    ext = set(brute_extensions(alphabet, rel))
    n1 = set()
    for p in itertools.permutations(sorted(alphabet)):
        if p in ext:
            continue
        for i in range(len(p) - 1):
            q = list(p)
            q[i], q[i + 1] = q[i + 1], q[i]
            if tuple(q) in ext:
                n1.add(p)
                break
    return n1


def brute_ks(trace, rel):
    """Enabled-counts k_j of the racing clock, computed independently."""
    ks = []
    for j in range(len(trace)):
        rem = set(trace[j:])
        ks.append(sum(1 for y in rem if not any((x, y) in rel for x in rem)))
    return ks


def full_perm_log(alphabet):
    """GroupedLog whose distinct traces are ALL permutations (one per group)."""
    return GroupedLog([[p] for p in itertools.permutations(sorted(alphabet))])


# ---------------------------------------------------------------------------
# GroupedLog bookkeeping: counts, n_g, pbar
# ---------------------------------------------------------------------------

def test_bookkeeping_counts_ng_pbar():
    abc, acb = ("a", "b", "c"), ("a", "c", "b")
    log = GroupedLog([[abc, abc, acb], [abc]])
    assert log.alphabet == ["a", "b", "c"]
    assert log.m == 3
    assert log.m_fact == 6
    assert log.traces == [abc, acb]  # sorted distinct traces
    assert log.T == 2 and log.G == 2
    assert np.array_equal(log.counts, np.array([[2.0, 1.0], [1.0, 0.0]]))
    assert np.array_equal(log.n_g, np.array([3.0, 1.0]))
    assert np.array_equal(log.pbar, np.array([0.75, 0.25]))
    assert log.pbar.sum() == 1.0


def test_pbar_sums_to_one_nonuniform():
    perms = list(itertools.permutations("abcd"))
    # pad every group with an extra copy of the first permutation -> pbar
    # is non-uniform but must still be a probability vector
    log = GroupedLog([[p, perms[0]] for p in perms])
    assert log.T == 24
    assert abs(log.pbar.sum() - 1.0) < 1e-12
    assert log.pbar[log.tidx[perms[0]]] == pytest.approx(25.0 / 48.0)
    assert np.array_equal(log.n_g, np.full(24, 2.0))


# ---------------------------------------------------------------------------
# normalisation of trace_p over all of S_4
# ---------------------------------------------------------------------------

EPS_ETA = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.3), (0.25, 0.5), (0.7, 0.2),
           (1.0, 0.0), (0.0, 1.0)]


@pytest.mark.parametrize("rel", [frozenset(), AB, CHAIN4, N_REL],
                         ids=["antichain", "a<b", "chain", "N"])
@pytest.mark.parametrize("eps,eta", EPS_ETA)
def test_trace_p_normalises_uniform_kernel(rel, eps, eta):
    log = full_perm_log("abcd")
    atom = make_atom(ABCD, rel, eps, eta)
    s = log.trace_p(atom).sum()
    assert abs(s - 1.0) < 1e-12


@pytest.mark.parametrize("rel", [AB, CHAIN4, N_REL], ids=["a<b", "chain", "N"])
@pytest.mark.parametrize("eps,eta", [(0.1, 0.0), (0.5, 0.3), (1.0, 0.0),
                                     (0.25, 0.5)])
def test_trace_p_normalises_swap_kernel(rel, eps, eta):
    # swap kernel: eps mass sits on N1(rel), which is non-empty for these rels
    log = full_perm_log("abcd")
    atom = make_atom(ABCD, rel, eps, eta, noise_kernel="swap")
    assert len(brute_n1("abcd", rel)) > 0  # precondition for exact normalisation
    s = log.trace_p(atom).sum()
    assert abs(s - 1.0) < 1e-12


def test_trace_p_normalises_with_nonuniform_pbar():
    perms = list(itertools.permutations("abcd"))
    log = GroupedLog([[p, perms[0]] for p in perms])  # non-uniform pbar
    atom = make_atom(ABCD, N_REL, 0.2, 0.4)
    assert abs(log.trace_p(atom).sum() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# in_L indicator vs brute force
# ---------------------------------------------------------------------------

def test_in_L_matches_brute_force_all_posets_on_abcd():
    log = full_perm_log("abcd")
    rels = enumerate_posets("abcd")
    assert len(rels) == 219  # OEIS A001035
    for rel in rels:
        ind = log.in_L(rel)
        expect = np.array([1.0 if brute_respects(t, rel) else 0.0
                           for t in log.traces])
        assert np.array_equal(ind, expect)
        # number of extensions consistent with the ideal-lattice DP
        assert int(ind.sum()) == count_linear_extensions(ABCD, rel)
        assert int(ind.sum()) == len(brute_extensions("abcd", rel))


# ---------------------------------------------------------------------------
# swap kernel
# ---------------------------------------------------------------------------

def test_swap_kernel_matches_brute_force_all_posets_on_abcd():
    log = full_perm_log("abcd")
    for rel in enumerate_posets("abcd"):
        ind, size = log.swap_kernel(rel)
        n1 = brute_n1("abcd", rel)
        # indicator over the 24 distinct traces matches the brute set
        expect = np.array([1.0 if t in n1 else 0.0 for t in log.traces])
        assert np.array_equal(ind, expect)
        assert size == max(len(n1), 1)
        # N1 disjoint from L(rel)
        inL = log.in_L(rel)
        assert not np.any((ind == 1.0) & (inL == 1.0))
        # every flagged trace is one adjacent transposition from an extension
        for ti, t in enumerate(log.traces):
            if ind[ti] != 1.0:
                continue
            ok = False
            for i in range(len(t) - 1):
                q = list(t)
                q[i], q[i + 1] = q[i + 1], q[i]
                if brute_respects(tuple(q), rel):
                    ok = True
                    break
            assert ok


def test_swap_kernel_empty_rel_clamps_size():
    # rel = antichain: L = S_m, so N1 is empty and size clamps to 1
    log = full_perm_log("abc")
    ind, size = log.swap_kernel(frozenset())
    assert np.array_equal(ind, np.zeros(6))
    assert size == 1


def test_swap_kernel_large_alphabet_raises():
    log = GroupedLog([[tuple("abcdefghi")]])  # m = 9 > 8
    with pytest.raises(ValueError):
        log.swap_kernel(frozenset())


def test_swap_kernel_cached():
    log = full_perm_log("abc")
    r1 = log.swap_kernel(AB)
    r2 = log.swap_kernel(AB)
    assert r1 is r2  # cache hit returns the same tuple
    assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1]


# ---------------------------------------------------------------------------
# group_logf: hand examples, -inf semantics, naive cross-check
# ---------------------------------------------------------------------------

def test_group_logf_hand_example():
    # one group, two traces, eps = eta = 0; P = {a<b} on {a,b,c} has e = 3
    log = GroupedLog([[("a", "b", "c"), ("a", "c", "b")]])
    atom = make_atom(ABC, AB, 0.0, 0.0)
    assert atom.e == 3
    logf = log.group_logf(atom)
    assert logf.shape == (1,)
    assert math.isclose(logf[0], 2.0 * math.log(1.0 / 3.0), rel_tol=1e-12)


def test_group_logf_minus_inf_and_finite():
    abc, acb = ("a", "b", "c"), ("a", "c", "b")
    log = GroupedLog([[abc], [abc, acb]])
    chain_atom = make_atom(ABC, CHAIN3, 0.0, 0.0)
    assert chain_atom.e == 1
    logf = log.group_logf(chain_atom)
    assert logf[0] == 0.0  # log(1): the chain gives abc probability 1
    assert logf[1] == -np.inf  # group 1 contains a zero-probability trace
    # with eps > 0 nothing is dead any more; hand value for both groups
    atom_eps = make_atom(ABC, CHAIN3, 0.1, 0.0)
    logf2 = log.group_logf(atom_eps)
    assert np.all(np.isfinite(logf2))
    p_abc = 0.9 * 1.0 + 0.1 / 6.0
    p_acb = 0.1 / 6.0
    assert math.isclose(logf2[0], math.log(p_abc), rel_tol=1e-12)
    assert math.isclose(logf2[1], math.log(p_abc) + math.log(p_acb),
                        rel_tol=1e-12)


@pytest.mark.parametrize("noise", ["uniform", "swap"])
def test_group_logf_matches_naive_per_group(noise):
    abc, acb = ("a", "b", "c"), ("a", "c", "b")
    bac, cab = ("b", "a", "c"), ("c", "a", "b")
    groups = [[abc, abc, acb], [acb, bac], [cab]]
    log = GroupedLog(groups)
    atom = make_atom(ABC, AB, 0.2, 0.3, noise_kernel=noise)
    p = log.trace_p(atom)
    assert np.all(p[np.array([log.tidx[t] for g in groups for t in g])] > 0)
    naive = np.array([sum(math.log(p[log.tidx[t]]) for t in g) for g in groups])
    assert np.allclose(log.group_logf(atom), naive, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# make_atom
# ---------------------------------------------------------------------------

def test_make_atom_general_class_non_sp_rel():
    assert is_partial_order(ABCD, N_REL)
    atom = make_atom(ABCD, N_REL, 0.05, 0.1)
    assert atom is not None and isinstance(atom, Atom)
    assert atom.e == len(brute_extensions("abcd", N_REL)) == 5
    assert atom.eps == 0.05 and atom.eta == 0.1
    assert atom.noise_kernel == "uniform" and atom.lam == 1.0
    assert atom.desc != ""  # Hasse form for a non-SP order
    assert "<" in atom.desc


def test_make_atom_sp_class_rejects_non_sp():
    assert make_atom(ABCD, N_REL, 0.05, 0.1, poset_class="sp") is None
    assert make_atom(ABCD, N_REL, 0.05, 0.1, poset_class=SP) is None


def test_make_atom_sp_class_accepts_sp():
    sp_rel = tree_relations(series("a", parallel("b", "c"), "d"))
    atom = make_atom(ABCD, sp_rel, 0.0, 0.0, poset_class="sp")
    assert atom is not None
    assert atom.e == len(brute_extensions("abcd", sp_rel)) == 2
    assert atom.desc != "" and "->" in atom.desc and "||" in atom.desc
    # general class agrees on e for the same relation set
    gen = make_atom(ABCD, sp_rel, 0.0, 0.0, poset_class=GENERAL)
    assert gen.e == atom.e


def test_make_atom_e_matches_brute_for_all_posets_on_abc():
    for rel in enumerate_posets("abc"):
        atom = make_atom(ABC, rel, 0.0, 0.0)
        assert atom is not None
        assert atom.e == len(brute_extensions("abc", rel))
        assert atom.desc != ""


# ---------------------------------------------------------------------------
# TimedGroupedLog
# ---------------------------------------------------------------------------

def _tiny_timed_log():
    return TimedGroupedLog([
        [(("a", "b"), (0.3, 0.7))],
        [(("b", "a"), (0.5, 0.2))],
    ])


def test_timed_gaps_length_validation():
    with pytest.raises(ValueError):
        TimedGroupedLog([[(("a", "b"), (0.5,))]])  # too few gaps
    with pytest.raises(ValueError):
        TimedGroupedLog([[(("a", "b"), (0.5, 0.2, 0.1))]])  # too many gaps


def test_timed_pooled_rate():
    tl = _tiny_timed_log()
    assert tl.pooled_rate == pytest.approx(4.0 / 1.7, rel=1e-15)


def test_timed_clean_density_hand_example():
    tl = _tiny_timed_log()
    # P = a || b : both traces are extensions, k = (2, 1) for each
    par_atom = make_atom(frozenset("ab"), frozenset(), 0.0, 0.0, lam=1.7)
    logf = tl.group_logf(par_atom)
    assert math.isclose(logf[0], 2 * math.log(1.7) - 1.7 * (2 * 0.3 + 1 * 0.7),
                        rel_tol=1e-12)
    assert math.isclose(logf[1], 2 * math.log(1.7) - 1.7 * (2 * 0.5 + 1 * 0.2),
                        rel_tol=1e-12)
    # P = a -> b : trace ab has k = (1, 1); trace ba is not an extension
    chain_atom = make_atom(frozenset("ab"), frozenset({("a", "b")}),
                           0.0, 0.0, lam=0.9)
    logf2 = tl.group_logf(chain_atom)
    assert math.isclose(logf2[0], 2 * math.log(0.9) - 0.9 * (0.3 + 0.7),
                        rel_tol=1e-12)
    assert logf2[1] == -np.inf


def test_timed_noisy_density_matches_independent_formula():
    tl = _tiny_timed_log()
    eps, eta, lam = 0.2, 0.3, 1.3
    atom = make_atom(frozenset("ab"), frozenset({("a", "b")}), eps, eta, lam=lam)
    lbar, m = tl.pooled_rate, tl.m
    expect = []
    for g in tl.timed_groups:
        tot = 0.0
        for t, gaps in g:
            ks = brute_ks(t, frozenset({("a", "b")}))
            gbar = lbar ** m * math.exp(-lbar * sum(gaps))
            fclean = (lam ** m
                      * math.exp(-lam * sum(k * x for k, x in zip(ks, gaps)))
                      if brute_respects(t, frozenset({("a", "b")})) else 0.0)
            p = ((1 - eta) * ((1 - eps) * fclean + eps * gbar / math.factorial(m))
                 + eta * tl.pbar[tl.tidx[t]] * gbar)
            tot += math.log(p)
        expect.append(tot)
    assert np.allclose(tl.group_logf(atom), np.array(expect), rtol=1e-12)


def test_timed_k_vectors_match_brute_and_marginal_normalises():
    perms = list(itertools.permutations("abc"))
    tl = TimedGroupedLog([[(p, (0.2, 0.5, 0.3))] for p in perms])
    for rel in enumerate_posets("abc"):
        ks = tl._k_vectors(rel)
        # brute-force enabled counts per distinct trace
        for ti, t in enumerate(tl.traces):
            assert np.array_equal(ks[ti], np.array(brute_ks(t, rel), dtype=float))
        # racing-clock ordinal marginal prod_j 1/k_j sums to 1 over L(rel)
        inL = tl.in_L(rel)
        total = sum(float(np.prod(1.0 / ks[ti]))
                    for ti in range(tl.T) if inL[ti])
        assert abs(total - 1.0) < 1e-12


def test_timed_swap_kernel_rejected():
    tl = _tiny_timed_log()
    atom = make_atom(frozenset("ab"), frozenset(), 0.1, 0.0, noise_kernel="swap")
    with pytest.raises(ValueError):
        tl.group_logf(atom)


def test_timed_k_vector_cache():
    tl = _tiny_timed_log()
    ks1 = tl._k_vectors(AB)
    ks2 = tl._k_vectors(AB)
    assert ks1 is ks2  # cached: identical object
    assert all(np.array_equal(a, b) for a, b in zip(ks1, ks2))


def test_timed_ordinal_marginal_is_racing_clock_not_uniform():
    # P = (a -> b) || c on {a,b,c}: extensions abc, acb, cab (e = 3), but the
    # racing-clock ordinal marginal is prod_j 1/k_j = (1/4, 1/4, 1/2), NOT the
    # uniform 1/3 of the untimed model.
    tree = parallel(series("a", "b"), "c")
    assert tree_relations(tree) == AB
    mix = TrueMixture([tree], [1.0])
    groups, _ = sample_timed_grouped_log(mix, G=2000, n_g=2, lams=[1.0], seed=123)
    counts = Counter(t for g in groups for (t, _) in g)
    n = sum(counts.values())
    assert n == 4000
    expected = {("a", "b", "c"): 0.25, ("a", "c", "b"): 0.25,
                ("c", "a", "b"): 0.5}
    assert set(counts) <= set(expected)
    for t, p in expected.items():
        assert abs(counts[t] / n - p) < 0.03
    # explicitly refute the uniform-over-extensions (1/e) marginal
    assert counts[("c", "a", "b")] / n > 1.0 / 3.0 + 0.05


# ---------------------------------------------------------------------------
# audited-defect guards (DESIGN_REVIEW W6, W11, W21)
# ---------------------------------------------------------------------------

def test_grouped_log_rejects_non_permutation_traces():
    # W11: the likelihood assumes complete, duplicate-free traces everywhere;
    # violations used to produce KeyErrors or silently wrong densities with a
    # clean certificate.
    with pytest.raises(ValueError, match="missing activities"):
        GroupedLog([[("a", "b")], [("a", "b", "c")]])
    with pytest.raises(ValueError, match="repeated activity labels"):
        GroupedLog([[("a", "a", "b")]])


def test_timed_log_rejects_degenerate_gaps():
    # W6: zero gaps (same-timestamp ties), negative, or non-finite gaps
    # violate the continuous racing-clock model and must refuse at ingestion.
    good = (("a", "b"), (0.5, 1.0))
    for bad_gaps in ((0.0, 1.0), (-0.1, 1.0), (float("nan"), 1.0),
                     (float("inf"), 1.0)):
        with pytest.raises(ValueError, match="strictly positive"):
            TimedGroupedLog([[good, (("b", "a"), bad_gaps)]])


def test_timed_logf_survives_extreme_gap_sums():
    # W6: the linear-space evaluation underflowed exp(-lam <k, gaps>) to an
    # exact 0.0 beyond ~745, turning a legitimately in-L trace into a
    # spurious -inf group density.  In log space the value is exact:
    # for a chain (k_j = 1), log f = m log(lam) - lam * sum(gaps).
    trace = ("a", "b", "c", "d")
    gaps = (300.0, 300.0, 300.0, 300.0)
    tlog = TimedGroupedLog([[(trace, gaps)]])
    atom = make_atom(frozenset("abcd"), rel_from_trace(trace), 0.0, 0.0,
                     lam=1.0)
    (got,) = tlog.group_logf(atom)
    assert np.isfinite(got)
    assert math.isclose(got, -1200.0, rel_tol=0, abs_tol=1e-9)


def test_make_atom_asserts_partial_order():
    # W21: a relation set that is not transitively closed gives silently
    # wrong e(P) downstream; make_atom now asserts (debug builds).
    broken = frozenset({("a", "b"), ("b", "c")})  # missing (a, c)
    with pytest.raises(AssertionError, match="transitively closed"):
        make_atom(frozenset("abc"), broken, 0.0, 0.0)


def test_noise_kernel_deprecation_shim():
    # Phase-4 item 4: noise -> noise_kernel behind deprecation shims.
    ab = frozenset("ab")
    # canonical spelling: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = make_atom(ab, frozenset(), 0.1, 0.0, noise_kernel="swap")
    assert a.noise_kernel == "swap"

    # deprecated make_atom(noise=) still works, but warns
    with pytest.warns(DeprecationWarning, match="noise_kernel"):
        b = make_atom(ab, frozenset(), 0.1, 0.0, noise="swap")
    assert b.noise_kernel == "swap"

    # deprecated .noise read returns noise_kernel, but warns
    with pytest.warns(DeprecationWarning, match="noise_kernel"):
        assert b.noise == "swap"

    # passing both (non-default) is a TypeError
    with pytest.raises(TypeError, match="not both"):
        make_atom(ab, frozenset(), 0.1, 0.0, noise_kernel="swap", noise="uniform")
