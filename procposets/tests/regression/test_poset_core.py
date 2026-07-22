"""Tests for procposets/poset.py.

Oracle strategy: brute force over itertools.permutations, OEIS counts
(A001035 for labeled posets), analytic identities (Szpilrajn: every poset is
the meet of its linear extensions), and cross-checks between the general
ideal-lattice DP and the SP-tree (Valdes-Tarjan-Lawler) code paths.

All randomness is seeded (random.Random) so the suite is deterministic.
"""

import itertools
import random

import pytest

from procposets import (
    GENERAL,
    SP,
    count_linear_extensions,
    decompose,
    describe,
    enumerate_posets,
    enumerate_sp,
    extension_count,
    get_poset_class,
    is_partial_order,
    meet,
    meet_closure,
    parallel,
    rel_from_trace,
    respects,
    sample_extension_tree,
    sample_linear_extension,
    series,
    transitive_reduction,
    tree_relations,
)
from procposets.rel import refines

# ---------------------------------------------------------------------------
# Helpers (test-local oracles)
# ---------------------------------------------------------------------------


def _els(m):
    return frozenset("abcdef"[:m])


def _transitive_closure(rel):
    """Tiny independent transitive-closure helper (naive fixpoint)."""
    out = set(rel)
    while True:
        add = {(a, d) for (a, b) in out for (c, d) in out if b == c} - out
        if not add:
            return frozenset(out)
        out |= add


def _brute_extensions(elements, rel):
    """All linear extensions by filtering every permutation (the oracle)."""
    return [
        p for p in itertools.permutations(sorted(elements)) if respects(p, rel)
    ]


def _chi2(counts, expected):
    return sum((c - expected) ** 2 / expected for c in counts)


# The canonical non-SP poset on 4 elements: the "N" (a<c, b<c, b<d).
N_ELS = frozenset("abcd")
N_REL = frozenset({("a", "c"), ("b", "c"), ("b", "d")})


# ---------------------------------------------------------------------------
# enumerate_posets vs OEIS A001035
# ---------------------------------------------------------------------------


def test_enumerate_posets_counts_match_oeis_a001035():
    # A001035: labeled posets on m elements (skip m=6: slow).
    expected = {1: 1, 2: 3, 3: 19, 4: 219, 5: 4231}
    for m, want in expected.items():
        posets = enumerate_posets(_els(m))
        assert len(posets) == want, f"m={m}"
        assert len(set(posets)) == want, f"m={m}: duplicates found"


def test_enumerate_posets_all_valid_and_transitively_closed():
    for m in (1, 2, 3, 4, 5):
        els = _els(m)
        for rel in enumerate_posets(els):
            assert is_partial_order(els, rel)
            assert _transitive_closure(rel) == rel


# ---------------------------------------------------------------------------
# enumerate_sp and decompose
# ---------------------------------------------------------------------------


def test_enumerate_sp_m4_count_and_subset():
    els = _els(4)
    sp = enumerate_sp(els)
    assert len(sp) == 195
    assert len(set(sp)) == 195
    general = set(enumerate_posets(els))
    assert set(sp) <= general
    # Every SP member decomposes into an SP tree that reproduces the relation.
    for rel in sp:
        tree = decompose(els, rel)
        assert tree is not None
        assert tree_relations(tree) == rel
        assert sorted(tree.elements()) == sorted(els)
    # The 219 - 195 = 24 non-SP posets are exactly the labelings of the N,
    # and none of them decomposes.
    non_sp = general - set(sp)
    assert len(non_sp) == 24
    for rel in non_sp:
        assert decompose(els, rel) is None
    assert N_REL in non_sp


def test_enumerate_sp_small_m_all_posets_are_sp():
    # On <= 3 elements no N fits, so SP == general: 1, 3, 19.
    for m in (1, 2, 3):
        assert enumerate_sp(_els(m)) == enumerate_posets(_els(m))


def test_enumerate_sp_m5_matches_decompose_filter():
    # Cross-check the two independent characterizations of SP on m=5:
    # exhaustive SP construction == {general posets that decompose}.
    els = _els(5)
    sp = set(enumerate_sp(els))
    via_decompose = {
        rel for rel in enumerate_posets(els) if decompose(els, rel) is not None
    }
    assert sp == via_decompose


# ---------------------------------------------------------------------------
# count_linear_extensions
# ---------------------------------------------------------------------------


def test_count_linear_extensions_bruteforce_all_m4():
    els = _els(4)
    for rel in enumerate_posets(els):
        assert count_linear_extensions(els, rel) == len(
            _brute_extensions(els, rel)
        ), describe(els, rel)


def test_count_linear_extensions_matches_sp_recursion_m4():
    els = _els(4)
    for rel in enumerate_sp(els):
        tree = decompose(els, rel)
        assert tree is not None
        assert count_linear_extensions(els, rel) == extension_count(tree)


def test_count_linear_extensions_m5_spotcheck():
    els = _els(5)
    posets = enumerate_posets(els)
    # Deterministic sample: first, last, and every 400th of the 4231.
    picks = posets[::400] + [posets[0], posets[-1]]
    for rel in picks:
        assert count_linear_extensions(els, rel) == len(
            _brute_extensions(els, rel)
        )
    # Analytic anchors: antichain -> 5!, chain -> 1.
    assert count_linear_extensions(els, frozenset()) == 120
    assert count_linear_extensions(els, rel_from_trace("abcde")) == 1


# ---------------------------------------------------------------------------
# sample_linear_extension (general DP sampler)
# ---------------------------------------------------------------------------


def test_sample_linear_extension_only_produces_extensions():
    rng = random.Random(0)
    els5 = _els(5)
    cases = [
        (N_ELS, N_REL),
        (els5, frozenset({("a", "b"), ("c", "d"), ("a", "e"), ("c", "e")})),
        (els5, rel_from_trace("abcde")),
    ]
    for els, rel in cases:
        for _ in range(200):
            t = sample_linear_extension(els, rel, rng)
            assert sorted(t) == sorted(els)  # a permutation of the alphabet
            assert respects(t, rel)


def test_sample_linear_extension_uniform_on_n_poset():
    # The N poset has e = 5 linear extensions; chi-squared with df = 4.
    exts = _brute_extensions(N_ELS, N_REL)
    assert len(exts) == 5
    rng = random.Random(12345)
    n = 5000
    counts = {e: 0 for e in exts}
    for _ in range(n):
        counts[sample_linear_extension(N_ELS, N_REL, rng)] += 1
    assert set(counts) == set(exts)  # support is exactly L(P)
    assert all(c > 0 for c in counts.values())
    # chi2_{0.9999, df=4} ~ 23.5; generous threshold.
    assert _chi2(counts.values(), n / 5) < 25.0


def test_sample_linear_extension_uniform_on_sp_poset():
    # (a -> b) || (c -> d): e = C(4,2) = 6 extensions; df = 5.
    els = _els(4)
    rel = frozenset({("a", "b"), ("c", "d")})
    exts = _brute_extensions(els, rel)
    assert len(exts) == 6
    rng = random.Random(2024)
    n = 6000
    counts = {e: 0 for e in exts}
    for _ in range(n):
        counts[sample_linear_extension(els, rel, rng)] += 1
    assert set(counts) == set(exts)
    # chi2_{0.9999, df=5} ~ 25.7; generous threshold.
    assert _chi2(counts.values(), n / 6) < 27.0


# ---------------------------------------------------------------------------
# sample_extension (SP tree sampler)
# ---------------------------------------------------------------------------


def test_sample_extension_uniform_on_sp_tree():
    tree = parallel(series("a", "b"), series("c", "d"))
    rel = tree_relations(tree)
    exts = _brute_extensions(_els(4), rel)
    assert extension_count(tree) == len(exts) == 6
    rng = random.Random(7)
    n = 6000
    counts = {e: 0 for e in exts}
    for _ in range(n):
        t = sample_extension_tree(tree, rng)
        assert respects(t, rel)
        counts[t] += 1
    assert set(counts) == set(exts)
    assert _chi2(counts.values(), n / 6) < 27.0


def test_sample_extension_series_tree_is_deterministic():
    tree = series("a", "b", "c")
    rng = random.Random(3)
    for _ in range(10):
        assert sample_extension_tree(tree, rng) == ("a", "b", "c")


# ---------------------------------------------------------------------------
# meet
# ---------------------------------------------------------------------------


def test_meet_is_intersection_idempotent_commutative():
    p = rel_from_trace("abcd")
    q = N_REL
    m = meet(p, q)
    assert m == p & q
    assert m <= p and m <= q
    assert is_partial_order(N_ELS, m)
    assert meet(p, p) == p  # idempotent
    assert meet(p, q) == meet(q, p)  # commutative
    assert meet(p) == p  # single argument
    # variadic == folded binary
    r = rel_from_trace("badc")
    assert meet(p, q, r) == meet(meet(p, q), r)


def test_meet_of_chains_is_their_common_order():
    m = meet(rel_from_trace("abcd"), rel_from_trace("abdc"))
    expected = frozenset(
        {("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")}
    )
    assert m == expected
    # which is exactly a -> b -> (c || d)
    assert m == tree_relations(series("a", "b", parallel("c", "d")))
    # meets of totally-unrelated chains give the antichain
    assert meet(rel_from_trace("abc"), rel_from_trace("cba")) == frozenset()


def test_meet_of_all_extensions_recovers_poset():
    # Szpilrajn / Dushnik-Miller: P is the intersection of its linear
    # extensions.  Check for every labeled poset on 4 elements.
    els = _els(4)
    for rel in enumerate_posets(els):
        chains = [rel_from_trace(t) for t in _brute_extensions(els, rel)]
        assert meet(*chains) == rel
    # And spot-check a few m=5 posets, including a non-SP one.
    els5 = _els(5)
    for rel in [
        frozenset({("a", "c"), ("b", "c"), ("b", "d")}),  # N plus isolated e
        rel_from_trace("edcba"),
        tree_relations(series("a", parallel("b", "c", "d"), "e")),
    ]:
        chains = [rel_from_trace(t) for t in _brute_extensions(els5, rel)]
        assert meet(*chains) == rel


# ---------------------------------------------------------------------------
# transitive_reduction
# ---------------------------------------------------------------------------


def test_transitive_reduction_of_chain():
    covers = transitive_reduction(rel_from_trace("abcd"))
    assert covers == frozenset({("a", "b"), ("b", "c"), ("c", "d")})


def test_transitive_reduction_roundtrips_all_m4():
    els = _els(4)
    for rel in enumerate_posets(els):
        red = transitive_reduction(rel)
        assert red <= rel
        assert _transitive_closure(red) == rel


# ---------------------------------------------------------------------------
# meet_closure
# ---------------------------------------------------------------------------


def test_meet_closure_contains_generators_and_is_closed():
    gens = [rel_from_trace(t) for t in ("abcd", "badc", "cdab", "dcba")]
    closure, hit_cap = meet_closure(gens)
    assert hit_cap is False
    closed = set(closure)
    assert set(gens) <= closed
    # closed under pairwise meet (hence all finite meets)
    for r1 in closure:
        for r2 in closure:
            assert (r1 & r2) in closed
    # every member is a genuine partial order
    for rel in closure:
        assert is_partial_order(_els(4), rel)
    # sorted output, no duplicates
    assert len(closed) == len(closure)
    assert closure == sorted(closure, key=lambda r: (len(r), sorted(r)))


def test_meet_closure_tiny_cap_hits_cap():
    gens = [rel_from_trace(t) for t in ("abcd", "badc", "cdab")]
    # meets of these chains produce new orders, so a cap equal to the
    # generator count is immediately exceeded.
    _, hit_cap = meet_closure(gens, cap=3)
    assert hit_cap is True


# ---------------------------------------------------------------------------
# respects / rel_from_trace / refines
# ---------------------------------------------------------------------------


def test_rel_from_trace_basics():
    assert rel_from_trace("abc") == frozenset(
        {("a", "b"), ("a", "c"), ("b", "c")}
    )
    assert rel_from_trace("a") == frozenset()
    assert rel_from_trace("") == frozenset()
    assert len(rel_from_trace("abcde")) == 10  # C(5,2): a total order


def test_respects_basics():
    rel = rel_from_trace("abc")
    assert respects(("a", "b", "c"), rel)
    assert not respects(("a", "c", "b"), rel)
    assert not respects(("c", "b", "a"), rel)
    assert respects(("c", "a", "b"), frozenset())  # antichain: anything goes
    assert respects(("a", "b", "c", "d"), N_REL)
    assert not respects(("c", "a", "b", "d"), N_REL)


def test_refines_is_subset_order():
    chain = rel_from_trace("abc")
    sub = frozenset({("a", "c")})
    assert refines(frozenset(), chain)  # antichain refines everything
    assert refines(sub, chain)
    assert not refines(chain, sub)
    assert refines(chain, chain)
    p, q = rel_from_trace("abcd"), rel_from_trace("abdc")
    assert refines(meet(p, q), p) and refines(meet(p, q), q)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_sp_and_non_sp():
    # SP orders get the SP-tree string.
    assert describe(_els(3), rel_from_trace("abc")) == "(a -> b -> c)"
    diamond = tree_relations(series("a", parallel("b", "c"), "d"))
    assert describe(_els(4), diamond) == "(a -> (b || c) -> d)"
    # The N is not SP: cover-relation braces form (sorted covers).
    assert describe(N_ELS, N_REL) == "{a<c, b<c, b<d}"


# ---------------------------------------------------------------------------
# Hypothesis class objects and get_poset_class
# ---------------------------------------------------------------------------


def test_general_and_sp_class_consistency():
    els = _els(4)
    sp_set = set(enumerate_sp(els))
    assert GENERAL.name == "general"
    assert SP.name == "sp"
    assert GENERAL.enumerate(els) == enumerate_posets(els)
    assert SP.enumerate(els) == enumerate_sp(els)
    for rel in enumerate_posets(els):
        assert GENERAL.contains(els, rel)
        assert SP.contains(els, rel) == (rel in sp_set)
        assert GENERAL.extension_count(els, rel) == count_linear_extensions(
            els, rel
        )
        if rel in sp_set:
            assert SP.extension_count(els, rel) == GENERAL.extension_count(
                els, rel
            )


def test_sp_class_rejects_non_sp():
    assert not SP.contains(N_ELS, N_REL)
    with pytest.raises(ValueError):
        SP.extension_count(N_ELS, N_REL)
    with pytest.raises(ValueError):
        SP.sample_extension(N_ELS, N_REL, random.Random(0))


def test_class_sample_extension_consistency():
    els = _els(4)
    rel = frozenset({("a", "b"), ("c", "d")})  # SP: (a->b) || (c->d)
    rng = random.Random(99)
    for _ in range(100):
        t = GENERAL.sample_extension(els, rel, rng)
        assert sorted(t) == sorted(els) and respects(t, rel)
        t = SP.sample_extension(els, rel, rng)
        assert sorted(t) == sorted(els) and respects(t, rel)
    # GENERAL samples non-SP posets too.
    t = GENERAL.sample_extension(N_ELS, N_REL, rng)
    assert respects(t, N_REL)


def test_get_poset_class():
    assert get_poset_class("general") is GENERAL
    assert get_poset_class("sp") is SP
    assert get_poset_class(GENERAL) is GENERAL  # instance passthrough
    assert get_poset_class(SP) is SP
    with pytest.raises(ValueError):
        get_poset_class("bogus")


# ---------------------------------------------------------------------------
# audited-defect guards (DESIGN_REVIEW W12.1, W12.4, W18, W21)
# ---------------------------------------------------------------------------

def test_ideal_dp_refuses_wide_posets():
    # W12.1: the ideal-lattice DP is exponential in width; a wide poset must
    # refuse loudly *before* recursing, not hang or exhaust memory.
    from procposets.rel import (
        IdealBudgetExceeded,
        count_linear_extensions,
        rel_from_trace,
        sample_linear_extension,
    )

    wide = frozenset(f"x{i:02d}" for i in range(30))  # 30-antichain: 2^30 ideals
    with pytest.raises(IdealBudgetExceeded):
        count_linear_extensions(wide, frozenset())
    with pytest.raises(IdealBudgetExceeded):
        sample_linear_extension(wide, frozenset(), random.Random(0))
    # a 30-chain has 31 ideals: fine on the same budget
    chain = tuple(sorted(wide))
    assert count_linear_extensions(wide, rel_from_trace(chain)) == 1


def test_meet_closure_truncation_is_hashseed_deterministic():
    # W12.4: a cap hit used to keep whatever subset the hash seed's BFS order
    # produced; truncation must be identical across PYTHONHASHSEED values.
    import subprocess
    import sys

    script = (
        "from procposets.rel import meet_closure, rel_from_trace\n"
        "import itertools\n"
        "chains = [rel_from_trace(t) for t in"
        " itertools.permutations('abcdef')][:12]\n"
        "closed, hit = meet_closure(chains, cap=20)\n"
        "assert hit\n"
        "print(sorted(sorted(r) for r in closed))\n"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outs) == 1


def test_poset_class_capability_flags():
    # W18: the oracle dispatches on declared capability flags, not pointer
    # identity -- the flags document what the reduction theorem needs.
    from procposets.rel import GeneralPosets, SPPosets

    fresh = GeneralPosets()
    assert fresh.contains_all_posets and fresh.closed_under_meet
    sp = SPPosets()
    assert not sp.contains_all_posets and not sp.closed_under_meet


def test_get_poset_class_rejects_non_protocol_objects():
    # W21: arbitrary objects used to pass straight through get_poset_class
    # and fail deep inside atom construction (or mis-dispatch silently).
    with pytest.raises(TypeError, match="PosetClass"):
        get_poset_class(42)
    with pytest.raises(TypeError, match="missing"):
        get_poset_class(object())
