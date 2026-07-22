"""Repeated-label support: id-based constructor + v2 label-preserving-iso prime canonicalisation."""
from procposets.moddecomp import Prime, decompose, tiling
from procposets.poset import from_edges, leaf, par


def test_from_edges_allows_repeated_labels():
    # an N-shape with TWO 'a's -- from_dag cannot express this (it keys nodes by label)
    P = from_edges({"a1": "a", "b": "b", "a2": "a", "d": "d"},
                   [("a1", "a2"), ("b", "a2"), ("b", "d")])
    assert len(P) == 4
    assert sorted(P.labels.values()) == ["a", "a", "b", "d"]
    assert isinstance(decompose(P), Prime)         # still a prime (non-SP), now with a repeated label


def test_v2_prime_canonical_matches_isomorphic():
    # two label-preserving-isomorphic repeated-label primes, built with different keys/orderings
    P1 = from_edges({"x": "a", "y": "b", "z": "a", "w": "d"},
                    [("x", "z"), ("y", "z"), ("y", "w")])
    P2 = from_edges({"a1": "a", "b0": "b", "a2": "a", "d0": "d"},
                    [("a1", "a2"), ("b0", "a2"), ("b0", "d0")])
    assert decompose(P1).canonical() == decompose(P2).canonical()


def test_v2_prime_canonical_separates_non_isomorphic():
    # same label multiset {a,a,b,d}, DIFFERENT order structure -> must get different canonical strings
    P1 = from_edges({"a1": "a", "b": "b", "a2": "a", "d": "d"},
                    [("a1", "a2"), ("b", "a2"), ("b", "d")])   # b below a2 and d
    P2 = from_edges({"a1": "a", "b": "b", "a2": "a", "d": "d"},
                    [("a1", "a2"), ("a1", "d"), ("b", "a2")])  # a1 below a2 and d
    assert decompose(P1).canonical() != decompose(P2).canonical()


def test_distinct_label_prime_keeps_readable_form():
    # the fast path is unchanged (backward compatible with the readable N{...} string)
    P = from_edges({"a": "a", "b": "b", "c": "c", "d": "d"},
                   [("a", "c"), ("b", "c"), ("b", "d")])
    assert decompose(P).canonical() == "N{a<c, b<c, b<d}"


def test_repeated_label_fibre_counts_in_traces():
    from procposets.traces import trace_distribution
    # a (x) a (x) b -> {aab, aba, baa} each 1/3 (3 distinct words, not 6 element-orderings)
    d = trace_distribution([(par(leaf("a"), leaf("a"), leaf("b")), 1.0)])
    assert set(d) == {("a", "a", "b"), ("a", "b", "a"), ("b", "a", "a")}
    assert all(abs(v - 1 / 3) < 1e-9 for v in d.values())


def test_parallel_repeated_label_block_is_atomic():
    # (a (x) a) is one atomic concurrent block; canonical is order-insensitive
    assert tiling(par(leaf("a"), leaf("a"))) == "(a * a)"
