"""Context-depth (VLMC order-k) tests for the block-transition matrix."""
from procposets.distance import smd
from procposets.matrix import END, START, build
from procposets.poset import leaf, par, then

# a repeated-label sequence: the two 'a' occurrences conflate at depth 1, separate at depth 2.
REP = [(then(leaf("a"), leaf("b"), leaf("a"), leaf("c")), 1.0)]


def test_depth1_is_backward_compatible():
    # depth-1 reproduces the memoryless block chain exactly (states = bare block labels)
    m, states = build(REP, context_depth=1)
    assert states == {START, END, "a", "b", "c"}
    assert m["a"] == {"b": 0.5, "c": 0.5}   # the two a-occurrences merged into one row
    assert m[START] == {"a": 1.0}
    assert m["b"] == {"a": 1.0}
    assert m["c"] == {END: 1.0}


def test_depth2_separates_repeated_block():
    # depth-2 keys the second 'a' by its predecessor 'b', so no conflation
    m, states = build(REP, context_depth=2)
    assert "a" in states and "b|a" in states          # first a vs second a are distinct states
    assert m["a"] == {"a|b": 1.0}                      # first a -> deterministic
    assert m["b|a"] == {"a|c": 1.0}                    # second a -> deterministic
    assert "a" not in m["b|a"]                         # the conflated {b:.5,c:.5} row is gone


def test_depth1_default_matches_explicit():
    assert build(REP)[0] == build(REP, context_depth=1)[0]


def test_self_distance_zero_at_any_depth():
    for k in (1, 2, 3):
        assert smd(REP, REP, context_depth=k)[0] < 1e-12


def test_deeper_context_is_sharper_or_equal():
    # two models sharing a prefix but differing later: deeper context never *decreases* separation
    A = [(then(leaf("a"), leaf("b"), leaf("a"), leaf("c")), 1.0)]
    B = [(then(leaf("a"), leaf("b"), leaf("a"), leaf("d")), 1.0)]
    d1 = smd(A, B, context_depth=1)[0]
    d2 = smd(A, B, context_depth=2)[0]
    assert d2 >= d1 - 1e-12


def test_concurrency_stays_atomic_at_depth2():
    # a parallel block is one token at every depth (orthogonality of the two knobs)
    conc = [(par(leaf("a"), leaf("b")), 1.0)]
    _, states = build(conc, context_depth=2)
    assert "(a * b)" in states
