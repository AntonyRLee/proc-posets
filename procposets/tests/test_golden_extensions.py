"""Polish check: one shared guarded ideal-lattice engine (procposets._extensions)
now backs BOTH the Rel view and the canonical Poset.

Pins that the de-dup changed no values: the shared engine agrees with the
word enumerator and with the original NPMLE Rel counter, on the same posets;
and the canonical Poset gains a guarded counter that refuses (rather than
hangs on) a too-wide poset.
"""

import pathlib
import random
import sys

import pytest

NPMLE = pathlib.Path("/home/arl/Research/poset-mixture-npmle")


def test_three_counters_agree_on_distinct_label_posets():
    """count_extensions(Poset) == len(traces.linear_extensions(Poset))
    == rel.count_linear_extensions(elements, rel) -- one engine, three views."""
    from procposets import rel, bridge, poset, traces
    alpha = frozenset("abcd")
    for r in rel.enumerate_posets(alpha):
        P = bridge.from_rel(alpha, r)
        n_engine = poset.count_extensions(P)
        n_words = len(traces.linear_extensions(P))
        n_rel = rel.count_linear_extensions(alpha, r)
        assert n_engine == n_words == n_rel


def test_rel_counter_still_matches_npmle_after_dedup():
    """The re-export must not have changed the Rel counter's values."""
    if not (NPMLE / "poset_mixture" / "posets.py").is_file():
        pytest.skip("NPMLE not checked out")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "orig_np_posets", NPMLE / "poset_mixture" / "posets.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["orig_np_posets"] = old
    spec.loader.exec_module(old)
    from procposets import rel
    alpha = frozenset("abcde")
    rng = random.Random(0)
    sample = [rng.choice(rel.enumerate_posets(alpha)) for _ in range(80)]
    for r in sample:
        assert (rel.count_linear_extensions(alpha, r)
                == old.count_linear_extensions(alpha, r))


def test_poset_counter_handles_repeated_labels():
    """The canonical Poset's guarded counter works where a Rel cannot -- a
    poset with two identically-labelled elements (ids are distinct)."""
    from procposets import poset
    # an N with two 'a's: a1 < a2, b < a2, b < d  (4 distinct elements)
    P = poset.from_edges({"a1": "a", "b": "b", "a2": "a", "d": "d"},
                         [("a1", "a2"), ("b", "a2"), ("b", "d")])
    assert poset.count_extensions(P) >= 1
    # sampling returns a permutation of the element ids
    ext = poset.sample_extension(P, random.Random(1))
    assert sorted(ext) == sorted(P.elements)


def test_guard_refuses_a_too_wide_poset():
    """The guard travels with the counter: a very wide antichain trips
    IdealBudgetExceeded rather than hanging."""
    from procposets import poset
    from procposets._extensions import IdealBudgetExceeded
    # 30-element antichain: 2^30 ideals >> MAX_IDEAL_STATES
    P = poset.par(*[poset.leaf(f"x{i}") for i in range(30)])
    with pytest.raises(IdealBudgetExceeded):
        poset.count_extensions(P)


def test_linear_extensions_refuses_materialisation_over_budget():
    """Phase-7 #6: the materialisation guard closes the gap the ideal-budget guard
    leaves open.  A width-13 antichain PASSES count_extensions' ideal guard (2^13
    ideals << MAX_IDEAL_STATES) and its e(P) is returned cheaply, but has 13! (~6e9)
    words -- so traces.linear_extensions refuses rather than exhausting memory."""
    from procposets import poset, traces
    from procposets._extensions import IdealBudgetExceeded
    P = poset.par(*[poset.leaf(f"x{i}") for i in range(13)])
    # count_extensions stays cheap and does NOT raise: 2^13 ideals is under budget,
    # even though e(P) = 13! is huge (the ideal-DP cost is ~2^width, not e(P)).
    assert poset.count_extensions(P) == 6_227_020_800          # 13!
    assert 6_227_020_800 > traces.MAX_LINEAR_EXTENSIONS         # precondition
    # ... but materialising 13! words is refused (the OOM this fix prevents).
    with pytest.raises(IdealBudgetExceeded, match="MAX_LINEAR_EXTENSIONS"):
        traces.linear_extensions(P)
    # trace_distribution calls linear_extensions per variant, so it refuses too.
    with pytest.raises(IdealBudgetExceeded):
        traces.trace_distribution([(P, 1.0)])


def test_linear_extensions_below_budget_byte_unchanged():
    """Below the materialisation budget the guard is transparent: the enumerated
    word list is identical (and equal in length to the cheap e(P) count), so the
    guard changed no values on any realistic input."""
    from procposets import poset, traces
    P = poset.par(*[poset.leaf(f"x{i}") for i in range(6)])   # 6! = 720 words
    les = traces.linear_extensions(P)
    assert len(les) == poset.count_extensions(P) == 720
    assert len(set(les)) == 720                              # all distinct (distinct labels)


def test_transitive_closure_helper_is_shared():
    """from_dag and from_edges route through the one helper and still build
    the correct closed order."""
    from procposets import poset, bridge
    P = poset.from_dag([("a", "b"), ("b", "c")])
    # a<b, b<c  =>  a<c must be present (closure)
    r = bridge.to_rel(P)
    assert ("a", "c") in r and ("a", "b") in r and ("b", "c") in r
