"""Phase-1 golden cross-checks: procposets' A0 poset core must reproduce the
ORIGINAL implementations value-for-value before the originals are deleted.

The originals still live in their repos; this test imports both and asserts
equality on a fixed corpus.  It is skipped (not failed) if the sibling repos
are not checked out, so procposets stays independently testable.
"""

import importlib.util
import itertools
import pathlib
import random

import pytest

RESEARCH = pathlib.Path("/home/arl/Research")
NPMLE = RESEARCH / "poset-mixture-npmle"
SPME = RESEARCH / "stochastic_process_mining" / "experiments"


def _load(name, path):
    if not path.is_file():
        pytest.skip(f"original not checked out: {path}")
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses (SPTree) resolve via sys.modules
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# NPMLE Rel engine (procposets.rel) == poset_mixture.posets
# ---------------------------------------------------------------------------

def _orig_npmle():
    # posets.py is self-contained (stdlib only), so load it directly
    return _load("orig_npmle_posets", NPMLE / "poset_mixture" / "posets.py")


ALPHABETS = [tuple("abc"), tuple("abcd"), tuple("abcde")]


def test_enumerate_posets_matches_npmle():
    from procposets import rel as new
    old = _orig_npmle()
    for alpha in (tuple("abc"), tuple("abcd")):
        a = sorted(set(new.enumerate_posets(alpha)))
        b = sorted(set(old.enumerate_posets(alpha)))
        assert a == b, f"enumerate_posets differs on {alpha}"


def test_count_linear_extensions_matches_npmle():
    from procposets import rel as new
    old = _orig_npmle()
    alpha = frozenset("abcd")
    for r in new.enumerate_posets(alpha):
        assert (new.count_linear_extensions(alpha, r)
                == old.count_linear_extensions(alpha, r))


def test_meet_and_transitive_reduction_match_npmle():
    from procposets import rel as new
    old = _orig_npmle()
    alpha = frozenset("abcd")
    rels = new.enumerate_posets(alpha)
    rng = random.Random(0)
    for _ in range(200):
        r1, r2 = rng.choice(rels), rng.choice(rels)
        assert new.meet(r1, r2) == old.meet(r1, r2)
        assert new.transitive_reduction(r1) == old.transitive_reduction(r1)


def test_decompose_and_sp_tree_match_npmle():
    from procposets import rel as new
    old = _orig_npmle()
    alpha = frozenset("abcd")
    for r in new.enumerate_posets(alpha):
        tn, to = new.decompose(alpha, r), old.decompose(alpha, r)
        assert (tn is None) == (to is None)
        if tn is not None:
            assert str(tn) == str(to)                      # ->/|| renderer, byte-identical
            assert new.extension_count(tn) == old.extension_count(to)
            assert new.tree_relations(tn) == old.tree_relations(to)
        assert new.is_sp(alpha, r) == old.is_sp(alpha, r)


def test_sample_linear_extension_matches_npmle():
    from procposets import rel as new
    old = _orig_npmle()
    alpha = frozenset("abcd")
    diamond = new.tree_relations(new.series("a", new.parallel("b", "c"), "d"))
    for seed in range(20):
        assert (new.sample_linear_extension(alpha, diamond, random.Random(seed))
                == old.sample_linear_extension(alpha, diamond, random.Random(seed)))


def test_describe_matches_npmle():
    from procposets import rel as new
    old = _orig_npmle()
    alpha = frozenset("abcd")
    for r in new.enumerate_posets(alpha):
        assert new.describe(alpha, r) == old.describe(alpha, r)


# ---------------------------------------------------------------------------
# SPME Poset / moddecomp / traces  (procposets.{poset,moddecomp,traces})
# ---------------------------------------------------------------------------

def _orig_spm(mod):
    # spm modules are a package (relative imports); add experiments/ to path
    import sys
    if str(SPME) not in sys.path:
        sys.path.insert(0, str(SPME))
    if not (SPME / "spm" / f"{mod}.py").is_file():
        pytest.skip(f"SPME not checked out: {mod}")
    return importlib.import_module(f"spm.{mod}")


def test_spme_moddecomp_canonical_strings_match():
    import procposets.moddecomp as new
    import procposets.poset as npos
    old = _orig_spm("moddecomp")
    opos = _orig_spm("poset")
    # build the same shapes on both and compare the ;/* canonical render
    for build_new, build_old in [
        (lambda: npos.then(npos.leaf("a"), npos.par(npos.leaf("b"), npos.leaf("c")), npos.leaf("d")),
         lambda: opos.then(opos.leaf("a"), opos.par(opos.leaf("b"), opos.leaf("c")), opos.leaf("d"))),
        (lambda: npos.n_poset(), lambda: opos.n_poset()),
    ]:
        assert new.tiling(build_new()) == old.tiling(build_old())


def test_spme_linear_extensions_match():
    import procposets.traces as new
    import procposets.poset as npos
    old = _orig_spm("traces")
    opos = _orig_spm("poset")
    pn = npos.then(npos.leaf("a"), npos.par(npos.leaf("b"), npos.leaf("c")), npos.leaf("d"))
    po = opos.then(opos.leaf("a"), opos.par(opos.leaf("b"), opos.leaf("c")), opos.leaf("d"))
    assert sorted(new.linear_extensions(pn)) == sorted(old.linear_extensions(po))


# ---------------------------------------------------------------------------
# The Rel <-> Poset bridge (new capability; internal consistency)
# ---------------------------------------------------------------------------

def test_bridge_roundtrip_on_distinct_labels():
    from procposets import bridge, rel, poset
    alpha = frozenset("abcd")
    for r in rel.enumerate_posets(alpha):
        P = bridge.from_rel(alpha, r)
        assert bridge.to_rel(P) == r
        # element labels round-trip
        assert bridge.rel_elements(P) == alpha


def test_bridge_refuses_repeated_labels():
    from procposets import bridge, poset
    # an N with two 'a's: not representable as a Rel
    P = poset.from_edges({"a1": "a", "b": "b", "a2": "a", "d": "d"},
                         [("a1", "a2"), ("b", "a2"), ("b", "d")])
    with pytest.raises(bridge.LabelCollision):
        bridge.to_rel(P)


def test_bridge_count_agrees_with_rel_engine():
    """A distinct-label poset's extension count via the Rel engine equals its
    count via the SPME word-enumerator through the bridge."""
    from procposets import bridge, rel, traces
    alpha = frozenset("abcd")
    for r in rel.enumerate_posets(alpha):
        n_rel = rel.count_linear_extensions(alpha, r)
        n_words = len(traces.linear_extensions(bridge.from_rel(alpha, r)))
        assert n_rel == n_words
