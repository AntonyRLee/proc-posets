"""Tests for procposets.grouping and procposets.simulate.

Oracle style: brute-force linear-extension sets via itertools.permutations +
respects(), analytic exponential-clock means, and exact hand-built blockings.
All rngs are seeded; statistical tolerances are ~5 sigma.
"""

import itertools
import random
import re
from math import sqrt


from procposets import (
    TrueMixture,
    group_by_key,
    parallel,
    respects,
    sample_grouped_log,
    sample_keyed_log,
    sample_timed_grouped_log,
    series,
    tree_relations,
)

# ---------------------------------------------------------------------------
# Fixtures / ground truth
# ---------------------------------------------------------------------------

ALPHABET = ("a", "b", "c", "d")

P1 = series("a", parallel("b", "c"), "d")  # a -> (b || c) -> d, e = 2
P2 = series("a", "c", "b", "d")            # chain a c b d,      e = 1
CHAIN_FWD = series("a", "b", "c", "d")     # chain a b c d
CHAIN_REV = series("d", "c", "b", "a")     # chain d c b a


def brute_extensions(tree):
    """All linear extensions of an SP tree, by brute force over permutations."""
    rel = tree_relations(tree)
    els = sorted(tree.elements())
    return {p for p in itertools.permutations(els) if respects(p, rel)}


def test_brute_extension_oracle_sanity():
    # e(P1) = 2 (the two interleavings of b, c); the chains have exactly one.
    assert brute_extensions(P1) == {("a", "b", "c", "d"), ("a", "c", "b", "d")}
    assert brute_extensions(P2) == {("a", "c", "b", "d")}
    assert brute_extensions(CHAIN_FWD) == {("a", "b", "c", "d")}
    assert brute_extensions(CHAIN_REV) == {("d", "c", "b", "a")}


# ---------------------------------------------------------------------------
# group_by_key
# ---------------------------------------------------------------------------

def test_group_by_key_blocking_and_leftovers(capsys):
    t = [(f"t{i}",) for i in range(9)]  # distinct one-element traces
    rows = [
        ("k2", t[0]),
        ("k1", t[1]),
        ("k2", t[2]),
        ("k3", t[3]),
        ("k1", t[4]),
        ("k2", t[5]),
        ("k1", t[6]),
        ("k3", t[7]),
        ("k1", t[8]),
    ]
    # sizes: k1 -> 4, k2 -> 3, k3 -> 2; default min_group_size = 3
    groups, leftovers = group_by_key(rows, "unit-test assumption: shared key")
    # groups in sorted key order, traces in row order within each bucket
    assert groups == [
        [t[1], t[4], t[6], t[8]],  # k1
        [t[0], t[2], t[5]],        # k2
    ]
    assert leftovers == [[t[3], t[7]]]  # k3
    out = capsys.readouterr().out
    assert "[grouping] declared assumption: unit-test assumption: shared key" in out
    assert "2 groups of size >= 3" in out
    assert "1 undersized blocks set aside" in out


def test_group_by_key_min_group_size_boundary(capsys):
    rows = [("a", (1,)), ("a", (2,)), ("b", (3,))]
    # min_group_size = 2: exactly size-2 buckets qualify (>= is inclusive)
    groups, leftovers = group_by_key(rows, "x", min_group_size=2)
    assert groups == [[(1,), (2,)]]
    assert leftovers == [[(3,)]]
    # min_group_size = 1: nothing is left over
    groups, leftovers = group_by_key(rows, "x", min_group_size=1)
    assert groups == [[(1,), (2,)], [(3,)]]
    assert leftovers == []
    # min_group_size = 5: everything is left over
    groups, leftovers = group_by_key(rows, "x", min_group_size=5)
    assert groups == []
    assert leftovers == [[(1,), (2,)], [(3,)]]
    out = capsys.readouterr().out
    assert "0 groups of size >= 5" in out


def test_group_by_key_deterministic_sorted_key_order(capsys):
    # Buckets come out in sorted-key order regardless of row arrival order.
    keys = ["z", "m", "a", "q"]
    rows = [(k, (k, str(i))) for k in keys for i in range(3)]
    rng = random.Random(7)
    for _ in range(5):
        rng.shuffle(rows)
        groups, leftovers = group_by_key(rows, "order test")
        assert leftovers == []
        assert [g[0][0] for g in groups] == sorted(keys)
        for g in groups:
            assert len(g) == 3
            assert all(tr[0] == g[0][0] for tr in g)
    capsys.readouterr()  # drain


def test_group_by_key_iterable_input(capsys):
    # Accepts any iterable, not just lists.
    rows = iter([("k", ("x",)), ("k", ("y",)), ("k", ("z",))])
    groups, leftovers = group_by_key(rows, "generator input", min_group_size=3)
    assert groups == [[("x",), ("y",), ("z",)]]
    assert leftovers == []
    capsys.readouterr()


# ---------------------------------------------------------------------------
# TrueMixture.sample_component / sample_trace
# ---------------------------------------------------------------------------

def test_sample_component_frequencies_match_weights():
    mix = TrueMixture(trees=[CHAIN_FWD, P1, P2], weights=[0.2, 0.3, 0.5])
    rng = random.Random(42)
    n = 20_000
    counts = [0, 0, 0]
    for _ in range(n):
        k = mix.sample_component(rng)
        assert k in (0, 1, 2)
        counts[k] += 1
    for k, w in enumerate(mix.weights):
        sigma = sqrt(w * (1 - w) / n)  # <= 0.0035 here
        assert abs(counts[k] / n - w) < 5 * sigma + 0.005, (k, counts)


def test_sample_trace_no_noise_is_uniform_extension():
    mix = TrueMixture(trees=[P1], weights=[1.0])
    rng = random.Random(0)
    exts = brute_extensions(P1)
    n = 4000
    seen = {e: 0 for e in exts}
    for _ in range(n):
        t = mix.sample_trace(0, rng, eps_sim=0.0)
        assert t in exts  # never leaves the extension set
        seen[t] += 1
    # uniform over the 2 extensions: 0.5 +- 5 sigma (sigma ~ 0.0079)
    for e in exts:
        assert abs(seen[e] / n - 0.5) < 0.045, seen


def test_sample_trace_eps_one_always_one_adjacent_transposition():
    # Chain component: the noiseless trace is unique, so eps_sim = 1.0 must
    # yield exactly the base with one adjacent swap applied -- Kendall
    # distance exactly 1, i.e. one of the 3 single-swap neighbours.
    base = ("a", "b", "c", "d")
    neighbours = {
        ("b", "a", "c", "d"),
        ("a", "c", "b", "d"),
        ("a", "b", "d", "c"),
    }
    mix = TrueMixture(trees=[CHAIN_FWD], weights=[1.0])
    rng = random.Random(11)
    seen = set()
    for _ in range(300):
        t = mix.sample_trace(0, rng, eps_sim=1.0)
        assert t != base  # never the un-swapped trace
        assert t in neighbours  # exactly one adjacent transposition
        seen.add(t)
    # the swap position is uniform over the 3 slots: all appear in 300 draws
    assert seen == neighbours


def test_sample_trace_eps_zero_never_swaps():
    mix = TrueMixture(trees=[P2], weights=[1.0])
    rng = random.Random(3)
    for _ in range(200):
        assert mix.sample_trace(0, rng, eps_sim=0.0) == ("a", "c", "b", "d")


def test_sample_trace_eps_one_general_component_distance_at_most_one():
    # For a non-chain component the swap may land back inside the extension
    # set; but the output is always within one adjacent transposition of some
    # extension, and is always a permutation of the alphabet.
    mix = TrueMixture(trees=[P1], weights=[1.0])
    rng = random.Random(5)
    exts = brute_extensions(P1)
    one_swap = set()
    for e in exts:
        for i in range(3):
            t = list(e)
            t[i], t[i + 1] = t[i + 1], t[i]
            one_swap.add(tuple(t))
    for _ in range(300):
        t = mix.sample_trace(0, rng, eps_sim=1.0)
        assert sorted(t) == list(ALPHABET)
        assert t in one_swap


def test_sample_trace_single_element_tree():
    # len(t) <= 1 guard: swapping is impossible, trace is returned unchanged.
    mix = TrueMixture(trees=[series("a")], weights=[1.0])
    rng = random.Random(1)
    assert mix.sample_trace(0, rng, eps_sim=1.0) == ("a",)


# ---------------------------------------------------------------------------
# sample_grouped_log
# ---------------------------------------------------------------------------

def test_sample_grouped_log_shapes_and_purity():
    mix = TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
    rels = [tree_relations(t) for t in mix.trees]
    G, n_g = 40, 5
    groups, z = sample_grouped_log(mix, G, n_g, seed=123, eps_sim=0.0, eta_sim=0.0)
    assert len(groups) == G
    assert len(z) == G
    assert all(k in (0, 1) for k in z)
    assert {0, 1} <= set(z)  # both components drawn at these weights/seed
    for g, k in zip(groups, z):
        assert len(g) == n_g
        for tr in g:
            assert sorted(tr) == list(ALPHABET)  # permutation of the alphabet
            assert respects(tr, rels[k])  # extension of the group's component


def test_sample_grouped_log_deterministic_in_seed():
    mix = TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
    a = sample_grouped_log(mix, 10, 4, seed=99, eps_sim=0.3, eta_sim=0.3)
    b = sample_grouped_log(mix, 10, 4, seed=99, eps_sim=0.3, eta_sim=0.3)
    assert a == b


def test_sample_grouped_log_eta_mixes_at_declared_rate():
    # Two reversed chains: a trace respects its group's relations iff it was
    # NOT replaced by the other component, so the violation rate is exactly
    # eta_sim * P(fresh component differs) = eta_sim * 0.5.
    mix = TrueMixture(trees=[CHAIN_FWD, CHAIN_REV], weights=[0.5, 0.5])
    rels = [tree_relations(t) for t in mix.trees]
    G, n_g, eta = 400, 5, 0.5
    groups, z = sample_grouped_log(mix, G, n_g, seed=7, eps_sim=0.0, eta_sim=eta)
    n = G * n_g
    violations = sum(
        not respects(tr, rels[k]) for g, k in zip(groups, z) for tr in g
    )
    # every trace is still one of the two chains
    for g in groups:
        for tr in g:
            assert tr in {("a", "b", "c", "d"), ("d", "c", "b", "a")}
    p = eta * 0.5
    sigma = sqrt(p * (1 - p) / n)  # ~ 0.0097
    assert abs(violations / n - p) < 5 * sigma, violations / n
    assert violations > 0  # eta_sim > 0 really does mix


def test_sample_grouped_log_eta_zero_never_mixes():
    mix = TrueMixture(trees=[CHAIN_FWD, CHAIN_REV], weights=[0.5, 0.5])
    rels = [tree_relations(t) for t in mix.trees]
    groups, z = sample_grouped_log(mix, 60, 5, seed=21, eps_sim=0.0, eta_sim=0.0)
    for g, k in zip(groups, z):
        for tr in g:
            assert respects(tr, rels[k])


# ---------------------------------------------------------------------------
# sample_keyed_log
# ---------------------------------------------------------------------------

KEY_RE = re.compile(r"^block(\d{3})$")


def blocks_of(rows):
    out = {}
    for key, tr in rows:
        out.setdefault(key, []).append(tr)
    return out


def test_sample_keyed_log_int_traces_per_block():
    mix = TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
    n_blocks, per = 20, 4
    rows = sample_keyed_log(mix, n_blocks, per, defect_prob=0.1, seed=5)
    assert len(rows) == n_blocks * per
    blocks = blocks_of(rows)
    assert len(blocks) == n_blocks
    ids = set()
    for key, traces in blocks.items():
        m = KEY_RE.match(key)
        assert m, key  # keys well-formed: block###
        ids.add(int(m.group(1)))
        assert len(traces) == per
        for tr in traces:
            assert sorted(tr) == list(ALPHABET)
    assert ids == set(range(n_blocks))  # contiguous block ids 0..n_blocks-1


def test_sample_keyed_log_range_traces_per_block():
    mix = TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
    n_blocks, lo, hi = 50, 2, 6
    rows = sample_keyed_log(mix, n_blocks, (lo, hi), defect_prob=0.0, seed=8)
    blocks = blocks_of(rows)
    assert len(blocks) == n_blocks
    sizes = [len(traces) for traces in blocks.values()]
    assert all(lo <= s <= hi for s in sizes)  # inclusive (lo, hi) range
    assert len(set(sizes)) > 1  # volume actually varies
    assert n_blocks * lo <= len(rows) <= n_blocks * hi
    assert {int(KEY_RE.match(k).group(1)) for k in blocks} == set(range(n_blocks))


def test_sample_keyed_log_defect_zero_blocks_pure():
    # With defect_prob = 0 and no recording noise, every block's traces are
    # extensions of a SINGLE component.  Reversed chains make the check sharp:
    # a block is pure iff all its traces are identical.
    mix = TrueMixture(trees=[CHAIN_FWD, CHAIN_REV], weights=[0.5, 0.5])
    rels = [tree_relations(t) for t in mix.trees]
    rows = sample_keyed_log(mix, 40, (3, 6), defect_prob=0.0, seed=13)
    for key, traces in blocks_of(rows).items():
        assert any(
            all(respects(tr, rel) for tr in traces) for rel in rels
        ), key


def test_sample_keyed_log_defect_prob_produces_impure_blocks():
    mix = TrueMixture(trees=[CHAIN_FWD, CHAIN_REV], weights=[0.5, 0.5])
    rels = [tree_relations(t) for t in mix.trees]
    rows = sample_keyed_log(mix, 40, 6, defect_prob=0.5, seed=17)
    impure = sum(
        not any(all(respects(tr, rel) for tr in traces) for rel in rels)
        for traces in blocks_of(rows).values()
    )
    assert impure > 0  # defectors really appear


# ---------------------------------------------------------------------------
# sample_timed_grouped_log
# ---------------------------------------------------------------------------

def test_sample_timed_grouped_log_shapes_and_validity():
    mix = TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
    rels = [tree_relations(t) for t in mix.trees]
    G, n_g = 30, 4
    groups, z = sample_timed_grouped_log(
        mix, G, n_g, lams=[1.0, 2.0], seed=2, eta_sim=0.0
    )
    assert len(groups) == G
    assert len(z) == G
    assert all(k in (0, 1) for k in z)
    for g, k in zip(groups, z):
        assert len(g) == n_g
        for trace, gaps in g:
            assert isinstance(trace, tuple) and isinstance(gaps, tuple)
            assert len(gaps) == len(trace) == len(ALPHABET)  # one gap/activity
            assert sorted(trace) == list(ALPHABET)
            assert all(gp > 0 for gp in gaps)
            assert respects(trace, rels[k])  # trace extends the component


def test_sample_timed_grouped_log_chain_gaps_are_exponential():
    # Pure chain: exactly one activity is enabled at every step, so each of
    # the 4 gaps is Exp(lam) with mean 1/lam.
    lam = 2.0
    mix = TrueMixture(trees=[CHAIN_FWD], weights=[1.0])
    G, n_g = 250, 5
    groups, z = sample_timed_grouped_log(mix, G, n_g, lams=[lam], seed=31)
    assert set(z) == {0}
    n = G * n_g
    for j in range(4):
        vals = [gaps[j] for g in groups for (trace, gaps) in g]
        # every trace of a chain component is the chain itself
        assert all(trace == ("a", "b", "c", "d") for g in groups for (trace, _) in g)
        mean = sum(vals) / n
        sigma = (1 / lam) / sqrt(n)  # Exp sd = mean; ~0.0141 here
        assert abs(mean - 1 / lam) < 5 * sigma, (j, mean)


def test_sample_timed_grouped_log_parallel_step_halves_the_gap():
    # P1 = a -> (b || c) -> d.  Enabled-set sizes along any trace: 1,2,1,1,
    # so the gap after 'a' races two Exp(lam) clocks -> Exp(2 lam), mean
    # 1/(2 lam); the other three gaps have mean 1/lam.
    lam = 1.5
    mix = TrueMixture(trees=[P1], weights=[1.0])
    G, n_g = 250, 5
    groups, _ = sample_timed_grouped_log(mix, G, n_g, lams=[lam], seed=37)
    n = G * n_g
    per_pos = [[gaps[j] for g in groups for (_, gaps) in g] for j in range(4)]
    expected = [1 / lam, 1 / (2 * lam), 1 / lam, 1 / lam]
    for j, (vals, mu) in enumerate(zip(per_pos, expected)):
        mean = sum(vals) / n
        sigma = mu / sqrt(n)
        assert abs(mean - mu) < 5 * sigma, (j, mean, mu)
    # and the finisher of the b/c race is a fair coin: both orders appear
    # at ~50% (uniform over the enabled set).
    n_bc = sum(tr.index("b") < tr.index("c") for g in groups for (tr, _) in g)
    sigma = sqrt(0.25 / n)
    assert abs(n_bc / n - 0.5) < 5 * sigma, n_bc / n


def test_sample_timed_grouped_log_eta_mixes():
    # Reversed chains again: with eta_sim > 0 some timed traces inside a
    # group come from the other component.
    mix = TrueMixture(trees=[CHAIN_FWD, CHAIN_REV], weights=[0.5, 0.5])
    rels = [tree_relations(t) for t in mix.trees]
    G, n_g, eta = 200, 5, 0.5
    groups, z = sample_timed_grouped_log(
        mix, G, n_g, lams=[1.0, 1.0], seed=41, eta_sim=eta
    )
    n = G * n_g
    violations = sum(
        not respects(trace, rels[k])
        for g, k in zip(groups, z)
        for (trace, _) in g
    )
    p = eta * 0.5
    sigma = sqrt(p * (1 - p) / n)
    assert violations > 0
    assert abs(violations / n - p) < 5 * sigma, violations / n
