"""OCCN.filtered (opt-in binding filter) + occn_generator_counts.

A hand-built OCCN with skewed binding counts: `a` has two output structures
(frequent -> b, rare -> c). Filtering must drop only the rare alternative,
never an activity's last group; counts must align generator-for-generator
with occn_to_signature's construction.
"""
from procposets.occn import OCCN, Marker, occn_generator_counts, occn_to_signature
from procposets.occn.fhm import OCDG


def _occn() -> OCCN:
    ocdg = OCDG(activities=frozenset({"a", "b", "c"}), otypes=("t",),
                arcs=frozenset(), starts={"t": "START_t"}, ends={"t": "END_t"})
    m = lambda act: frozenset({Marker(act, "t", 1, 1, 0)})  # noqa: E731
    return OCCN(
        ocdg,
        input_groups={"a": [], "b": [(m("a"), 90)], "c": [(m("a"), 10)]},
        output_groups={"a": [(m("b"), 90), (m("c"), 10)], "b": [], "c": []},
    )


def test_filtered_drops_rare_group_only():
    occn = _occn()
    f = occn.filtered(min_rel=0.2)  # 10/100 < 0.2 -> rare a->c output goes
    assert [c for _, c in f.output_groups["a"]] == [90]
    # consumers' input groups are filtered per-activity: c's only group survives
    # (most-frequent-group guard: filtering never erases a side entirely)
    assert [c for _, c in f.input_groups["c"]] == [10]
    assert occn.output_groups["a"] != f.output_groups["a"]  # original untouched


def test_filtered_min_count_and_guard():
    occn = _occn()
    f = occn.filtered(min_count=1000)  # everything below -> guard keeps top-1
    assert [c for _, c in f.output_groups["a"]] == [90]
    assert [c for _, c in f.input_groups["b"]] == [90]


def test_counts_align_with_signature():
    occn = _occn()
    sig = occn_to_signature(occn, bindings=True)
    counts = occn_generator_counts(occn, bindings=True)
    interior = {g for g in sig.generators
                if not g.label.startswith(("START_", "END_"))}
    assert set(counts) == interior
    by_label = {}
    for g, (ic, oc) in counts.items():
        by_label.setdefault(g.label, set()).add((ic, oc))
    assert by_label["a"] == {(None, 90), (None, 10)}
    assert by_label["b"] == {(90, None)}


def test_filtered_signature_is_subset():
    occn = _occn()
    full = occn_to_signature(occn)
    thin = occn_to_signature(occn.filtered(min_rel=0.2))
    assert thin.generators < full.generators
