"""Golden test: the engine reproduces the 22-generator signature of the
canonical running example (``.claude/RUNNING_EXAMPLE.md``).

We build the LM-graph directly from the section-3 OCPN place structure (17
places, mode-refined transitions s_c/s_b, r_i/r_n/r_in, G2_1..6, transparent
tau mediators ta/tcon/tbox) and assert the extracted signature equals the
per-generator interface table g1..g22 exactly.  This is the gate: nothing
downstream is trusted until it passes.
"""

from __future__ import annotations

from procposets.cospan.engine import extract_signature
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.signature import Generator, Port


def build_running_example() -> LMGraph:
    g = LMGraph()
    # absorbing transitions (label defaults to name; modes share a label) ----
    for t in ("G1", "a", "b", "c", "d", "i", "n"):
        g.add_activity(t)
    g.add_activity("s_c", "s")
    g.add_activity("s_b", "s")
    g.add_activity("r_i", "r")
    g.add_activity("r_n", "r")
    g.add_activity("r_in", "r")
    for k in range(1, 7):
        g.add_activity(f"G2_{k}", "G2")
    # transparent silent transitions (pass-through mediators) ----------------
    for tau in ("ta", "tcon", "tbox"):
        g.add_mediator(tau, Kind.SEQ)
    # places: all XOR mediators (choice over consumers / producers) ----------
    for p in range(17):
        g.add_mediator(f"p{p}", Kind.XOR)

    E = g.add_edge
    # p0 source -> G1
    E("p0", "G1")
    # p1 (ord): G1 -> {a, ta}
    E("G1", "p1", "ord"); E("p1", "a", "ord"); E("p1", "ta", "ord")
    # p2 (item routing): G1 -> {tcon, tbox}
    E("G1", "p2"); E("p2", "tcon"); E("p2", "tbox")
    # p3 (con): tcon -> c ; p4 (box): tbox -> d
    E("tcon", "p3", "con"); E("p3", "c", "con")
    E("tbox", "p4", "box"); E("p4", "d", "box")
    # p5 (ord): {a, ta} -> b
    E("a", "p5", "ord"); E("ta", "p5", "ord"); E("p5", "b", "ord")
    # p6 (ord): b -> {s_c, s_b}
    E("b", "p6", "ord"); E("p6", "s_c", "ord"); E("p6", "s_b", "ord")
    # p7 (con): c -> s_c ; p8 (box): d -> s_b
    E("c", "p7", "con"); E("p7", "s_c", "con")
    E("d", "p8", "box"); E("p8", "s_b", "box")
    # p9 (ord): {s_c, s_b} -> {r_i, r_n, r_in}
    E("s_c", "p9", "ord"); E("s_b", "p9", "ord")
    E("p9", "r_i", "ord"); E("p9", "r_n", "ord"); E("p9", "r_in", "ord")
    # p10 (con): s_c -> {G2_1, G2_2, G2_3} ; p11 (box): s_b -> {G2_4, G2_5, G2_6}
    E("s_c", "p10", "con")
    for k in (1, 2, 3):
        E("p10", f"G2_{k}", "con")
    E("s_b", "p11", "box")
    for k in (4, 5, 6):
        E("p11", f"G2_{k}", "box")
    # p12 (ord): {r_i, r_in} -> i ; p13 (ord): {r_n, r_in} -> n
    E("r_i", "p12", "ord"); E("r_in", "p12", "ord"); E("p12", "i", "ord")
    E("r_n", "p13", "ord"); E("r_in", "p13", "ord"); E("p13", "n", "ord")
    # p14 (ord): i -> {G2_1, G2_3, G2_4, G2_6}
    E("i", "p14", "ord")
    for k in (1, 3, 4, 6):
        E("p14", f"G2_{k}", "ord")
    # p15 (ord): n -> {G2_2, G2_3, G2_5, G2_6}
    E("n", "p15", "ord")
    for k in (2, 3, 5, 6):
        E("p15", f"G2_{k}", "ord")
    # p16 sink: {G2_1..6} -> .
    for k in range(1, 7):
        E(f"G2_{k}", "p16")

    g.validate()
    return g


def _gen(label, left, right) -> Generator:
    L = frozenset(Port(*p) for p in left)
    R = frozenset(Port(*p) for p in right)
    return Generator(label, L, R)


# Expected signature transcribed from RUNNING_EXAMPLE.md
# (per-generator interface table; port index 0/16 = empty boundary side).
EXPECTED = {
    _gen("G1", [], [("G1", "ord", "a"), ("G1", "con", "c")]),
    _gen("G1", [], [("G1", "ord", "a"), ("G1", "box", "d")]),
    _gen("G1", [], [("G1", "ord", "b"), ("G1", "con", "c")]),
    _gen("G1", [], [("G1", "ord", "b"), ("G1", "box", "d")]),
    _gen("a", [("G1", "ord", "a")], [("a", "ord", "b")]),
    _gen("b", [("a", "ord", "b")], [("b", "ord", "s")]),
    _gen("b", [("G1", "ord", "b")], [("b", "ord", "s")]),
    _gen("c", [("G1", "con", "c")], [("c", "con", "s")]),
    _gen("d", [("G1", "box", "d")], [("d", "box", "s")]),
    _gen("s", [("b", "ord", "s"), ("c", "con", "s")], [("s", "ord", "r"), ("s", "con", "G2")]),
    _gen("s", [("b", "ord", "s"), ("d", "box", "s")], [("s", "ord", "r"), ("s", "box", "G2")]),
    _gen("r", [("s", "ord", "r")], [("r", "ord", "i")]),
    _gen("r", [("s", "ord", "r")], [("r", "ord", "n")]),
    _gen("r", [("s", "ord", "r")], [("r", "ord", "i"), ("r", "ord", "n")]),
    _gen("i", [("r", "ord", "i")], [("i", "ord", "G2")]),
    _gen("n", [("r", "ord", "n")], [("n", "ord", "G2")]),
    _gen("G2", [("i", "ord", "G2"), ("s", "con", "G2")], []),
    _gen("G2", [("n", "ord", "G2"), ("s", "con", "G2")], []),
    _gen("G2", [("i", "ord", "G2"), ("n", "ord", "G2"), ("s", "con", "G2")], []),
    _gen("G2", [("i", "ord", "G2"), ("s", "box", "G2")], []),
    _gen("G2", [("n", "ord", "G2"), ("s", "box", "G2")], []),
    _gen("G2", [("i", "ord", "G2"), ("n", "ord", "G2"), ("s", "box", "G2")], []),
}


def test_running_example_22_generators():
    sig = extract_signature(build_running_example())
    got = set(sig.generators)
    assert len(EXPECTED) == 22
    missing = EXPECTED - got
    extra = got - EXPECTED
    assert not missing, "missing generators:\n" + "\n".join(map(str, sorted(missing, key=str)))
    assert not extra, "unexpected generators:\n" + "\n".join(map(str, sorted(extra, key=str)))
    assert len(sig) == 22


def test_per_label_counts():
    sig = extract_signature(build_running_example())
    counts = {lab: len(sig.by_label(lab)) for lab in sig.labels()}
    assert counts == {
        "G1": 4, "a": 1, "b": 2, "c": 1, "d": 1,
        "s": 2, "r": 3, "i": 1, "n": 1, "G2": 6,
    }
