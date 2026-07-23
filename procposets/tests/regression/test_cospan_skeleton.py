"""Golden cross-check: ``extract_skeleton(...).materialise()`` must equal the
slow ``extract_signature`` **exactly** (as a Signature, i.e. the full concrete
generator frozenset -- ports, labels, everything) on every fixture the slow
engine can reach, for every extraction mode.  This is the migration-discipline
seam for the factored skeleton: the per-arc families ARE the slow pipeline
stopped before ``engine._and``'s cross-arc product, so materialising them must
reproduce it byte-for-byte, including the two terminus modes (per-arc strip
commutes with the union; the pure-terminus collapse is deferred) and the
``kappa`` admissibility filter (applied at pairing time instead of extraction
time).
"""

from __future__ import annotations

import pytest

from procposets.cospan.engine import extract_signature
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.skeleton import extract_skeleton
from procposets.cospan.typebalance import Profile

from .test_cospan_engine_running_example import build_running_example
from .test_cospan_extract_fast import _typed_hub
from .test_cospan_surface_termini import _mixed_graph

_FIXTURES = [
    ("running_example", build_running_example),
    ("mixed_graph", _mixed_graph),
    ("mixed_graph_terminus", lambda: _mixed_graph(terminus_label="gamma2")),
    ("typed_hub", _typed_hub),
]


@pytest.mark.parametrize("name,factory", _FIXTURES, ids=[n for n, _ in _FIXTURES])
@pytest.mark.parametrize("surface", [False, True], ids=["strip", "surface"])
def test_materialise_equals_slow_signature(name, factory, surface):
    g = factory()
    slow = extract_signature(g, surface_termini=surface)
    fact = extract_skeleton(g, surface_termini=surface).materialise()
    missing = slow.generators - fact.generators
    extra = fact.generators - slow.generators
    assert fact == slow, (
        f"[{name}, surface_termini={surface}] materialised skeleton diverges "
        f"from the slow engine:\nMISSING:\n"
        + "\n".join(map(str, sorted(missing, key=str)))
        + "\nEXTRA:\n"
        + "\n".join(map(str, sorted(extra, key=str)))
    )


@pytest.mark.parametrize(
    "kappa",
    [
        {"X": Profile(creates=frozenset({"order"}))},  # licenses X's order creation
        {"X": Profile()},                              # rejects every X context
    ],
    ids=["licensed", "rejecting"],
)
def test_materialise_applies_kappa_like_slow(kappa):
    """``kappa`` deferred to pairing time lands on the identical filtered set as
    the slow engine's extraction-time filter.  ``X`` is fed by bare sources
    (zero-left), so in strip mode its context is ``(empty, {order})`` --
    a creation of ``order`` that needs a licence."""
    g = _mixed_graph()
    slow = extract_signature(g, kappa)
    fact = extract_skeleton(g, kappa).materialise()
    assert fact == slow
    # the two profiles must actually differ on X (the seam is non-vacuous)
    if kappa["X"].creates:
        assert slow.by_label("X")
    else:
        assert not slow.by_label("X")


def _silent_sync_graph():
    """A *synchronising* silent (2-in/2-out SEQ tau, marked ``silent=True``):
    kept, it AND-combines its branches ({A,B} jointly to {C,D} jointly);
    contracted, the XOR places route each branch independently -- so the two
    ``remove_silent`` modes yield genuinely different signatures (the running
    example marks no mediator silent, which would make this seam vacuous)."""
    g = LMGraph()
    for a in ("A", "B", "C", "D"):
        g.add_activity(a)
    for m in ("ma", "mb", "mc", "md"):
        g.add_mediator(m, Kind.XOR)
    g.add_mediator("tau", Kind.SEQ, silent=True)
    g.add_edge("A", "ma", "t"); g.add_edge("ma", "tau", "t")
    g.add_edge("B", "mb", "t"); g.add_edge("mb", "tau", "t")
    g.add_edge("tau", "mc", "t"); g.add_edge("mc", "C", "t")
    g.add_edge("tau", "md", "t"); g.add_edge("md", "D", "t")
    g.validate()
    return g


def test_remove_silent_matches_slow():
    """The silent-contraction preamble is shared (``_prepare_extraction``);
    both modes must agree with the slow engine -- and differ from each other
    (non-vacuity: the silent actually synchronises)."""
    slows = {}
    for remove in (True, False):
        g = _silent_sync_graph()
        slow = extract_signature(g, remove_silent=remove)
        fact = extract_skeleton(g, remove_silent=remove).materialise()
        assert fact == slow
        slows[remove] = slow
    assert slows[True] != slows[False]


def test_factored_generator_counts_stay_per_arc():
    """The skeleton itself is O(sum over arcs), never the product: the running
    example's widest activities keep one family entry per arc."""
    fs = extract_skeleton(build_running_example())
    by_label = {}
    for fg in fs:
        by_label.setdefault(fg.label, []).append(fg)
    # G1: one in-arc from the bare source place (its only alternative the empty
    # bundle -- a zero-left source), two out-arcs (ord fan + item routing)
    (g1,) = by_label["G1"]
    assert len(g1.left) == 1 and g1.left_can_be_empty() and len(g1.right) == 2
    # s_c / s_b stay separate factored entries under the shared label "s"
    assert len(by_label["s"]) == 2
