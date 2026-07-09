"""Validate the Sigma -> composite-diagrams composer against the canonical
running example (``.claude/RUNNING_EXAMPLE.md``).

The OCPT view states the process has exactly 2 (a/skip-a) x 2 (c/d item type)
x 3 (i/n/i+n router mode) = 12 valid resolved sub-trees. ``compose_signature``
should recover exactly those 12 connected composite diagrams directly from
the 22-generator Sigma, with no gateway nodes reintroduced.
"""

from __future__ import annotations

from procposets.cospan.compose import compose_signature
from procposets.cospan.engine import extract_signature

from .test_cpm_engine_running_example import build_running_example


def test_twelve_composite_diagrams():
    sig = extract_signature(build_running_example())
    composites = compose_signature(sig, start_label="G1", end_label="G2")
    assert len(composites) == 12
    for c in composites:
        assert c.placements[0].label == "G1" and not c.placements[0].left
        assert c.placements[-1].label == "G2" and not c.placements[-1].right


def test_router_mode_distribution():
    """Each of the 6 G2 boundary contexts (i/n/i+n x con/box) should appear
    exactly twice across the 12 composites -- once per a/skip-a choice,
    which doesn't affect the item-routing or sync/router path."""
    sig = extract_signature(build_running_example())
    composites = compose_signature(sig, start_label="G1", end_label="G2")
    end_gens = [c.placements[-1] for c in composites]
    assert len(set(end_gens)) == 6
    counts = {g: end_gens.count(g) for g in set(end_gens)}
    assert set(counts.values()) == {2}


def test_no_duplicate_composites():
    sig = extract_signature(build_running_example())
    composites = compose_signature(sig, start_label="G1", end_label="G2")
    seen = {c.placements for c in composites}
    assert len(seen) == 12
