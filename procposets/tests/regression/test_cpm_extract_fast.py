"""Golden cross-check: ``extract_signature_fast`` (output-sensitive) must emit the
**same CanonKey set** as the slow ``extract_signature`` on every fixture the slow
engine can reach -- the migration-discipline seam for the fast extractor
(docs/2026-07-20-fast-signature-extraction.md). The fast path returns one
representative generator per CanonKey (no bindings), so we compare the
``canonical_generators`` keys, which is exactly its stated contract.
"""

from __future__ import annotations

from procposets.cospan.engine import extract_signature
from procposets.cospan.engine_fast import extract_signature_fast
from procposets.cospan.from_ocpn import signature_from_ocpn
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.signature_compare import canon_key, canonical_generators

from .test_cpm_engine_running_example import build_running_example
from .test_cpm_surface_termini import _mixed_graph


# --- pm4py-free fakes for the OCPN wrapper (add_petri only duck-types the net:
#     .places / .transitions / .arcs, place.name, transition.label/.name,
#     arc.source / arc.target; transitions are dict keys so must be hashable). ---
class _P:
    def __init__(self, name): self.name = name


class _T:
    def __init__(self, name, label): self.name, self.label = name, label


class _Arc:
    def __init__(self, source, target): self.source, self.target = source, target


class _Net:
    def __init__(self, places, transitions, arcs):
        self.places, self.transitions, self.arcs = places, transitions, arcs


def _fake_ocpn() -> dict:
    """Two per-type nets (``t1``, ``t2``) sharing a hub activity ``H`` by label -- the
    object-centric typed merge, in miniature: ``H`` acquires typed in/out ports from
    both types, exactly the shape whose cross-type product the fast path collapses."""
    p1s, p1m, p1e = _P("p1s"), _P("p1m"), _P("p1e")
    a, h1 = _T("A", "A"), _T("H1", "H")
    net1 = _Net({p1s, p1m, p1e}, {a, h1},
                [_Arc(p1s, a), _Arc(a, p1m), _Arc(p1m, h1), _Arc(h1, p1e)])
    p2s, p2m, p2e = _P("p2s"), _P("p2m"), _P("p2e")
    b, h2 = _T("B", "B"), _T("H2", "H")
    net2 = _Net({p2s, p2m, p2e}, {b, h2},
                [_Arc(p2s, b), _Arc(b, p2m), _Arc(p2m, h2), _Arc(h2, p2e)])
    return {"petri_nets": {"t1": (net1, None, None), "t2": (net2, None, None)}}


def _typed_hub() -> LMGraph:
    """Small-scale analogue of the Bundestag hub blow-up: activity ``H`` with in- and
    out-arcs of two object types, each type reaching two endpoints through an XOR.
    The slow engine materialises |B|x|F| = 4x4 = 16 generators; both sides collapse to
    a single (t1:1, t2:1) type-multiset, so there is exactly ONE CanonKey for ``H``.
    This is the clean fully-typed case the plan validated on."""
    g = LMGraph()
    for a in ("H", "P1a", "P1b", "P2a", "P2b", "S1a", "S1b", "S2a", "S2b"):
        g.add_activity(a)
    for m in ("mp1", "mp2", "mq1", "mq2"):
        g.add_mediator(m, Kind.XOR)
    # backward: two predecessors per type feed H through a per-type XOR
    g.add_edge("P1a", "mp1", "t1"); g.add_edge("P1b", "mp1", "t1"); g.add_edge("mp1", "H", "t1")
    g.add_edge("P2a", "mp2", "t2"); g.add_edge("P2b", "mp2", "t2"); g.add_edge("mp2", "H", "t2")
    # forward: two successors per type
    g.add_edge("H", "mq1", "t1"); g.add_edge("mq1", "S1a", "t1"); g.add_edge("mq1", "S1b", "t1")
    g.add_edge("H", "mq2", "t2"); g.add_edge("mq2", "S2a", "t2"); g.add_edge("mq2", "S2b", "t2")
    g.validate()
    return g


def _canon_keys(sig) -> set:
    return set(canonical_generators(sig).keys())


def _fmt(keys) -> str:
    return "\n".join(f"  {k.label}: {k.arity_str()}" for k in sorted(keys))


def _assert_same_canonkeys(name, factory, kw):
    g = factory()
    slow = _canon_keys(extract_signature(g, **kw))
    fast = _canon_keys(extract_signature_fast(g, **kw))
    missing = slow - fast     # CanonKeys the fast path failed to produce
    extra = fast - slow       # CanonKeys the fast path invented
    msg = (
        f"[{name}] fast CanonKey set diverges from slow "
        f"(slow={len(slow)}, fast={len(fast)}):\n"
        f"MISSING from fast:\n{_fmt(missing)}\n"
        f"EXTRA in fast:\n{_fmt(extra)}"
    )
    assert fast == slow, msg


def test_running_example_canonkeys_match():
    _assert_same_canonkeys("running_example", build_running_example, {})


def test_running_example_surface_canonkeys_match():
    _assert_same_canonkeys("running_example_surface", build_running_example,
                           {"surface_termini": True})


def test_mixed_graph_canonkeys_match():
    _assert_same_canonkeys("mixed_graph", _mixed_graph, {})


def test_mixed_graph_surface_canonkeys_match():
    _assert_same_canonkeys("mixed_graph_surface", _mixed_graph, {"surface_termini": True})


def test_mixed_graph_terminus_surface_canonkeys_match():
    _assert_same_canonkeys("mixed_graph_terminus_surface",
                           lambda: _mixed_graph(terminus_label="gamma2"),
                           {"surface_termini": True})


def test_typed_hub_canonkeys_match():
    _assert_same_canonkeys("typed_hub", _typed_hub, {})


def test_typed_hub_surface_canonkeys_match():
    _assert_same_canonkeys("typed_hub_surface", _typed_hub, {"surface_termini": True})


def test_typed_hub_has_single_canonkey_for_hub():
    """The 16-generator hub collapses to one CanonKey for ``H`` in both engines."""
    g = _typed_hub()
    slow = extract_signature(g)
    hub_keys = {canon_key(gen) for gen in slow.by_label("H")}
    assert len(hub_keys) == 1
    fast_hub = {k for k in _canon_keys(extract_signature_fast(g)) if k.label == "H"}
    assert fast_hub == hub_keys


def test_signature_from_ocpn_canonical_matches_full():
    """The ``signature_from_ocpn`` opt-in: ``canonical=True`` (fast) and the default
    full extractor agree on the CanonKey set, for both ``surface_termini`` values --
    the wrapper's ocpn->lmgraph->dispatch plumbing on a real typed-merge net."""
    ocpn = _fake_ocpn()
    for surf in (False, True):
        slow = set(canonical_generators(
            signature_from_ocpn(ocpn, canonical=False, surface_termini=surf)).keys())
        fast = set(canonical_generators(
            signature_from_ocpn(ocpn, canonical=True, surface_termini=surf)).keys())
        assert fast == slow, f"surface_termini={surf}: fast={fast} slow={slow}"
        # the shared hub H appears with one typed-merge CanonKey
        assert any(k.label == "H" for k in slow)
