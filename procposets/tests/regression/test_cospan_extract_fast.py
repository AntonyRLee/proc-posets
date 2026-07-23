"""Golden cross-check: ``extract_signature_fast`` (output-sensitive) must emit the
**same CanonKey set** as the slow ``extract_signature`` on every fixture the slow
engine can reach -- the migration-discipline seam for the fast extractor
(docs/fast-extraction.md). The fast path returns one representative generator
per CanonKey (no bindings), so we compare the ``canonical_generators`` keys,
which is exactly its stated contract.
"""

from __future__ import annotations

from procposets.cospan.engine import extract_signature
from procposets.cospan.engine_fast import extract_signature_fast
from procposets.cospan.from_ocpn import signature_from_ocpn
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.signature_compare import canon_key, canonical_generators

from .test_cospan_engine_running_example import build_running_example
from .test_cospan_surface_termini import _mixed_graph


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


def _optional_hub(k: int, m: int) -> LMGraph:
    """Hub ``H`` with one in-arc per object type ``t1..tk``, each an XOR over
    ``m`` same-type predecessors AND a bare-source empty alternative (the type
    is optional) -- the Bundestag-``Beratung`` shape in miniature: the
    concrete-bundle cross-arc product is ``(m+1)^k`` while the canon-profile
    state space is ``2^k``."""
    g = LMGraph()
    g.add_activity("H")
    g.add_activity("E")
    for i in range(1, k + 1):
        t = f"t{i}"
        g.add_mediator(f"q{i}", Kind.XOR)
        g.add_mediator(f"r{i}", Kind.XOR)   # dead-end feeder: the empty alternative
        g.add_edge(f"r{i}", f"q{i}", t)
        for j in range(m):
            g.add_activity(f"P{i}_{j}")
            g.add_edge(f"P{i}_{j}", f"q{i}", t)
        g.add_edge(f"q{i}", "H", t)
    g.add_mediator("he", Kind.XOR)
    g.add_edge("H", "he", "t1")
    g.add_edge("he", "E", "t1")
    g.validate()
    return g


def _coupled_terminus() -> LMGraph:
    """A *coupling* out-arc (untyped XOR fanning into two object types) whose
    alternatives include a bare-sink ``gamma2`` branch: ``X`` via untyped ``p``
    reaches ``A`` (t1), ``B`` (t2), or a dead-end mediator (terminus).  This is
    the one shape the clean per-type factoring cannot take -- the arc must be
    folded jointly -- with the terminus modes live on the SAME arc."""
    g = LMGraph()
    for a in ("X", "A", "B"):
        g.add_activity(a)
    for m in ("p", "ma", "mb", "mdead"):
        g.add_mediator(m, Kind.XOR)
    g.add_edge("X", "p")            # untyped: the coupling source
    g.add_edge("p", "ma"); g.add_edge("ma", "A", "t1")
    g.add_edge("p", "mb"); g.add_edge("mb", "B", "t2")
    g.add_edge("p", "mdead")        # bare dead end -> gamma2 terminus branch
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


def test_coupled_terminus_canonkeys_match():
    _assert_same_canonkeys("coupled_terminus", _coupled_terminus, {})


def test_coupled_terminus_surface_canonkeys_match():
    _assert_same_canonkeys("coupled_terminus_surface", _coupled_terminus,
                           {"surface_termini": True})


def test_wide_optional_hub_stays_output_sensitive():
    """The wide-OCPN hang guard: k=12 optional types x m=4 alternatives give a
    ``5^12 ~ 2.4e8`` concrete-bundle cross-arc product -- a regression of the
    fast path back to concrete-bundle products hangs the suite budget -- while
    the canon-profile space is ``2^12 = 4096``, done in milliseconds.  The
    same shape is cross-checked exactly against the slow engine at k=6/m=2
    (``3^6 = 729`` concrete bundles, still cheap)."""
    _assert_same_canonkeys("optional_hub_small", lambda: _optional_hub(6, 2), {})
    fast = _canon_keys(extract_signature_fast(_optional_hub(12, 4)))
    assert len({k for k in fast if k.label == "H"}) == 2 ** 12


def test_signature_from_ocpn_defaults_to_canonical():
    """The flip: ``signature_from_ocpn``'s default is the output-sensitive
    canonical extractor (cannot hang on a wide net), byte-equal to an explicit
    ``canonical=True`` and distinct from the full extraction (whose generators
    carry concrete neighbour ports, not canonical placeholders).  Behaviour-
    level consumers opt out via ``canonical=False`` (``discover.py`` pins it)."""
    from procposets.cospan.engine_fast import extract_signature_fast as fast
    from procposets.cospan.from_ocpn import lmgraph_from_ocpn

    ocpn = _fake_ocpn()
    default = signature_from_ocpn(ocpn)
    assert default == signature_from_ocpn(ocpn, canonical=True)
    assert default == fast(lmgraph_from_ocpn(ocpn))
    assert default != signature_from_ocpn(ocpn, canonical=False)


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
