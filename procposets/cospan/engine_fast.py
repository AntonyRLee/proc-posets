"""Output-sensitive signature extraction: same ``Sigma`` (one representative
generator per distinct CanonKey) as :func:`engine.extract_signature`, without
materialising the cross-type ``|B|x|F|`` product that makes the slow engine
intractable on wide object-centric nets (a hub shared across ``k`` object types
blows up exponentially in ``k``).  See
``docs/2026-07-23-decorated-cospan-extractor-PLAN.md`` for the derivation and the
wider factored-skeleton plan this is the first step of.

**Exactness contract.**  The CanonKey set is **byte-identical** to
:func:`engine.extract_signature` on every fixture the slow engine can reach
(verified by ``tests/regression/test_cospan_extract_fast.py``).  The speed comes
from working at the *canon-profile* level, never at the concrete-bundle level.

A CanonKey reads only, per boundary side, the **multiset of endpoint object types**
(and, when termini are surfaced, whether a leg is a ``gamma2`` terminus) -- never
*which* concrete predecessor/successor.  So we combine an activity's arcs one at a
time, carrying not the concrete bundles but the *canon partial-state* -- the
per-type ``(real, gamma2)`` leg counts so far -- and dedup on it at every step.
Two partial states with the same per-type counts have identical futures, so the
state set stays bounded by the number of distinct **profiles**, i.e. by the output,
never by the ``product-over-arcs`` of concrete endpoint choices.  The Bundestag hub
``Beratung`` (26 typed arcs, ~15 independently optional) yields its 32,768 profiles
in seconds where the concrete-bundle product blew up / hung for minutes.

The terminus handling is folded into the state summary so it matches
``engine._strip_termini`` (default: ``gamma2`` legs absorbed) and
``engine._collapse_pure_terminus`` (``surface_termini``: ``gamma2`` legs kept and
counted by type, an all-``gamma2`` side collapsing to the empty right) exactly.

Returns one representative :class:`Generator` per CanonKey (no bindings) -- for
``compare`` / type-level views.  For the full per-context generator set
(splice / behavioural semantics) use :func:`engine.extract_signature`.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product

from .engine import GAMMA2, _prepare_extraction, _traverse
from .lmgraph import LMGraph
from .signature import Generator, Port, Signature
from .signature_compare import CanonKey


def _side_profiles(g: LMGraph, a: str, *, forward: bool, surface_termini: bool) -> set:
    """Achievable set of canon *profiles* for one boundary side of ``a`` -- each a
    ``frozenset`` of ``(object_type, arity)`` legs (``frozenset()`` = the empty side).

    Combines the side's arcs incrementally, carrying the per-type ``(real, gamma2)``
    leg counts and deduping on them, so the state set is bounded by the number of
    distinct profiles (the output), not the concrete ``product-over-arcs``.  Endpoint
    types are the *resolved* types :func:`engine._traverse` returns (which need not
    equal an arc's declared ``e.typ`` -- e.g. an untyped choice fanning into several
    typed successors), so this is exact for arbitrary LM-graphs, not only clean OCPNs."""
    edges = g.out_edges(a) if forward else g.in_edges(a)
    collapse = forward and surface_termini
    # Split arcs: a *clean* arc reaches a single object type across all its bundle
    # choices (the OCPN norm -- one typed net per type), so it factors into that
    # type's independent contribution; a *coupling* arc's bundles span >1 type (only
    # an untyped choice node produces this) and must be folded jointly.
    clean: dict = defaultdict(list)
    coupling: list = []
    for e in edges:
        node = e.tgt if forward else e.src
        choices: list = []
        typs: set = set()
        for b in _traverse(g, node, (e.typ,), frozenset(), forward=forward):
            rc: dict = {}
            gc: dict = {}
            for lab_, typ in b:
                d = gc if lab_ == GAMMA2 else rc
                d[typ] = d.get(typ, 0) + 1
                typs.add(typ)
            choices.append((rc, gc))
        (coupling if len(typs) > 1 else clean[next(iter(typs), None)]).append(choices)
    # Per clean type: the achievable scalar (real, gamma2) counts over that type's arcs.
    types_order: list = []
    opts_lists: list = []
    for t, arcs in clean.items():
        opts: set = set()
        for combo in product(*arcs):
            real = sum(sum(rc.values()) for rc, _ in combo)
            g2 = sum(sum(gc.values()) for _, gc in combo)
            opts.add((real, g2))
        types_order.append(t)
        opts_lists.append(sorted(opts))
    # One product across the independent types (the prototype's fast path), then fold
    # the rare coupling arcs incrementally.  state = (real_items, g2_items).
    states: set = set()
    for combo in product(*opts_lists):
        rc = {}
        gc = {}
        for t, (real, g2) in zip(types_order, combo):
            if real:
                rc[t] = real
            if g2:
                gc[t] = g2
        states.add((frozenset(rc.items()), frozenset(gc.items())))
    for choices in coupling:
        nxt: set = set()
        for rprev, gprev in states:
            for rc, gc in choices:
                nr = dict(rprev)
                for t, c in rc.items():
                    nr[t] = nr.get(t, 0) + c
                ng = dict(gprev)
                for t, c in gc.items():
                    ng[t] = ng.get(t, 0) + c
                nxt.add((frozenset(nr.items()), frozenset(ng.items())))
        states = nxt
    out: set = set()
    for rprev, gprev in states:
        if collapse:
            if (rprev or gprev) and not rprev:      # all-gamma2 side -> pure terminus
                out.add(frozenset())
                continue
            merged = dict(rprev)
            for t, c in gprev:
                merged[t] = merged.get(t, 0) + c
            out.add(frozenset(merged.items()))
        else:                                       # strip: gamma2 legs absorbed
            out.add(frozenset(rprev))
    return out


def _gen_from_profile(lab: str, left_profile: frozenset, right_profile: frozenset) -> Generator:
    """A representative :class:`Generator` realising the given canon profiles: one
    distinct :class:`Port` per typed leg (the concrete neighbour label is a synthetic
    placeholder -- the fast signature carries no bindings, and a CanonKey reads only
    the per-type leg count)."""
    left = frozenset(
        Port(f"·{i}:{t}", t, lab) for (t, ar) in left_profile for i in range(ar)
    )
    right = frozenset(
        Port(lab, t, f"·{i}:{t}") for (t, ar) in right_profile for i in range(ar)
    )
    return Generator(lab, left, right)


def extract_signature_fast(g: LMGraph, *, surface_termini: bool = False,
                           remove_silent: bool = True) -> Signature:
    """Output-sensitive twin of :func:`engine.extract_signature`: one representative
    :class:`Generator` per distinct CanonKey, without the cross-type ``|B|x|F|``
    blow-up.  See the module docstring for the exactness contract."""
    g, surface_termini = _prepare_extraction(g, remove_silent, surface_termini)

    def _side_key(profile: frozenset) -> tuple:
        # exactly signature_compare._type_multiset's canonical ((type, count), ...) form
        return tuple(sorted(profile, key=lambda kv: (str(kv[0]), kv[1])))

    best: dict = {}
    for a in sorted(g.activities):
        lab = g.lab(a)
        in_profiles = _side_profiles(g, a, forward=False, surface_termini=surface_termini)
        out_profiles = _side_profiles(g, a, forward=True, surface_termini=surface_termini)
        out_keyed = [(_side_key(ov), ov) for ov in out_profiles]
        for iv in in_profiles:
            ik = _side_key(iv)
            for ok, ov in out_keyed:
                key = CanonKey(lab, ik, ok)
                if key not in best:               # build the representative Generator once per key
                    best[key] = _gen_from_profile(lab, iv, ov)
    return Signature(frozenset(best.values()))
