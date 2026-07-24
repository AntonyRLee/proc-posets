"""Output-sensitive signature extraction: same ``Sigma`` (one representative
generator per distinct CanonKey) as :func:`engine.extract_signature`, without
materialising the cross-type ``|B|x|F|`` product that makes the slow engine
intractable on wide object-centric nets (a hub shared across ``k`` object types
blows up exponentially in ``k``).  See ``docs/fast-extraction.md`` for the
derivation and the wider factored-skeleton architecture this is part of.

**Exactness contract.**  The CanonKey set is **byte-identical** to
:func:`engine.extract_signature` on every fixture the slow engine can reach
(verified by ``tests/regression/test_cospan_extract_fast.py``).  The speed comes
from working at the *canon-profile* level, never at the concrete-bundle level.

A CanonKey reads only, per boundary side, the **multiset of endpoint object
types** (a surfaced terminus contributing its merged per-type count once the
pure-terminus collapse has been decided) -- never *which* concrete
predecessor/successor.  So we combine an activity's arcs one at a time in the
**output space**: the partial state carries, per type, only the merged
``real + gamma2`` leg count, plus one side-level saturating "has a real leg"
bit (which decides the pure-terminus collapse) under ``surface_termini`` --
real counts only under the default strip -- deduping after every fold step.
Two partial states equal in that reduced form have identical futures, so the
state set stays bounded by the number of distinct **profiles**, i.e. by the
output, never by the ``product-over-arcs`` of concrete endpoint choices (nor
by the raw real-vs-``gamma2`` split space, which can run 16x larger than the
profile space).  The Bundestag hub ``Beratung`` (26 typed arcs, ~15
independently optional) yields its 32,768 profiles in under two seconds where
the concrete-bundle product blew up / hung for minutes.

The terminus handling is folded into the state summary so it matches
``engine._strip_termini`` (default: ``gamma2`` legs absorbed) and
``engine._collapse_pure_terminus`` (``surface_termini``: ``gamma2`` legs kept and
counted by type, an all-``gamma2`` side collapsing to the empty right) exactly.

Returns one representative :class:`Generator` per CanonKey (no bindings) -- for
``compare`` / type-level views.  For the full per-context generator set
(splice / behavioural semantics) use :func:`engine.extract_signature`.
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import product

from .engine import GAMMA2, _prepare_extraction, _traverse
from .lmgraph import LMGraph
from .signature import Generator, Port, Signature
from .signature_compare import CanonKey

# the synthetic neighbour-label form _gen_from_profile mints (``·i:type``)
_PLACEHOLDER = re.compile(r"^·\d+:")


def is_canonical_signature(sig: Signature) -> bool:
    """True iff ``sig`` carries canonical representatives' synthetic placeholder
    neighbours (``·i:type``) -- i.e. it came from :func:`extract_signature_fast`
    (or ``signature_from_ocpn``'s ``canonical=True`` default).

    Such ports can never connect (a producer's right ``(lab, t, ·i:t)`` never
    equals a consumer's left ``(·j:t, t, lab)``), so behaviour-level machinery
    fed one would return silently-empty results -- ``compose_signature`` and
    ``extract_classes`` call this to fail loudly instead."""
    return any(
        _PLACEHOLDER.match(p.src) or _PLACEHOLDER.match(p.tgt)
        for g in sig for p in (*g.left, *g.right)
    )


def _side_profiles(g: LMGraph, a: str, *, forward: bool, surface_termini: bool) -> set:
    """Achievable set of canon *profiles* for one boundary side of ``a`` -- each a
    ``frozenset`` of ``(object_type, arity)`` legs (``frozenset()`` = the empty side).

    Combines the side's arcs incrementally in the reduced output space (per-type
    merged counts + the side-level has-a-real-leg bit under ``surface_termini``,
    real counts only under strip -- see the module docstring), deduping after
    every fold, so the state set is bounded by the number of distinct profiles
    (the output), not the concrete ``product-over-arcs``.  Endpoint
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
    # Dedup in the OUTPUT space, not the raw (real, gamma2) split space.  The
    # final profile keeps, per type, only the merged ``real + gamma2`` count --
    # plus one side-level "has a real leg" bit deciding the pure-terminus
    # collapse -- under ``surface_termini``, and only the real count under the
    # default strip.  That reduction is additive per arc and OR-saturating on
    # the bit, so applying it to each per-type option BEFORE the cross-arc /
    # cross-type folds reaches exactly the reduced form of every raw state
    # while the state count stays bounded by the output (x2 for the bit),
    # never by the raw real-vs-gamma2 splits (which reach 2^19 on the
    # Bundestag hub side where the output has ~2^15 profiles; measured
    # 10.5s -> ~1s for the fold+product).  Totals fold arc-by-arc with dedup
    # after every arc -- output-sensitive *within* a type too.
    frag_lists: list = []
    for t, arcs in clean.items():
        if collapse:
            opts: set = {(0, False)}
            for arc_choices in arcs:
                contribs = {(sum(rc.values()) + sum(gc.values()), bool(rc))
                            for rc, gc in arc_choices}
                opts = {(n + dn, hr or dhr) for (n, hr) in opts for (dn, dhr) in contribs}
            frag_lists.append(sorted(((((t, n),) if n else ()), hr) for n, hr in opts))
        else:
            opts = {0}
            for arc_choices in arcs:
                contribs = {sum(rc.values()) for rc, _ in arc_choices}
                opts = {n + dn for n in opts for dn in contribs}
            frag_lists.append(sorted((((t, n),) if n else ()) for n in opts))
    # One product across the independent types (the prototype's fast path), then
    # fold the rare coupling arcs incrementally, both in the reduced space.
    states: set = set()
    if collapse:
        for combo in product(*frag_lists):
            items: tuple = ()
            has_real = False
            for fr, hr in combo:
                items += fr
                has_real = has_real or hr
            states.add((frozenset(items), has_real))
        for choices in coupling:
            red = {
                (tuple(sorted(_merge_counts(rc, gc).items())), bool(rc))
                for rc, gc in choices
            }
            nxt: set = set()
            for prev, has_real in states:
                base = dict(prev)
                for add, hr in red:
                    nn = dict(base)
                    for t, c in add:
                        nn[t] = nn.get(t, 0) + c
                    nxt.add((frozenset(nn.items()), has_real or hr))
            states = nxt
        out: set = set()
        for prev, has_real in states:
            if prev and not has_real:               # all-gamma2 side -> pure terminus
                out.add(frozenset())
            else:
                out.add(prev)
        return out
    for combo in product(*frag_lists):
        items = ()
        for fr in combo:
            items += fr
        states.add(frozenset(items))
    for choices in coupling:
        red = {tuple(sorted(rc.items())) for rc, _ in choices}  # strip: gamma2 absorbed
        nxt = set()
        for prev in states:
            base = dict(prev)
            for add in red:
                nn = dict(base)
                for t, c in add:
                    nn[t] = nn.get(t, 0) + c
                nxt.add(frozenset(nn.items()))
        states = nxt
    return states


def _merge_counts(rc: dict, gc: dict) -> dict:
    """Per-type ``real + gamma2`` totals of one bundle (the surface-mode merge)."""
    out = dict(rc)
    for t, c in gc.items():
        out[t] = out.get(t, 0) + c
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
                           remove_silent: bool = False) -> Signature:
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
