"""Output-sensitive signature extraction: same ``Sigma`` (one representative
generator per distinct CanonKey) as :func:`engine.extract_signature`, without
materialising the cross-type ``|B|x|F|`` product that makes the slow engine
intractable on wide object-centric nets (a hub shared across ``k`` object types
blows up exponentially in ``k``).  See
``docs/2026-07-20-fast-signature-extraction.md`` for the derivation.

**Exactness contract.**  For every activity whose *per-coupled-component* arc
product stays under :data:`_ENUM_CAP` -- i.e. every case the slow engine can
itself reach -- the CanonKey set is **byte-identical** to
:func:`engine.extract_signature` (verified by
``tests/regression/test_cpm_extract_fast.py``).  The speed comes from two moves
the slow engine does not make:

1. *Coupled-component decomposition.*  Arcs that can reach a common object type
   are enumerated together; type-disjoint components factor (their endpoint sets
   are disjoint), so achievable side-bundles are the Cartesian product across
   components.  This is where the cross-type blow-up dies: the slow engine
   AND-products *all* arcs of *all* types at once.
2. *Collapse to the CanonKey-relevant key.*  A CanonKey only reads the per-type
   endpoint *count* (and, once termini are surfaced, whether a leg is a
   ``gamma2`` terminus), never *which* endpoint.  So each component keeps one
   representative bundle per distinct ``(is_gamma2, type)``-multiset -- a handful,
   not the ``10^k`` raw bundles.

Post-processing (terminus strip / pure-terminus collapse) reuses the slow
engine's own :func:`engine._strip_termini` / :func:`engine._collapse_pure_terminus`
on the representatives, so it is identical by construction.

**Above the cap** (same-type high fan-out -- variable/double arcs -- which the
slow engine also cannot finish): a *single-type* component falls back to the
exact ``[min-hitting, max-matching]`` arity interval; a *multi-type* component
over the cap can only arise from an untyped choice node fanning into a large
coupled type-web (never produced by a discovered OCPN), so it raises rather than
risk a silently-wrong answer.

Returns one representative :class:`Generator` per CanonKey (no bindings) -- for
``compare`` / type-level views.  For the full per-context generator set
(splice / behavioural semantics) use :func:`engine.extract_signature`.
"""
from __future__ import annotations

from itertools import combinations, product

from .engine import GAMMA2, _collapse_pure_terminus, _strip_termini, _traverse
from .lmgraph import LMGraph
from .signature import Generator, Port, Signature
from .signature_compare import canon_key

_ENUM_CAP = 200_000  # per-coupled-component arc product budget


def _arc_bundlesets(g: LMGraph, a: str, forward: bool) -> list:
    """Per arc of one side of ``a``: the arc's reachable bundle-set as a list of
    ``(label, type)`` frozensets -- exactly the slow engine's per-arc ``_traverse``."""
    edges = g.out_edges(a) if forward else g.in_edges(a)
    out = []
    for e in edges:
        node = e.tgt if forward else e.src
        Se = _traverse(g, node, (e.typ,), frozenset(), forward=forward)
        out.append([frozenset(b) for b in Se])
    return out


def _reachable_types(opts: list) -> set:
    """The object types an arc can contribute across all its bundle options."""
    return {t for b in opts for (_lab, t) in b}


def _components(arc_opts: list) -> list:
    """Partition arcs into connected components under *shared reachable type*.

    Two arcs that can reach a common object type are coupled: their per-type
    endpoint counts interact (dedup of a shared endpoint, or a joint choice
    forced through an untyped node), so they must be enumerated together.
    Type-disjoint components factor -- their endpoint sets are disjoint, so the
    achievable full-side bundles are the Cartesian product of the components'.
    Over-coupling only ever costs enumeration, never correctness; *under*-coupling
    loses distinct CanonKeys (the untyped-choice bug this file fixes)."""
    n = len(arc_opts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_type: dict = {}
    for i, opts in enumerate(arc_opts):
        for t in _reachable_types(opts):
            if t in by_type:
                union(i, by_type[t])
            else:
                by_type[t] = i
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [[arc_opts[i] for i in idxs] for idxs in groups.values()]


def _max_matching(reach: list) -> int:
    match: dict = {}

    def aug(i, seen):
        for ep in reach[i]:
            if ep in seen:
                continue
            seen.add(ep)
            if ep not in match or aug(match[ep], seen):
                match[ep] = i
                return True
        return False

    size = 0
    for i in range(len(reach)):
        if reach[i] and aug(i, set()):
            size += 1
    return size


def _min_hitting(must: list) -> int:
    if not must:
        return 0
    universe = sorted(set().union(*must), key=str)
    for s in range(1, len(must) + 1):
        for cand in combinations(universe, s):
            cs = set(cand)
            if all(cs & r for r in must):
                return s
    return len(must)


def _endpoint_key(bundle: frozenset) -> tuple:
    """The finest thing strip / collapse / :func:`canon_key` can distinguish about a
    bundle: the multiset of ``(is_gamma2, type)`` over its endpoints.  Two bundles
    with the same key yield identical post-processing and identical CanonKeys."""
    return tuple(sorted(
        ((ep[0] == GAMMA2, ep[1]) for ep in bundle),
        key=lambda x: (x[0], str(x[1])),
    ))


def _component_bundles(comp: list) -> set:
    """Achievable union-bundles within one coupled component, one representative
    per distinct :func:`_endpoint_key`.  Enumerate the arc product when it is under
    the cap; above it a single-type component uses the exact
    ``[min-hitting, max-matching]`` arity interval, and a multi-type component is
    unsupported (raises -- it cannot come from a discovered OCPN)."""
    total = 1
    for o in comp:
        total *= max(1, len(o))
    if total <= _ENUM_CAP:
        seen: dict = {}
        for combo in product(*comp):
            u = frozenset().union(*combo) if combo else frozenset()
            seen.setdefault(_endpoint_key(u), u)
        return set(seen.values())
    types = set().union(*(_reachable_types(o) for o in comp))
    if len(types) > 1:
        raise NotImplementedError(
            "multi-type coupled component exceeds the enumeration cap: an untyped "
            "choice node fans into a large coupled type-web. Discovered OCPNs do "
            "not produce this shape; use engine.extract_signature for it."
        )
    reach = [{ep for b in opts for ep in b} for opts in comp]
    must = [r for opts, r in zip(comp, reach) if frozenset() not in opts and r]
    lo, hi = _min_hitting(must), _max_matching(reach)
    universe = sorted(set().union(*reach), key=str) if reach else []
    g2eps = [e for e in universe if e[0] == GAMMA2]
    reps = set()
    for ar in range(max(lo, 0), hi + 1):
        reps.add(frozenset(g2eps[:ar]) if len(g2eps) >= ar else frozenset(universe[:ar]))
    return reps or {frozenset()}


def _side_bundles(g: LMGraph, a: str, forward: bool) -> set:
    """Achievable full-side bundle-set (pre strip/collapse) for one side of ``a``:
    the Cartesian product over type-disjoint components of each component's
    collapsed representative bundles -- output-sensitive, and equal set-for-set to
    the slow engine's ``_and`` on every below-cap component."""
    reps = {frozenset()}
    for comp in _components(_arc_bundlesets(g, a, forward)):
        comp_bundles = _component_bundles(comp)
        reps = {r | c for r in reps for c in comp_bundles}
    return reps


def extract_signature_fast(g: LMGraph, *, surface_termini: bool = False,
                           remove_silent: bool = True) -> Signature:
    """Output-sensitive twin of :func:`engine.extract_signature`: one representative
    :class:`Generator` per distinct CanonKey, without the cross-type ``|B|x|F|``
    blow-up.  See the module docstring for the exactness contract."""
    if remove_silent and g.silent:
        g = g.without_silent()
    if surface_termini and any(
        g.lab(a) == GAMMA2 or g.lab(a).startswith("END_") for a in g.activities
    ):
        surface_termini = False
    best: dict = {}
    for a in g.activities:
        lab = g.lab(a)
        B = _side_bundles(g, a, forward=False)  # backward carries no gamma2 -> no post-proc
        F = _side_bundles(g, a, forward=True)
        F = _collapse_pure_terminus(F) if surface_termini else _strip_termini(F)
        for P in B:
            for S in F:
                left = frozenset(Port(p, t, lab) for (p, t) in P)
                right = frozenset(Port(lab, t, s) for (s, t) in S)
                gen = Generator(lab, left, right)
                best.setdefault(canon_key(gen), gen)
    return Signature(frozenset(best.values()))
