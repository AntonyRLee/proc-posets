"""The section 2b engine: LM-graph -> generator-cospan signature ``Sigma``.

Implements the four-step pipeline of ``thm:canonical-presentation``:

  search   -- forward/backward bundle expansion through mediators
  filter   -- admissibility ``><`` (default: type-balance)
  generate -- one generator cospan per admissible context
  compose  -- (downstream, in F(Sigma); not needed to build Sigma)

A *bundle* is a frozenset of ``(label, type)`` endpoints reachable as one
committed firing choice.  Mediators combine the bundle-sets of their branches:

  XOR -> union (choose one branch),   AND -> product (combine all branches),
  OR  -> union over nonempty subsets of branches (XOR-of-AND-bundles),
  SEQ -> pass-through (transparent).

An activity's own multiple ports are combined with AND (a firing acts on all
its ports); branching/choice lives in the mediators it points at.  Memoisation
on the visited-mediator set gives termination on cyclic graphs
(``lem:gf-termination``); loop refinement (``a in L cap R``) is future work.
"""

from __future__ import annotations

from itertools import combinations, product

from ._boundary import GAMMA2
from .lmgraph import Kind, LMGraph
from .signature import Generator, Port, Signature
from .typebalance import Kappa, admissible

Bundle = frozenset  # frozenset[tuple[str, str | None]]
BundleSet = set  # set[Bundle]

# ``GAMMA2`` (imported from the dependency-free ``_boundary`` leaf) is the boundary
# endpoint label for a bare sink *place* reached forward -- a discovered model with no
# explicit terminus activity (§40 single-boundary convention). There is deliberately no
# GAMMA1 counterpart here: backward dead-ends stay zero-left origins, see ``_traverse``.


def _resolve_type(types: tuple[str | None, ...]) -> str | None:
    """The object type of a causal link along a (possibly untyped-stepped)
    path.  Untyped (``None``) edges inherit the path's unique typed value; this
    is how a transparent ``tau`` mediator carries e.g. ``con`` typing across an
    otherwise untyped source edge."""
    seen = [t for t in types if t is not None]
    if not seen:
        return None
    if len(set(seen)) == 1:
        return seen[0]
    return seen[-1]  # heterogeneous path: type nearest the endpoint


def _and(children: list[BundleSet]) -> BundleSet:
    if not children:
        return {frozenset()}
    out: BundleSet = set()
    for combo in product(*children):
        out.add(frozenset().union(*combo) if combo else frozenset())
    return out


def _or(children: list[BundleSet]) -> BundleSet:
    out: BundleSet = set()
    n = len(children)
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            out |= _and([children[j] for j in idx])
    return out


def _xor(children: list[BundleSet]) -> BundleSet:
    out: BundleSet = set()
    for c in children:
        out |= c
    return out if out else {frozenset()}


def _combine(kind: Kind, children: list[BundleSet]) -> BundleSet:
    if kind is Kind.XOR:
        return _xor(children)
    if kind is Kind.AND:
        return _and(children)
    if kind is Kind.OR:
        return _or(children)
    # SEQ: transparent pass-through (1-ary AND)
    return children[0] if len(children) == 1 else _and(children)


def _traverse(
    g: LMGraph,
    node: str,
    types: tuple[str | None, ...],
    visited: frozenset[str],
    *,
    forward: bool,
) -> BundleSet:
    """Bundle-set reachable from ``node`` in one direction.  Absorbing nodes are
    boundary endpoints (base case); mediators recurse per their kind."""
    if g.is_activity(node):
        return {frozenset({(g.lab(node), _resolve_type(types))})}
    if node in visited:
        return {frozenset()}  # cycle cut
    visited = visited | {node}
    nbrs = g.out_edges(node) if forward else g.in_edges(node)
    if not nbrs:
        if forward:
            # bare sink *place* reached forward: a gamma2 boundary endpoint, typed by the
            # path so far. Returning this rather than ``{frozenset()}`` keeps the
            # terminating object alive when AND-combined, so an activity with a MIX of
            # continuing and terminating out-branches (the discovered OCPN's ``s``: order
            # -> ``r`` continues, carrier -> bare sink terminates) keeps the carrier as a
            # gamma2 leg instead of having it silently absorbed. A PURE terminus -- whose
            # every branch is boundary, e.g. the running example's explicit ``G2`` -- is
            # collapsed back to the empty (zero-right) bundle in ``forward_bundles``.
            return {frozenset({(GAMMA2, _resolve_type(types))})}
        # bare source *place* reached backward: a zero-left **origin**, kept as the empty
        # bundle (NOT a gamma1 endpoint). Origins are bare zero-left *activities* of
        # distinct labels that co-fire one-per-label (§40a); minting a same-label ``gamma1``
        # leg here would instead force a single origin per closing and break the legal
        # multi-type co-start (and the skip-route origin of ``b``). The display γ1 is folded
        # in afterwards by ``gamma_normalize`` / the family boundary, not here.
        return {frozenset()}
    children = [
        _traverse(
            g,
            e.tgt if forward else e.src,
            types + (e.typ,),
            visited,
            forward=forward,
        )
        for e in nbrs
    ]
    return _combine(g.mediators[node], children)


def _collapse_pure_terminus(bundles: BundleSet) -> BundleSet:
    """Collapse any bundle that is *entirely* ``gamma2`` endpoints to the empty bundle.

    An all-``gamma2`` bundle means the activity is a pure terminus -- it *is* a boundary
    generator (e.g. the running example's explicit ``G2``), whose canonical interface is
    zero-right. A MIXED bundle (real successors alongside a terminating object) is kept
    as-is: that terminating object is a genuine gamma2 leg (the OCPN carrier into
    ``gamma2`` beside the order into ``r``)."""
    out: BundleSet = set()
    for b in bundles:
        if b and all(endpoint[0] == GAMMA2 for endpoint in b):  # endpoint = (label, type)
            out.add(frozenset())
        else:
            out.add(b)
    return out


def _strip_termini(bundles: BundleSet) -> BundleSet:
    """Drop every ``gamma2`` endpoint -- a terminating object is *absorbed*, not surfaced as
    a boundary leg. The original behaviour, kept as the default (``surface_termini=False``):
    the per-type flattened PN/PT/BPMN/CN are compared on interior structure (a Petri net's
    token-conservation sink and a process tree's bare consumption would otherwise disagree on
    termini), and the golden running example carries its boundary as explicit ``G1``/``G2``
    activities. Exactly recovers the pre-surfacing result (an absorbed terminus contributed
    ``frozenset()`` to the AND)."""
    return {frozenset(e for e in b if e[0] != GAMMA2) for b in bundles}


def forward_bundles(g: LMGraph, a: str, *, surface_termini: bool = False) -> BundleSet:
    """``F_a``: successor-activity bundles of ``a`` (AND over its out-ports).

    ``surface_termini`` keeps an object that terminates at a bare sink *place* as a
    ``gamma2`` boundary leg (the OCPN carrier into the final marking), collapsing only a
    *pure* terminus to zero-right; off (default) it absorbs every terminus as before."""
    F = _and([_traverse(g, e.tgt, (e.typ,), frozenset(), forward=True) for e in g.out_edges(a)])
    return _collapse_pure_terminus(F) if surface_termini else _strip_termini(F)


def backward_bundles(g: LMGraph, a: str) -> BundleSet:
    """``B_a``: predecessor-activity bundles of ``a`` (AND over its in-ports)."""
    return _and(
        [_traverse(g, e.src, (e.typ,), frozenset(), forward=False) for e in g.in_edges(a)]
    )


def extract_signature(
    g: LMGraph, kappa: Kappa | None = None, remove_silent: bool = True,
    *, surface_termini: bool = False,
) -> Signature:
    """Run the pipeline over every activity and collect ``Sigma``.

    ``surface_termini`` (default ``False``) keeps an object that terminates at a bare sink
    *place* as a ``gamma2`` right leg -- the object-centric final marking (the OCPN carrier
    into ``gamma2``, matching the OCCN's ``END_<ot>`` and what a faithful cospan boundary
    needs). Off for the per-type flattened PN/PT/BPMN/CN (compared on interior structure) and
    the golden running example (whose boundary is explicit ``G1``/``G2`` activities).

    ``remove_silent`` (default ``True``) first contracts every silent (tau)
    mediator via :meth:`LMGraph.without_silent` -- model -> signature drops
    silent transitions by design, so the generators carry only real activity
    labels. Pass ``remove_silent=False`` to keep silents as transparent
    mediators (signature-neutral for the degree-(1,1) silents that discovered
    models emit; differs only when a silent genuinely synchronises).

    Admissibility ``⋈`` (``def:contexts-general``) is the type-balance filter of
    :mod:`.typebalance`: when a per-activity create/consume profile ``kappa`` is
    supplied, any context that would mint or absorb a type without a licence is
    rejected before it can enter ``Sigma`` (this forbids arbitrary type
    conversion in the OC notations).  With ``kappa=None`` the filter is inert --
    activities are unconstrained, which is the prior behaviour and what the
    golden running-example test exercises.  (Type-preservation at *mediators* is
    enforced separately and unconditionally by ``LMGraph.validate``.)"""
    if remove_silent and g.silent:
        g = g.without_silent()
    if surface_termini and any(
        g.lab(a) == GAMMA2 or g.lab(a).startswith("END_") for a in g.activities
    ):
        # the model already carries an explicit terminus *activity* (a master spec
        # simulated into a log re-introduces ``gamma2`` as an event; BPMN/OCCN-style
        # ``END_<ot>``): that activity already is the final marking, so surfacing bare-sink
        # termini would double it (and collide on the ``gamma2`` label). Leave them absorbed.
        surface_termini = False
    gens: set[Generator] = set()
    for a in sorted(g.activities):
        lab = g.lab(a)
        B = backward_bundles(g, a)
        F = forward_bundles(g, a, surface_termini=surface_termini)
        for P in B:
            for S in F:
                if kappa is not None and not admissible(lab, P, S, kappa):
                    continue
                left = frozenset(Port(p, t, lab) for (p, t) in P)
                right = frozenset(Port(lab, t, s) for (s, t) in S)
                gens.add(Generator(lab, left, right))
    return Signature(frozenset(gens))
