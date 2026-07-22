"""Filters for discovery artifacts that are properties of the mined
``Signature`` itself, not of any one notation's adapter -- so they belong
upstream of `class_extraction.extract_classes`, not in the post-hoc
`morphism_schema`/`signature_diff` layers (which only ever see whatever the
signature already contains).

Promoted from an OCPN-specific signature extractor (originally written for the
mined OCPN signature only, but the check itself never assumed anything
OCPN-specific) when the same artifact showed up as a literal generator in a
cross-signature diff.
"""

from __future__ import annotations

from .signature import Generator, Port, Signature


def forget_provenance(sig: Signature) -> Signature:
    """The fungible-token *behavioural quotient*: forget each port's producer
    ``src`` while keeping ``(typ, tgt)`` (object type + consumer activity).

    Unlike :func:`degenerate_filtered` this is **not** a lossy artifact filter
    -- it is a sound semantic quotient. A ``Port=(src,typ,tgt)`` records
    ``src`` (which activity produced the token, i.e. provenance/past) and
    ``tgt`` (which activity will consume it, i.e. destination/future). For
    *fungible* place tokens the future depends only on ``(typ,tgt)``, never on
    ``src`` (Markov), so generators that differ only in ``src`` denote the same
    behaviour and are merged. The result keeps **activity-referenced**
    boundaries (``tgt`` is a consumer activity, shared across notations), so
    cross-notation comparison still works -- this is the comparison-granularity
    quotient, *not* the rejected place-identity boundary.

    Effect: on a mined OCPN it collapses the
    predecessor-provenance cross-product that the engine's fine
    ``(src,typ,tgt)`` ports generate (ED OCPN 2514 -> 270 generators) and lets
    the discovered loop close as a real cycle instead of spiralling through
    raw-distinct-but-isomorphic per-round frontiers. It is a no-op on a
    provenance-clean signature (the hand-authored ED master: 12 -> 12), and on
    OCCN it merges only behavioural duplicates (54 -> 32) while preserving every
    distinct closing (e.g. ``M(1,trop)``). It only ever *merges* -- never adds a
    composition -- so it cannot manufacture behaviour.

    Applied as a comparison-time transform over the §02b-faithful fine
    signature (which is left intact); see the discovery/comparison scripts.
    """
    from .constraints import remap

    def quotient(p: Port) -> Port:
        return Port(src="*", typ=p.typ, tgt=p.tgt)

    def quotient_weights(g: Generator) -> frozenset:
        """Carry §38 leg weights through the quotient (sum when ports collapse), so a
        *grounded* weighted signature is not silently de-grounded by the merge. A no-op
        for the common unweighted case (empty ``weights``)."""
        if not g.weights:
            return frozenset()
        agg: dict[Port, int] = {}
        for p, w in g.weights:
            q = quotient(p)
            agg[q] = agg.get(q, 0) + w
        return frozenset(agg.items())

    gens = {
        Generator(
            label=g.label,
            left=frozenset(quotient(p) for p in g.left),
            right=frozenset(quotient(p) for p in g.right),
            # carry the §32 leg constraints, remapping their ports through the
            # same quotient so cardinality/distribution survives the behavioural merge
            constraints=frozenset(remap(c, quotient) for c in g.constraints),
            weights=quotient_weights(g),
        )
        for g in sig.generators
    }
    return Signature(frozenset(gens))


def gamma_normalize(sig: Signature, *, gamma1: str = "gamma1", gamma2: str = "gamma2") -> Signature:
    """Present every notation's boundary in the **common γ1/γ2 convention** (the
    master's), so the per-notation ``cospans.svg`` and the comparison line up like-for-
    like. Two adapter conventions are folded:

    1. **Explicit start/end generators** (OCCN's ``START_<ot>``/``END_<ot>``, §13) are
       *relabelled* to ``gamma1``/``gamma2`` -- as the generator label **and** as the
       port ``src``/``tgt`` everywhere they appear, so the wires stay balanced (an
       interior leg ``(START_order, order, a)`` becomes ``(gamma1, order, a)``, matching
       the master). Constraint leg-ports are remapped the same way.
    2. **Open boundaries** (an OCPN source activity is *zero-left*, a sink *zero-right*;
       it carries no boundary generator at all): a ``gamma1`` source leg is **synthesised**
       into each zero-left activity (one per object type it emits) and a ``gamma2`` sink
       leg out of each zero-right activity, with the matching γ generator added. This
       turns an OCPN ``a: · → order`` into ``a: (gamma1,order,a) → order`` -- the same
       generator the master/OCCN carry.

    Existing γ generators (already labelled ``gamma1``/``gamma2``) are left untouched, so
    the hand-authored master is a **no-op**. The γ generators themselves may still differ
    in arity across notations (the master *joins* an order + carrier origin in one γ1;
    discovery finds *per-type* origins) -- that is a real structural difference the
    comparison should show, now under one shared label rather than two encodings."""
    from .constraints import remap

    def rename(name: str) -> str:
        if name.startswith("START_"):
            return gamma1
        if name.startswith("END_"):
            return gamma2
        return name

    def relabel_port(p: Port) -> Port:
        return Port(rename(p.src), p.typ, rename(p.tgt))

    # (1) relabel explicit START_*/END_* boundary -> gamma1/gamma2
    relabelled: set[Generator] = set()
    for g in sig.generators:
        relabelled.add(Generator(
            label=rename(g.label),
            left=frozenset(relabel_port(p) for p in g.left),
            right=frozenset(relabel_port(p) for p in g.right),
            constraints=frozenset(remap(c, relabel_port) for c in g.constraints),
            weights=frozenset((relabel_port(p), w) for p, w in g.weights),
        ))

    # (2) synthesise γ for open (OCPN-style) boundaries; skip generators that are
    # already a γ source/sink.
    gens: set[Generator] = set()
    g1_legs: set[Port] = set()
    g2_legs: set[Port] = set()
    for g in relabelled:
        if g.label in (gamma1, gamma2):
            gens.add(g)
            continue
        if not g.left and g.right:  # zero-left source activity -> feed it from gamma1
            new_left = {Port(gamma1, ot, g.label) for ot in {p.typ for p in g.right}}
            g1_legs |= new_left
            gens.add(Generator(g.label, frozenset(new_left), g.right, g.constraints, g.weights))
        elif not g.right and g.left:  # zero-right sink activity -> drain it into gamma2
            new_right = {Port(g.label, ot, gamma2) for ot in {p.typ for p in g.left}}
            g2_legs |= new_right
            gens.add(Generator(g.label, g.left, frozenset(new_right), g.constraints, g.weights))
        else:
            gens.add(g)

    for leg in g1_legs:
        gens.add(Generator(gamma1, frozenset(), frozenset({leg})))
    for leg in g2_legs:
        gens.add(Generator(gamma2, frozenset({leg}), frozenset()))
    return Signature(frozenset(gens))


def close_gamma2_termini(sig: Signature, *, gamma2: str = "gamma2") -> Signature:
    """Give every **unconsumed** ``gamma2``-terminus leg an explicit ``gamma2`` drain
    generator, so a cospan whose interior activity emits a *terminating* object can still
    close.

    The discovered OCPN's ``s`` sends its carrier (container/box) to a bare final place;
    ``engine._traverse`` now surfaces that as a ``(s, carrier, gamma2)`` right leg (the
    carrier-drop fix) instead of silently absorbing it. But the closing DP
    (``extract_dp``) drains a frontier port only through a *consuming* generator, and a bare
    sink has none -- so without a drain the carrier leg keeps the frontier non-empty and the
    OCPN yields **zero** closings. This adds one zero-right ``gamma2`` generator per such
    leg.

    No-op where the terminus is already consumed: the hand-authored master's own ``gamma2``
    generators, and the OCCN's ``END_<ot>`` (a different terminus marker). Deliberately one-
    sided -- there is **no** ``gamma1`` counterpart: origins stay bare zero-left *activities*
    that co-fire one-per-label (§40a), whereas a ``gamma2`` drain is zero-*right* and so is
    not single-firing-constrained; several may co-drain in one closing."""
    consumed = {p for g in sig.generators for p in g.left}
    termini = {
        p for g in sig.generators for p in g.right
        if p.tgt == gamma2 and p not in consumed
    }
    if not termini:
        return sig
    drains = {Generator(gamma2, frozenset({p}), frozenset()) for p in termini}
    return Signature(frozenset(sig.generators | drains))


def degenerate_filtered(sig: Signature) -> Signature:
    """Drop degree-(1,1) self-bounce contexts.

    An activity is degree-(1,1) if every one of its generators has exactly
    one left port and one right port. For such an activity, a context is a
    self-bounce iff the single predecessor label or the single successor
    label equals the activity's own label -- there is no second leg that
    could ever justify a genuine self-reference, so this is a discovery
    artifact, not real behaviour. (An activity that legitimately repeats --
    e.g. one whose generators always carry several simultaneous ports --
    is never degree-(1,1) in the first place, so this filter cannot remove
    a real self-loop; confirmed directly against ground truth for the ED
    OCEL: `re_examine`'s self-loop generators all
    carry 3 ports and survive; `troponin_neg`'s lone degree-(1,1) self-loop
    generator, which never appears as a literal adjacent repeat anywhere in
    the underlying log, does not.)
    """
    gens = list(sig.generators)
    by_label: dict[str, list[Generator]] = {}
    for g in gens:
        by_label.setdefault(g.label, []).append(g)

    degree_11_labels = {
        lab for lab, gs in by_label.items()
        if all(len(g.left) == 1 and len(g.right) == 1 for g in gs)
    }

    def is_self_bounce(g: Generator) -> bool:
        if g.label not in degree_11_labels:
            return False
        (pred,) = g.left
        (succ,) = g.right
        return pred.src == g.label or succ.tgt == g.label

    return Signature(frozenset(g for g in gens if not is_self_bounce(g)))
