"""Discover models from an OCEL and convert each to a cospan signature.

For one OCEL cell we flatten on every object type and, per model class,
discover a per-type *raw* pm4py model and feed it directly to that
notation's own LM-graph adapter (no detour through Petri nets):

  PN   inductive Petri net               -> from_petri
  PT   inductive process tree            -> from_process_tree
  BPMN inductive tree -> BPMN             -> from_bpmn
  CN   heuristics net (causal-net analogue) -> from_heuristics   [independent miner]

Each model class is converted by its own native adapter so that PN/PT/BPMN/CN
agreement (or disagreement) is a genuine empirical finding, not a foregone
conclusion of funnelling everything through one converter first. (pm4py's own
model-to-model converters can be exercised separately as a secondary soundness
check.)

BPMN has no "transparent" node class (``def:bpmn``): its start/end events are
real activities. So a BPMN signature carries two extra boundary generators
(``start``, ``end``) that PN/PT/CN never expose -- a real, documented vocabulary
difference, not an adapter bug; see ``equivalence.drop_boundary_activities`` for
a like-for-like comparison.
"""

from __future__ import annotations

import pm4py

from .adapters.from_bpmn import lmgraph_from_bpmn_diagrams
from .cospan.from_heuristics import lmgraph_from_heuristics_nets
from .cospan.from_ocpn import signature_from_ocpn as _signature_from_ocpn
from .cospan.from_petri import lmgraph_from_petri_nets
from .adapters.from_process_tree import lmgraph_from_process_trees
from .cospan.discovery_cleanup import forget_provenance
from .cospan.engine import extract_signature
from .cospan.extract_dp import extract_classes
from .cospan.signature import Signature
from .cospan.splice import SpliceRepresentation
from .occn import mine_occn, occn_to_signature

MODEL_CLASSES = ("PN", "PT", "BPMN", "CN")


def discover_raw(cls: str, flat) -> object | None:
    """Discover one per-type *native* model of the given class; ``None`` on
    failure (e.g. heuristics miner declining on a degenerate flattened log)."""
    try:
        if cls == "PN":
            net, _, _ = pm4py.discover_petri_net_inductive(flat)
            return net
        if cls == "PT":
            return pm4py.discover_process_tree_inductive(flat)
        if cls == "BPMN":
            tree = pm4py.discover_process_tree_inductive(flat)
            return pm4py.convert_to_bpmn(tree)
        if cls == "CN":
            return pm4py.discover_heuristics_net(flat)
    except Exception:
        return None
    raise ValueError(f"unknown model class {cls}")


_LMGRAPH_BUILDER = {
    "PN": lmgraph_from_petri_nets,
    "PT": lmgraph_from_process_trees,
    "BPMN": lmgraph_from_bpmn_diagrams,
    "CN": lmgraph_from_heuristics_nets,
}


def discover_raw_models(ocel) -> dict[str, dict[str, object]]:
    """Map each model class to ``{otype: native pm4py model}`` for one OCEL."""
    otypes = pm4py.ocel_get_object_types(ocel)
    flats = {ot: pm4py.ocel_flattening(ocel, ot) for ot in otypes}
    out: dict[str, dict[str, object]] = {}
    for cls in MODEL_CLASSES:
        models = {}
        for ot, flat in flats.items():
            model = discover_raw(cls, flat)
            if model is not None:
                models[ot] = model
        out[cls] = models
    return out


def signatures_from_raw(raw: dict[str, dict[str, object]]) -> dict[str, Signature]:
    return {
        cls: extract_signature(_LMGRAPH_BUILDER[cls](models))
        for cls, models in raw.items()
    }


def signature_from_ocpn(ocel, *, canonical: bool = False) -> Signature:
    """Signature of the *discovered object-centric Petri net* itself.

    Unlike the ``PN`` class (which re-discovers a per-type inductive net per
    flattened log), this reads ``pm4py.discover_oc_petri_net``'s OCPN object
    directly via :func:`.cospan.from_ocpn.lmgraph_from_ocpn`.

    ``canonical`` defaults ``False`` here (unlike the underlying
    :func:`.cospan.from_ocpn.signature_from_ocpn`): established consumers of
    this entry point feed the result to the splice/behavioural machinery
    (``extract_classes``), which needs the full concrete-port signature -- the
    migration-discipline byte-for-byte contract pins that.  Pass
    ``canonical=True`` for the CanonKey-level views (compare, inventory,
    localisation) on wide OCPNs, where the full extraction is intractable."""
    ocpn = pm4py.discover_oc_petri_net(ocel)
    # surface_termini: the OCPN is the object-centric net whose full final marking matters --
    # an object that ends at a transition's final place (the ``s`` carrier) becomes a
    # ``gamma2`` leg, matching the OCCN's ``END_<ot>`` and the master's ``gamma2`` carrier.
    # Delegate the lmgraph_from_ocpn + extract pipeline to its single home in from_ocpn.
    return _signature_from_ocpn(ocpn, surface_termini=True, canonical=canonical)


def signature_from_occn(ocel, *, bindings: bool = True) -> Signature:
    """Signature of the discovered object-centric causal net (Liss et al.),
    via the paper-faithful miner :func:`.occn.mine_occn` and the cospan
    adapter :func:`.occn.occn_to_signature`. ``bindings=False`` gives the plain
    1-1, key-free signature (see :func:`.occn.occn_to_signature`)."""
    return occn_to_signature(mine_occn(ocel), bindings=bindings)


# object-centric model classes are discovered from the WHOLE ocel (not per
# flattened type like MODEL_CLASSES), so they have their own producers.
OC_MODEL_CLASSES = ("OCPN", "OCCN")
_OC_SIGNATURE = {"OCPN": signature_from_ocpn, "OCCN": signature_from_occn}


def discover_model(ocel, cls: str):
    """Return ``(raw_model, signature)`` for one class. ``raw_model`` is the
    native discovered object kept for visualisation -- the **OCCN**
    (:func:`.occn.mine_occn`, drawn by :func:`.viz.occn_vis.draw_occn`) or the
    **OCPN** (``pm4py.discover_oc_petri_net``, drawn by ``pm4py.save_vis_ocpn``);
    ``None`` for the per-type classes, whose visualisation falls back to the
    generator string-diagram grid. Mining the model and its signature together
    avoids re-discovering the (expensive) OC model just to draw it."""
    if cls == "OCCN":
        occn = mine_occn(ocel)
        return occn, occn_to_signature(occn)
    if cls == "OCPN":
        ocpn = pm4py.discover_oc_petri_net(ocel)
        # canonical=False: discover_model's signature feeds behaviour-level
        # consumers (splice/extract_classes need concrete ports, not the fast
        # path's placeholder representatives).
        return ocpn, _signature_from_ocpn(ocpn, surface_termini=True, canonical=False)
    return None, discover_signatures(ocel, object_centric=False).get(cls)


def splice_representation_from_signature(
    sig: Signature, *, name: str, quotient: bool = True, one_origin: bool = False,
    prune_bound: int | None = None,
) -> SpliceRepresentation:
    """The splice representation of one discovered signature -- its ``M(m,σ)``
    family catalogue (concrete pomsets + algebraic skeletons), the finite
    generating grammar of the trace language.

    ``quotient`` applies :func:`forget_provenance` (the behavioural quotient that
    makes loops close as finite cycles) -- mandatory for OCPN tractability and the
    right semantic object; on a clean signature it is a no-op.

    ``one_origin`` enforces the single-origin / one-process-instance rule: at most
    one zero-left (``gamma1``) generator fires per closing, so degenerate-label
    ``gamma1`` cospans (e.g. the container-route vs box-route origins of an
    object-centric master) are read as XOR alternatives, not composed into a
    2-instance closing. Off by default (legacy behaviour).

    ``prune_bound``: if set, drop families whose accumulated leg-constraint system
    is infeasible at that bound -- rejects multi-instance over-generation via
    contradictory cardinality constraints (the constraint-based analogue of
    ``one_origin``, and the only one that works when object types co-start)."""
    s = forget_provenance(sig) if quotient else sig
    # NB: ``extract_classes`` adds the gamma2 termini-drains (``close_gamma2_termini``) so the
    # OCPN carrier can close; no need to repeat it here.
    return SpliceRepresentation.from_extraction_result(
        extract_classes(s, one_origin=one_origin),
        name=name, quotient="forget_provenance" if quotient else "none",
        prune_bound=prune_bound,
    )


def discover_signatures(ocel, object_centric: bool = True) -> dict[str, Signature]:
    """Map each model class to its cospan signature, built directly from that
    notation's native discovered model.

    Always returns PN/PT/BPMN/CN. With ``object_centric`` (default), also returns
    OCPN and OCCN -- the genuine object-centric notations. The OC producers are
    best-effort here: a failure yields an empty signature rather than aborting the
    whole call (the standalone ``signature_from_*`` functions do surface errors)."""
    sigs = signatures_from_raw(discover_raw_models(ocel))
    if object_centric:
        for cls, producer in _OC_SIGNATURE.items():
            try:
                sigs[cls] = producer(ocel)
            except Exception:
                sigs[cls] = Signature(frozenset())
    return sigs
