"""Adapter: a pm4py *object-centric* Petri net -> logic-mediated LM-graph.

``pm4py.discover_oc_petri_net`` returns an OCPN dict whose ``petri_nets`` field
is ``{otype: (PetriNet, initial_marking, final_marking)}`` -- per-type nets that
share activity labels.  That is exactly the typed-merge structure
:func:`procposets.cospan.from_petri.lmgraph_from_petri_nets` already consumes, so this
adapter just extracts the per-type nets and reuses it.

Distinction from a per-type *re-discovery* model: such a model *re-discovers*
a per-type inductive Petri net on each flattened log, whereas this
consumes the **discovered OCPN object itself** -- its silent transitions,
variable (double) arcs and shared places included -- so the resulting signature
is the algebraic presentation of the OCPN as discovered, not of a re-derivation.
"""
from __future__ import annotations

from .engine import extract_signature
from .engine_fast import extract_signature_fast
from .from_petri import lmgraph_from_petri_nets
from .lmgraph import LMGraph
from .signature import Signature


def lmgraph_from_ocpn(ocpn: dict) -> LMGraph:
    """Build a typed LM-graph from a ``discover_oc_petri_net`` result."""
    nets = {otype: triple[0] for otype, triple in ocpn["petri_nets"].items()}
    return lmgraph_from_petri_nets(nets)


def signature_from_ocpn(ocpn: dict, *, canonical: bool = True,
                        surface_termini: bool = False,
                        remove_silent: bool = True) -> Signature:
    """OCPN dict -> generator signature ``Sigma`` (the ``lmgraph_from_ocpn`` +
    extractor pipeline in one call).

    ``canonical=True`` (default) runs the output-sensitive
    :func:`engine_fast.extract_signature_fast`: one representative generator per
    CanonKey, which stays tractable on *wide* object-centric nets where the full
    extractor's cross-type ``|B|x|F|`` product blows up (a hub shared across many
    object types is exponential).  This is the right object for the type-level
    views (:func:`signature_compare.compare`, inventory, localisation) -- the
    dominant OCPN use -- and agrees exactly with the full extractor on the
    CanonKey set, so the default cannot hang on a wide discovered net.

    ``canonical=False`` runs :func:`engine.extract_signature` -- the full
    per-context signature carrying every concrete firing context.  Required for
    the splice/behavioural semantics (``extract_classes`` and friends consume
    concrete ports; the canonical representatives carry synthetic placeholder
    neighbours), and only tractable where ``|B|x|F|`` is.  For *composition*
    behaviour prefer :func:`skeleton.extract_skeleton` +
    :func:`compose.compose_signature`, which never materialise that product.

    OCPN callers typically want ``surface_termini=True`` (keep an object that
    terminates at a bare sink place as a ``gamma2`` right leg -- the object-centric
    final marking); it defaults ``False`` here to match :func:`extract_signature`."""
    g = lmgraph_from_ocpn(ocpn)
    extract = extract_signature_fast if canonical else extract_signature
    return extract(g, surface_termini=surface_termini, remove_silent=remove_silent)
