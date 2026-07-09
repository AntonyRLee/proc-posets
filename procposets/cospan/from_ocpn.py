"""Adapter: a pm4py *object-centric* Petri net -> logic-mediated LM-graph.

``pm4py.discover_oc_petri_net`` returns an OCPN dict whose ``petri_nets`` field
is ``{otype: (PetriNet, initial_marking, final_marking)}`` -- per-type nets that
share activity labels.  That is exactly the typed-merge structure
:func:`cpm.cospan.from_petri.lmgraph_from_petri_nets` already consumes, so this
adapter just extracts the per-type nets and reuses it.

Distinction from the ``PN`` model class in :mod:`cpm.discover`: ``PN``
*re-discovers* a per-type inductive Petri net on each flattened log, whereas this
consumes the **discovered OCPN object itself** -- its silent transitions,
variable (double) arcs and shared places included -- so the resulting signature
is the algebraic presentation of the OCPN as discovered, not of a re-derivation.
"""
from __future__ import annotations

from .from_petri import lmgraph_from_petri_nets
from .lmgraph import LMGraph


def lmgraph_from_ocpn(ocpn: dict) -> LMGraph:
    """Build a typed LM-graph from a ``discover_oc_petri_net`` result."""
    nets = {otype: triple[0] for otype, triple in ocpn["petri_nets"].items()}
    return lmgraph_from_petri_nets(nets)
