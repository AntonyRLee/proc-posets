"""procposets: reusable poset / cospan / string-diagram core.

Extracted from three research repos (poset-mixture-npmle, sim/cpm,
stochastic_process_mining/spm); see README.md and the consumer repos'
.claude/REFACTOR_DESIGN.md.

Two poset views coexist:

- ``Rel = frozenset[(label, label)]`` with a free-function toolkit
  (``rel`` module) -- the estimation vocabulary, distinct-label only.
- the canonical id+label ``Poset`` object (``poset`` module) -- carries
  repeated labels; ``bridge.to_rel`` / ``bridge.from_rel`` convert.

Layers land phase by phase: this exposes A0 (the pure-stdlib poset core).
Estimation (A1, numpy) and the cospan algebra (B0+) arrive with later phases.
"""

__version__ = "0.1.0"

# --- Rel toolkit (distinct-label view; NPMLE engine) -----------------------
from .rel import (
    GENERAL,
    SP,
    Rel,
    SPTree,
    count_linear_extensions,
    decompose,
    describe,
    enumerate_posets,
    enumerate_sp,
    extension_count,
    get_poset_class,
    is_partial_order,
    is_sp,
    meet,
    meet_closure,
    parallel,
    refines,
    rel_from_trace,
    respects,
    sample_extension,
    sample_linear_extension,
    series,
    transitive_reduction,
    tree_relations,
)

# --- canonical Poset object (id+label; SPME base) --------------------------
from .poset import Poset, from_dag, from_edges, leaf, n_poset, par, then

# --- total modular decomposition (SPME) ------------------------------------
from .moddecomp import Leaf, Parallel, Prime, Series
from .moddecomp import decompose as modular_decompose
from .moddecomp import tiling

# --- trace-level views (SPME) ----------------------------------------------
from .traces import linear_extensions, trace_bhattacharyya, trace_distribution

# --- grouping (NPMLE) ------------------------------------------------------
from .grouping import group_by_key

# --- the certified Rel <-> Poset bridge ------------------------------------
from .bridge import LabelCollision, from_rel, rel_elements, to_rel

__all__ = [
    # Rel toolkit
    "Rel", "rel_from_trace", "respects", "meet", "refines", "is_partial_order",
    "transitive_reduction", "count_linear_extensions", "sample_linear_extension",
    "enumerate_posets", "meet_closure", "describe", "get_poset_class",
    "GENERAL", "SP",
    # SP-tree view
    "SPTree", "decompose", "is_sp", "extension_count", "sample_extension",
    "tree_relations", "series", "parallel", "enumerate_sp",
    # canonical Poset
    "Poset", "leaf", "then", "par", "n_poset", "from_dag", "from_edges",
    # total modular decomposition
    "Leaf", "Series", "Parallel", "Prime", "modular_decompose", "tiling",
    # traces
    "linear_extensions", "trace_distribution", "trace_bhattacharyya",
    # grouping
    "group_by_key",
    # bridge
    "to_rel", "from_rel", "rel_elements", "LabelCollision",
]
