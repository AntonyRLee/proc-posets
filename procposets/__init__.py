"""procposets: reusable poset / cospan / string-diagram core.

Extracted from three research repos (poset-mixture-npmle, sim/cpm,
stochastic_process_mining/spm); see README.md and the consumer repos'
.claude/REFACTOR_DESIGN.md.

Two poset views coexist:

- ``Rel = frozenset[(label, label)]`` with a free-function toolkit
  (``rel`` module) -- the estimation vocabulary, distinct-label only.
- the canonical id+label ``Poset`` object (``poset`` module) -- carries
  repeated labels; ``bridge.to_rel`` / ``bridge.from_rel`` convert.

Layers land phase by phase: this exposes A0 (the pure-stdlib poset core) and
A1 (numpy estimation + stdlib stochastic distance).  The cospan algebra (B0+)
arrives with later phases.
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
from .poset import count_extensions  # guarded e(P) on the canonical Poset
from ._extensions import IdealBudgetExceeded

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

# ===========================================================================
# Layer A1 — estimation (numpy) + stochastic distance (stdlib)
# ===========================================================================

# NPMLE certified mixture estimator + M9 moment initialiser (numpy)
from .likelihood import Atom, GroupedLog, TimedGroupedLog, make_atom
from .npmle import (
    FitResult,
    fit,
    polish_nuisances,
    refit_weights,
    trivial_chain_loglik,
)
from .oracle import Oracle
from .initialiser import (
    find_margin_equivalences,
    margin_equivalent,
    moment_seed,
    poset_moment,
)
from .simulate import (
    TrueMixture,
    sample_grouped_log,
    sample_keyed_log,
    sample_timed_grouped_log,
)
from .diagnostics import (
    bootstrap_weights,
    identifiability_report,
    recovery_report,
)

# SPME stochastic-matrix distance + known-law EM/counting (stdlib)
from .distance import bhattacharyya_angle, smd, smd_pairwise, smd_rows
from .matrix import build as build_block_matrix
from .matrix import normal_form_distribution
from .estimate import (
    log_likelihood,
    mixture_law,
    reweight,
    rho_counting,
    rho_mle,
    variant_laws,
)
from .loops import empirical_loop_model, loop_limit, loop_model, unrolling

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
    "count_extensions", "IdealBudgetExceeded",
    # total modular decomposition
    "Leaf", "Series", "Parallel", "Prime", "modular_decompose", "tiling",
    # traces
    "linear_extensions", "trace_distribution", "trace_bhattacharyya",
    # grouping
    "group_by_key",
    # bridge
    "to_rel", "from_rel", "rel_elements", "LabelCollision",
    # A1 estimation (numpy)
    "Atom", "GroupedLog", "TimedGroupedLog", "make_atom",
    "FitResult", "fit", "polish_nuisances", "refit_weights",
    "trivial_chain_loglik", "Oracle",
    "moment_seed", "poset_moment", "margin_equivalent",
    "find_margin_equivalences",
    "TrueMixture", "sample_grouped_log", "sample_keyed_log",
    "sample_timed_grouped_log",
    "recovery_report", "identifiability_report", "bootstrap_weights",
    # A1 stochastic distance + known-law estimators (stdlib)
    "smd", "smd_rows", "smd_pairwise", "bhattacharyya_angle",
    "build_block_matrix", "normal_form_distribution",
    "variant_laws", "reweight", "mixture_law", "rho_counting", "rho_mle",
    "log_likelihood",
    "unrolling", "loop_model", "loop_limit", "empirical_loop_model",
]
