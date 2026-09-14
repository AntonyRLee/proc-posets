"""procposets: reusable poset / cospan / string-diagram core.

Extracted and unified from three research codebases into a single
standalone library; see README.md.

Two poset views coexist:

- ``Rel = frozenset[(label, label)]`` with a free-function toolkit
  (``rel`` module) -- the estimation vocabulary, distinct-label only.
- the canonical id+label ``Poset`` object (``poset`` module) -- carries
  repeated labels; ``bridge.to_rel`` / ``bridge.from_rel`` convert.

**Dependency-free core.**  ``import procposets`` and every stdlib module
(poset algebra, decomposition, traces, grouping, the stochastic-matrix
distance and known-law estimators) pull **no third-party package**.  The
numpy layer (
``initialiser``/``diagnostics``) is loaded **lazily** and needs the
``[estimate]`` extra; networkx/pm4py/matplotlib sit behind ``[graph]`` /
``[pm4py]`` / ``[viz]``.  So the top-level names ``fit``, ``GroupedLog``,
``Oracle``, ``moment_seed``, ``recovery_report`` etc. resolve on first access
(PEP 562) and only then import numpy -- consumers that never touch the
estimator (e.g. the stochastic-distance stack) install numpy-free.
"""

__version__ = "0.1.0"

# --- Rel toolkit (distinct-label view) ------------------------------------
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
    join,
    meet,
    concept_intents,
    meet_closure,
    parallel,
    refines,
    rel_from_trace,
    respects,
    sample_extension_tree,
    sample_linear_extension,
    series,
    transitive_reduction,
    tree_relations,
    width,
)

# --- canonical Poset object (id+label; SPME base) --------------------------
from .poset import (
    Model,
    Poset,
    from_dag,
    from_edges,
    leaf,
    n_poset,
    par,
    sample_extension,
    then,
)
from .poset import count_extensions  # guarded e(P) on the canonical Poset
from ._extensions import IdealBudgetExceeded

# --- total modular decomposition (SPME) ------------------------------------
from .moddecomp import Leaf, Parallel, Prime, Series
from .moddecomp import decompose as modular_decompose
from .moddecomp import tiling

# --- trace-level views (SPME) ----------------------------------------------
from .traces import linear_extensions, trace_bhattacharyya, trace_distribution


# --- choosing a grouping key, and pricing the design it implies ------------
# Both stdlib-only, so they sit in the numpy-free core: `keys` grades candidate
# keys before a fit, `planning` says what the resulting design can resolve.
from .loopscan import RecurrenceSpectrum, format_spectrum, recurrence_spectrum

# --- the certified Rel <-> Poset bridge ------------------------------------
from .bridge import LabelCollision, from_rel, rel_elements, to_rel

# ===========================================================================
# Layer A1 (stdlib) — simulation, stochastic-matrix distance, known-law EM
# (numpy-free; eager)
# ===========================================================================

from .simulate import (
    TrueMixture,
    sample_grouped_log,
    sample_keyed_log,
    sample_timed_grouped_log,
)
from .distance import Mode, bhattacharyya_angle, smd, smd_pairwise, smd_rows
from .matrix import build as build_block_matrix
from .matrix import normal_form_distribution
from .estimate import (
    Law,
    Trace,
    log_likelihood,
    mixture_law,
    reweight,
    rho_counting,
    rho_mle,
    variant_laws,
)
from .loops import empirical_loop_model, loop_limit, loop_model, unrolling

# ===========================================================================
# Layer A1 (numpy) — MCMC extension sampling, loaded LAZILY so
# `import procposets` stays numpy-free.  Each name below resolves on first
# access (PEP 562 __getattr__), importing its module (which imports numpy)
# only then.  Needs the [estimate] extra.
# ===========================================================================

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-only re-exports: give type-checkers the real signatures of the
    # lazily-loaded numpy layer.  TYPE_CHECKING is False at runtime, so this
    # never imports numpy -- the PEP 562 __getattr__ below does the actual lazy
    # resolution and the numpy-free import contract (test_lazy_numpy) is intact.
    from .approx_count import (
        CountEstimate,
        approx_count_extensions,
        sample_extensions_mcmc,
    )

_LAZY = {  # top-level name -> (submodule, attribute)
    **{n: ("approx_count", n) for n in
       ("CountEstimate", "approx_count_extensions", "sample_extensions_mcmc")},
    # Label-conditional summaries of a fit.  numpy layer, so lazy.
    # Grading a candidate grouping key before any fit.  Stdlib-only; lazy for
    # the same reason as the screen below -- not needed to import the library.
    # Input screening. Stdlib-only, so lazy here purely to keep package import cheap --
    # nothing below pulls in the XML parser unless a screen is actually asked for.
    **{n: ("logscreen", n) for n in
       ("TieReport", "OverlapReport", "LoopReport", "tie_composition",
        "interval_overlap", "loop_composition", "iter_cases",
        "DEFAULT_TERMINATORS")},
}


def __getattr__(name: str) -> object:  # PEP 562: lazy numpy-layer resolution
    info = _LAZY.get(name)
    if info is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    val = getattr(importlib.import_module(f"{__name__}.{info[0]}"), info[1])
    globals()[name] = val  # cache so subsequent access skips __getattr__
    return val


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))

__all__ = [
    # Rel toolkit
    "Rel", "rel_from_trace", "respects", "meet", "join", "width",
    "refines", "is_partial_order",
    "transitive_reduction", "count_linear_extensions", "sample_linear_extension",
    "enumerate_posets", "meet_closure", "concept_intents", "describe", "get_poset_class",
    "GENERAL", "SP",
    # SP-tree view
    "SPTree", "decompose", "is_sp", "extension_count",
    "sample_extension_tree",
    "tree_relations", "series", "parallel", "enumerate_sp",
    # canonical Poset
    "Poset", "Model", "leaf", "then", "par", "n_poset", "from_dag", "from_edges",
    "sample_extension", "count_extensions", "IdealBudgetExceeded",
    # (ε,δ) wide-poset extension-count estimator (numpy; lazy)
    "CountEstimate", "approx_count_extensions", "sample_extensions_mcmc",
    # total modular decomposition
    "Leaf", "Series", "Parallel", "Prime", "modular_decompose", "tiling",
    # traces
    "linear_extensions", "trace_distribution", "trace_bhattacharyya",
    # grouping-key diagnostics (stdlib)
    "recurrence_spectrum", "RecurrenceSpectrum", "format_spectrum",
    # bridge
    "to_rel", "from_rel", "rel_elements", "LabelCollision",
    # A1 estimation (numpy)
    "TrueMixture", "sample_grouped_log", "sample_keyed_log",
    "sample_timed_grouped_log",
    # A1 stochastic distance + known-law estimators (stdlib)
    "smd", "smd_rows", "smd_pairwise", "bhattacharyya_angle", "Mode",
    "build_block_matrix", "normal_form_distribution",
    "variant_laws", "reweight", "mixture_law", "rho_counting", "rho_mle",
    "log_likelihood", "Trace", "Law",
    "unrolling", "loop_model", "loop_limit", "empirical_loop_model",
    # input screening: does this log carry within-case concurrency at all? (stdlib)
    "TieReport", "OverlapReport", "LoopReport", "tie_composition",
    "interval_overlap", "loop_composition", "iter_cases",
    "DEFAULT_TERMINATORS",
]
