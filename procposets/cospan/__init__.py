"""procposets.cospan -- the cospan / string-diagram algebra.

Modules are imported by full path (matching the source repo's usage), so
``import procposets.cospan`` pulls nothing heavy.  Two sub-layers live here:

- **B0 (pure stdlib):** signature, lmgraph, engine, engine_fast, typebalance,
  constraints, feasibility, compose, class_extraction, morphism_schema,
  signature_compare, signature_diff, discovery_cleanup, from_petri, from_ocpn,
  unroll_core.  Usable with the numpy-only core, no extra.
- **B1 (needs the [graph] extra -- networkx):** occurrence, extract_dp,
  dag_diff, splice, trace_language, from_heuristics, equivalence.
  (class_extraction pulls extract_dp lazily, so it stays B0 until its extraction
  path is called.)

The pm4py model adapters live in ``procposets.adapters``; ``procposets.occn``
is the object-centric causal-net miner.  ``cospan.equivalence`` is the [graph]
structural-equivalence surface.
"""

# The output-sensitive extractor is a light B0-only convenience surfaced at the
# package root (the exact CanonKey twin of engine.extract_signature for wide
# object-centric nets); everything else is imported by full module path.
from .engine_fast import extract_signature_fast  # noqa: E402,F401

__all__ = ["extract_signature_fast"]
