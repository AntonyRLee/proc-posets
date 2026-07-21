"""Compatibility shim: this module moved to ``procposets.cospan.equivalence``.

``equivalence`` is a networkx-backed, ``cospan.signature``-dependent B1 module
that was mis-placed beside the numpy-free core; it now lives under ``cospan/``.
This shim re-exports it so ``import procposets.equivalence`` keeps working (the
public import path used by the golden cross-checks).  New code should import
from ``procposets.cospan.equivalence`` directly.
"""

from .cospan.equivalence import (
    TraceCheck,
    diff,
    drop_boundary_activities,
    equal,
    isomorphic,
    jaccard,
    trace_language_check,
)

__all__ = [
    "TraceCheck",
    "diff",
    "drop_boundary_activities",
    "equal",
    "isomorphic",
    "jaccard",
    "trace_language_check",
]
