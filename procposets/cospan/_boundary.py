"""Boundary-label constants for the cospan algebra -- the single source of truth.

Dependency-free leaf module (imports nothing from ``cospan``) so that both the
numpy-only core (``engine``/``engine_fast``) and the ``[graph]``-layer occurrence
/splice logic can share these labels. ``engine`` previously carried its own local
``GAMMA2 = "gamma2"`` copy expressly to avoid importing the networkx-bearing
``occurrence`` module (the import cycle noted in its old comment); routing both
through this leaf breaks that cycle cleanly without pulling networkx into the core.

* ``GAMMA1``/``GAMMA2`` -- the §40 single master boundary source/sink labels.
  ASCII (the canonical-key WL hash encodes node labels as ASCII); a renderer may
  prettify them to ``γ1``/``γ2`` for display only.
* ``BOUNDARY_PREFIXES`` -- the OCCN per-type origin/terminus wrappers, shared with
  the splice-site/skeleton logic (``splice._is_wrapper_label``, ``signature_diff``).
  Must stay ``START_``/``END_`` only, or those coordinate systems shift.
"""
from __future__ import annotations

GAMMA1 = "gamma1"
GAMMA2 = "gamma2"

BOUNDARY_PREFIXES = ("START_", "END_")
