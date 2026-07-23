"""Object-Centric Causal Net discovery (Liss et al., CAiSE 2025), paper-faithful.

Two phases:
  * :mod:`.fhm`        -- Flexible Heuristics Miner -> typed object-centric
                          dependency graph (:class:`.fhm.OCDG`).
  * :mod:`.markers`    -- Listing-1.1 marker-group discovery -> :class:`.markers.OCCN`.
And :mod:`.to_signature` lifts a mined OCCN into the cospan ``Signature`` used by
the rest of :mod:`procposets.cospan`.

Validated against the authors' reference implementation; the validation harness
(oracle bridge, structural diff, visualisers, generated logs) is maintained separately.

Scope (migration WS 2.4): this subpackage is an **optional inbound adapter** --
it turns an object-centric event log into a cospan ``Signature`` for the rest of
the library to work on. It is deliberately kept OUT of the top-level
``procposets`` ``__all__`` (import it explicitly as ``procposets.occn``); the
numpy-free algebraic core neither imports nor depends on it. The miner itself is
stdlib-only, but the logs it consumes are produced upstream via the ``[pm4py]``
path, so treat it as the ``[pm4py]``-adjacent discovery front-end.
"""
from .fhm import OCDG, mine_ocdg
from .markers import OCCN, Marker, MarkerGroup, mine_occn
from .to_signature import occn_generator_counts, occn_to_signature
from .unroll_occn import gamma_boundary, ground_occn, ground_run

__all__ = [
    "OCDG",
    "mine_ocdg",
    "OCCN",
    "Marker",
    "MarkerGroup",
    "mine_occn",
    "occn_to_signature",
    "occn_generator_counts",
    "ground_occn", "ground_run", "gamma_boundary",
]
