"""Object-Centric Causal Net discovery (Liss et al., CAiSE 2025), paper-faithful.

Two phases:
  * :mod:`.fhm`        -- Flexible Heuristics Miner -> typed object-centric
                          dependency graph (:class:`.fhm.OCDG`).
  * :mod:`.markers`    -- Listing-1.1 marker-group discovery -> :class:`.markers.OCCN`.
And :mod:`.to_signature` lifts a mined OCCN into the cospan ``Signature`` used by
the rest of :mod:`cpm.cospan`.

Validated against the authors' reference implementation; the validation harness
(oracle bridge, structural diff, visualisers, generated logs) lives in
``sim/occn_dev/``. See ``sim/occn_dev/OCCN_DEV.md`` for the design/finding log.
"""
from .fhm import OCDG, mine_ocdg
from .markers import OCCN, Marker, MarkerGroup, mine_occn
from .to_signature import occn_to_signature

__all__ = [
    "OCDG",
    "mine_ocdg",
    "OCCN",
    "Marker",
    "MarkerGroup",
    "mine_occn",
    "occn_to_signature",
]
