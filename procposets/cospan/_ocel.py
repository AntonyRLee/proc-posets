"""Shared OCEL column vocabulary + frame assembly for the cospan simulators.

The pm4py OCEL column keys and the events/objects/relations DataFrame build
common to :mod:`cospan.simulate` (cospan -> OCEL sampler) and
:mod:`cospan.faithful_simulate` (constraint-driven faithful sampler). Lives
under ``cospan`` behind the ``[pm4py]`` extra (both importers already need
pm4py). The per-module simulation epoch (``_BASE``/``_STEP``) is deliberately
NOT shared -- it differs between the two samplers -- so only the column
vocabulary and the row-assembly tail live here.
"""
from __future__ import annotations

import pandas as pd
from pm4py.objects.ocel.obj import OCEL

EID, ACT, TS = "ocel:eid", "ocel:activity", "ocel:timestamp"
OID, OTYPE = "ocel:oid", "ocel:type"


def _ocel_from_rows(rows: list[tuple]) -> OCEL:
    """Assemble an :class:`OCEL` from ``(eid, activity, timestamp, oid, otype)``
    rows: the relations table verbatim, with the events and objects tables
    deduplicated out of it."""
    rel = pd.DataFrame(rows, columns=[EID, ACT, TS, OID, OTYPE])
    events = rel[[EID, ACT, TS]].drop_duplicates().reset_index(drop=True)
    objects = rel[[OID, OTYPE]].drop_duplicates().reset_index(drop=True)
    return OCEL(events=events, objects=objects, relations=rel)
