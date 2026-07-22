"""Re-simulate an OCEL from a cospan **signature**, closing the loop
``spec -> log -> discover``: a hand-authored master signature can then drive the
full pipeline (generate a synthetic log, discover OCCN/OCPN, splice).

Each run of the signature's trace language becomes one object-centric process
execution. Object channels come from the **port types**: an activity touches an
object type iff one of its generators carries a port of that type, so each type
involved in a run gets one object threaded through exactly the events that touch
it (a per-type-thread model derived from the signature). Deterministic.

Caveat: one object per involved type per run (cardinality >1 / convergence is
not modelled), which is correct for single-channel models; richer multiplicities
go through :mod:`.faithful_simulate`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from pm4py.objects.ocel.obj import OCEL

from .signature import Signature
from .splice import SpliceRepresentation
from .trace_language import model_traces

EID, ACT, TS = "ocel:eid", "ocel:activity", "ocel:timestamp"
OID, OTYPE = "ocel:oid", "ocel:type"
_BASE = datetime(2025, 1, 1)


def activity_object_types(sig: Signature) -> dict[str, set[str]]:
    """``{activity label -> object types it touches}``, read off the port types
    of that activity's generators."""
    return {
        lab: {p.typ for g in sig.by_label(lab) for p in (set(g.left) | set(g.right)) if p.typ is not None}
        for lab in sig.labels()
    }


def signature_traces(sig: Signature, *, max_loops: int, quotient: bool, one_origin: bool = False) -> list[tuple]:
    """The signature's trace language (sorted, deterministic) up to a loop
    cut-off -- the runs to realise as process executions. ``one_origin`` enforces
    the single-origin rule (one ``gamma1`` per run)."""
    from ..discover import splice_representation_from_signature

    rep: SpliceRepresentation = splice_representation_from_signature(
        sig, name="sim", quotient=quotient, one_origin=one_origin
    )
    return sorted(model_traces(rep, max_loops=max_loops).all_traces())


def _emit_pomset_case(rows: list, eid: list, case: int, events: tuple, edges: tuple) -> None:
    """Emit one OCEL case from a family's **pomset**: per-event object types come
    from the *typed cover edges incident to that event* -- route-exact, so a
    container-route pomset (no box edges) yields no box object. This is the fix
    for degenerate-label pollution: ``activity_object_types`` unions a label's
    generators (so ``gamma1``/``s`` look like they touch every carrier type),
    but the pomset records only the types that actually flow in this run."""
    import networkx as nx

    id2label = dict(events)
    g = nx.DiGraph()
    g.add_nodes_from(id2label)
    ev_types: dict[int, set] = {n: set() for n in id2label}
    for (u, v, typs) in edges:
        if u in id2label and v in id2label:
            g.add_edge(u, v)
        for t in typs:
            if u in ev_types:
                ev_types[u].add(t)
            if v in ev_types:
                ev_types[v].add(t)
    order = list(nx.topological_sort(g))
    types_present = sorted({t for s in ev_types.values() for t in s})
    oid = {ot: f"{ot}_{case}" for ot in types_present}
    base = _BASE + timedelta(hours=case * 6)
    for i, n in enumerate(order):
        t = base + timedelta(minutes=5 * (i + 1))
        e = f"e{eid[0]}"
        eid[0] += 1
        for ot in sorted(ev_types[n]):
            rows.append((e, id2label[n], t, oid[ot], ot))


def ocel_from_signature(
    sig: Signature, *, max_loops: int = 3, repeats: int = 1, quotient: bool = True,
    one_origin: bool = False, faithful: bool | None = None,
) -> OCEL:
    """Build a synthetic OCEL from ``sig``: one process execution per run (each
    repeated ``repeats`` times), with one object per involved type threaded
    through the events that touch it.

    For a **loop-free** signature each family's pomset *is* a run, so we emit it
    directly (:func:`_emit_pomset_case`) -- route-exact per-event typing, no
    degenerate-label pollution. When the model has loops (families with splice
    sites, or recorded loops) we fall back to the label-trace expansion (correct
    for single-type-set labels).

    **Multiplicity dispatch.** ``faithful=None`` (default) auto-detects: if the
    signature carries real N-linear bindings (a leg count >1/``*`` or a key split), the
    one-object-per-type emission below is lossy, so we delegate to
    :func:`.faithful_simulate.faithful_ocel_from_signature` -- the constraint-driven
    token game that realises bundles, batching and key distribution. Pure-1-1 signatures
    are untouched. Pass ``faithful=False`` to force the 1-1 path (the
    structural/origin-rule tests want the exact loop-free pomset, not multiplicity)."""
    from .faithful_simulate import faithful_ocel_from_signature, needs_faithful

    if faithful is None:
        faithful = needs_faithful(sig)
    if faithful:
        return faithful_ocel_from_signature(sig, n_runs=100 * max(1, repeats), seed=7)

    from ..discover import splice_representation_from_signature

    rep: SpliceRepresentation = splice_representation_from_signature(
        sig, name="sim", quotient=quotient, one_origin=one_origin
    )
    has_loops = bool(rep.loops) or any(f.splices for f in rep.families)

    rows: list[tuple] = []
    eid = [0]
    case = 0
    if not has_loops:
        for fam in rep.families:
            for _ in range(repeats):
                _emit_pomset_case(rows, eid, case, fam.pomset.events, fam.pomset.edges)
                case += 1
    else:
        act_types = activity_object_types(sig)
        traces = sorted(model_traces(rep, max_loops=max_loops).all_traces())
        for trace in traces:
            for _ in range(repeats):
                involved = sorted({ot for a in trace for ot in act_types.get(a, ())})
                oid = {ot: f"{ot}_{case}" for ot in involved}
                t = _BASE + timedelta(hours=case * 6)
                for activity in trace:
                    t += timedelta(minutes=5)
                    e = f"e{eid[0]}"
                    eid[0] += 1
                    for ot in sorted(act_types.get(activity, ())):
                        rows.append((e, activity, t, oid[ot], ot))
                case += 1

    rel = pd.DataFrame(rows, columns=[EID, ACT, TS, OID, OTYPE])
    events = rel[[EID, ACT, TS]].drop_duplicates().reset_index(drop=True)
    objects = rel[[OID, OTYPE]].drop_duplicates().reset_index(drop=True)
    return OCEL(events=events, objects=objects, relations=rel)
