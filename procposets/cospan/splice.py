"""The **splice representation** of a discovered model: its closing catalogue
collapsed to ``M(m,σ)`` families, recorded *both* concretely (the boundary-
rooted occurrence-net pomsets) *and* algebraically (a minimal boundary-stripped
step-skeleton with loop splice-points), in one canonical, serializable object.

This is the productized, deterministic, LLM-free form of what
``dag_render.render_catalogue_dags`` computes ad hoc. It is the **finite
generating grammar of the model's trace language** (§27c): a run γ1→γ2 is a
family baseline with its anchored loops spliced in (any multiplicity). The
companion :mod:`procposets.cospan.trace_language` consumes the *algebraic* form to emit
traces up to a finite loop cut-off.

Two views per fragment:

* **Algebraic** (:class:`AlgebraicTerm`) -- the boundary-stripped step sequence
  (atoms and ``@``-concurrent tensor groups), i.e. the ``M(m,σ)`` skeleton.
  Stripping ``START_``/``END_`` wrappers makes the layered reading match the
  causal pomset for series-parallel families (the OCCN ``72→2`` fix, §27).
* **Concrete** (:class:`Pomset`) -- the occurrence-net DAG (events + typed cover
  edges), the exact causal order. Used for non-SP families (where the algebraic
  layering is only an over-ordering) and for the SVG renderers.

Canonical ids (``σ1..``, ``ℓ1..``) are assigned in ``canonical_key`` order, so
``to_dict`` is byte-stable across runs and hash seeds (§27b determinism).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx

from .._extensions import count_extensions as _count_ext
from .class_extraction import (
    ExtractionResult,
    NamedMorphism,
    _fire,
    _flatten_generators,
    _to_counter,
)
from .constraints import union as _union_constraints
from .feasibility import FeasibilityTooLarge, feasible
from .dag_diff import closing_spine
from .morphism_schema import _expand, _frames
from .occurrence import (
    IN,
    OUT,
    BOUNDARY_PREFIXES,
    anchor_types,
    canonical_key,
    loop_cycle_dag,
    to_event_dag,
)
from .signature_diff import label_skeleton, strip_boundary_wrapper

# An algebraic step: a single activity label, or a tensor group of concurrent
# labels (canonically sorted). A loop/spine body is a sequence of these.
Step = str | tuple[str, ...]


def _is_wrapper_label(label: str) -> bool:
    return label.startswith(BOUNDARY_PREFIXES)


@dataclass(frozen=True)
class AlgebraicTerm:
    """A boundary-stripped step sequence (the ``M(m,σ)`` skeleton of a spine or
    loop body): ``;``-sequential across steps, ``@``-concurrent within a tuple
    step. The minimal symbolic form the trace module expands."""

    steps: tuple

    def render(self) -> str:
        return ";".join("(" + "@".join(s) + ")" if isinstance(s, tuple) else s for s in self.steps)


@dataclass(frozen=True)
class Pomset:
    """The concrete companion: a labelled occurrence-net DAG. ``events`` are
    ``(id, label)``; ``edges`` are ``(u, v, typ-tuple)`` typed cover arcs (the
    IN/OUT boundary roots are dropped -- a source/sink is just a boundary
    event)."""

    events: tuple
    edges: tuple

    @staticmethod
    def from_event_dag(dag) -> "Pomset":
        g = dag.graph
        # node ids mix int (real events) and str (the gamma1/gamma2 boundary, §40), so
        # sort by stringified id to stay type-safe. IN/OUT loop-anchor sentinels are
        # dropped; gamma1/gamma2 are kept -- they are the rendered boundary nodes.
        events = tuple(sorted(
            ((n, d["label"]) for n, d in g.nodes(data=True) if d["label"] not in (IN, OUT)),
            key=lambda e: str(e[0])))
        edges = tuple(sorted(
            ((u, v, tuple(d.get("typs", ())))
             for u, v, d in g.edges(data=True) if u not in (IN, OUT) and v not in (IN, OUT)),
            key=lambda e: (str(e[0]), str(e[1]), e[2])))
        return Pomset(events=events, edges=edges)


@dataclass(frozen=True)
class Splice:
    """A loop attachment point on a family's spine: the loop bodies in
    ``loop_ids`` may be spliced after ``site`` spine steps (``spine[:site] ⋉
    loop ⋉ spine[site:]``)."""

    site: int
    loop_ids: tuple


@dataclass(frozen=True)
class LoopFragment:
    """One distinct loop **phasing** (a cut of a cycle; provenance/name duplicates
    collapsed): its boundary-stripped body skeleton, its anchor type-multiset, and the
    concrete pomset. ``cycle`` is the shared cut-invariant identity (``Cn``) -- phasings that
    are rotations of the same trace ``Tr(g)`` carry the same ``cycle`` (a loop is one cycle
    categorically; the grammar realises it as several phasings, kept distinct for sound
    generation but presented as one ``Cn`` in the figures)."""

    loop_id: str
    term: AlgebraicTerm
    anchor: tuple
    key: str
    pomset: Pomset
    cycle: str = ""


@dataclass(frozen=True)
class Family:
    """One ``M(m,σ)`` closing family: the loop-free baseline (or shortest
    member, ``loop_free=False``) as both algebraic spine and concrete pomset,
    plus where loops splice in. ``sp_exact`` records whether the algebraic
    (layered) reading equals the causal pomset order -- if not, the trace module
    must use ``pomset`` for exact linearization (§27c caveat)."""

    spine_id: str
    term: AlgebraicTerm
    splices: tuple
    key: str
    loop_free: bool
    sp_exact: bool
    pomset: Pomset
    constraints: tuple = ()  # the family's accumulated leg-multiplicity system (§32)


@dataclass(frozen=True)
class SpliceRepresentation:
    """The whole catalogue of one model as splice families + loops."""

    name: str
    quotient: str
    truncated: bool
    families: tuple
    loops: tuple

    # -- construction -------------------------------------------------------

    @staticmethod
    def from_extraction_result(
        result: ExtractionResult, *, name: str, quotient: str = "forget_provenance",
        prune_bound: int | None = None,
    ) -> "SpliceRepresentation":
        """``prune_bound`` (§32): if set, drop families whose accumulated leg
        constraint system is infeasible at that bound -- this rejects multi-instance
        over-generation closings (whose cardinality constraints contradict, e.g. one
        wire forced both ``==1`` and ``∈[2,5]`` by two routes). ``None`` = keep all."""
        loops_nm = result.loops()
        frags = result.fragments
        loops_out, loop_id_of_name = _build_loop_fragments(loops_nm, frags)

        # closings grouped into families by loop-free spine
        families_by_spine: dict[tuple, list[NamedMorphism]] = {}
        for nm in result.closing():
            families_by_spine.setdefault(closing_spine(nm, loops_nm, frags), []).append(nm)

        fams: list[tuple] = []
        for spine, members in families_by_spine.items():
            baseline = _baseline_of(members, frags)
            dag = to_event_dag(baseline, frags)
            term = AlgebraicTerm(spine)
            site_loops = _spine_step_splices(baseline, members, loops_nm, frags)
            splices = tuple(
                Splice(site=s, loop_ids=tuple(sorted({loop_id_of_name[n] for n in names})))
                for s, names in sorted(site_loops.items())
            )
            loop_free = strip_boundary_wrapper(label_skeleton(baseline.body, frags)) == spine
            # §32: the family's leg-multiplicity system = the union of its baseline
            # generators' constraints (cardinality intervals + key-distribution sums).
            cons = _union_constraints(_flatten_generators(baseline.body, frags))
            if prune_bound is not None:
                try:
                    if not feasible(cons, bound=prune_bound):
                        continue  # infeasible system -> multi-instance over-generation
                except FeasibilityTooLarge:
                    pass  # undecidable at this bound -> keep, don't silently drop
            fams.append(
                (
                    canonical_key(dag),
                    Family(
                        spine_id="",  # assigned below in canonical order
                        term=term,
                        splices=splices,
                        key=canonical_key(dag),
                        loop_free=loop_free,
                        sp_exact=_sp_exact(term, dag),
                        pomset=Pomset.from_event_dag(dag),
                        constraints=tuple(sorted(cons, key=str)),
                    ),
                )
            )

        families_out = tuple(
            Family(**{**f.__dict__, "spine_id": f"σ{i}"})
            for i, (_, f) in enumerate(sorted(fams, key=lambda kf: kf[0]), start=1)
        )
        return SpliceRepresentation(
            name=name,
            quotient=quotient,
            truncated=result.truncated,
            families=families_out,
            loops=tuple(loops_out),
        )

    # -- serialization (canonical, byte-stable) -----------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "quotient": self.quotient,
            "truncated": self.truncated,
            "loops": [
                {
                    "loop_id": lp.loop_id,
                    "cycle": lp.cycle,
                    "term": _term_to_json(lp.term),
                    "anchor": _anchor_to_json(lp.anchor),
                    "key": lp.key,
                    "pomset": _pomset_to_json(lp.pomset),
                }
                for lp in self.loops
            ],
            "families": [
                {
                    "spine_id": f.spine_id,
                    "term": _term_to_json(f.term),
                    "splices": [{"site": s.site, "loop_ids": list(s.loop_ids)} for s in f.splices],
                    "key": f.key,
                    "loop_free": f.loop_free,
                    "sp_exact": f.sp_exact,
                    "pomset": _pomset_to_json(f.pomset),
                    "constraints": [str(c) for c in f.constraints],
                }
                for f in self.families
            ],
        }


# --- helpers -----------------------------------------------------------------


def _build_loop_fragments(loops_nm, frags):
    """Loop phasings -> ``(list[LoopFragment], name->loop_id map)``.

    Distinct loop **phasings** = (anchor, dag-key) classes: one ℓ per cut. A loop is a
    trace Tr(g) and the rotation-cuts of one cycle ARE one loop categorically -- but the
    finite splice GRAMMAR realises it through these phasings, and trace generation inserts
    each ℓ's own ``term`` at its site (``trace_language._expanded_sequences``). A phasing's
    term is only a valid insertion at its own anchor frontier, so the phasings MUST stay
    distinct here: merging them to one representative term inserts it at sites that matched
    a different phasing, manufacturing unrealisable traces (an over-generation the
    completeness check catches). The shared cyclic identity is exposed for *display* via
    ``LoopFragment.cycle``, without disturbing this sound per-phasing generation.
    """
    struct: dict[tuple, list[NamedMorphism]] = {}
    for lp in loops_nm:
        struct.setdefault((anchor_types(lp), canonical_key(to_event_dag(lp, frags))), []).append(lp)
    loop_id_of_name: dict[str, str] = {}
    phasings: list[tuple] = []  # (loop_id, rep, key)
    for i, (key, members) in enumerate(sorted(struct.items()), start=1):
        lid = f"ℓ{i}"
        rep = min(members, key=lambda m: m.name)
        for m in members:
            loop_id_of_name[m.name] = lid
        phasings.append((lid, rep, key))
    # cyclic identity (display only): group phasings by their cut-invariant closed-cycle
    # key, so the catalogue/gallery can present the rotation-cuts as one loop ``C`` while
    # generation stays per-phasing. ``Cn`` numbered by first appearance over the sorted ℓ.
    cycle_of_key: dict[str, str] = {}
    loops_out: list[LoopFragment] = []
    for lid, rep, (anchor, ckey) in phasings:
        cyc_key = canonical_key(loop_cycle_dag(rep, frags))
        cycle = cycle_of_key.setdefault(cyc_key, f"C{len(cycle_of_key) + 1}")
        loops_out.append(
            LoopFragment(
                loop_id=lid,
                term=AlgebraicTerm(strip_boundary_wrapper(label_skeleton(rep.body, frags))),
                anchor=anchor,
                key=ckey,
                pomset=Pomset.from_event_dag(to_event_dag(rep, frags)),
                cycle=cycle,
            )
        )
    return loops_out, loop_id_of_name


def _baseline_of(members: list[NamedMorphism], by_name: dict) -> NamedMorphism:
    """The family's drawn rep: the loop-free member if any (its stripped
    skeleton equals the spine), else the shortest member; ties by name."""

    def size(nm: NamedMorphism) -> int:
        return len(to_event_dag(nm, by_name).graph)

    return min(members, key=lambda nm: (size(nm), nm.name))


def _spine_step_splices(
    baseline: NamedMorphism, members: list[NamedMorphism], loops: list[NamedMorphism], by_name: dict
) -> dict[int, set]:
    """``{insertion_index: {loop_name}}`` in boundary-stripped spine-step
    coordinates. Two mechanisms unioned (the §22h committed-vs-neutral split, at
    step granularity): (a) the index where a loop's skeleton was removed to
    reach the spine in some unrolled member (committed baselines expose nothing
    themselves, their m=2 sibling does); (b) the spine step after which the
    baseline's port-frontier first ⊇ a loop's anchor (neutral baselines, e.g.
    the master)."""
    spine = closing_spine(baseline, loops, by_name)
    out: dict[int, set] = {}

    # (a) removal-index from family members (single contiguous loop occurrence)
    loop_skels = [(lp.name, strip_boundary_wrapper(label_skeleton(lp.body, by_name))) for lp in loops]
    for nm in members:
        skel = strip_boundary_wrapper(label_skeleton(nm.body, by_name))
        for lname, lsk in loop_skels:
            m = len(lsk)
            if not m:
                continue
            for i in range(len(skel) - m + 1):
                if skel[i : i + m] == lsk and skel[:i] + skel[i + m :] == spine:
                    out.setdefault(i, set()).add(lname)
                    break

    # (b) port-frontier replay on the baseline (catches neutral baselines)
    loop_anchors = [(lp.name, _to_counter(lp.boundary)) for lp in loops if lp.boundary]
    frontier: Counter = Counter()
    step = 0
    for frame in _frames(_expand(baseline.body, by_name)):
        wrapper_frame = all(_is_wrapper_label(g.label) for g in frame)
        for g in frame:
            frontier = _fire(frontier, g)
        if not wrapper_frame:
            step += 1
        for lname, anchor in loop_anchors:
            if anchor and all(frontier[p] >= c for p, c in anchor.items()):
                out.setdefault(step, set()).add(lname)
    return out


def _layered_dag(term: AlgebraicTerm) -> nx.DiGraph:
    """The layered (series-parallel) DAG of a step skeleton: every event of step
    i precedes every event of step i+1; within a tensor step events are
    concurrent."""
    g = nx.DiGraph()
    nid = 0
    prev: list[int] = []
    for stepv in term.steps:
        labels = stepv if isinstance(stepv, tuple) else (stepv,)
        cur = []
        for _ in labels:
            g.add_node(nid)
            for p in prev:
                g.add_edge(p, nid)
            cur.append(nid)
            nid += 1
        prev = cur
    return g


def _count_linear_extensions(g: nx.DiGraph, cap_nodes: int = 11) -> int | None:
    # >cap_nodes fragments stay "not verifiably exact" (None -- the contract
    # _sp_exact reads). Below the cap, count via the guarded ideal-lattice DP
    # (_extensions.count_extensions, O(2^width) states) instead of enumerating
    # every width! topological sort: an 11-wide antichain is 2^11=2048 states
    # vs 11!~4e7 sorts. The count is identical -- linear extensions of the DAG's
    # reachability order -- and no budget refusal can fire below 11 nodes.
    if g.number_of_nodes() > cap_nodes:
        return None
    return _count_ext(list(g.nodes()), set(g.edges()))


def _sp_exact(term: AlgebraicTerm, dag) -> bool:
    """True iff the algebraic (layered) reading has the same linear-extension
    count as the causal pomset -- i.e. the family is series-parallel and the
    cheap algebraic linearization is exact. Computed only for small fragments
    (the trace module falls back to ``pomset`` otherwise)."""
    causal = nx.DiGraph()
    causal.add_nodes_from(n for n, d in dag.graph.nodes(data=True) if d["label"] not in (IN, OUT))
    causal.add_edges_from((u, v) for u, v in dag.graph.edges() if u not in (IN, OUT) and v not in (IN, OUT))
    a = _count_linear_extensions(_layered_dag(term))
    b = _count_linear_extensions(causal)
    return a is not None and b is not None and a == b


def _term_to_json(t: AlgebraicTerm) -> list:
    return [list(s) if isinstance(s, tuple) else s for s in t.steps]


def _anchor_to_json(anchor: tuple) -> list:
    # anchor_types: (((is_loop, typ), count), ...) -> fully list-ified for JSON
    return [[[bool(b), t], n] for ((b, t), n) in anchor]


def _pomset_to_json(p: Pomset) -> dict:
    return {
        "events": [list(e) for e in p.events],
        "edges": [[u, v, list(typs)] for (u, v, typs) in p.edges],
    }
