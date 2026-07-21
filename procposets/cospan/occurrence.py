"""Occurrence-net (boundary-rooted labelled DAG) view of a ``NamedMorphism``.

Design: ``CLASS_EXTRACTION.md`` §19. The canonical object for comparing
morphisms *under label projection* is **not** the ``;``/``@`` token tree
(many-to-one onto the morphism: associativity, interleaving, loop-cut all
give different trees for one morphism) but the morphism's wiring graph: one
node per atomic generator occurrence, one edge per produced->consumed port,
with the boundary as distinguished roots/sinks. Ports are projected to their
endpoint *activity labels* (arcs kept, ``Port`` identity forgotten) so the
result is comparable *across* adapters whose ``Port`` conventions differ
(§14), while still encoding causality -- a real poset comparison, not the
lossy label-skeleton string comparison (§19c/§19d).

This reuses ``morphism_schema``'s replay verbatim in spirit: the same
``_enabled``/``_fire`` multiset bookkeeping, tagging each consumed port with
its provenance (``BOUNDARY`` or the ``(frame, slot)`` of the producing step),
but *emits a graph* instead of hashing the provenance into a shape key.

Acyclic by construction for closing instances and single loop bodies; a
loop's cyclicity is handled upstream by comparing ``(anchor_types, body-DAG)``
(§19e), not represented as a back-edge here.

Reading convention (§23b): every edge is **conjunctive (AND)**. At each event
node *all* incoming arcs are required preconditions and *all* outgoing arcs are
produced simultaneously -- a fan-out is a fork (all branches, in parallel), a
fan-in is a join (all inputs needed), never an exclusive choice. The figure is
therefore a conflict-free occurrence net / causal net (a pomset = labelled
partial order, i.e. pure and-causality); alternatives between outcomes are
represented as *separate* DAGs, never as a branch inside one.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

from .class_extraction import NamedMorphism, _to_counter
from .morphism_schema import _expand, _frames
from .signature import Port

# networkx >=3.5 emits a one-time UserWarning that its *directed*-graph WL hash
# changed in v3.5 (a bugfix tracking in/out edges separately). Our keys are only
# ever computed and compared within a single run -- never persisted or compared
# across networkx versions -- so the change is irrelevant here. Suppress just
# that message (not warnings in general) so it does not spam CLI/test output.
warnings.filterwarnings(
    "ignore",
    message="The hashes produced for directed graphs changed",
    category=UserWarning,
)

IN = "__IN__"
OUT = "__OUT__"
# The single boundary nodes of a *closing* pomset (§40): all of a notation's origin
# markers (master ``gamma1`` / OCCN ``START_<ot>``) collapse onto the one ``GAMMA1``
# source, all of its termination markers (``gamma2`` / ``END_<ot>``) onto the one
# ``GAMMA2`` sink, and any bare source/sink (a discovered OCPN's real first/last
# activities) is attached to them -- so every notation renders one connected boundary.
# (``IN``/``OUT`` remain the *loop-anchor* frontier sentinels, a different role.)
# ASCII labels (the canonical-key WL hash encodes node labels as ASCII); the renderer
# may prettify ``gamma1``/``gamma2`` to ``γ1``/``γ2`` for display.
GAMMA1 = "gamma1"
GAMMA2 = "gamma2"


@dataclass
class EventDag:
    """A boundary-rooted labelled DAG (the label-pomset / occurrence net of one
    fragment). ``graph`` is a ``networkx.DiGraph`` whose nodes carry a
    ``label`` attribute (an activity name, or the sentinels :data:`IN`/
    :data:`OUT` for the boundary roots/sinks) and whose edges carry a ``typ``
    attribute (the object type the produced->consumed wire is typed by)."""

    graph: nx.DiGraph
    name: str = ""

    def event_nodes(self) -> list[int]:
        return [n for n, d in self.graph.nodes(data=True) if d["label"] not in (IN, OUT)]


# ``BOUNDARY_PREFIXES`` is the OCCN per-type origin/terminus wrappers, shared with the
# splice-site/skeleton logic (``splice._is_wrapper_label``, ``signature_diff``) -- it must
# stay START_/END_ only, or those coordinate systems shift.
BOUNDARY_PREFIXES = ("START_", "END_")
# ``to_event_dag`` additionally contracts the explicit ``gamma1``/``gamma2`` origin/terminus
# (exact match) some logs carry as events, so every notation's *occurrence DAG* renders one
# connected γ1 source / γ2 sink (§40). This is a render/canonical-key concern only; it does
# NOT touch the splice spine. Pass ``strip_prefixes=()`` for the raw DAG (no contraction).
DAG_BOUNDARY_MARKERS = ("START_", "END_", "gamma1", "gamma2")


def _typ_str(typ: str | None) -> str:
    return "*" if typ is None else str(typ)


def to_event_dag(
    nm: NamedMorphism,
    by_name: dict[str, NamedMorphism],
    *,
    strip_prefixes: tuple[str, ...] = DAG_BOUNDARY_MARKERS,
) -> EventDag:
    """Build the occurrence-net DAG of ``nm`` by replaying the live search's
    own bookkeeping from ``nm``'s domain boundary.

    Each atomic generator occurrence becomes a node (``label`` = its activity);
    each consumed port becomes an in-edge -- from the producing event if the
    port was produced internally, or from the synthetic :data:`IN` root if it
    came from the fragment's boundary. Right ports never consumed inside the
    fragment become out-edges to the synthetic :data:`OUT` sink. ``Port``
    identity is dropped: only labels and typed arcs survive, which is exactly
    the §19d label-pomset projection.

    Parallel wires between the *same* node pair are preserved as a sorted
    type-multiset on a single edge (``typs`` attribute) -- a plain ``DiGraph``
    would silently collapse them, which is precisely the wiring information
    §19c needs to keep (it is what distinguishes an N-poset from a complete
    bipartite order).

    ``strip_prefixes`` names boundary-wrapper activities (default the OCCN
    ``START_<ot>``/``END_<ot>`` markers, §13/§14 B3). These are not merely
    undrawn -- they are **fully absorbed**: the wire a contracted ``START_``
    induces into the first real activity (and the wire the last real activity
    feeds into a contracted ``END_``) is *dropped*, so that activity becomes a
    true source/sink, exactly as master/OCPN's zero-input ``gamma1`` /
    zero-output ``gamma2`` already are. Contracting only the wrapper *node*
    while leaving its stub edge would make OCCN's prefix wiring differ from
    master/OCPN's by that one edge -- which is the §20 finding that kept the
    closings overlay from ever going green. A loop's *real* anchor boundary
    (``nm.boundary``, the :data:`IN` tokens) is kept, not absorbed -- only
    wrapper-induced wires are dropped."""
    frames = _frames(_expand(nm.body, by_name))

    # Origin/termination markers map onto the single γ1 source / γ2 sink (§40). A loop's
    # real anchor tokens stay keyed plain IN/OUT (a different role -- the loop frontier).
    # ``strip_prefixes`` governs the whole contraction: an exact match (``gamma1``) or a
    # ``_``-suffixed prefix (``START_<ot>``); empty ``strip_prefixes`` => raw DAG.
    def boundary_role(label: str) -> str | None:
        for m in strip_prefixes:
            if label == m or (m.endswith("_") and label.startswith(m)):
                return GAMMA2 if (m.upper().startswith("END") or m == "gamma2") else GAMMA1
        return None

    # accumulate typed wires per ordered node pair, then emit one edge each
    wires: dict[tuple, list[str | None]] = {}

    # pool[Port] -> FIFO list of producer node-keys (IN for real anchor tokens)
    pool: dict[Port, list] = {}
    for p, n in _to_counter(nm.boundary).items():
        pool.setdefault(p, []).extend([IN] * n)

    labels: dict[int, str] = {}
    node_id = 0
    for frame in frames:
        # consume (left) before produce (right) -- matches morphism_schema's
        # frame ordering so the reconstructed wiring is identical.
        produced_this_frame: list[tuple] = []
        for gen in frame:
            role = boundary_role(gen.label)
            key = role if role is not None else node_id
            if role is None:
                labels[node_id] = gen.label
                node_id += 1
            # weight-aware (§38): a leg of multiplicity w wires w tokens, so a
            # grounded bundle-consumer pops as many producer-tokens as it consumes.
            for p in gen.left:
                for _ in range(gen.weight(p)):
                    src = pool[p].pop(0)
                    wires.setdefault((src, key), []).append(p.typ)
            for p in gen.right:
                for _ in range(gen.weight(p)):
                    produced_this_frame.append((key, p))
        for key, p in produced_this_frame:
            pool.setdefault(p, []).append(key)

    # anything left unconsumed is an open output boundary wire (a loop body's frontier)
    for p, producers in pool.items():
        for src in producers:
            wires.setdefault((src, OUT), []).append(p.typ)

    return _assemble_event_dag(wires, labels, strip_prefixes, nm.name)


def _assemble_event_dag(wires: dict, labels: dict, strip_prefixes: tuple,
                        name: str) -> EventDag:
    """Build the ``EventDag`` from the replayed ``wires`` (typed arcs per ordered
    node pair) and ``labels``: one typed edge per pair (dropping bare
    anchor->anchor and boundary-internal self-loop wires), reattach any bare
    source/sink to the single γ1/γ2 boundary, then drop unused sentinels."""
    g = nx.DiGraph()
    for sentinel in (IN, OUT, GAMMA1, GAMMA2):
        g.add_node(sentinel, label=sentinel)
    for nid, lab in labels.items():
        g.add_node(nid, label=lab)
    for (u, v), typs in wires.items():
        if u == IN and v == OUT:
            continue  # a bare anchor->anchor wire carries no structure
        if u == v and u in (GAMMA1, GAMMA2):
            continue  # a wire *internal* to the collapsed boundary (e.g. a discovered
            # OCCN's ``START_<ot> -> gamma1`` two-layer origin): both ends map to the one
            # γ1 node, so the wire becomes a self-loop -- drop it, the boundary is one node
        g.add_edge(u, v, typs=tuple(sorted(_typ_str(t) for t in typs)))

    # Attach any *bare* source/sink (a model with no origin marker -- e.g. a discovered
    # OCPN, whose first/last activities are real) to the single γ1 / γ2, so every closing
    # pomset has exactly one connected boundary pair. A loop body's interior sources are
    # fed by its IN anchor (in-degree > 0), so they are never bare and are not reattached.
    # Skipped when boundary contraction is off (``strip_prefixes=()`` => raw DAG).
    _BND = (IN, OUT, GAMMA1, GAMMA2)
    if strip_prefixes:
        for v in [n for n in g.nodes if n not in _BND]:
            if g.in_degree(v) == 0:
                typs = sorted({t for _, _, d in g.out_edges(v, data=True) for t in d["typs"]})
                g.add_edge(GAMMA1, v, typs=tuple(typs) or ("*",))
            if g.out_degree(v) == 0:
                typs = sorted({t for _, _, d in g.in_edges(v, data=True) for t in d["typs"]})
                g.add_edge(v, GAMMA2, typs=tuple(typs) or ("*",))

    # drop any boundary sentinel that ended up unused (e.g. γ1/γ2 on a loop body, or the
    # IN/OUT anchors on a closing) so it neither renders nor enters the canonical key
    for b in (IN, OUT, GAMMA1, GAMMA2):
        if g.degree(b) == 0:
            g.remove_node(b)

    return EventDag(graph=g, name=name)


def loop_cycle_dag(nm: NamedMorphism, by_name: dict[str, NamedMorphism]) -> EventDag:
    """The loop's **closed** occurrence graph: like :func:`to_event_dag`, but each carried-out
    boundary token is wired back to the matching carried-in consumer (per port, FIFO) instead
    of being rooted at synthetic IN/OUT. So the graph is **invariant to where the cycle was
    cut**.

    A loop is a trace ``Tr(g)``; the traced-monoidal sliding (yanking) axiom makes ``Tr``
    rotation-invariant, so its canonical bucket key must be too. This is the dedup key that
    collapses the rotation-cuts of one cycle (re_examine-first / ecg-first / …) into a single
    loop structure -- the genuine quotient the engine owes the theory. The result is genuinely
    cyclic (no IN/OUT sentinels); WL hashing and VF2 handle cycles unchanged."""
    frames = _frames(_expand(nm.body, by_name))
    wires: dict[tuple, list] = {}
    pool: dict[Port, list] = {}
    carried_in: dict[Port, list] = {}  # port -> [consumer node], in pop order (the wrap targets)
    BIN = object()  # sentinel marking a carried-in (boundary) token
    for p, n in _to_counter(nm.boundary).items():
        pool.setdefault(p, []).extend([BIN] * n)

    labels: dict[int, str] = {}
    node_id = 0
    for frame in frames:
        produced_this_frame: list[tuple] = []
        for gen in frame:
            key = node_id
            labels[node_id] = gen.label
            node_id += 1
            for p in gen.left:
                for _ in range(gen.weight(p)):
                    src = pool[p].pop(0)
                    if src is BIN:
                        carried_in.setdefault(p, []).append(key)
                    else:
                        wires.setdefault((src, key), []).append(p.typ)
            for p in gen.right:
                for _ in range(gen.weight(p)):
                    produced_this_frame.append((key, p))
        for key, p in produced_this_frame:
            pool.setdefault(p, []).append(key)

    # close the cycle: each carried-OUT token (leftover producer) wraps to a carried-IN
    # consumer of the same port, FIFO-matched (a balanced cycle has equal counts per port).
    for p, producers in pool.items():
        outs = [s for s in producers if s is not BIN]
        for src, dst in zip(outs, carried_in.get(p, [])):
            wires.setdefault((src, dst), []).append(p.typ)

    g = nx.DiGraph()
    for nid, lab in labels.items():
        g.add_node(nid, label=lab)
    for (u, v), typs in wires.items():
        g.add_edge(u, v, typs=tuple(sorted(_typ_str(t) for t in typs)))
    return EventDag(graph=g, name=nm.name)


def _node_match(a: dict, b: dict) -> bool:
    return a["label"] == b["label"]


def _edge_match(a: dict, b: dict) -> bool:
    return a["typs"] == b["typs"]


def canonical_key(dag: EventDag) -> str:
    """A cheap, boundary-aware canonical bucket key for ``dag``: the
    Weisfeiler-Lehman graph hash over node ``label`` and the edge type-multiset
    ``typs``. WL is not a perfect certificate (rare non-isomorphic collisions),
    so equality must be confirmed with :func:`is_isomorphic` within a bucket --
    at these sizes (closings ~5-9 events) that confirmation is trivially cheap
    and the pair (hash bucket + VF2) is exact."""
    return nx.weisfeiler_lehman_graph_hash(
        dag.graph, node_attr="label", edge_attr="typs", iterations=4
    )


def is_isomorphic(a: EventDag, b: EventDag) -> bool:
    """Exact boundary-rooted labelled-DAG isomorphism (VF2, label- and
    type-respecting). The roots/sinks (:data:`IN`/:data:`OUT`) carry distinct
    sentinel labels, so they can only map to each other -- this is what makes
    the comparison *boundary-rooted* (§19a): admissible isos fix the
    boundary."""
    return nx.is_isomorphic(
        a.graph, b.graph, node_match=_node_match, edge_match=_edge_match
    )


def history_keys(dag: EventDag) -> dict:
    """A Merkle-style causal-history key per node: bottom-up over the
    topological order, ``key(v) = hash(label(v), sorted (typs, key(u)) over
    incoming edges)``. Two nodes get the same key iff their causal cones are
    isomorphic *including the typed wiring* -- so merging by this key (the
    occurrence-net union / prefix-trie, §19/§20d) collapses shared prefixes
    without ever conflating differently-wired events.

    :data:`IN` is the shared base; :data:`OUT` is forced to a single shared
    sink (``"OUT"``) so all endings funnel into one terminal node visually,
    while their distinct source histories keep the in-edges separate."""
    import hashlib

    g = dag.graph
    keys: dict = {}
    for n in nx.topological_sort(g):
        lab = g.nodes[n]["label"]
        if lab == IN:
            keys[n] = "IN"
            continue
        if lab == OUT:
            keys[n] = "OUT"
            continue
        contribs = sorted(
            (tuple(g.edges[u, n]["typs"]), keys[u]) for u in g.predecessors(n)
        )
        digest = hashlib.sha1(f"{lab}|{contribs!r}".encode()).hexdigest()[:12]
        keys[n] = digest
    return keys


def anchor_types(nm: NamedMorphism) -> tuple:
    """The §19e loop anchor projected to a type-multiset: the sorted
    ``(typ, count)`` pairs of the fragment's boundary ports. Comparable across
    adapters (unlike raw ``Port``s), so two loops anchor-match iff they splice
    into a frontier with the same typed shape."""
    counter = _to_counter(nm.boundary)
    types = Counter(p.typ for p, c in counter.items() for _ in range(c))
    return tuple(sorted(((t is None, t), n) for t, n in types.items()))


@dataclass
class CanonClass:
    """A bucket of fragment DAGs that are all mutually isomorphic: one
    canonical structure, possibly reached by several fragments/models."""

    key: str
    rep: EventDag
    members: list[EventDag] = field(default_factory=list)


def canonicalize(dags: list[EventDag]) -> dict[str, list[CanonClass]]:
    """Group ``dags`` into isomorphism classes. Returns a map from WL bucket
    key to the list of distinct :class:`CanonClass`es in that bucket (usually
    one; >1 only when WL collides two genuinely non-isomorphic graphs, split
    apart by the VF2 confirmation)."""
    buckets: dict[str, list[CanonClass]] = {}
    for dag in dags:
        key = canonical_key(dag)
        classes = buckets.setdefault(key, [])
        for cls in classes:
            if is_isomorphic(cls.rep, dag):
                cls.members.append(dag)
                break
        else:
            classes.append(CanonClass(key=key, rep=dag, members=[dag]))
    return buckets
