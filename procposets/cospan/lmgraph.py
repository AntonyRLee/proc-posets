"""Logic-mediated multipartite graph (LM-graph).

Faithful Python encoding of the abstraction in section 2b of the paper
(``def:lm-graph``): a finite directed graph whose vertices are partitioned into
*absorbing* activities ``A`` (boundary endpoints, observable) and *mediating*
nodes carrying local routing logic (AND / XOR / OR / SEQ).  Edges carry a port
type ``theta: E -> O`` (``None`` = untyped, i.e. the ``O = 1`` specialisation on
that edge).  Multiple absorbing nodes may share a *label* via ``ell`` -- this is
the mode-refinement device (e.g. Petri ``s_c``/``s_b`` both labelled ``s``).

The engine (``engine.py``) reads this structure; adapters (``from_*.py``) build
it from discovered pm4py models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Kind(Enum):
    """Mediator firing-rule kind (catalogue of ``def:firing-rule``).

    ``SEQ`` is the transparent / pass-through mediator (1-ary AND); silent
    transitions (tau) are modelled as ``SEQ`` mediators and contribute zero
    generators while still routing tokens and carrying type information.
    """

    XOR = "XOR"  # exclusive choice: union over branches
    AND = "AND"  # conjunction: product over branches
    OR = "OR"    # XOR-of-AND-bundles: union over nonempty subsets
    SEQ = "SEQ"  # transparent pass-through


@dataclass(frozen=True)
class Edge:
    src: str
    tgt: str
    typ: str | None = None


@dataclass
class LMGraph:
    """A logic-mediated multipartite graph.

    ``activities`` is the absorbing set ``A``; ``mediators`` maps each mediating
    node to its :class:`Kind`; ``label`` is ``ell`` (defaults to identity);
    ``obj_types`` accumulates the object-type set ``O`` from edge typings.
    """

    activities: set[str] = field(default_factory=set)
    mediators: dict[str, Kind] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    label: dict[str, str] = field(default_factory=dict)
    obj_types: set[str] = field(default_factory=set)
    silent: set[str] = field(default_factory=set)  # silent-transition (tau) mediators

    # --- builders -----------------------------------------------------------
    def add_activity(self, name: str, label: str | None = None) -> str:
        self.activities.add(name)
        self.label[name] = name if label is None else label
        return name

    def add_mediator(self, name: str, kind: Kind | str, silent: bool = False) -> str:
        self.mediators[name] = kind if isinstance(kind, Kind) else Kind(kind)
        if silent:
            self.silent.add(name)
        return name

    def add_edge(self, src: str, tgt: str, typ: str | None = None) -> None:
        self.edges.append(Edge(src, tgt, typ))
        if typ is not None:
            self.obj_types.add(typ)

    # --- queries ------------------------------------------------------------
    def out_edges(self, n: str) -> list[Edge]:
        return [e for e in self.edges if e.src == n]

    def in_edges(self, n: str) -> list[Edge]:
        return [e for e in self.edges if e.tgt == n]

    def lab(self, n: str) -> str:
        return self.label.get(n, n)

    def is_activity(self, n: str) -> bool:
        return n in self.activities

    # --- silent-transition elimination --------------------------------------
    def without_silent(self) -> LMGraph:
        """Return a copy with every silent (tau) mediator contracted out.

        Each silent mediator is spliced: every ``in -> silent -> out`` path
        becomes a direct ``in -> out`` edge whose type is the path's resolved
        object type (an untyped step inherits the path's typed value, mirroring
        ``engine._resolve_type``). Chains of silent mediators contract
        transitively. For the degree-(1,1) transparent silents that discovered
        models produce this is signature-neutral; for higher-degree silents it
        removes their synchronisation -- which is exactly "remove the silent
        transition". This realises the engine's default: model -> signature
        drops silent transitions (the canonical generators live on real
        activities only)."""
        g = LMGraph(
            activities=set(self.activities),
            mediators=dict(self.mediators),
            edges=list(self.edges),
            label=dict(self.label),
            obj_types=set(self.obj_types),
            silent=set(self.silent),
        )
        for s in list(g.silent):
            for ie in g.in_edges(s):
                for oe in g.out_edges(s):
                    if ie.src == oe.tgt:
                        continue  # don't mint a self-loop from a silent on a cycle
                    typ = ie.typ if ie.typ is not None else oe.typ
                    spliced = Edge(ie.src, oe.tgt, typ)
                    if spliced not in g.edges:
                        g.edges.append(spliced)
            g.edges = [e for e in g.edges if e.src != s and e.tgt != s]
            g.mediators.pop(s, None)
            g.silent.discard(s)
        return g

    def validate(self) -> None:
        """Structural rules of ``def:lm-graph``:

        1. No edge between two absorbing nodes (every activity-to-activity
           link passes through a mediator).
        2. **Mediators are type-preserving.** Object-type creation/consumption
           is a privilege reserved for activities (``def:contexts-general``'s
           admissibility no longer constrains them, precisely because they are
           the designated locus of type change); a gateway/router (AND, XOR,
           OR, SEQ) may not silently convert one object type into another --
           every edge incident to a single mediator must share one object
           type (untyped edges are exempt). This is a modelling decision for
           OC-BPMN and the other OC notations here, not an inherited
           literature constraint, so we enforce it structurally rather than
           leaving it to convention.
        """
        for e in self.edges:
            if e.src in self.activities and e.tgt in self.activities:
                raise ValueError(
                    f"edge {e.src}->{e.tgt} joins two absorbing nodes; "
                    "insert a mediator between them"
                )
        for m in self.mediators:
            types = {e.typ for e in self.in_edges(m) + self.out_edges(m) if e.typ is not None}
            if len(types) > 1:
                raise ValueError(
                    f"mediator {m!r} touches multiple object types {types}; "
                    "gateways must be type-preserving -- only activities may "
                    "create or consume an object type"
                )
