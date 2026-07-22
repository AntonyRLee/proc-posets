"""Generator-cospan signature datatypes.

A generator cospan ``g_{a,c}`` (``def:generator-cospan-general``) for activity
label ``a`` in context ``c = (P, S)`` has a left boundary ``P``, an apex
carrying a single hyperedge labelled ``a``, and a right boundary ``S``.  We
record boundaries as sets of typed :class:`Port` triples ``(src, type, tgt)``
-- the canonical shared interface of ``RUNNING_EXAMPLE.md`` (the index 0..16
port map is just an enumeration of these triples).

The full signature ``Sigma`` (``def:generator-cospan-general``) is a frozenset
of :class:`Generator`.  Two signatures being equal as frozensets is the
label-aware equivalence; structural (label-blind) isomorphism is in
``equivalence.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Port:
    """A typed causal link ``(src) --type--> (tgt)`` between activity labels.

    A ``Port`` is the *wire identity* used for composition (type-balance matches an
    output port against an equal input port). Object-multiplicity is NOT part of
    this identity -- a producer leg of cardinality ``[1,5]`` must still compose with
    a consumer leg of ``[1,1]`` -- so cardinality/key live in :class:`Binding` on
    the generator, keyed by port, not as ``Port`` fields (§32)."""

    src: str
    typ: str | None
    tgt: str

    def __str__(self) -> str:
        return f"({self.src},{self.typ},{self.tgt})"


_REL = {"<=", ">=", "=="}


@dataclass(frozen=True)
class LinearConstraint:
    """A linear (in)equality over a generator's leg-multiplicity variables (§32).

    Each leg of a generator carries an integer variable ``n_leg`` (the number of
    objects on that wire), identified by its :class:`Port`. A constraint reads
    ``sum(coeff * n_leg for leg) <rel> rhs`` with ``rel`` in ``{"<=", ">=", "=="}``
    and ``rhs`` an integer. Legs absent from ``terms`` are unconstrained *by this
    inequality*; a leg mentioned by **no** constraint at all defaults to count 1.

    This is the general form -- arbitrary linear type-inequality constraints. The
    OCCN bindings are the special case (see :mod:`procposets.cospan.constraints`):
    cardinality ``[cmin,cmax]`` on leg ``p`` = ``{p:1} >= cmin`` and
    ``{p:1} <= cmax``; a shared-key partition of input leg ``q`` over output legs
    ``p1..pk`` = ``{p1:1,...,pk:1, q:-1} == 0``."""

    terms: frozenset  # frozenset[tuple[Port, int]]
    rel: str
    rhs: int

    def __post_init__(self):
        if self.rel not in _REL:
            raise ValueError(f"rel must be one of {_REL}, got {self.rel!r}")

    def coeffs(self) -> dict:
        return dict(self.terms)

    def ports(self) -> set:
        return {p for p, _ in self.terms}

    def __str__(self) -> str:
        lhs = " + ".join(
            (f"{c}*" if c != 1 else "") + str(p) for p, c in sorted(self.terms, key=lambda t: str(t[0]))
        )
        return f"{lhs} {self.rel} {self.rhs}"


@dataclass(frozen=True)
class Generator:
    """A generator cospan: label ``a`` with typed left/right boundary ports.

    ``constraints`` is the optional §32 decoration -- general linear inequalities
    over the leg-multiplicity variables (:class:`LinearConstraint`). Empty = every
    leg is pinned to count 1, so existing signatures are unchanged.

    ``weights`` is the optional §38 **grounding**: a boundary is a *multiset* of typed
    wires, so a leg may carry an integer object-count > 1 (e.g. a grounded ``s`` that
    consumes a bundle of 3 orders in one firing). It maps a boundary :class:`Port` to
    its object-count; ports absent from it carry weight 1. Empty (the default) = every
    leg weight 1, so a symbolic/ungrounded generator behaves exactly as before. The
    unroller (:mod:`procposets.cospan.unroll_core`) sets it from a satisfying assignment of
    ``constraints``; :func:`procposets.cospan.compose.compose_signature` reads it via
    :meth:`weight`."""

    label: str
    left: frozenset[Port]
    right: frozenset[Port]
    constraints: frozenset = frozenset()  # frozenset[LinearConstraint]
    weights: frozenset = frozenset()  # frozenset[tuple[Port, int]]

    def constrained_ports(self) -> set:
        """Legs mentioned by some constraint (the rest default to count 1)."""
        out: set = set()
        for c in self.constraints:
            out |= c.ports()
        return out

    def weight(self, port: Port) -> int:
        """Object-count on boundary leg ``port`` (§38 grounding); 1 if unset."""
        for p, w in self.weights:
            if p == port:
                return w
        return 1

    def __str__(self) -> str:
        L = "{" + ",".join(map(str, sorted(self.left))) + "}"
        R = "{" + ",".join(map(str, sorted(self.right))) + "}"
        base = f"{L} -[{self.label}]-> {R}"
        if self.constraints:
            base += "  |  " + " ; ".join(sorted(map(str, self.constraints)))
        return base


@dataclass(frozen=True)
class Signature:
    """The full generator signature ``Sigma`` for a model."""

    generators: frozenset[Generator]

    def __len__(self) -> int:
        return len(self.generators)

    def __iter__(self):
        return iter(self.generators)

    def labels(self) -> set[str]:
        return {g.label for g in self.generators}

    def by_label(self, label: str) -> set[Generator]:
        return {g for g in self.generators if g.label == label}

    def ports(self) -> set[Port]:
        out: set[Port] = set()
        for g in self.generators:
            out |= set(g.left) | set(g.right)
        return out

    def pretty(self) -> str:
        return "\n".join(str(g) for g in sorted(self.generators, key=str))
