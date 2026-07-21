"""Cross-notation generator comparison: line up master vs discovered signatures
as **N-linear-parameterised generator cospans** -- the object the user reads off
``model.svg`` ("a finite small collection of N-linear parameterised generators").

Where :mod:`cpm.cospan.signature_diff` compares *behavioural label-skeletons*
(deliberately dropping ports, types and multiplicity -- "same activity language"),
this keeps the **generator inventory with its §32 leg-multiplicity parameters
intact** and asks "did discovery rederive the master's parameterised generators".

The obstacle is that :class:`~cpm.cospan.signature.Port` ``src``/``tgt`` naming is
notation-specific -- the master's hand-authored ``gamma1``/``outcome`` vs the OCCN
miner's ``START_<ot>`` -- so a raw ``Port`` comparison aligns nothing. What *is*
shared (both adapters read it off the same OCEL) is the activity **label** and the
object **types** on each leg. So the canonical, notation-independent identity of a
generator is

    CanonKey = (label, multiset of input object-types, multiset of output types)

and under that key we compare the per-``(side, type)`` binding intervals (the §32
cardinalities) and any multi-leg relations (shared-key partitions). Two notations
*agree* on a generator iff those coincide.

Modular by construction: :func:`compare` takes ``{notation: Signature}`` for any
number of notations, so a newly-added notation is just another entry -- no code
change. Boundary generators (``gamma1``/``gamma2``/``START_<ot>``/``END_<ot>``) are
flagged (:func:`is_gamma_or_marker`) and sorted last, because their *encoding*
differs by adapter convention (master's single γ1/γ2 interface §39 vs OCCN's
per-type START/END §40) -- a real, documented structural difference the matrix
should show honestly rather than force-align.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .signature import Generator, Signature

# unbounded cmax sentinel (a `*` interval upper bound)
INF = None


def is_gamma_or_marker(label: str) -> bool:
    """ANY boundary generator: a master ``gamma1``/``gamma2`` OR an OCCN per-type
    ``START_<ot>``/``END_<ot>`` wrapper.  Whose encoding is adapter-specific, so it
    is grouped apart from the interior activities that genuinely align across
    notations.  NB the *gamma-inclusive* membership -- distinct from
    :func:`signature_diff._is_start_end_marker`, which is ``START_``/``END_`` only
    (the two were once both called ``[_]is_boundary_label``, a hazard)."""
    return (
        label in ("gamma1", "gamma2")
        or label.startswith("START_")
        or label.startswith("END_")
    )


# Back-compat alias for the pre-rename name (gamma-inclusive membership).
is_boundary_label = is_gamma_or_marker


def _type_multiset(ports) -> tuple:
    """The object-type multiset of a leg-set, as a sorted ``((type, count), ...)``
    tuple -- the notation-independent shape of one boundary side."""
    return tuple(sorted(Counter(p.typ for p in ports).items(), key=lambda kv: (str(kv[0]), kv[1])))


@dataclass(frozen=True, order=True)
class CanonKey:
    """Notation-independent generator identity: label + typed in/out arity."""

    label: str
    left: tuple   # ((otype, count), ...)
    right: tuple  # ((otype, count), ...)

    @property
    def boundary(self) -> bool:
        return is_gamma_or_marker(self.label)

    def arity_str(self) -> str:
        def side(ms: tuple) -> str:
            if not ms:
                return "·"
            return " ".join(f"{t}{'' if c == 1 else f'×{c}'}" for t, c in ms)
        return f"{side(self.left)} → {side(self.right)}"


def canon_key(g: Generator) -> CanonKey:
    return CanonKey(g.label, _type_multiset(g.left), _type_multiset(g.right))


@dataclass(frozen=True)
class BindingProfile:
    """The §32 N-linear parameters of a generator, keyed by ``(side, type)`` so it is
    comparable across notations.

    ``intervals`` -- one entry per leg carrying a non-trivial single-leg cardinality:
    ``(side, otype, (lo, hi))`` with ``hi = None`` an unbounded ``*``. A leg with no
    constraint is the default count 1 and is **omitted** (so an all-1:1 generator has
    an empty profile, exactly the ``model.svg`` "legs without a bind label are 1:1"
    convention). ``relations`` -- multi-leg constraints (shared-key partitions etc.),
    canonicalised to ``(((side, otype), coeff), ...) rel rhs``."""

    intervals: frozenset  # frozenset[(str, str|None, tuple[int, int|None])]
    relations: frozenset  # frozenset[(tuple, str, int)]

    def is_trivial(self) -> bool:
        return not self.intervals and not self.relations

    def render(self) -> str:
        """Compact ``↓order[2,5] ↑order=1`` form (↓ input, ↑ output); ``1:1`` when
        trivial."""
        if self.is_trivial():
            return "1:1"
        arrow = {"in": "↓", "out": "↑"}
        parts = []
        for side, typ, (lo, hi) in sorted(self.intervals, key=lambda x: (x[0], str(x[1]))):
            rng = f"={lo}" if hi == lo else f"[{lo},{'*' if hi is INF else hi}]"
            parts.append(f"{arrow[side]}{typ}{rng}")
        for agg, rel, rhs in sorted(self.relations):
            def coeff(c: int) -> str:
                return "" if c == 1 else ("-" if c == -1 else f"{c}")
            terms = "".join(
                f"{'+' if c > 0 and k else ''}{coeff(c)}{arrow.get(s, s)}{t}"
                for k, ((s, t), c) in enumerate(agg) if c
            )
            parts.append(f"⟨{terms}{rel}{rhs}⟩")
        return " ".join(parts)


def _side_of(g: Generator, port) -> str:
    return "in" if port in g.left else "out"


def binding_profile(g: Generator) -> BindingProfile:
    """Extract :class:`BindingProfile` from one generator's §32 constraints.

    Single-leg unit-coefficient constraints are folded per leg into a ``(lo, hi)``
    interval (mirroring :func:`cpm.vis._leg_card_by_port`): ``>=``/``==`` raise ``lo``,
    ``<=``/``==`` lower ``hi``; a leg with none defaults to 1:1 and is dropped.
    Multi-leg constraints become :attr:`BindingProfile.relations`."""
    by_port: dict = {}  # Port -> [lo, hi]
    relations: set = set()
    for c in g.constraints:
        terms = tuple(c.terms)
        if len(terms) == 1 and terms[0][1] == 1:
            p = terms[0][0]
            lo, hi = by_port.get(p, [0, INF])
            if c.rel in (">=", "=="):
                lo = max(lo, c.rhs)
            if c.rel in ("<=", "=="):
                hi = c.rhs if hi is INF else min(hi, c.rhs)
            by_port[p] = [lo, hi]
        else:
            agg: dict = {}
            for p, coeff in terms:
                key = (_side_of(g, p), p.typ)
                agg[key] = agg.get(key, 0) + coeff
            relations.add((tuple(sorted(agg.items())), c.rel, c.rhs))
    # an exact (1,1) interval is the *default* count-1 of an unconstrained leg, so it
    # is dropped: a leg the master leaves implicit (1:1) and a leg the OCCN miner pins
    # explicitly with `exactly == 1` are the same binding, and must compare equal.
    intervals = frozenset(
        (_side_of(g, p), p.typ, (lo, hi)) for p, (lo, hi) in by_port.items() if (lo, hi) != (1, 1)
    )
    return BindingProfile(intervals=intervals, relations=frozenset(relations))


def _merge_profiles(profiles: list[BindingProfile]) -> BindingProfile:
    """Union the binding profiles of a notation's generators that share one
    :class:`CanonKey` (e.g. the checked/skip-a routing variants of ``b``, both 1:1).
    They are normally identical or trivial; a union keeps every distinct binding the
    notation attaches to that generator."""
    inter: set = set()
    rels: set = set()
    for p in profiles:
        inter |= p.intervals
        rels |= p.relations
    return BindingProfile(intervals=frozenset(inter), relations=frozenset(rels))


def canonical_generators(sig: Signature) -> dict[CanonKey, BindingProfile]:
    """One :class:`BindingProfile` per :class:`CanonKey` in ``sig`` (within-signature
    duplicates merged, :func:`_merge_profiles`)."""
    groups: dict[CanonKey, list[BindingProfile]] = {}
    for g in sig.generators:
        groups.setdefault(canon_key(g), []).append(binding_profile(g))
    return {k: _merge_profiles(v) for k, v in groups.items()}


# -- comparison report -------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One generator's status in one notation.

    ``status``: ``ref`` (the reference column), ``match`` (== reference),
    ``diff`` (present but binding-params differ from reference), ``absent`` (the
    notation lacks this generator), ``novel`` (present, but absent in the reference)."""

    profile: BindingProfile | None
    status: str


@dataclass(frozen=True)
class GenRow:
    key: CanonKey
    cells: tuple  # ((notation, Cell), ...) in column order

    def verdict(self) -> str:
        statuses = {c.status for _, c in self.cells}
        if "diff" in statuses:
            return "param-diff"
        if "absent" in statuses or "novel" in statuses:
            return "partial"
        return "match"


@dataclass(frozen=True)
class ComparisonReport:
    notations: tuple  # column order (reference first)
    reference: str
    rows: tuple  # (GenRow, ...) interior generators first, boundary last

    def summary(self) -> dict[str, int]:
        out = {"match": 0, "param-diff": 0, "partial": 0}
        for r in self.rows:
            out[r.verdict()] += 1
        return out

    def per_notation(self) -> dict[str, dict[str, int]]:
        """Per non-reference notation, the cell-status tally **against the reference**:
        ``match`` (rederived the master generator, params agree), ``diff`` (present,
        params differ), ``absent`` (master generator not found), ``novel`` (extra
        generator the master lacks). This is the direct "did notation X rederive the
        master signature" answer, read per column rather than as the all-column row
        verdict."""
        out: dict[str, dict[str, int]] = {}
        for n in self.notations:
            if n == self.reference:
                continue
            tally = {"match": 0, "diff": 0, "absent": 0, "novel": 0}
            for r in self.rows:
                cells = dict(r.cells)
                ref_present = cells[self.reference].status == "ref"
                st = cells[n].status
                # skip rows that are neither the reference's nor this notation's (a *third*
                # notation's generator) -- they are irrelevant to "did n rederive master".
                if st == "absent" and not ref_present:
                    continue
                tally[st] += 1
            out[n] = tally
        return out


def compare(named_sigs: dict[str, Signature], *, reference: str | None = None) -> ComparisonReport:
    """Align the generators of several notations' signatures by :class:`CanonKey` and
    classify each against ``reference`` (default: ``"master"`` if present, else the
    first notation). Columns are ordered reference-first; rows are interior
    generators (sorted by label/arity) then boundary generators."""
    notations = list(named_sigs)
    if reference is None:
        reference = "master" if "master" in named_sigs else notations[0]
    if reference not in named_sigs:
        raise ValueError(f"reference {reference!r} not among {notations}")
    order = [reference] + [n for n in notations if n != reference]

    per_notation = {n: canonical_generators(s) for n, s in named_sigs.items()}
    all_keys = set().union(*per_notation.values()) if per_notation else set()
    ref_gens = per_notation[reference]

    rows: list[GenRow] = []
    for key in all_keys:
        cells = []
        ref_prof = ref_gens.get(key)
        for n in order:
            prof = per_notation[n].get(key)
            if n == reference:
                status = "ref" if prof is not None else "absent"
            elif prof is None:
                status = "absent"
            elif ref_prof is None:
                status = "novel"
            else:
                status = "match" if prof == ref_prof else "diff"
            cells.append((n, Cell(profile=prof, status=status)))
        rows.append(GenRow(key=key, cells=tuple(cells)))

    rows.sort(key=lambda r: (r.key.boundary, r.key.label, r.key.left, r.key.right))
    return ComparisonReport(notations=tuple(order), reference=reference, rows=tuple(rows))


def report_to_dict(report: ComparisonReport) -> dict:
    """JSON-serialisable comparison (byte-stable): the canonical generator table with
    per-notation binding profiles and verdicts."""
    def prof_dict(p: BindingProfile | None) -> object:
        if p is None:
            return None
        return {
            "intervals": sorted(
                [side, typ, lo, ("*" if hi is INF else hi)] for side, typ, (lo, hi) in p.intervals
            ),
            "relations": sorted(
                [[[list(k), c] for k, c in agg], rel, rhs] for agg, rel, rhs in p.relations
            ),
            "render": p.render(),
        }

    rows = []
    for r in report.rows:
        rows.append({
            "label": r.key.label,
            "in": [[t, c] for t, c in r.key.left],
            "out": [[t, c] for t, c in r.key.right],
            "boundary": r.key.boundary,
            "verdict": r.verdict(),
            "cells": {n: {"status": cell.status, "binding": prof_dict(cell.profile)}
                      for n, cell in r.cells},
        })
    return {
        "reference": report.reference,
        "notations": list(report.notations),
        "summary": report.summary(),
        "per_notation": report.per_notation(),
        "generators": rows,
    }


def report_text(report: ComparisonReport) -> str:
    """A plain-text rendering of the comparison table for ``traces.txt``-style dumps
    and the console."""
    cols = report.notations
    w_lbl = max([len("generator")] + [len(f"{r.key.label}  {r.key.arity_str()}") for r in report.rows], default=9)
    w = {n: max(len(n), 8) for n in cols}
    for r in report.rows:
        for n, cell in r.cells:
            txt = "—" if cell.profile is None else cell.profile.render()
            w[n] = max(w[n], len(txt))

    def line(label: str, vals: list[str]) -> str:
        return "  ".join([label.ljust(w_lbl)] + [v.ljust(w[n]) for v, n in zip(vals, cols)])

    out = [
        f"# generator comparison — reference: {report.reference}",
        f"# row verdicts (all columns): {report.summary()}",
    ]
    for n, t in report.per_notation().items():
        out.append(f"# vs {report.reference} — {n}: {t['match']} rederived, {t['diff']} param-diff, "
                   f"{t['absent']} absent, {t['novel']} novel")
    out += [
        "",
        line("generator", list(cols)),
        line("-" * 9, ["-" * w[n] for n in cols]),
    ]
    last_boundary = False
    for r in report.rows:
        if r.key.boundary and not last_boundary:
            out.append(line("· boundary ·", ["" for _ in cols]))
            last_boundary = True
        vals = ["—" if c.profile is None else c.profile.render() for _, c in r.cells]
        marker = {"match": "", "param-diff": "  ✗", "partial": "  ~"}[r.verdict()]
        out.append(line(f"{r.key.label}  {r.key.arity_str()}", vals) + marker)
    return "\n".join(out) + "\n"
