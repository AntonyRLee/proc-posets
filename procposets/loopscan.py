"""The loop layer, read as a *grouping* rather than a model feature.

A loop needs no model of its own.  Reading a case as a **group** and an
iteration as a **trace** makes rework depth the group size :math:`n_g` and the
loop body an ordinary component, so the estimator that already exists fits it
unchanged.  What that costs is a preparation step: the log must be cut into
pre-block / iterations / post-block, and the cut is a function of one thing --
which activities belong to the loop **body**.

This module supplies the measurement that answers it, and deliberately does
*not* supply a threshold.  Body membership is usually described by a recurrence
rate :math:`\\tau`, but :math:`\\tau` is not a parameter to pick: the map from
:math:`\\tau` to the body alphabet is a step function, constant on each plateau
between consecutive observed rates.  Every value inside a plateau returns
literally the same alphabet, so choosing one discards the plateau's width --
which is the only quantity that says whether the reading is supported at all.
:func:`recurrence_spectrum` therefore reports the whole step function.

Distinct from :mod:`procposets.loops`, which models a loop as a geometric
family of unrollings: that is the object the estimator no longer needs once a
loop is read as a grouping.  This module is *screening* -- it measures the log
and reports what a loop reading would rest on, without fitting anything.

The discriminator is *how* an activity repeats, not merely that it does.  A
logging duplicate lands **adjacent** to its original; a loop iteration puts the
rest of the body in between.  Splitting the two gives the noise floor and the
body signal as separate measurements instead of one rate that confounds them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

__all__ = [
    "ActivityRecurrence",
    "Plateau",
    "RecurrenceSpectrum",
    "recurrence_spectrum",
    "format_spectrum",
]

Trace = Sequence[Hashable]


@dataclass(frozen=True)
class ActivityRecurrence:
    """How one activity repeats, split by whether the repeat is adjacent."""

    activity: Hashable
    n_cases: int
    interleaved_cases: int
    """Cases that **rework** and in which it repeats -- the loop signal.

    Reworking is judged for the case as a whole (some activity recurs with
    other work in between), then membership by this activity's own
    multiplicity, so an iteration boundary that happens to place two copies
    back to back still counts.
    """
    immediate_cases: int
    """Cases where it recurs adjacently -- the logging-jitter signal."""
    cases: frozenset = frozenset()
    """Indices of the cases it recurs in, interleaved.  Two activities of one
    loop body recur in the *same* cases; two activities of different loops do
    not, which is what :meth:`RecurrenceSpectrum.cohesion` measures."""

    @property
    def interleaved_rate(self) -> float:
        return self.interleaved_cases / self.n_cases if self.n_cases else 0.0

    @property
    def immediate_rate(self) -> float:
        return self.immediate_cases / self.n_cases if self.n_cases else 0.0


@dataclass(frozen=True)
class Plateau:
    """A maximal interval of :math:`\\tau` over which the body is unchanged.

    ``lo`` is exclusive and ``hi`` inclusive: the body is the activities whose
    interleaved rate is at least :math:`\\tau`, so the alphabet changes only
    when :math:`\\tau` crosses an observed rate.
    """

    lo: float
    hi: float
    body: tuple

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, tau: float) -> bool:
        return self.lo < tau <= self.hi


@dataclass(frozen=True)
class RecurrenceSpectrum:
    """The whole :math:`\\tau \\mapsto \\text{body}` step function, measured."""

    n_cases: int
    activities: tuple[ActivityRecurrence, ...]
    """Every activity, ordered by interleaved rate, descending."""

    # ----------------------------------------------------------------
    # the two well-posedness bounds, both measured rather than chosen
    # ----------------------------------------------------------------

    @property
    def jitter_rate(self) -> float:
        """The largest adjacent-repeat rate: how dirty the log is.

        Adjacent repeats are what duplicate-logging produces.  This is a
        **cleanliness diagnostic, not a bound on the interleaved scale**:
        adjacency is collapsed before an interleaved repeat is counted, so
        jitter of this kind cannot inflate the spectrum the body is read off.
        It matters when it approaches the body's own rate, because then the
        two mechanisms are no longer visibly different.
        """
        return max((a.immediate_rate for a in self.activities), default=0.0)

    @property
    def rework_rate(self) -> float:
        """:math:`q` -- the largest interleaved-repeat rate in the log.

        The rate the body itself recurs at.  A :math:`\\tau` above it empties
        the body alphabet, dropping activities that demonstrably looped.
        """
        return max((a.interleaved_rate for a in self.activities), default=0.0)

    @property
    def admissible(self) -> tuple[float, float] | None:
        """The widest plateau's interval, or ``None`` when nothing recurs.

        This is the interval :math:`\\tau` may be chosen from, and every value
        in it returns the same body.  It is read off the interleaved spectrum
        alone -- deliberately *not* as ``(jitter_rate, rework_rate)``, which
        would compare an adjacent rate against interleaved ones and so mix two
        scales that measure different mechanisms.
        """
        widest = self.widest
        return None if widest is None else (widest.lo, widest.hi)

    # ----------------------------------------------------------------
    # the step function itself
    # ----------------------------------------------------------------

    def body_at(self, tau: float) -> tuple:
        """Activities whose interleaved rate is at least ``tau``."""
        return tuple(a.activity for a in self.activities
                     if a.interleaved_rate >= tau)

    @property
    def plateaus(self) -> tuple[Plateau, ...]:
        """Every interval of ``tau`` giving a distinct non-empty body.

        Widest first.  More than one means the log holds loops running at
        different rates: each plateau is then correct *for a different body*,
        and :math:`\\tau` selects which is in view rather than being right or
        wrong.  The answer in that case is all of them.
        """
        rates = sorted({a.interleaved_rate for a in self.activities
                        if a.interleaved_rate > 0.0}, reverse=True)
        out = []
        for i, hi in enumerate(rates):
            lo = rates[i + 1] if i + 1 < len(rates) else 0.0
            out.append(Plateau(lo, hi, self.body_at(hi)))
        return tuple(sorted(out, key=lambda p: (-p.width, -p.hi)))

    @property
    def widest(self) -> Plateau | None:
        return self.plateaus[0] if self.plateaus else None

    @property
    def margin(self) -> float:
        """Width of the widest plateau over the width of the next widest.

        Reported rather than thresholded.  Body activities never share a rate
        exactly at finite sample size, so the small differences between them
        open narrow plateaus that are sampling noise, not second loops; the
        margin is what separates those from a genuine second body running at
        its own rate.  ``inf`` when there is only one plateau.
        """
        widths = [p.width for p in self.plateaus]
        if len(widths) < 2:
            return float("inf")
        return float("inf") if widths[1] == 0 else widths[0] / widths[1]

    def cohesion(self, body: Sequence[Hashable]) -> float:
        """Smallest pairwise Jaccard of "cases I recur in", over ``body``.

        One loop body recurs as a unit: whenever the case rework, *every* body
        activity repeats, so the case sets coincide and the Jaccard is near 1.
        Two loops running at different rates give a low value, and that is the
        one thing the rate spectrum alone cannot see -- it orders activities by
        rate, so two separate loops appear as **nested** alphabets rather than
        as two bodies.  Measured on a two-loop log the widest plateau reports
        the fast body and the next reports the *union* of both, which is not a
        body at all.  This is the check that catches it.

        Returns 1.0 for a body of fewer than two activities.
        """
        sets = [a.cases for a in self.activities if a.activity in set(body)]
        if len(sets) < 2:
            return 1.0
        worst = 1.0
        for i, left in enumerate(sets):
            for right in sets[i + 1:]:
                union = len(left | right)
                worst = min(worst, len(left & right) / union if union else 1.0)
        return worst

    def suggested_split(self, body: Sequence[Hashable]) -> tuple[tuple, ...]:
        """Partition ``body`` by **exact** equality of its co-recurrence sets.

        When several loops coexist, the rate spectrum merges them: it orders
        activities by rate, so two loops surface as *nested* alphabets and
        their union appears as one plateau.  Activities of one body recur in
        exactly the same cases, so grouping by that set separates them, and a
        clean multi-loop log splits perfectly (each group has cohesion 1).

        Exact equality is used because it needs no threshold, and its failure
        mode is visible rather than silent: under noisy co-recurrence the
        partition fragments towards singletons, which says "do not trust this
        split" without a tuning knob deciding it.  Groups come back largest
        first.
        """
        wanted = set(body)
        groups: dict = {}
        for a in self.activities:
            if a.activity in wanted:
                groups.setdefault(a.cases, []).append(a.activity)
        return tuple(sorted((tuple(g) for g in groups.values()),
                            key=lambda g: (-len(g), str(g[0]))))

    @property
    def verdict(self) -> str:
        widest = self.widest
        if widest is None:
            return "no loop (no interleaved repeats anywhere)"
        body = ", ".join(str(b) for b in widest.body)
        head = (f"body = [{body}] for tau in ({widest.lo:.3f}, "
                f"{widest.hi:.3f}], width {widest.width:.3f}")
        cohesion = self.cohesion(widest.body)
        if cohesion < 1.0:
            head += f", cohesion {cohesion:.2f}"
        if cohesion == 0.0 and len(widest.body) > 1:
            split = " | ".join("[" + ", ".join(str(x) for x in g) + "]"
                               for g in self.suggested_split(widest.body))
            return (head + " -- NOT ONE LOOP: these activities never recur in "
                    "the same case, so they are separate loops the rate "
                    f"spectrum has merged; split = {split}")
        if len(self.plateaus) == 1:
            return head + " -- the only plateau, so tau cannot matter"
        margin = self.margin
        rest = (f"; {len(self.plateaus) - 1} narrower plateau(s), widest "
                f"{margin:.3g}x the next")
        if self.jitter_rate >= widest.hi:
            rest += (" -- WARNING: adjacent-repeat jitter reaches the body's "
                     "own rate, so the two mechanisms are not visibly distinct")
        return head + rest


def recurrence_spectrum(traces: Sequence[Trace]) -> RecurrenceSpectrum:
    """Measure how every activity repeats across a case log.

    ``traces`` is one activity sequence per **case** (not per iteration -- the
    cut into iterations is what this measurement exists to inform).
    """
    interleaved: Counter = Counter()
    immediate: Counter = Counter()
    seen: set = set()
    members: dict = {}
    for case_index, raw in enumerate(traces):
        trace = list(raw)
        counts = Counter(trace)
        seen.update(counts)
        adjacent = {a for a, b in zip(trace, trace[1:]) if a == b}
        # Does this case rework at all?  Adjacency is collapsed first, so a
        # case that merely double-logged an event does not qualify.  The test
        # is at CASE level, not per activity, because two iterations can place
        # the same activity back to back ("A B B A" is two iterations of a
        # concurrent body, not a duplicated B) -- judging each activity by its
        # own adjacency would discard that repeat as jitter.
        reworks = _reworks(trace)
        for activity, multiplicity in counts.items():
            if multiplicity < 2:
                continue
            if activity in adjacent:
                immediate[activity] += 1
            if reworks:
                interleaved[activity] += 1
                members.setdefault(activity, set()).add(case_index)
    n = len(traces)
    rows = [ActivityRecurrence(a, n, interleaved[a], immediate[a],
                               frozenset(members.get(a, ())))
            for a in seen]
    rows.sort(key=lambda r: (-r.interleaved_rate, -r.immediate_rate,
                             str(r.activity)))
    return RecurrenceSpectrum(n_cases=n, activities=tuple(rows))


def _reworks(trace: list) -> bool:
    """Does this case run some activity twice with other work in between?

    Adjacent duplicates are collapsed first, so ``a a b`` is a double-logged
    ``a`` and does not count, while ``a b a`` and ``a b b a`` do.  This is the
    signature that separates a loop from logging jitter: a duplicate lands
    beside its original, an iteration puts the rest of the body in between.
    """
    collapsed = [e for i, e in enumerate(trace) if i == 0 or e != trace[i - 1]]
    return len(collapsed) != len(set(collapsed))


def format_spectrum(spectrum: RecurrenceSpectrum) -> str:
    """The step function as text, closing on what it means for ``tau``."""
    lines = [f"recurrence spectrum over {spectrum.n_cases} cases",
             "",
             f"  {'activity':<20s} {'interleaved':>12s} {'adjacent':>10s}"]
    for a in spectrum.activities:
        lines.append(f"  {str(a.activity):<20s} {a.interleaved_rate:>12.3f} "
                     f"{a.immediate_rate:>10.3f}")
    lines.append("")
    lines.append(f"  jitter rate (max adjacent)        = "
                 f"{spectrum.jitter_rate:.3f}   [log cleanliness only]")
    lines.append(f"  rework rate (max interleaved)     = "
                 f"{spectrum.rework_rate:.3f}")
    window = spectrum.admissible
    lines.append("  admissible tau                    = "
                 + ("none" if window is None
                    else f"({window[0]:.3f}, {window[1]:.3f}]"))
    lines.append("")
    if spectrum.plateaus:
        lines.append("  plateaus (widest first) -- tau anywhere inside one "
                     "gives the same body:")
        for pl in spectrum.plateaus:
            body = ", ".join(str(b) for b in pl.body)
            cohesion = spectrum.cohesion(pl.body)
            lines.append(f"    ({pl.lo:.3f}, {pl.hi:.3f}]  width "
                         f"{pl.width:.3f}   cohesion "
                         f"{cohesion:.2f}   body = [{body}]")
            if cohesion < 1.0 and len(pl.body) > 1:
                split = " | ".join("[" + ", ".join(str(x) for x in g) + "]"
                                   for g in spectrum.suggested_split(pl.body))
                lines.append(f"{'':>42s}not one loop; split = {split}")
        lines.append("")
    lines.append(f"  VERDICT: {spectrum.verdict}")
    return "\n".join(lines)
