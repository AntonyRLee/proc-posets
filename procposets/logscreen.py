"""Screening an event log for within-case concurrency.

Every construction in this package reads a case as a *partial order* of activity occurrences. Most
event logs cannot supply one: they record a total order, or they record ties that look like
simultaneity but are one actor writing several fields at a single instant. Running the machinery on
such a log yields a definite answer computed from structure the log never contained.

This module answers, for a given log, the question a user should ask before anything else: **does
this log carry within-case concurrency at all?** It is deliberately a falsification report -- it
tells you when to stop -- and it is the input-side counterpart of :mod:`procposets.diagnostics`,
which falsifies a *fit*.

Two independent readings, because "two things happened at once" has two meanings in event data:

:func:`tie_composition`
    **Simultaneity by timestamp equality.** Two events of one case carry the same stamp. This is the
    reading used when events are instants. It is only as good as the clock: at a one-second ceiling
    a tie may be genuine simultaneity or same-second batching, and nothing in the log distinguishes
    them, so the granularity check gates everything else.

:func:`interval_overlap`
    **Simultaneity by overlapping duration.** Activity instances carrying a lifecycle (``start``
    then an end) occupy intervals, and two intervals of one case intersect. This reading needs no
    sub-second clock, but needs lifecycle instrumentation.

A log can pass one and fail the other. Report both.

Both functions stream with ``iterparse`` and hold one case at a time, so peak memory is flat on
multi-hundred-megabyte logs. Standard library only: no numpy, no pandas, no pm4py.

Command line::

    python -m procposets.logscreen LOG.xes.gz --report ties
    python -m procposets.logscreen LOG.xes.gz --report intervals
"""
from __future__ import annotations

import collections
import gzip
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional, Sequence
import xml.etree.ElementTree as ET

__all__ = [
    "TieReport", "OverlapReport", "LoopReport",
    "tie_composition", "interval_overlap", "loop_composition",
    "iter_cases", "DEFAULT_TERMINATORS",
]

_NS = "{http://www.xes-standard.org/}"
_FRAC = re.compile(r"\.(\d+)")

_STARTERS = frozenset({"start"})
_RESUMERS = frozenset({"resume"})
_PAUSERS = frozenset({"suspend"})

#: Lifecycle transitions that end an activity instance.
#:
#: ``complete`` alone is the naive choice and is usually wrong. In a log where work items can be
#: abandoned, most instances never complete, so pairing on ``complete`` leaves their ``start``
#: events unmatched and marries them to a later, unrelated completion -- manufacturing long
#: intervals that overlap everything nearby. Accepting the abandonment transitions removes those
#: false pairs; it does not invent real ones. The instance occupied its resource either way.
DEFAULT_TERMINATORS = frozenset({"complete", "ate_abort", "withdraw"})


def _opener(path: str):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def _has_subsecond(stamp: Optional[str]) -> bool:
    """True iff the stamp carries a non-zero fractional-second component."""
    m = _FRAC.search(stamp or "")
    return bool(m) and int(m.group(1)) != 0


def _parse_ts(stamp: Optional[str]) -> Optional[datetime]:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(stamp, fmt)
            except ValueError:
                pass
    return None


def iter_cases(path: str, keys: Sequence[str]) -> Iterator[list[tuple]]:
    """Yield one case at a time as a list of tuples of the requested attribute ``keys``.

    Streaming: each element is cleared as it closes, so memory stays flat regardless of log size.
    """
    keys = tuple(keys)
    cur: Optional[list] = None
    with _opener(path) as fh:
        for ev, el in ET.iterparse(fh, events=("start", "end")):
            tag = el.tag.replace(_NS, "")
            if ev == "start" and tag == "trace":
                cur = []
            elif ev == "end" and tag == "event":
                if cur is not None:
                    d = {c.get("key"): c.get("value") for c in el}
                    cur.append(tuple(d.get(k) for k in keys))
                el.clear()
            elif ev == "end" and tag == "trace":
                yield cur or []
                cur = None
                el.clear()


# --------------------------------------------------------------------------- ties


@dataclass
class TieReport:
    """Simultaneity read as timestamp equality within a case."""

    cases: int = 0
    events: int = 0
    variants: int = 0
    singletons: int = 0
    subsecond_stamps: int = 0
    coarse_stamps: int = 0
    subsecond_tie_cases: int = 0
    subsecond_tied_events: int = 0
    subsecond_blocks: int = 0
    coarse_tie_cases: int = 0
    coarse_tied_events: int = 0
    coarse_blocks: int = 0
    scored_blocks: int = 0
    distinct_activity_blocks: int = 0
    cross_resource_blocks: int = 0
    single_resource_blocks: int = 0
    no_resource_blocks: int = 0
    scored_stratum: str = "subsecond"

    @property
    def subsecond_pct(self) -> float:
        return _pct(self.subsecond_stamps, self.events)

    @property
    def singleton_pct(self) -> float:
        """Share of cases that are their own trace variant. Near 100% saturates any order
        comparison: every case is unique, so there is nothing for a distance to generalise over."""
        return _pct(self.singletons, self.cases)

    @property
    def clock_too_coarse(self) -> bool:
        """No sub-second stamp exists anywhere. A schema property, not a sampling one: no amount
        of extra data will produce one, so tie-based simultaneity is undecidable on this log."""
        return self.subsecond_stamps == 0

    @property
    def distinct_activity_pct(self) -> float:
        return _pct(self.distinct_activity_blocks, self.scored_blocks)

    @property
    def cross_resource_pct(self) -> float:
        """The discriminator. A tie block spanning two resources is two actors working at once; a
        block within one resource is one actor emitting several fields at one instant."""
        return _pct(self.cross_resource_blocks, self.scored_blocks)

    @property
    def single_resource_pct(self) -> float:
        return _pct(self.single_resource_blocks, self.scored_blocks)

    def verdict(self, min_distinct_activity: float = 80.0,
                min_cross_resource: float = 50.0) -> str:
        """``"pass"``, ``"fail"``, or ``"undecidable"``.

        Undecidable is not a failure of the data collector and not a pass: it means the clock
        cannot represent the distinction, so the question cannot be answered from this log.
        """
        if self.scored_stratum == "coarse" or self.clock_too_coarse:
            return "undecidable"
        if not self.scored_blocks:
            return "fail"
        return "pass" if (self.distinct_activity_pct >= min_distinct_activity
                          and self.cross_resource_pct >= min_cross_resource) else "fail"


def _pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else 0.0


def tie_composition(path: str, resource_key: str = "org:resource",
                    drop_resources: Sequence[str] = (),
                    score_coarse_stratum: bool = False) -> TieReport:
    """Measure within-case timestamp ties and what they are made of.

    Ties are read as timestamp **equality**, never interval overlap -- see :func:`interval_overlap`
    for the other reading.

    Sub-second and coarse ties are counted in separate strata and never merged: they are different
    evidence. A tie at a one-second ceiling is consistent with same-second batching, so composition
    is scored on the sub-second stratum only, unless ``score_coarse_stratum`` is set -- which
    records *which kind* of undecidable a log is, and never yields a pass.

    ``drop_resources`` removes non-human actors (a system account that stamps everything) from the
    cross-resource test, where they would otherwise fake multi-actor blocks.
    """
    drop = set(drop_resources)
    r = TieReport(scored_stratum="coarse" if score_coarse_stratum else "subsecond")
    variants: collections.Counter = collections.Counter()

    for evs in iter_cases(path, ("time:timestamp", "concept:name", resource_key)):
        r.cases += 1
        r.events += len(evs)
        for stamp, _, _ in evs:
            if _has_subsecond(stamp):
                r.subsecond_stamps += 1
            else:
                r.coarse_stamps += 1
        variants[tuple(a for _, a, _ in evs)] += 1

        blocks: dict = collections.defaultdict(list)
        for stamp, act, res in evs:
            blocks[stamp].append((act, res))

        case_sub = case_coarse = False
        for stamp, blk in blocks.items():
            if len(blk) < 2:
                continue
            sub = _has_subsecond(stamp)
            if sub:
                case_sub = True
                r.subsecond_blocks += 1
                r.subsecond_tied_events += len(blk)
            else:
                case_coarse = True
                r.coarse_blocks += 1
                r.coarse_tied_events += len(blk)
            if sub or score_coarse_stratum:
                acts = {a for a, _ in blk}
                res = {x for _, x in blk if x is not None}
                r.scored_blocks += 1
                r.distinct_activity_blocks += len(acts) > 1
                r.cross_resource_blocks += len(res - drop) > 1
                r.single_resource_blocks += len(res) == 1
                r.no_resource_blocks += not res
        r.subsecond_tie_cases += case_sub
        r.coarse_tie_cases += case_coarse

    r.variants = len(variants)
    r.singletons = sum(1 for v in variants.values() if v == 1)
    return r


# ----------------------------------------------------------------------- intervals


@dataclass
class OverlapReport:
    """Simultaneity read as overlapping activity-instance durations within a case."""

    cases: int = 0
    instances: int = 0
    segments: int = 0
    cases_without_interval: int = 0
    envelope_overlap_cases: int = 0
    envelope_overlap_distinct_activity: int = 0
    active_overlap_cases: int = 0
    active_overlap_distinct_activity: int = 0
    overlapping_pairs: collections.Counter = field(default_factory=collections.Counter)

    @property
    def envelope_pct(self) -> float:
        """Share of cases where two instances were both *open*, suspended time included."""
        return _pct(self.envelope_overlap_cases, self.cases)

    @property
    def active_pct(self) -> float:
        """Share of cases where two instances were both *being worked*, suspensions excluded.
        The tighter reading, and the one that speaks to genuine simultaneity."""
        return _pct(self.active_overlap_cases, self.cases)


def _case_intervals(evs, terminators, activity_prefix=None):
    """Return ``(envelope, active)`` interval lists for one case.

    Each interval is ``(start, end, activity, instance_id)``. Instances of the same activity are
    paired by queue -- earliest open closes first -- since XES carries no instance identifier.
    Unterminated instances are dropped: an open-ended interval would overlap everything after it,
    which is an artefact of the log ending rather than a measurement.
    """
    evs = [e for e in evs if e[0] is not None]
    evs.sort(key=lambda e: e[0])
    open_q: dict = collections.defaultdict(collections.deque)
    done = []
    counter = 0
    for ts, act, lt, _res in evs:
        if act is None or (activity_prefix and not act.startswith(activity_prefix)):
            continue
        lt = (lt or "").lower()
        if lt in _STARTERS:
            counter += 1
            open_q[act].append({"id": counter, "act": act, "first": ts,
                                "open": ts, "segs": [], "end": None})
        elif lt in _RESUMERS:
            if open_q[act] and open_q[act][0]["open"] is None:
                open_q[act][0]["open"] = ts
        elif lt in _PAUSERS:
            if open_q[act] and open_q[act][0]["open"] is not None:
                inst = open_q[act][0]
                inst["segs"].append((inst["open"], ts))
                inst["open"] = None
        elif lt in terminators:
            if open_q[act]:
                inst = open_q[act].popleft()
                if inst["open"] is not None:
                    inst["segs"].append((inst["open"], ts))
                    inst["open"] = None
                inst["end"] = ts
                done.append(inst)
    envelope = [(i["first"], i["end"], i["act"], i["id"]) for i in done if i["end"] is not None]
    active = [(a, b, i["act"], i["id"]) for i in done for (a, b) in i["segs"] if b > a]
    return envelope, active


def _first_overlap(intervals, distinct_activity=False):
    """Sweep for a strictly overlapping pair from two different instances."""
    live: list = []
    for s, e, act, iid in sorted(intervals, key=lambda x: x[0]):
        live = [x for x in live if x[0] > s]
        for end, other_act, other_id in live:
            if other_id == iid or (distinct_activity and other_act == act):
                continue
            if end > s:
                return (other_act, act)
        live.append((e, act, iid))
    return None


def interval_overlap(path: str, terminators: Sequence[str] = (),
                     activity_prefix: Optional[str] = None,
                     collect_pairs: bool = False) -> OverlapReport:
    """Measure within-case overlap of activity-instance intervals.

    An instance runs from its ``start`` to its end. ``terminators`` names the transitions that end
    one; it defaults to :data:`DEFAULT_TERMINATORS`, which accepts abandonment as well as
    completion. See that constant for why ``complete`` alone is usually the wrong choice.

    ``suspend``/``resume`` split an instance into worked segments, giving two readings: the
    *envelope* from first start to end, and the *active* segments only. Both are reported, since
    the gap between them says how much apparent overlap is really queued or paused time.

    Overlap is strict: intervals must intersect in positive length.
    """
    terms = frozenset(terminators) if terminators else DEFAULT_TERMINATORS
    r = OverlapReport()
    for evs in iter_cases(path, ("time:timestamp", "concept:name",
                                 "lifecycle:transition", "org:resource")):
        evs = [(_parse_ts(t), a, lt, res) for t, a, lt, res in evs]
        r.cases += 1
        envelope, active = _case_intervals(evs, terms, activity_prefix)
        r.instances += len(envelope)
        r.segments += len(active)
        if not envelope:
            r.cases_without_interval += 1
        hit = _first_overlap(envelope)
        if hit:
            r.envelope_overlap_cases += 1
            if collect_pairs:
                r.overlapping_pairs[tuple(sorted(hit))] += 1
        if _first_overlap(envelope, distinct_activity=True):
            r.envelope_overlap_distinct_activity += 1
        if _first_overlap(active):
            r.active_overlap_cases += 1
        if _first_overlap(active, distinct_activity=True):
            r.active_overlap_distinct_activity += 1
    return r


# ----------------------------------------------------------------------------- cli


def _format_ties(path: str, r: TieReport) -> str:
    L = [f"=== {path}",
         f"cases {r.cases}   events {r.events}   variants {r.variants}",
         "",
         "clock granularity",
         f"  sub-second stamps      {r.subsecond_stamps:>9}  ({r.subsecond_pct:5.2f}%)",
         f"  coarse stamps          {r.coarse_stamps:>9}  ({_pct(r.coarse_stamps, r.events):5.2f}%)",
         f"  -> {'no sub-second stamp exists: ties are undecidable' if r.clock_too_coarse else 'sub-second stamps present'}",
         "",
         "variant saturation",
         f"  singleton traces       {r.singletons:>9}  ({r.singleton_pct:5.2f}%)",
         "",
         "tie rate (strata are separate evidence and are never merged)",
         f"  sub-second  cases {r.subsecond_tie_cases:>8} ({_pct(r.subsecond_tie_cases, r.cases):5.2f}%)"
         f"   events {r.subsecond_tied_events:>9} ({_pct(r.subsecond_tied_events, r.events):5.2f}%)"
         f"   blocks {r.subsecond_blocks}",
         f"  coarse      cases {r.coarse_tie_cases:>8} ({_pct(r.coarse_tie_cases, r.cases):5.2f}%)"
         f"   events {r.coarse_tied_events:>9} ({_pct(r.coarse_tied_events, r.events):5.2f}%)"
         f"   blocks {r.coarse_blocks}",
         "",
         f"composition of the {r.scored_blocks} scored {r.scored_stratum} tie blocks"]
    if r.scored_blocks:
        L += [f"  distinct-activity      {r.distinct_activity_blocks:>9}  ({r.distinct_activity_pct:5.1f}%)",
              f"  cross-resource         {r.cross_resource_blocks:>9}  ({r.cross_resource_pct:5.1f}%)",
              f"  single-resource        {r.single_resource_blocks:>9}  ({r.single_resource_pct:5.1f}%)"]
    else:
        L.append("  none in the scored stratum")
    v = r.verdict()
    L += ["", f"verdict: {v.upper()}"]
    if v == "undecidable":
        L.append("  the clock cannot represent the distinction, so this log cannot answer the question")
    elif v == "fail":
        L.append("  ties are an emission artefact, not observed simultaneity")
    return "\n".join(L)


def _format_intervals(path: str, r: OverlapReport) -> str:
    L = [f"=== {path}",
         f"cases {r.cases}   paired instances {r.instances}   worked segments {r.segments}",
         f"cases with no paired interval: {r.cases_without_interval}"
         f" ({_pct(r.cases_without_interval, r.cases):.2f}%)",
         "",
         "within-case interval overlap (cases with at least one strictly overlapping pair)",
         f"  envelope  [first start -> end]            {r.envelope_overlap_cases:>8}  ({r.envelope_pct:5.2f}%)",
         f"    distinct activities only                {r.envelope_overlap_distinct_activity:>8}"
         f"  ({_pct(r.envelope_overlap_distinct_activity, r.cases):5.2f}%)",
         f"  active    worked segments only            {r.active_overlap_cases:>8}  ({r.active_pct:5.2f}%)",
         f"    distinct activities only                {r.active_overlap_distinct_activity:>8}"
         f"  ({_pct(r.active_overlap_distinct_activity, r.cases):5.2f}%)"]
    if r.overlapping_pairs:
        L += ["", "  most common overlapping pairs"]
        for (a, b), n in r.overlapping_pairs.most_common(8):
            L.append(f"    {n:>6}  {a} || {b}")
    return "\n".join(L)


# --------------------------------------------------------------------------- loops


@dataclass
class LoopReport:
    """Repetition read as activity multiplicity within a case.

    The third reading, and the one that decides whether the estimator can see
    this log at all: every construction downstream reads a case as a partial
    order of activity occurrences, and ``GroupedLog`` refuses a trace whose
    activity labels repeat.  A repeat means the case is a *pomset*, not a
    poset -- the same label occupies several positions -- and that is a
    different object with a different estimator (bounded unrolling).

    Repetition is not one phenomenon, so the report separates two:

    **Immediate repetition** (``a a``, adjacent).  Usually instrumentation:
    a batch write, a retried emit, a status refresh.  It inflates
    multiplicity without implying any cycle in the process, and collapsing
    runs often makes an otherwise-usable log usable.

    **Cycles** (``a … a`` with work in between).  A genuine rework loop, whose
    *body* is the activity set strictly between two consecutive occurrences.
    This is the structure the loop law models, and it is what makes ``K_loop`` a
    modelling parameter rather than a nuisance.
    """

    cases: int = 0
    events: int = 0
    events_dropped_lifecycle: int = 0
    loop_free_cases: int = 0
    immediate_only_cases: int = 0
    cycle_cases: int = 0
    max_multiplicity: int = 0
    case_max_multiplicity: Counter = field(default_factory=Counter)
    """max activity multiplicity in a case -> number of cases with that maximum."""
    repeating_activities: Counter = field(default_factory=Counter)
    """activity -> number of cases in which it occurs more than once."""
    activity_max_multiplicity: dict = field(default_factory=dict)
    loop_bodies: Counter = field(default_factory=Counter)
    """loop body (the activities recurring in a case) -> number of cases."""
    incoherent_body_cases: int = 0
    """Cycling cases whose recurring activities do NOT share one multiplicity.

    Under the declared conventions -- positional occurrence alignment, do-while
    -- a coherent body is not a precondition to check and fall back from, it is
    **definitional**: a pomset has no optional behaviour, so every iteration
    runs the whole body. A case where ``b`` recurs three times and ``c`` twice
    is therefore *evidence against* the single-loop model, not a case to align
    more cleverly. It means the body is smaller than it looks, the extra
    occurrences sit outside the loop, or the case mixes two loop components.
    """

    @property
    def loop_free_pct(self) -> float:
        """Share of cases the current input contract already accepts."""
        return _pct(self.loop_free_cases, self.cases)

    @property
    def cycle_pct(self) -> float:
        return _pct(self.cycle_cases, self.cases)

    @property
    def immediate_only_pct(self) -> float:
        return _pct(self.immediate_only_cases, self.cases)

    @property
    def body_coherent_pct(self) -> float:
        """Share of cycling cases whose body runs complete every iteration.

        Positional occurrence alignment is exactly correct on these and
        mislabels the rest, so this is the number that says whether the
        declared alignment convention is safe on this log.
        """
        return _pct(self.cycle_cases - self.incoherent_body_cases, self.cycle_cases)

    def k_loop(self, coverage: float = 95.0, cycling_only: bool = False) -> int:
        """Smallest ``K`` whose bounded unrolling covers ``coverage``% of cases.

        The truncation parameter bounded unrolling needs, read off the
        data rather than guessed.  Note what taking it from the log costs: the
        occurrence alphabet -- and therefore the ``eps`` base measure -- becomes
        data-dependent, so ``K`` is a *declared* choice informed by this number,
        not a number the estimator may quietly adopt.

        ``cycling_only`` selects which quantile the **do-while** convention
        wants.  Under do-while the loop-free cases are a separate component
        with their own weight, so the loop component's truncation is the
        quantile among cases that actually cycle; counting the loop-free cases
        as depth 1 drags ``K`` down toward the majority that never looped.  The
        two can differ a lot -- on a log that is 65% loop-free, 3 against 5,
        and therefore an occurrence alphabet of 9 against 13.
        """
        hist = ({k: n for k, n in self.case_max_multiplicity.items() if k > 1}
                if cycling_only else dict(self.case_max_multiplicity))
        total = sum(hist.values())
        if not total:
            return 1
        target = coverage / 100.0 * total
        seen = 0
        for k in sorted(hist):
            seen += hist[k]
            if seen >= target:
                return max(k, 1)
        return max(self.max_multiplicity, 1)

    def occurrence_alphabet_size(self, k: Optional[int] = None) -> int:
        """Size of the occurrence alphabet at truncation ``k``.

        ``sum_a min(max multiplicity of a, k)`` -- the ``m`` every cost in the
        pipeline is exponential or factorial in, so this is the number that
        says whether unrolling this log is affordable at all.
        """
        if k is None:
            k = self.k_loop()
        return sum(min(v, k) for v in self.activity_max_multiplicity.values())

    def verdict(self) -> str:
        """``"loop-free"``, ``"batching"``, ``"loops"``, or ``"empty"``."""
        if not self.cases:
            return "empty"
        if not self.cycle_cases and not self.immediate_only_cases:
            return "loop-free"
        if not self.cycle_cases:
            return "batching"
        return "loops"


def _case_repetition(activities: Sequence[str]):
    """``(max multiplicity, {repeated activity: count}, immediate?, has cycle?)``.

    The **loop body** is ``set(repeated)`` -- the activities occurring more than
    once. It is *not* the content between consecutive repeats: for a body
    ``{b, c}`` the gap between two ``b``s is ``{c}`` and vice versa, so that
    measure returns ``n`` fragments of size ``n-1`` for a body of size ``n``,
    which is an artifact of the measure rather than the loop.
    """
    counts = Counter(activities)
    repeated = {a: n for a, n in counts.items() if n > 1}
    if not repeated:
        return 1, {}, False, False
    positions: dict = {}
    for i, a in enumerate(activities):
        positions.setdefault(a, []).append(i)
    immediate = has_cycle = False
    for a in repeated:
        for i, j in zip(positions[a], positions[a][1:]):
            if j == i + 1:
                immediate = True          # `a a`: nothing happened in between
            else:
                has_cycle = True
    return max(repeated.values()), repeated, immediate, has_cycle


def loop_composition(path: str, terminators: Sequence[str] = (),
                     count_all_lifecycle: bool = False) -> LoopReport:
    """Measure within-case activity repetition -- does this log carry loops?

    **The lifecycle trap, handled by default.**  In a log carrying
    ``lifecycle:transition``, every activity instance emits at least a
    ``start`` and a ``complete``, so *every* activity repeats and a naive
    screen reports loops everywhere.  Events are therefore counted only at
    their terminating transition (``terminators``, defaulting to
    :data:`DEFAULT_TERMINATORS`); events with no lifecycle attribute are always
    counted, so an instant log is unaffected.  ``count_all_lifecycle=True``
    disables the filter and counts raw events -- which measures instrumentation
    volume, not process structure, and will call almost any lifecycle log
    looped.  The number of events the filter removed is reported either way.
    """
    terms = frozenset(terminators) if terminators else DEFAULT_TERMINATORS
    r = LoopReport()
    for evs in iter_cases(path, ("concept:name", "lifecycle:transition")):
        r.cases += 1
        r.events += len(evs)
        if count_all_lifecycle:
            acts = [a for a, _ in evs if a is not None]
        else:
            acts = [a for a, lt in evs
                    if a is not None and (lt is None or lt in terms)]
            r.events_dropped_lifecycle += len(evs) - len(acts)

        top, repeated, immediate, has_cycle = _case_repetition(acts)
        r.case_max_multiplicity[top] += 1
        r.max_multiplicity = max(r.max_multiplicity, top)
        for a, n in Counter(acts).items():
            if n > r.activity_max_multiplicity.get(a, 0):
                r.activity_max_multiplicity[a] = n
        if not repeated:
            r.loop_free_cases += 1
            continue
        for a in repeated:
            r.repeating_activities[a] += 1
        if has_cycle:
            r.cycle_cases += 1
            r.loop_bodies[tuple(sorted(repeated))] += 1
            if len(set(repeated.values())) > 1:
                r.incoherent_body_cases += 1
        elif immediate:
            r.immediate_only_cases += 1
    return r


def _format_loops(path: str, r: LoopReport) -> str:
    L = [f"=== {path}",
         f"cases {r.cases}   events counted {r.events - r.events_dropped_lifecycle}"
         f"   (dropped as non-terminating lifecycle: {r.events_dropped_lifecycle})",
         "",
         "within-case activity repetition",
         f"  loop-free cases (usable as posets today)  {r.loop_free_cases:>8}"
         f"  ({r.loop_free_pct:5.2f}%)",
         f"  immediate repetition only  `a a`          {r.immediate_only_cases:>8}"
         f"  ({r.immediate_only_pct:5.2f}%)",
         f"  cycles  `a ... a` with work between       {r.cycle_cases:>8}"
         f"  ({r.cycle_pct:5.2f}%)",
         f"  greatest multiplicity of one activity     {r.max_multiplicity:>8}",
         "",
         f"  verdict: {r.verdict()}"]
    if r.verdict() == "loops":
        k_all = r.k_loop(95.0)
        k_cyc = r.k_loop(95.0, cycling_only=True)
        L += ["",
              f"  body coherence (every iteration runs the whole body)"
              f"   {r.body_coherent_pct:5.2f}%",
              f"    incoherent cycling cases                {r.incoherent_body_cases:>8}"
              f"   -- evidence AGAINST a single loop, not a",
              "                                                    "
              "  case to align more cleverly",
              "",
              "  bounded unrolling (bounded unrolling) -- a DECLARED truncation, informed here",
              f"    K at 95% of cases that CYCLE (do-while) {k_cyc:>8}"
              f"   -> occurrence alphabet m = {r.occurrence_alphabet_size(k_cyc)}",
              f"    K at 95% of all cases (0-or-more)       {k_all:>8}"
              f"   -> occurrence alphabet m = {r.occurrence_alphabet_size(k_all)}"]
    if r.repeating_activities:
        L += ["", "  most-repeated activities (cases in which they recur)"]
        for a, n in r.repeating_activities.most_common(8):
            L.append(f"    {n:>6}  {a}  (max {r.activity_max_multiplicity.get(a, 1)} per case)")
    if r.loop_bodies:
        L += ["", "  commonest loop bodies (the activities that recur in a case)"]
        for body, n in r.loop_bodies.most_common(6):
            L.append(f"    {n:>6}  ({', '.join(body)})")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="python -m procposets.logscreen",
        description="Does this event log carry within-case concurrency?",
    )
    ap.add_argument("log", help="path to a .xes or .xes.gz file")
    ap.add_argument("--report", choices=("ties", "intervals", "loops", "all", "both"),
                    default="both")
    ap.add_argument("--count-all-lifecycle", action="store_true",
                    help="count every event in the loop screen instead of only "
                         "terminating transitions; measures instrumentation volume, "
                         "not process structure")
    ap.add_argument("--resource-key", default="org:resource")
    ap.add_argument("--drop-resource", nargs="*", default=(),
                    help="resource ids to treat as non-human in the cross-resource test")
    ap.add_argument("--coarse-stratum", action="store_true",
                    help="score composition on coarse ties when no sub-second stamp exists; "
                         "records which kind of undecidable a log is, never a pass")
    ap.add_argument("--terminators", nargs="*", default=(),
                    help=f"lifecycle transitions ending an instance (default: "
                         f"{' '.join(sorted(DEFAULT_TERMINATORS))})")
    ap.add_argument("--activity-prefix", default=None)
    ap.add_argument("--pairs", action="store_true", help="list the commonest overlapping pairs")
    ap.add_argument("--json", help="also write the measurements to this path")
    a = ap.parse_args(argv)

    out: dict = {"log": a.log}
    if a.report in ("ties", "both", "all"):
        t = tie_composition(a.log, a.resource_key, a.drop_resource, a.coarse_stratum)
        print(_format_ties(a.log, t))
        out["ties"] = {**vars(t), "subsecond_pct": t.subsecond_pct,
                       "singleton_pct": t.singleton_pct,
                       "distinct_activity_pct": t.distinct_activity_pct,
                       "cross_resource_pct": t.cross_resource_pct,
                       "verdict": t.verdict()}
    if a.report in ("intervals", "both", "all"):
        o = interval_overlap(a.log, a.terminators, a.activity_prefix, collect_pairs=a.pairs)
        print(("\n" if a.report == "both" else "") + _format_intervals(a.log, o))
        d = {k: v for k, v in vars(o).items() if k != "overlapping_pairs"}
        out["intervals"] = {**d, "envelope_pct": o.envelope_pct, "active_pct": o.active_pct}
    if a.report in ("loops", "all"):
        lp = loop_composition(a.log, a.terminators, a.count_all_lifecycle)
        print(("\n" if a.report == "all" else "") + _format_loops(a.log, lp))
        d = {k: (dict(v) if isinstance(v, Counter) else v)
             for k, v in vars(lp).items()}
        out["loops"] = {**d, "loop_free_pct": lp.loop_free_pct,
                        "cycle_pct": lp.cycle_pct,
                        "k_loop_95": lp.k_loop(95.0),
                        "k_loop_95_cycling": lp.k_loop(95.0, cycling_only=True),
                        "body_coherent_pct": lp.body_coherent_pct,
                        "occurrence_alphabet_95":
                            lp.occurrence_alphabet_size(lp.k_loop(95.0)),
                        "occurrence_alphabet_95_cycling":
                            lp.occurrence_alphabet_size(
                                lp.k_loop(95.0, cycling_only=True)),
                        "verdict": lp.verdict()}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
