"""Three-way structural comparison of model catalogues as occurrence-net DAGs.

Where ``signature_diff`` compares the
lossy *label-skeleton* (a string projection that erases wiring and so reports
false agreement on, e.g., an N-poset vs a complete bipartite order, §19c),
this module compares the **boundary-rooted labelled DAG** (the label-pomset of
§19d) up to isomorphism -- a genuine poset comparison.

Two models are *cospan-equivalent under label projection* iff their canonical
DAG-class sets are equal. Closing instances are compared as whole DAGs; loops
by the §19e certificate ``(anchor_types, body-DAG)`` so a loop's splice point
is part of its identity. The comparison is exact (WL bucket + VF2), never a
linearization-language equality (which would conflate distinct pomsets that
share a language, §19d).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .class_extraction import ExtractionResult
from .occurrence import EventDag, anchor_types, canonical_key, is_isomorphic, to_event_dag


@dataclass
class DagClass:
    """One canonical structure and which models exhibit it. ``anchor`` is the
    §19e loop anchor-type signature (``()`` for closing instances, whose
    boundary is empty). ``members`` maps a model name to the fragment names in
    that model that realize this structure."""

    rep: EventDag
    anchor: tuple
    bucket: str
    members: dict[str, list[str]] = field(default_factory=dict)

    @property
    def models(self) -> frozenset[str]:
        return frozenset(self.members)


def _classify(
    tagged: list[tuple[str, EventDag, tuple]],
) -> list[DagClass]:
    """Merge ``(model, dag, anchor)`` triples into isomorphism classes,
    matching on anchor first (cheap) then VF2 (exact), bucketed by WL key."""
    by_bucket: dict[str, list[DagClass]] = {}
    classes: list[DagClass] = []
    for model, dag, anchor in tagged:
        bucket = canonical_key(dag)
        match = None
        for cls in by_bucket.get(bucket, []):
            if cls.anchor == anchor and is_isomorphic(cls.rep, dag):
                match = cls
                break
        if match is None:
            match = DagClass(rep=dag, anchor=anchor, bucket=bucket)
            by_bucket.setdefault(bucket, []).append(match)
            classes.append(match)
        match.members.setdefault(model, []).append(dag.name)
    return classes


@dataclass
class DagDiffReport:
    """The three-way structural diff: canonical DAG classes for closing
    instances and for loops, each tagged with which models exhibit it."""

    model_names: tuple[str, ...]
    closing_classes: list[DagClass]
    loop_classes: list[DagClass]

    def _partition(self, classes: list[DagClass]) -> dict[frozenset[str], list[DagClass]]:
        out: dict[frozenset[str], list[DagClass]] = {}
        for cls in classes:
            out.setdefault(cls.models, []).append(cls)
        return out

    def shared_by_all(self, *, loops: bool = False) -> list[DagClass]:
        classes = self.loop_classes if loops else self.closing_classes
        full = frozenset(self.model_names)
        return [c for c in classes if c.models == full]

    def summary(self) -> str:
        lines = []
        full = frozenset(self.model_names)
        for kind, classes in (("closings", self.closing_classes), ("loops", self.loop_classes)):
            part = self._partition(classes)
            lines.append(f"=== {kind}: {len(classes)} distinct structures ===")
            lines.append(f"  shared by all {len(self.model_names)}: {len(part.get(full, []))}")
            for models, cls_list in sorted(part.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
                if models == full:
                    continue
                tag = "+".join(sorted(models))
                lines.append(f"  only {tag}: {len(cls_list)}")
        return "\n".join(lines)


def diff_dags(results: dict[str, ExtractionResult]) -> DagDiffReport:
    """Build the three-way (or n-way) DAG diff over named model catalogues.

    ``results`` maps a model name (e.g. ``"master"``, ``"occn"``, ``"ocpn"``)
    to its :class:`ExtractionResult`. Closings are compared as whole DAGs
    (anchor ``()``); loops by ``(anchor_types, body-DAG)``."""
    closing_tagged: list[tuple[str, EventDag, tuple]] = []
    loop_tagged: list[tuple[str, EventDag, tuple]] = []
    for model, result in results.items():
        for nm in result.closing():
            closing_tagged.append((model, to_event_dag(nm, result.fragments), ()))
        for nm in result.loops():
            loop_tagged.append((model, to_event_dag(nm, result.fragments), anchor_types(nm)))
    return DagDiffReport(
        model_names=tuple(results),
        closing_classes=_classify(closing_tagged),
        loop_classes=_classify(loop_tagged),
    )


def diff_dag_families(results: dict[str, ExtractionResult]) -> DagDiffReport:
    """Family-level (splice-aware) variant of :func:`diff_dags` for the headline
    overlay/gallery figures (§22h/§23b).

    Each closing is collapsed to its loop-free **spine**, so the ``M(m,σ)`` loop
    unrollings fold onto their baseline instead of surfacing as separate
    structures. In the exact diff those unrollings show up as a spurious
    ``occn+ocpn`` combo -- OCCN/OCPN list explicit m=2 closings that the master
    generates by *splicing* its loop, so the master has no exact closing to
    match and the redundant realisation reads as a model difference. Collapsing
    to families removes it: master and OCCN become the same four families, OCPN
    adds its genuine over-generation families. Loops are unchanged (the
    repetition they capture is exactly what is factored out of the closings
    here) and are drawn by the loop gallery instead.

    Each family is one :class:`DagClass` whose ``rep`` is the family's loop-free
    baseline DAG -- or the shortest member when no member is loop-free (e.g.
    OCPN paths that always loop) -- and whose ``members`` records, per model, the
    closings that reduce to that spine. Grouping is by label-spine, but the
    loop-free baseline is unique up to iso for every shared family, so no
    distinct concurrency is conflated."""
    from .signature_diff import label_skeleton, strip_boundary_wrapper

    fam: dict[tuple, dict] = {}
    for model, result in results.items():
        loops = result.loops()
        for nm in result.closing():
            spine = closing_spine(nm, loops, result.fragments)
            dag = to_event_dag(nm, result.fragments)
            loop_free = strip_boundary_wrapper(label_skeleton(nm.body, result.fragments)) == spine
            size = len(dag.graph)
            entry = fam.setdefault(spine, {"members": {}, "rep": None, "size": 0, "loop_free": False})
            entry["members"].setdefault(model, []).append(nm.name)
            # rep = a loop-free baseline if any; among equals, the smallest DAG
            if entry["rep"] is None or (loop_free, -size) > (entry["loop_free"], -entry["size"]):
                entry["rep"], entry["size"], entry["loop_free"] = dag, size, loop_free

    closing_classes = [
        DagClass(rep=e["rep"], anchor=(), bucket="", members=e["members"]) for e in fam.values()
    ]
    loop_tagged = [
        (model, to_event_dag(nm, result.fragments), anchor_types(nm))
        for model, result in results.items()
        for nm in result.loops()
    ]
    return DagDiffReport(
        model_names=tuple(results),
        closing_classes=closing_classes,
        loop_classes=_classify(loop_tagged),
    )


# --- round-2: splice-aware (family-level) closing comparison -----------------
#
# ``diff_dags`` compares *exact* closings, so a model that enumerates a loop
# unrolling explicitly (OCCN/OCPN's ``M(2,σ)``) shows it as "only occn+ocpn":
# the master generates the same pomset by *splicing* its loop, never listing it,
# so there is no master closing to match. The
# splice-aware view factors loops *out* of each closing to its loop-free
# **spine** and compares spines: two closings are in the same ``M(m,σ)`` family
# iff they share a spine, so all unrollings collapse onto the baseline and the
# m-redundancy disappears. (Operates on the label-skeleton: loops match by label
# across models -- the committed-vs-neutral generator distinction is exactly
# what must be forgotten here, §22h.)


def _remove_first_contiguous(skel: tuple, sub: tuple) -> tuple:
    n, m = len(skel), len(sub)
    for i in range(n - m + 1):
        if skel[i : i + m] == sub:
            return skel[:i] + skel[i + m :]
    return skel


def closing_spine(nm, loops, by_name) -> tuple:
    """The loop-free spine of a closing: its label-skeleton with every loop's
    label-skeleton removed (repeatedly, as a contiguous sub-walk) and the
    boundary wrapper stripped. ``M(m,σ)`` for any ``m`` reduces to the same
    spine ``M(1,σ)``."""
    from .signature_diff import label_skeleton, strip_boundary_wrapper

    skel = label_skeleton(nm.body, by_name)
    loop_skels = [label_skeleton(lp.body, by_name) for lp in loops]
    changed = True
    while changed:
        changed = False
        for lsk in loop_skels:
            new = _remove_first_contiguous(skel, lsk)
            if new != skel:
                skel, changed = new, True
    return strip_boundary_wrapper(skel)


@dataclass
class SpineDiffReport:
    """Family-level closing diff: each distinct loop-free spine tagged with the
    models that produce some closing reducing to it."""

    model_names: tuple[str, ...]
    spine_members: dict  # spine (tuple) -> set[str]

    def shared_by_all(self) -> list:
        full = frozenset(self.model_names)
        return [sp for sp, ms in self.spine_members.items() if frozenset(ms) == full]

    def summary(self) -> str:
        full = frozenset(self.model_names)
        part: dict = {}
        for sp, ms in self.spine_members.items():
            part.setdefault(frozenset(ms), []).append(sp)
        lines = [f"=== closing families (loop-free spines): {len(self.spine_members)} distinct ==="]
        lines.append(f"  shared by all {len(self.model_names)}: {len(part.get(full, []))}")
        for models, sps in sorted(part.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
            if models == full:
                continue
            lines.append(f"  only {'+'.join(sorted(models))}: {len(sps)}")
        return "\n".join(lines)


def spine_diff(results: dict[str, ExtractionResult]) -> SpineDiffReport:
    """Round-2 splice-aware comparison: group closings across models by their
    loop-free spine, so ``M(m,σ)`` unrollings collapse onto the baseline."""
    spine_members: dict = {}
    for model, result in results.items():
        loops = result.loops()
        for nm in result.closing():
            sp = closing_spine(nm, loops, result.fragments)
            spine_members.setdefault(sp, set()).add(model)
    return SpineDiffReport(model_names=tuple(results), spine_members=spine_members)
