"""Cross-signature structural diff: compare two notations' discovered
`ExtractionResult`s (from `class_extraction.extract_classes`) for
structural, not merely linguistic, equivalence.

Design: ``CLASS_EXTRACTION.md`` §14. Unlike ``morphism_schema.py`` (which
recognizes sameness *within* one signature, where comparing ``Port``
identity is meaningful), this module compares *across* two signatures from
possibly-different notations/adapters, where ``Port`` naming/granularity
conventions are not compatible at all -- only activity *labels* survive
the comparison, since both adapters read those off the same underlying
OCEL directly.

Three levels, applied in order:

1. label inventory (``valid_generators`` label-set diff);
2. a within-signature dedup pre-pass reusing §12's ``schema_classes`` (one
   canonical representative per shape-class, before any cross-signature
   step touches the catalogue);
3. label-skeleton path diff: expand every fragment's ``Ref``s, collapse to
   the `@`-grouped sequence of activity labels only (drop all ports/types),
   and diff the two catalogues' skeletons as **sets**, not multisets --
   multiplicity is known to be search-order-dependent (CLASS_EXTRACTION.md
   §13's M1 vs M10), so a multiset diff would manufacture fake
   count-mismatches with no structural meaning.

Classifying *why* a residual difference exists (discovery-threshold
artifact, notation-granularity residue, adapter-convention difference,
degenerate miner noise, or genuine structural disagreement -- §14's B1-B5)
is left as a human-triage step over this module's output, not automated --
**with one named exception**: ``strip_boundary_wrapper`` (B3, the
``START_<ot>``/``END_<ot>`` boundary-generator convention added in §13) is
baked in as an explicit, documented normalization applied to closing-
instance paths before the diff, because it is mechanically justified (those
labels are provably an artifact of *how* one adapter represents "no
predecessor", confirmed by checking which adapter needs them and why --
not a guess) and because leaving it un-normalized was found to make every
single closing-instance path look "different" even when the two notations
fully agreed once the wrapper was discounted (verified by hand on the ED
OCEL: 3 of 5 distinct stripped OCCN paths then matched an OCPN path
exactly). Pass ``strip_boundary=False`` to ``diff_signatures`` to see the
raw, un-normalized diff instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .class_extraction import ExtractionResult, NamedMorphism, Ref, _compress
from .morphism_schema import _expand, shape_key

Skeleton = tuple


def label_skeleton(body: tuple, by_name: dict[str, NamedMorphism]) -> Skeleton:
    """Expand every ``Ref`` in ``body``, then collapse to just the
    `@`-grouped sequence of activity labels -- drop all ports/types."""
    out = []
    for step in _expand(body, by_name):
        if isinstance(step, frozenset):
            out.append(tuple(sorted(g.label for g in step)))
        else:
            out.append(step.label)
    return tuple(out)


def deduped_skeletons(
    fragments_subset: dict[str, NamedMorphism], by_name: dict[str, NamedMorphism]
) -> set[Skeleton]:
    """§12's within-signature dedup, applied to one subset of a signature's
    fragments (e.g. just its closing instances, or just its loops): group by
    `shape_key`, keep one representative per shape-class, then collapse
    each representative to its label skeleton. Returns a set, so multiple
    shape-classes that happen to share a label skeleton (e.g. the same
    activity sequence discovered via different port-context variants)
    collapse to one entry, as intended.
    """
    groups: dict[tuple, list[NamedMorphism]] = {}
    for nm in fragments_subset.values():
        key = shape_key(nm.body, nm.boundary, by_name)
        groups.setdefault(key, []).append(nm)
    skeletons: set[Skeleton] = set()
    for members in groups.values():
        rep = min(members, key=lambda nm: nm.name)
        skeletons.add(label_skeleton(rep.body, by_name))
    return skeletons


def _collapse_repeats(skeleton: Skeleton) -> list[tuple]:
    """Run-length-encode consecutive identical steps: ``(L1, L1, x)`` ->
    ``[(L1, 2), (x, 1)]``. The most common decomposed-body shape this fires
    on is a loop ``Ref`` repeated -- e.g. ``L1;L1`` collapses to one
    ``L1^2`` entry, an explicit `m`-summary for that path: an `M(m,σ)`-style
    rendering, just literal-repeat-only (no rotation-awareness, same scope
    limit as :func:`decompose_loops`)."""
    out: list[tuple] = []
    for step in skeleton:
        if out and out[-1][0] == step:
            out[-1] = (step, out[-1][1] + 1)
        else:
            out.append((step, 1))
    return out


def _step_str(step) -> str:
    if isinstance(step, Ref):
        return step.name
    if isinstance(step, tuple):
        return "(" + " x ".join(step) + ")"
    return step


def render_skeleton(skeleton: Skeleton) -> str:
    """Render a ``Skeleton`` (or a decomposed body containing ``Ref``s, see
    :func:`decompose_loops`/:func:`decompose_paths`) in ``a;b;(c x d);e``
    form: ``;`` between sequential steps, ``x`` between `@`-grouped
    (simultaneous) generators within one step, a ``Ref``'s name printed
    as-is. A run of `n` consecutive identical steps renders as ``X^n``
    (see :func:`_collapse_repeats`) -- the most direct `M(m,σ)`-style
    payoff, since this is exactly what a loop traversed `m` times looks
    like once :func:`decompose_paths` has folded it to a `Ref`. Otherwise
    purely a notation simplification for reading a diff by eye -- no
    structural meaning is added or lost versus the raw tuple."""
    parts = []
    for step, n in _collapse_repeats(skeleton):
        s = _step_str(step)
        parts.append(f"{s}^{n}" if n > 1 else s)
    return ";".join(parts)


def _step_labels(step) -> tuple[str, ...]:
    return step if isinstance(step, tuple) else (step,)


def label_alphabet(*skeleton_sets) -> dict[str, str]:
    """Assign each distinct *activity label* appearing anywhere in
    ``skeleton_sets`` a short character code (``a``, ``b``, ..., ``z``,
    ``aa``, ``ab``, ...), ordered alphabetically by label so codes are
    stable across re-runs on the same data. A second, finer-grained
    presentation aid than :func:`label_legend` -- that one compresses whole
    composites to one ID; this one compresses the activity names *inside* a
    composite's rendered form, so the legend lines themselves get shorter."""
    labels: set[str] = set()
    for skels in skeleton_sets:
        for skel in skels:
            for step in skel:
                labels.update(_step_labels(step))
    ordered = sorted(labels)
    alpha = "abcdefghijklmnopqrstuvwxyz"

    def code_for(i: int) -> str:
        if i < 26:
            return alpha[i]
        i -= 26
        return alpha[i // 26] + alpha[i % 26]

    return {lbl: code_for(i) for i, lbl in enumerate(ordered)}


def _step_str_compact(step, alphabet: dict[str, str]) -> str:
    if isinstance(step, Ref):
        return step.name
    if isinstance(step, tuple):
        return "(" + " x ".join(alphabet[lbl] for lbl in step) + ")"
    return alphabet[step]


def render_skeleton_compact(skeleton: Skeleton, alphabet: dict[str, str]) -> str:
    """Like :func:`render_skeleton`, but substituting each activity label
    with its short code from :func:`label_alphabet`. A ``Ref`` step (from a
    decomposed body) is left as its name -- it isn't an activity label.
    Also applies :func:`_collapse_repeats`' ``X^n`` notation."""
    parts = []
    for step, n in _collapse_repeats(skeleton):
        s = _step_str_compact(step, alphabet)
        parts.append(f"{s}^{n}" if n > 1 else s)
    return ";".join(parts)


def label_legend(*skeleton_sets: "set[Skeleton] | frozenset[Skeleton]", prefix: str) -> dict[Skeleton, str]:
    """Assign each distinct skeleton across all ``skeleton_sets`` a short,
    stable ID (``{prefix}1``, ``{prefix}2``, ...), ordered by its rendered
    form so the same skeleton always gets the same ID across re-runs on the
    same data. A pure presentation aid -- look up ``render_skeleton`` to see
    what an ID actually stands for."""
    all_skeletons = sorted(set().union(*skeleton_sets), key=render_skeleton)
    return {s: f"{prefix}{i + 1}" for i, s in enumerate(all_skeletons)}


def _id_sort_key(sid: str) -> tuple[str, int]:
    """Natural sort for legend IDs: 'P2' before 'P10', not lexicographic."""
    i = 0
    while i < len(sid) and not sid[i].isdigit():
        i += 1
    return (sid[:i], int(sid[i:]) if sid[i:] else 0)


class _NameHolder:
    """Minimal stand-in with a ``.name`` attribute -- the only thing
    `class_extraction._compress` reads off a dictionary value. Lets
    `decompose_loops`/`decompose_paths` reuse `_compress` unmodified at the
    label-skeleton level."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def decompose_loops(loop_legend: dict[Skeleton, str]) -> tuple[dict[str, tuple], dict[Skeleton, _NameHolder]]:
    """Try to express each loop in ``loop_legend`` as a concatenation of
    *other, shorter* loops already in the same legend, via
    `class_extraction._compress`'s greedy longest-match algorithm --
    applied here at the label-skeleton level (ports/types already dropped),
    not the `Generator`-identity level `morphism_schema`/`class_extraction`
    use. CLASS_EXTRACTION.md §6 designed this folding mechanism for exactly
    this purpose, but §9c/§11 found it never fires on a real signature at
    the identity level (port-typed variants of "the same" activity are
    literally different generators); checked here and confirmed it *does*
    fire once that distinction is dropped -- e.g. on the real ED OCEL, 9 of
    10 post-B4 loops decompose into a 1-2-step reference chain built from a
    single true primitive.

    Returns ``(bodies, fragments_by_body)``: ``bodies`` maps each loop ID to
    its decomposed body (a tuple possibly containing ``Ref``s to *other*
    loop IDs); ``fragments_by_body`` is the same growing dictionary, handed
    back so :func:`decompose_paths` can let closing-instance paths
    reference these same loops too.
    """
    ordered = sorted(loop_legend.items(), key=lambda kv: len(kv[0]))
    fragments_by_body: dict[Skeleton, _NameHolder] = {}
    bodies: dict[str, tuple] = {}
    for skel, sid in ordered:
        bodies[sid] = _compress(skel, fragments_by_body)
        fragments_by_body[skel] = _NameHolder(sid)
    return bodies, fragments_by_body


def decompose_paths(
    path_legend: dict[Skeleton, str], loop_fragments_by_body: dict[Skeleton, _NameHolder]
) -> dict[str, tuple]:
    """Express each closing-instance path as itself, with any contiguous
    sub-sequence that exactly matches a known loop (from
    :func:`decompose_loops`'s ``loop_fragments_by_body``) replaced by a
    ``Ref`` to that loop's ID -- e.g. a path that traverses the same loop
    twice renders as ``...;L1;L1;...`` instead of repeating the loop's full
    body twice inline."""
    return {sid: _compress(skel, loop_fragments_by_body) for skel, sid in path_legend.items()}


def _is_boundary_label(label: str) -> bool:
    return label.startswith("START_") or label.startswith("END_")


def strip_boundary_wrapper(skeleton: Skeleton) -> Skeleton:
    """B3 adapter-convention normalization (``CLASS_EXTRACTION.md`` §14):
    strip a leading/trailing run of steps that are *entirely*
    ``START_<ot>``/``END_<ot>`` labels.

    These are real generators in an adapter that needs them to represent
    "no predecessor"/"no successor" as a synthesizable boundary (OCCN, per
    §13's fix), but have no counterpart in an adapter whose own zero-left/
    zero-right generator already plays that role directly (OCPN's literal
    empty Petri-net source/sink place). Comparing across notations should
    not count a path as "different" purely because one adapter's
    *encoding* of "the process starts here" differs from the other's --
    that's a convention difference, not a behavioural one.
    """
    def is_boundary_step(step) -> bool:
        labels = step if isinstance(step, tuple) else (step,)
        return all(_is_boundary_label(lbl) for lbl in labels)

    steps = list(skeleton)
    while steps and is_boundary_step(steps[0]):
        steps.pop(0)
    while steps and is_boundary_step(steps[-1]):
        steps.pop()
    return tuple(steps)


@dataclass(frozen=True)
class DiffReport:
    labels_only_a: frozenset[str]
    labels_only_b: frozenset[str]
    labels_in_both: frozenset[str]
    paths_only_a: frozenset[Skeleton]
    paths_only_b: frozenset[Skeleton]
    paths_in_both: frozenset[Skeleton]
    loops_only_a: frozenset[Skeleton]
    loops_only_b: frozenset[Skeleton]
    loops_in_both: frozenset[Skeleton]
    boundary_stripped: bool

    def pretty(self, *, labelled: bool = True, compact: bool = True, decompose: bool = True) -> str:
        """``labelled=True`` (default): assign each distinct path/loop a
        short ID (``P1``, ``P2``, ... / ``L1``, ``L2``, ...) via
        :func:`label_legend`, print a legend mapping ID -> ``a;b;(c x d);e``
        form (:func:`render_skeleton`), then print the diff sections using
        IDs only -- a pure presentation simplification, not a structural
        change, meant to make the diff scannable by eye. ``labelled=False``
        prints the raw skeleton tuples instead (the original format).

        ``compact=True`` (default, only matters when ``labelled=True``):
        additionally substitutes each *activity label* inside the P/L
        legend lines with a short character code from
        :func:`label_alphabet`, printed as its own "activity legend" block
        first -- compresses the legend definitions themselves, on top of
        the ID compression of the composites. ``compact=False`` keeps full
        activity names in the P/L legend lines.

        ``decompose=True`` (default, only matters when ``labelled=True``):
        additionally expresses each loop/path as a concatenation of
        *already-shown shorter loops* where an exact match exists
        (:func:`decompose_loops`/:func:`decompose_paths`), referencing them
        by ID (e.g. ``L5 = L4;L1``) instead of repeating their full body --
        the third, structure-revealing compression layer, on top of ID
        compression and activity-code compression. ``decompose=False``
        always shows a legend entry's full skeleton.
        """
        def section(title: str, only_a, only_b, both, fmt, sort_key=str) -> str:
            lines = [f"-- {title} --"]
            lines.append(f"  only in A ({len(only_a)}):")
            lines += [f"    {fmt(x)}" for x in sorted(only_a, key=lambda x: sort_key(fmt(x)))]
            lines.append(f"  only in B ({len(only_b)}):")
            lines += [f"    {fmt(x)}" for x in sorted(only_b, key=lambda x: sort_key(fmt(x)))]
            lines.append(f"  in both ({len(both)}):")
            lines += [f"    {fmt(x)}" for x in sorted(both, key=lambda x: sort_key(fmt(x)))]
            return "\n".join(lines)

        note = (
            "(closing-instance paths: B3 START_<ot>/END_<ot> boundary wrapper stripped before diffing)"
            if self.boundary_stripped
            else "(closing-instance paths: raw, boundary wrapper NOT stripped)"
        )

        if not labelled:
            identity = lambda x: x
            return "\n\n".join(
                [
                    section("labels", self.labels_only_a, self.labels_only_b, self.labels_in_both, identity),
                    note,
                    section(
                        "closing-instance paths", self.paths_only_a, self.paths_only_b, self.paths_in_both, identity
                    ),
                    section("loops", self.loops_only_a, self.loops_only_b, self.loops_in_both, identity),
                ]
            )

        path_legend = label_legend(self.paths_only_a, self.paths_only_b, self.paths_in_both, prefix="P")
        loop_legend = label_legend(self.loops_only_a, self.loops_only_b, self.loops_in_both, prefix="L")

        if decompose:
            loop_bodies, loop_frags = decompose_loops(loop_legend)
            path_bodies = decompose_paths(path_legend, loop_frags)
        else:
            loop_bodies = {sid: skel for skel, sid in loop_legend.items()}
            path_bodies = {sid: skel for skel, sid in path_legend.items()}

        alphabet_block = ""
        render = render_skeleton
        if compact:
            alphabet = label_alphabet(path_legend.keys(), loop_legend.keys())
            render = lambda body: render_skeleton_compact(body, alphabet)
            lines = ["-- activity legend --"]
            for lbl, code in sorted(alphabet.items(), key=lambda kv: kv[1]):
                lines.append(f"  {code} = {lbl}")
            alphabet_block = "\n".join(lines)

        def legend_block(title: str, bodies: dict[str, tuple]) -> str:
            lines = [f"-- {title} legend --"]
            for sid in sorted(bodies, key=_id_sort_key):
                lines.append(f"  {sid} = {render(bodies[sid])}")
            return "\n".join(lines)

        sections = [
            section(
                "labels",
                self.labels_only_a,
                self.labels_only_b,
                self.labels_in_both,
                lambda x: x,
            ),
            note,
        ]
        if compact:
            sections.append(alphabet_block)
        sections.append(legend_block("closing-instance path", path_bodies))
        return "\n\n".join(
            [
                *sections,
                section(
                    "closing-instance paths",
                    self.paths_only_a,
                    self.paths_only_b,
                    self.paths_in_both,
                    lambda x: path_legend[x],
                    sort_key=_id_sort_key,
                ),
                legend_block("loop", loop_bodies),
                section(
                    "loops",
                    self.loops_only_a,
                    self.loops_only_b,
                    self.loops_in_both,
                    lambda x: loop_legend[x],
                    sort_key=_id_sort_key,
                ),
            ]
        )


def catalogue_pretty(
    result: ExtractionResult, *, compact: bool = True, decompose: bool = True, strip_boundary: bool = True
) -> str:
    """Render *one* signature's own M/L catalogue, standalone, in the same
    simplified notation `DiffReport.pretty` uses for a cross-signature diff
    -- no A/B partition, just "what did this signature's `extract_classes`
    find," in `a;b;(c x d);e` / `L1^m` form. Useful both for inspecting one
    notation's discovered catalogue on its own terms, and as the common
    rendering for comparing a discovered catalogue against the hand-
    authored master signature's own catalogue by eye (CLASS_EXTRACTION.md
    §14: "compare back to the master cospan M structure" is a separate axis
    from comparing two discovered signatures against each other -- this
    function and :func:`diff_signatures` both apply to that axis too, the
    master signature's `ExtractionResult` is not privileged in any way).
    """
    closing = {nm.name: nm for nm in result.closing()}
    loops = {nm.name: nm for nm in result.loops()}
    paths = deduped_skeletons(closing, result.fragments)
    if strip_boundary:
        paths = {strip_boundary_wrapper(s) for s in paths}
    loopsk = deduped_skeletons(loops, result.fragments)

    path_legend = label_legend(paths, prefix="P")
    loop_legend = label_legend(loopsk, prefix="L")

    if decompose:
        loop_bodies, loop_frags = decompose_loops(loop_legend)
        path_bodies = decompose_paths(path_legend, loop_frags)
    else:
        loop_bodies = {sid: skel for skel, sid in loop_legend.items()}
        path_bodies = {sid: skel for skel, sid in path_legend.items()}

    render = render_skeleton
    lines: list[str] = []
    if compact:
        alphabet = label_alphabet(path_legend.keys(), loop_legend.keys())
        render = lambda body: render_skeleton_compact(body, alphabet)
        lines.append("-- activity legend --")
        for lbl, code in sorted(alphabet.items(), key=lambda kv: kv[1]):
            lines.append(f"  {code} = {lbl}")
        lines.append("")

    lines.append(f"-- closing-instance paths ({len(path_bodies)}) --")
    for sid in sorted(path_bodies, key=_id_sort_key):
        lines.append(f"  {sid} = {render(path_bodies[sid])}")
    lines.append(f"-- loops ({len(loop_bodies)}) --")
    for sid in sorted(loop_bodies, key=_id_sort_key):
        lines.append(f"  {sid} = {render(loop_bodies[sid])}")
    return "\n".join(lines)


def diff_signatures(
    result_a: ExtractionResult, result_b: ExtractionResult, *, strip_boundary: bool = True
) -> DiffReport:
    """The full three-level diff between two ``extract_classes`` results,
    presumed to come from two different notations'/adapters' signatures
    mined from the same underlying log.

    ``strip_boundary`` (default ``True``) applies :func:`strip_boundary_wrapper`
    (B3) to closing-instance paths before diffing -- pass ``False`` to see
    the raw, un-normalized comparison instead. Loops are never stripped:
    they are interior structures and are not expected to touch the
    boundary wrapper in the same way.
    """
    labels_a = frozenset(g.label for g in result_a.valid_generators)
    labels_b = frozenset(g.label for g in result_b.valid_generators)

    closing_a = {nm.name: nm for nm in result_a.closing()}
    closing_b = {nm.name: nm for nm in result_b.closing()}
    loops_a = {nm.name: nm for nm in result_a.loops()}
    loops_b = {nm.name: nm for nm in result_b.loops()}

    paths_a = deduped_skeletons(closing_a, result_a.fragments)
    paths_b = deduped_skeletons(closing_b, result_b.fragments)
    if strip_boundary:
        paths_a = {strip_boundary_wrapper(s) for s in paths_a}
        paths_b = {strip_boundary_wrapper(s) for s in paths_b}
    loopsk_a = deduped_skeletons(loops_a, result_a.fragments)
    loopsk_b = deduped_skeletons(loops_b, result_b.fragments)

    return DiffReport(
        labels_only_a=frozenset(labels_a - labels_b),
        labels_only_b=frozenset(labels_b - labels_a),
        labels_in_both=frozenset(labels_a & labels_b),
        paths_only_a=frozenset(paths_a - paths_b),
        paths_only_b=frozenset(paths_b - paths_a),
        paths_in_both=frozenset(paths_a & paths_b),
        loops_only_a=frozenset(loopsk_a - loopsk_b),
        loops_only_b=frozenset(loopsk_b - loopsk_a),
        loops_in_both=frozenset(loopsk_a & loopsk_b),
        boundary_stripped=strip_boundary,
    )
