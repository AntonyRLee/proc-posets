"""Post-search schema recognition: group ``NamedMorphism`` fragments
(produced by ``class_extraction.extract_classes``) into ``M(m,sigma)``-style
families by an up-to-port-renaming equivalence.

This module runs strictly *after* the
live search has finished, never feeding back into ``_enabled``/``_fire`` --
the search stays exact (sound, no spurious composites); only the
*recognition* that two already-discovered, already-sound bodies are
instances of one schema is approximate, and the approximation is scoped
precisely: it forgets which external neighbour a boundary wire is attached
to (the literal ``src``/``tgt`` activity name), and nothing else. Internal
wiring (which produced port is consumed by which later step) is
reconstructed by replaying the same multiset bookkeeping the live search
used, and is preserved exactly -- two bodies with the same per-step port
*type* arity but different internal connectivity are NOT considered the
same schema.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .class_extraction import FrontierKey, NamedMorphism, Ref, _sort_key, _to_counter
from .signature import Generator, Port

BOUNDARY: tuple = (0,)


def _step_tag(frame_idx: int, slot: int) -> tuple:
    return (1, frame_idx, slot)


def _typ_key(typ: str | None) -> tuple:
    return (typ is None, typ)


def _expand(body: tuple, by_name: dict[str, NamedMorphism]) -> tuple:
    """Recursively inline every ``Ref`` so wiring can be reconstructed down
    to atomic generators, even through the §6 loop hierarchy."""
    out = []
    for step in body:
        if isinstance(step, Ref):
            out.extend(_expand(by_name[step.name].body, by_name))
        else:
            out.append(step)
    return tuple(out)


def _frames(expanded: tuple) -> list[list[Generator]]:
    """One frame per recorded step: a singleton for an atomic generator, or
    the canonically-sorted members of an `@`-group (canonical order makes
    "same frame" comparable across two different bodies positionally)."""
    frames = []
    for step in expanded:
        if isinstance(step, frozenset):
            frames.append(sorted(step, key=_sort_key))
        else:
            frames.append([step])
    return frames


def shape_key(body: tuple, boundary: FrontierKey, by_name: dict[str, NamedMorphism]):
    """Canonical, renaming-invariant description of ``body``, a fragment's
    recorded steps starting from its domain ``boundary``.

    Replays the live search's own ``_enabled``/``_fire`` bookkeeping to
    reconstruct internal wiring, tagging each consumed port occurrence with
    its provenance: ``BOUNDARY`` (an external attachment, free to vary) or
    the ``(frame, slot)`` of the step that produced it internally (fixed,
    part of the diagram's actual shape). Two bodies hash/compare equal here
    iff they are the same diagram up to renaming which external neighbour
    each boundary wire touches.
    """
    frames = _frames(_expand(body, by_name))

    boundary_counter = _to_counter(boundary)
    boundary_shape = tuple(
        sorted(
            (_typ_key(typ), n)
            for typ, n in Counter(p.typ for p, c in boundary_counter.items() for _ in range(c)).items()
        )
    )

    pool: dict[Port, list[tuple]] = {}
    for p, n in boundary_counter.items():
        pool[p] = [BOUNDARY] * n

    frame_descs = []
    for f_idx, frame in enumerate(frames):
        left_wires_per_gen: list[list[tuple]] = []
        for g in frame:
            wires = []
            for p, n in Counter(g.left).items():
                for _ in range(n):
                    tag = pool[p].pop(0)
                    wires.append((_typ_key(p.typ), tag))
            left_wires_per_gen.append(sorted(wires))

        right_types_per_gen: list[tuple] = []
        for slot, g in enumerate(frame):
            for p, n in Counter(g.right).items():
                pool.setdefault(p, [])
                for _ in range(n):
                    pool[p].append(_step_tag(f_idx, slot))
            right_types_per_gen.append(
                tuple(sorted((_typ_key(typ), n) for typ, n in Counter(p.typ for p in g.right).items()))
            )

        frame_descs.append(
            tuple(
                (g.label, tuple(left_wires_per_gen[i]), right_types_per_gen[i])
                for i, g in enumerate(frame)
            )
        )

    return (boundary_shape, tuple(frame_descs))


def schema_classes(
    fragments: dict[str, NamedMorphism],
) -> dict[tuple, list[NamedMorphism]]:
    """Group every fragment by :func:`shape_key`. Most classes will be
    singletons -- a fragment with no sibling elsewhere in the catalogue is
    the expected, honest outcome."""
    groups: dict[tuple, list[NamedMorphism]] = {}
    for nm in fragments.values():
        key = shape_key(nm.body, nm.boundary, fragments)
        groups.setdefault(key, []).append(nm)
    return groups


@dataclass(frozen=True)
class SchemaClass:
    """A recognized `M(m,σ)`-style family: ≥2 fragments sharing one shape,
    differing only in which external neighbour their boundary wires touch."""

    name: str
    members: tuple[NamedMorphism, ...]

    def __str__(self) -> str:
        return f"{self.name}: " + ", ".join(nm.name for nm in self.members)


def find_schema_classes(fragments: dict[str, NamedMorphism]) -> list[SchemaClass]:
    """The actually-recognized families: schema classes with ≥2 members.

    Naming/synthesizing the `σ`/`m` parameters that vary across a class's
    members is explicitly out of scope here (§12 scope note) -- this only
    answers "which fragments are the same schema."
    """
    groups = schema_classes(fragments)
    out = []
    for i, members in enumerate(
        sorted((m for m in groups.values() if len(m) >= 2), key=lambda ms: ms[0].name), start=1
    ):
        out.append(SchemaClass(f"S{i}", tuple(sorted(members, key=lambda nm: nm.name))))
    return out
