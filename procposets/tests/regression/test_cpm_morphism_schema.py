"""Tests for the post-search schema-recognition layer (``CLASS_EXTRACTION.md``
§12, ``cpm/cospan/morphism_schema.py``).

Core claim under test: two fragment bodies are the "same schema" iff they
agree on everything except *which external neighbour* a boundary wire is
attached to. In particular this is strictly more than matching per-step
port-*type* arity: two bodies can have identical arity per step while
differing in internal connectivity (which produced port is consumed by
which later step), and those must NOT be merged -- that's the refinement
this module adds over the naive "same label + same type counts" idea.
"""

from __future__ import annotations

from procposets.cospan.class_extraction import NamedMorphism, Ref
from procposets.cospan.morphism_schema import find_schema_classes, shape_key
from procposets.cospan.signature import Generator, Port


def _frag(name: str, boundary, body) -> NamedMorphism:
    return NamedMorphism(name, frozenset(boundary), body)


# --- renaming equivalence: same schema, different external neighbour ------


def test_same_schema_under_boundary_renaming():
    x_from_ctx1 = Generator("x", frozenset({Port("ctx1", "t", "x")}), frozenset({Port("x", "t", "y")}))
    y_to_ctx1 = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("ctx1", "t", "x")}))
    boundary1 = {(Port("ctx1", "t", "x"), 1)}
    body1 = (x_from_ctx1, y_to_ctx1)

    x_from_ctx2 = Generator("x", frozenset({Port("ctx2", "t", "x")}), frozenset({Port("x", "t", "y")}))
    y_to_ctx2 = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("ctx2", "t", "x")}))
    boundary2 = {(Port("ctx2", "t", "x"), 1)}
    body2 = (x_from_ctx2, y_to_ctx2)

    by_name = {}
    assert shape_key(body1, frozenset(boundary1), by_name) == shape_key(body2, frozenset(boundary2), by_name)


def test_different_neighbour_type_is_not_the_same_schema():
    x = Generator("x", frozenset({Port("ctx1", "t", "x")}), frozenset({Port("x", "t", "y")}))
    y = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("ctx1", "t", "x")}))
    boundary = {(Port("ctx1", "t", "x"), 1)}
    body = (x, y)

    x2 = Generator("x", frozenset({Port("ctx1", "u", "x")}), frozenset({Port("x", "t", "y")}))
    y2 = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("ctx1", "t", "x")}))
    boundary2 = {(Port("ctx1", "u", "x"), 1)}
    body2 = (x2, y2)

    by_name = {}
    assert shape_key(body, frozenset(boundary), by_name) != shape_key(body2, frozenset(boundary2), by_name)


# --- the refinement: same per-step type arity, different internal wiring --


def test_same_arity_but_different_internal_wiring_is_not_the_same_schema():
    boundary = {(Port("c1", "t", "x"), 1), (Port("c2", "t", "y"), 1)}

    # Body C: step2 consumes step1's internal output (an internal wire).
    step1_c = Generator("x", frozenset({Port("c1", "t", "x")}), frozenset({Port("x", "t", "y")}))
    step2_c = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("y", "t", "z")}))
    body_c = (step1_c, step2_c)

    # Body D: step2 consumes a second boundary port directly instead --
    # same per-step arity (1 left(t)/1 right(t) on each of "x" and "y"),
    # same boundary type-multiset (two type-t ports), but different wiring.
    step1_d = Generator("x", frozenset({Port("c1", "t", "x")}), frozenset({Port("x", "t", "y")}))
    step2_d = Generator("y", frozenset({Port("c2", "t", "y")}), frozenset({Port("y", "t", "z")}))
    body_d = (step1_d, step2_d)

    by_name = {}
    key_c = shape_key(body_c, frozenset(boundary), by_name)
    key_d = shape_key(body_d, frozenset(boundary), by_name)
    assert key_c != key_d, "different internal wiring must not be folded into one schema"


# --- identity case: literally the same body trivially matches itself ------


def test_identical_body_matches_itself():
    x = Generator("x", frozenset({Port("c", "t", "x")}), frozenset({Port("x", "t", "y")}))
    boundary = {(Port("c", "t", "x"), 1)}
    by_name = {}
    assert shape_key((x,), frozenset(boundary), by_name) == shape_key((x,), frozenset(boundary), by_name)


# --- Ref expansion: wiring is reconstructed through the loop hierarchy ----


def test_ref_expansion_matches_manually_inlined_equivalent():
    inner = Generator("inner", frozenset({Port("c", "t", "x")}), frozenset({Port("x", "t", "c")}))
    inner_frag = _frag("L1", {(Port("c", "t", "x"), 1)}, (inner,))

    outer_tail = Generator("z", frozenset({Port("x", "t", "c")}), frozenset({Port("c", "t", "x")}))
    body_with_ref = (Ref("L1"), outer_tail)
    body_inlined = (inner, outer_tail)

    by_name = {"L1": inner_frag}
    boundary = {(Port("c", "t", "x"), 1)}
    assert shape_key(body_with_ref, frozenset(boundary), by_name) == shape_key(
        body_inlined, frozenset(boundary), by_name
    )


# --- grouping API -----------------------------------------------------------


def test_schema_classes_groups_renamed_siblings_together():
    x1 = Generator("x", frozenset({Port("ctx1", "t", "x")}), frozenset({Port("x", "t", "y")}))
    y1 = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("ctx1", "t", "x")}))
    x2 = Generator("x", frozenset({Port("ctx2", "t", "x")}), frozenset({Port("x", "t", "y")}))
    y2 = Generator("y", frozenset({Port("x", "t", "y")}), frozenset({Port("ctx2", "t", "x")}))

    frag1 = _frag("L1", {(Port("ctx1", "t", "x"), 1)}, (x1, y1))
    frag2 = _frag("L2", {(Port("ctx2", "t", "x"), 1)}, (x2, y2))
    fragments = {"L1": frag1, "L2": frag2}

    classes = find_schema_classes(fragments)
    assert len(classes) == 1
    assert {nm.name for nm in classes[0].members} == {"L1", "L2"}


