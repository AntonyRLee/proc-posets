"""Hypothesis classes -- the estimator's declared search space (README
choice C): the abstract :class:`PosetClass` and the ``general`` / ``sp``
implementations, plus the ``get_poset_class`` resolver.

Split out of :mod:`procposets.rel` for size; ``rel`` re-exports every name
here.  numpy-free.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List, Tuple

from .rel import (
    Rel,
    count_linear_extensions,
    enumerate_posets,
    sample_linear_extension,
)
from .rel_sp import decompose, enumerate_sp, extension_count, sample_extension_tree


# ---------------------------------------------------------------------------
# Hypothesis classes: the declared search space of the estimator (choice C)
# ---------------------------------------------------------------------------

class PosetClass:
    """Abstract base class for a hypothesis class of partial orders (README choice C).

    A class declares two capability flags the oracle dispatches on -- these
    document exactly what the meet-closure reduction theorem needs, instead
    of encoding it as pointer identity (DESIGN_REVIEW W18):

    * ``contains_all_posets`` -- every partial order on the alphabet is in
      the class (so a sup over the class is a sup over all posets);
    * ``closed_under_meet``   -- meets of class members stay in the class
      (so the closure of the observed chains is inside the class).

    The exact meet-closure oracle regime requires BOTH (plus the uniform
    kernel, untimed); a class missing either falls to the heuristic regime.
    Required methods: ``contains``, ``extension_count``, ``enumerate``,
    ``sample_extension``; required attributes: ``name`` and the two flags.
    """

    name: str
    contains_all_posets: bool = False
    closed_under_meet: bool = False

    def contains(self, elements: FrozenSet[str], rel: Rel) -> bool:
        raise NotImplementedError

    def extension_count(self, elements: FrozenSet[str], rel: Rel) -> int:
        raise NotImplementedError

    def enumerate(self, elements: Iterable[str]) -> List[Rel]:
        raise NotImplementedError

    def sample_extension(self, elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
        raise NotImplementedError


class GeneralPosets(PosetClass):
    """All partial orders on the alphabet -- the default hypothesis class.

    Closed under meet (so the oracle's lattice moves never leave the class),
    exhaustively enumerable to m = 6, with exact e(P) by the ideal DP.
    """

    name = "general"
    contains_all_posets = True
    closed_under_meet = True

    def contains(self, elements: FrozenSet[str], rel: Rel) -> bool:
        return True  # rel is maintained transitively closed by construction

    def extension_count(self, elements: FrozenSet[str], rel: Rel) -> int:
        return count_linear_extensions(elements, rel)

    def enumerate(self, elements: Iterable[str]) -> List[Rel]:
        return enumerate_posets(elements)

    def sample_extension(self, elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
        return sample_linear_extension(elements, rel, rng)


class SPPosets(PosetClass):
    """Series-parallel orders only -- the original, smaller class.

    Kept for (i) speed at m = 6 (fewer atoms, linear-time e(P)) and (ii) as
    the declared class when the domain genuinely excludes N-shaped
    dependencies.  Meets can leave the class; the oracle skips those points.
    """

    name = "sp"
    contains_all_posets = False
    closed_under_meet = False

    def contains(self, elements: FrozenSet[str], rel: Rel) -> bool:
        return decompose(elements, rel) is not None

    def extension_count(self, elements: FrozenSet[str], rel: Rel) -> int:
        tree = decompose(elements, rel)
        if tree is None:
            raise ValueError("relation set is not series-parallel")
        return extension_count(tree)

    def enumerate(self, elements: Iterable[str]) -> List[Rel]:
        return enumerate_sp(elements)

    def sample_extension(self, elements: FrozenSet[str], rel: Rel, rng) -> Tuple[str, ...]:
        tree = decompose(elements, rel)
        if tree is None:
            raise ValueError("relation set is not series-parallel")
        return sample_extension_tree(tree, rng)


GENERAL = GeneralPosets()
SP = SPPosets()


def get_poset_class(spec):
    """Resolve a class spec: "general" | "sp" | a PosetClass instance.

    Arbitrary objects are rejected with the missing protocol members named
    (DESIGN_REVIEW W21): a class that reaches the oracle unvalidated would
    fail deep inside atom construction, or worse, silently mis-dispatch.
    """
    if isinstance(spec, str):
        try:
            return {"general": GENERAL, "sp": SP}[spec]
        except KeyError:
            raise ValueError(f"unknown poset class {spec!r}; use 'general' or 'sp'")
    missing = [
        member
        for member in ("name", "contains_all_posets", "closed_under_meet",
                       "contains", "extension_count", "enumerate",
                       "sample_extension")
        if not hasattr(spec, member)
    ]
    if missing:
        raise TypeError(
            f"poset class {spec!r} does not implement the PosetClass "
            f"protocol: missing {missing}"
        )
    return spec
