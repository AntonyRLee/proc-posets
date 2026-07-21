"""Group construction -- the one stage that cannot be made principled.

Co-membership of traces is not estimable from exchangeable trace data: any
rule that manufactures groups from a flat log is a *modelling assumption*,
not an estimation step.  This module therefore does exactly one thing --
blocks a flat log on a user-declared key -- and forces the caller to declare
it, so the assumption appears in code the same way it must appear in the
analysis: explicitly.

The estimation consequences of an imperfect key are then handled *inside*
the model, by the interloper rate eta (a mixing coordinate of the NPMLE, not
a preprocessing dial): if a fraction of each block's traces belong to other
components, atoms with the matching eta explain the blocks best and the
fitted eta reports the blocking quality back to you.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Tuple


def group_by_key(
    rows: Iterable[Tuple[str, Tuple[str, ...]]],
    declared_assumption: str,
    min_group_size: int = 3,
) -> Tuple[List[List[Tuple[str, ...]]], List[List[Tuple[str, ...]]]]:
    """Block a flat log of (key, trace) rows into groups.

    ``declared_assumption`` is mandatory and is printed: state *why* traces
    sharing this key should share a latent component (e.g. "traces from the
    same clinical episode follow one protocol variant").  Refusing a default
    here is deliberate.

    Returns (groups meeting min_group_size, undersized leftovers).  Groups of
    size < 3 carry no Kruskal identification; leftovers can still be used for
    weight efficiency in a second stage but are excluded here.
    """
    print(f"[grouping] declared assumption: {declared_assumption}")
    buckets = defaultdict(list)
    for key, trace in rows:
        buckets[key].append(trace)
    groups, leftovers = [], []
    for key in sorted(buckets):
        (groups if len(buckets[key]) >= min_group_size else leftovers).append(buckets[key])
    print(
        f"[grouping] {len(groups)} groups of size >= {min_group_size}, "
        f"{len(leftovers)} undersized blocks set aside"
    )
    return groups, leftovers
