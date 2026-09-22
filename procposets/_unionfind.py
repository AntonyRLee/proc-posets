"""A tiny union-find (disjoint-set) with path-halving.

Single home for the find/union/group boilerplate that ``moddecomp._components``
and ``cospan.engine_fast._components`` each inlined.  ``groups()`` returns the
classes as lists in original item order, the classes themselves ordered by the
first appearance of their root -- both callers depend on this ordering
(moddecomp's series rank and the arc-component Cartesian product are
order-sensitive).  Imports nothing, so any module may use it without a cycle.
"""
from __future__ import annotations


class UnionFind:
    def __init__(self, items):
        self._parent = {x: x for x in items}

    def find(self, x):
        p = self._parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, a, b) -> None:
        """Merge the classes of ``a`` and ``b`` (``a``'s root points at ``b``'s)."""
        self._parent[self.find(a)] = self.find(b)

    def groups(self) -> list:
        """The classes: each a list in insertion order, classes ordered by first
        appearance of their root over the original item order."""
        g: dict = {}
        for x in self._parent:
            g.setdefault(self.find(x), []).append(x)
        return list(g.values())
