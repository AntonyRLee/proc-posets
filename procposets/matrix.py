"""Block-transition stochastic matrix of a model.

A *model* is a weighted set of normal forms: list of (Poset, weight). Each normal form is
tiled by modular decomposition into atomic blocks (leaves, parallel/concurrent blocks, prime
blocks); a SERIES node chains its atomic parts. The matrix records, across the weighted normal
forms, the block-to-block composition transitions -- so concurrency stays atomic and models that
share a composition prefix share matrix rows. This is the object ``distance.smd`` compares.

`context_depth` (the VLMC dial): the state after emitting blocks b_0..b_i is the '|'-joined last
`context_depth` blocks (START/END stay bare sentinels). Depth 1 is the memoryless block chain
(default, backward-compatible); higher depth is a variable-order chain that separates repeated
blocks by their preceding context -- sharper and less row-sharing across models. Orthogonal to
concurrency (a parallel block stays one atomic token at every depth).
"""
from __future__ import annotations

from collections import defaultdict

from .moddecomp import Series, decompose
from .poset import Poset

START = "START"
END = "END"


def block_sequence(tree) -> list[str]:
    """Ordered atomic block labels of one normal form (series -> chain; else a single block)."""
    if isinstance(tree, Series):
        return [c.canonical() for c in tree.parts]  # series parts are atomic (mod. decomp. flattens)
    return [tree.canonical()]


def build(model: list[tuple[Poset, float]], context_depth: int = 1,
          *, context: str = "uniform", ceiling: int | None = None, tree=None):
    """Return (matrix, states). matrix[src] = {dst: prob}; states are order-`context_depth` block
    contexts ('|'-joined last <=k blocks) plus the bare sentinels START, END. context_depth=1 is
    the memoryless block chain (default; identical matrices to the pre-VLMC implementation).

    ``context="canonical"`` switches to the genuinely lossless chart of :mod:`procposets.context`,
    where the state is the shortest trailing run of blocks whose histories share a future. There
    ``context_depth`` is IGNORED and ``ceiling`` applies instead -- and it is a CEILING, not an
    order: ``None`` (the default) is the parameter-free lossless chart, an integer caps how far
    back a context may reach. Pass ``tree=`` to chart on a shared fleet tree.

    The two modes coincide at ``ceiling=1`` with anchoring off, which is what keeps the uniform
    default byte-identical."""
    if tree is not None or context == "canonical":
        from .context import build as _tree_build
        from .context import context_tree
        return _tree_build(model, tree if tree is not None else context_tree(model, ceiling=ceiling))
    if context != "uniform":
        raise ValueError(f"context must be 'uniform' or 'canonical', not {context!r}")
    k = max(1, context_depth)
    raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    states: set[str] = {START, END}
    for P, w in model:
        blocks = block_sequence(decompose(P))
        prev = START
        for i in range(len(blocks)):
            cur = "|".join(blocks[max(0, i - k + 1):i + 1])   # last <=k blocks
            states.add(cur)
            raw[prev][cur] += w
            prev = cur
        raw[prev][END] += w
    matrix: dict[str, dict[str, float]] = {}
    for s in states:
        row = raw.get(s, {})
        tot = sum(row.values())
        matrix[s] = {d: v / tot for d, v in row.items()} if tot > 0 else {}
    return matrix, states


def normal_form_distribution(model: list[tuple[Poset, float]]) -> dict[str, float]:
    """Flat distribution over normal-form tilings -- the Result-1 (brittle) object."""
    dist: dict[str, float] = defaultdict(float)
    tot = 0.0
    for P, w in model:
        dist[decompose(P).canonical()] += w
        tot += w
    return {k: v / tot for k, v in dist.items()}
