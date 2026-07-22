"""Ground-truth simulators for grouped logs.

Generates grouped logs from an SP-poset mixture with the two noise channels
of the model, plus a coarse-key generator for inferred-group recovery:

* eps_sim -- recording noise: with this probability a trace suffers one
  uniformly chosen adjacent transposition (a realistic logging error that the
  eps channel of the model absorbs).
* eta_sim -- interloper noise: with this probability a trace inside a group
  is drawn from a freshly sampled component instead of the group's own.
* coarse keys -- traces are emitted in "blocks" (e.g. days); each block has a
  dominant component and each trace defects from it with some probability.
  Blocking on the key then yields groups whose purity is exactly the eta the
  model must absorb.
"""

from __future__ import annotations

import random
from bisect import insort
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ._extensions import preds as _preds
from .rel import SPTree, sample_extension_tree


@dataclass
class TrueMixture:
    """A ground-truth mixture of SP posets -- ``trees`` weighted by ``weights`` --
    the generative model the simulators sample from."""

    trees: List[SPTree]
    weights: List[float]

    def sample_component(self, rng: random.Random) -> int:
        """Draw a component index ``k`` with probability ``weights[k]``."""
        return rng.choices(range(len(self.trees)), weights=self.weights)[0]

    def sample_trace(self, k: int, rng: random.Random, eps_sim: float = 0.0):
        """A uniform linear extension of component ``k`` as a label word, with an
        optional ``eps_sim`` chance of one adjacent transposition (recording noise)."""
        t = list(sample_extension_tree(self.trees[k], rng))
        if eps_sim > 0 and rng.random() < eps_sim and len(t) > 1:
            i = rng.randrange(len(t) - 1)
            t[i], t[i + 1] = t[i + 1], t[i]
        return tuple(t)


def sample_grouped_log(
    mix: TrueMixture,
    G: int,
    n_g: int,
    seed: int = 0,
    eps_sim: float = 0.0,
    eta_sim: float = 0.0,
) -> Tuple[List[List[Tuple[str, ...]]], List[int]]:
    """G groups of n_g traces each; returns (groups, true component per group)."""
    rng = random.Random(seed)
    groups, z = [], []
    for _ in range(G):
        k = mix.sample_component(rng)
        z.append(k)
        g = []
        for _ in range(n_g):
            kk = mix.sample_component(rng) if (eta_sim > 0 and rng.random() < eta_sim) else k
            g.append(mix.sample_trace(kk, rng, eps_sim))
        groups.append(g)
    return groups, z


def sample_timed_grouped_log(
    mix: TrueMixture,
    G: int,
    n_g: int,
    lams: Sequence[float],
    seed: int = 0,
    eta_sim: float = 0.0,
) -> Tuple[List[List[Tuple[Tuple[str, ...], Tuple[float, ...]]]], List[int]]:
    """G groups of n_g *timestamped* traces under racing-clock semantics.

    Component k runs each enabled activity on an independent Exp(lams[k])
    clock: at every step the gap is Exp(lam * #enabled) and the finisher is
    uniform over the enabled set.  Returns groups of (trace, gaps) pairs and
    the true component per group.
    """
    from .rel import tree_relations

    rng = random.Random(seed)
    rels = [tree_relations(t) for t in mix.trees]
    # Ground alphabet PER COMPONENT.  NPMLE mixtures normally share one activity
    # set (so every alphabets[k] is identical and this is byte-identical to the
    # old single ``sorted(trees[0].elements())`` hoist), but TrueMixture does not
    # enforce a common support and the untimed sampler ``sample_grouped_log``
    # already samples each component over its OWN elements -- so key every
    # component off its own tree rather than trees[0], instead of silently
    # dropping (or spuriously enabling) activities for an odd-one-out component.
    alphabets = [sorted(t.elements()) for t in mix.trees]
    # Hoist the per-component predecessor/successor indices out of one_timed
    # (it was rebuilding preds on each of its G*n_g calls); both are rng-free, so
    # the draw stream is untouched.  preds is transitively closed (tree_relations),
    # so preds[x] & rem empty <=> x is minimal in rem, and indeg[x] = |remaining
    # predecessors| lets the enabled frontier be maintained incrementally instead
    # of rescanned O(|rem|*deg) every step.
    preds_by_k = [_preds(alphabets[k], rel) for k, rel in enumerate(rels)]
    succs_by_k = [
        {x: [y for y in alphabets[k] if x in preds[y]] for x in alphabets[k]}
        for k, preds in enumerate(preds_by_k)
    ]

    def one_timed(k: int) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        preds, succs, lam = preds_by_k[k], succs_by_k[k], lams[k]
        alphabet = alphabets[k]
        rem = set(alphabet)
        indeg = {x: len(preds[x]) for x in alphabet}  # rem = full alphabet initially
        enabled = sorted(x for x in alphabet if indeg[x] == 0)  # minimal elements
        trace: List[str] = []
        gaps: List[float] = []
        while rem:
            # enabled == sorted(x for x in rem if not (preds[x] & rem)) by invariant,
            # so the rng draws below (len + choice) are byte-identical to the rescan.
            gaps.append(rng.expovariate(lam * len(enabled)))
            f = rng.choice(enabled)
            trace.append(f)
            rem.discard(f)
            enabled.remove(f)
            for y in succs[f]:  # f was minimal, so every descendant loses one pred
                indeg[y] -= 1
                if indeg[y] == 0:  # last predecessor of y just fired -> newly enabled
                    insort(enabled, y)
        return tuple(trace), tuple(gaps)

    groups, z = [], []
    for _ in range(G):
        k = mix.sample_component(rng)
        z.append(k)
        g = []
        for _ in range(n_g):
            kk = mix.sample_component(rng) if (eta_sim > 0 and rng.random() < eta_sim) else k
            g.append(one_timed(kk))
        groups.append(g)
    return groups, z


def sample_keyed_log(
    mix: TrueMixture,
    n_blocks: int,
    traces_per_block,
    defect_prob: float,
    seed: int = 0,
    eps_sim: float = 0.0,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Flat log of (coarse_key, trace) rows: each block (e.g. a day) has a
    dominant component; traces defect from it with defect_prob.  No case IDs.

    ``traces_per_block`` is an int, or an inclusive (lo, hi) range sampled
    uniformly per block -- variable daily volume, which realistically leaves
    some blocks below the Kruskal size 3 (they become second-stage weight
    evidence, see npmle.refit_weights).
    """
    rng = random.Random(seed)
    rows = []
    for b in range(n_blocks):
        k_block = mix.sample_component(rng)
        n = (
            traces_per_block
            if isinstance(traces_per_block, int)
            else rng.randint(*traces_per_block)
        )
        for _ in range(n):
            k = mix.sample_component(rng) if rng.random() < defect_prob else k_block
            rows.append((f"block{b:03d}", mix.sample_trace(k, rng, eps_sim)))
    return rows
