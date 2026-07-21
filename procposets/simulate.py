"""Ground-truth simulators for the demos.

Generates grouped logs from an SP-poset mixture with the two noise channels
of the model, plus a coarse-key generator for the inferred-groups demo:

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
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ._extensions import preds as _preds
from .rel import SPTree, sample_extension


@dataclass
class TrueMixture:
    trees: List[SPTree]
    weights: List[float]

    def sample_component(self, rng: random.Random) -> int:
        return rng.choices(range(len(self.trees)), weights=self.weights)[0]

    def sample_trace(self, k: int, rng: random.Random, eps_sim: float = 0.0):
        t = list(sample_extension(self.trees[k], rng))
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
    alphabet = sorted(mix.trees[0].elements())

    def one_timed(k: int) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        rel, lam = rels[k], lams[k]
        preds = _preds(alphabet, rel)  # {e: frozenset(predecessors)}
        rem = set(alphabet)
        trace: List[str] = []
        gaps: List[float] = []
        while rem:
            enabled = sorted(x for x in rem if not (preds[x] & rem))
            gaps.append(rng.expovariate(lam * len(enabled)))
            trace.append(rng.choice(enabled))
            rem.discard(trace[-1])
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
