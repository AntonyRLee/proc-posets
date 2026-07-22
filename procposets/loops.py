"""Loops: geometric unrolling + truncation -- the paper's loop treatment, live.

A loop is the one structure the causal-poset pipeline cannot hold as a finite poset: its
longest tiling word L is unbounded, so the depth bound of thm:faithful fails and full fidelity
would need an infinite state space. The paper's resolution (Sec. 5): the loop enters the
pipeline through its UNROLLINGS -- the variant family body^1, body^2, ... with geometrically
decaying weights (repeat probability q: unrolling n carries (1-q) q^(n-1)) -- truncated at a
finite K, "the longest unrolling the data can witness". The cut-off is physically motivated:
the mass beyond K is q^K, so the truncated block chain converges geometrically to the LIMIT
object `loop_limit`: the genuine cyclic block Markov chain in which the body's closing block
returns to its opening block with probability q. That limit chain is exactly what a
practitioner would draw for the loop.

Conventions:
  * the body runs AT LEAST once (a do-while; a 0-or-more loop is the mixture of this family
    with the loop-free variant);
  * truncation either renormalises the geometric law over n <= K (`tail="renorm"`, the
    conditional law given the witnessed horizon) or lumps the tail mass onto the deepest
    unrolling (`tail="lump"`); both converge to the same limit at the same geometric rate;
  * `loop_limit` is the depth-1 limit chain, so the block names of pre + body + post must be
    pairwise distinct (a repeated block would need context depth, prop:minimal-depth).

Pure stdlib, like the rest of the core.
"""
from __future__ import annotations

import random
from collections import Counter

from .matrix import END, START, block_sequence
from .moddecomp import decompose
from .poset import Model, Poset, then


def truncated_geometric(q: float, K: int, tail: str = "renorm") -> list[float]:
    """Weights of unrollings n = 1..K under repeat probability q. "renorm" conditions the
    geometric law on n <= K; "lump" keeps it exact below K and gives the tail mass q^(K-1)
    to the deepest unrolling."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"repeat probability q must be in (0,1), got {q}")
    if K < 1:
        raise ValueError("need at least one unrolling")
    w = [(1.0 - q) * q ** (n - 1) for n in range(1, K + 1)]
    if tail == "lump":
        w[-1] = q ** (K - 1)
    elif tail == "renorm":
        tot = sum(w)                       # = 1 - q^K
        w = [x / tot for x in w]
    else:
        raise ValueError(f"unknown tail convention {tail!r}")
    return w


def unrolling(body: Poset, n: int, pre: tuple = (), post: tuple = ()) -> Poset:
    """The n-th unrolling: pre ; body^n ; post as one causal poset."""
    if n < 1:
        raise ValueError("the loop body runs at least once")
    return then(*pre, *([body] * n), *post)


def loop_model(body: Poset, q: float, K: int, pre: tuple = (), post: tuple = (),
               tail: str = "renorm") -> Model:
    """The truncated unrolling family: the loop as a finite weighted variant set."""
    w = truncated_geometric(q, K, tail)
    return [(unrolling(body, n, pre, post), w[n - 1]) for n in range(1, K + 1)]


def _word(parts) -> list[str]:
    out: list[str] = []
    for p in parts:
        out += block_sequence(decompose(p))
    return out


def loop_limit(body: Poset, q: float, pre: tuple = (), post: tuple = ()):
    """The K -> infinity limit of `build(loop_model(...), context_depth=1)`: the cyclic block
    chain. Deterministic along pre ; body ; post, except the body's closing block, which
    returns to its opening block with probability q and continues with 1 - q. Returns
    (matrix, states) in the `matrix.build` format (compare via `distance.smd_rows`)."""
    body_word = _word([body])
    pre_word, post_word = _word(pre), _word(post)
    word = pre_word + body_word + post_word
    if len(set(word)) != len(word):
        raise ValueError("depth-1 limit chain needs pairwise-distinct block names across "
                         f"pre + body + post, got {word}")
    chain = [START] + word + [END]
    matrix = {chain[i]: {chain[i + 1]: 1.0} for i in range(len(chain) - 1)}
    closing, opening = body_word[-1], body_word[0]
    after = post_word[0] if post_word else END
    matrix[closing] = {opening: q, after: 1.0 - q}
    return matrix, {START, END, *word}


def sample_repeats(q: float, n: int, rng: random.Random) -> list[int]:
    """n i.i.d. repetition counts from the UNBOUNDED geometric loop (P(k) = (1-q) q^(k-1)) --
    the forward model an event log actually comes from. Attributing a trace to its unrolling
    is deterministic (unrollings are trace-disjoint by length: Regime-1 counting is exact),
    so sampling the counts directly IS sampling the log, as far as rho_hat is concerned."""
    out = []
    for _ in range(n):
        k = 1
        while rng.random() < q:
            k += 1
        out.append(k)
    return out


def empirical_loop_model(body: Poset, repeats: list[int], pre: tuple = (),
                         post: tuple = ()) -> Model:
    """E_N = P(rho_hat_N) for a loop log: the witnessed unrollings at their observed
    frequencies. The truncation depth is the deepest unrolling the log witnessed -- the
    paper's data-driven cut-off, with no K to choose."""
    counts = Counter(repeats)
    n = len(repeats)
    return [(unrolling(body, k, pre, post), c / n) for k, c in sorted(counts.items())]
