"""Regenerate the two Chapter 4 figures of the thesis with this package.

Chapter 4 (*A distance function for stochastic matrices*) compares four distances by how
well each clusters Dirichlet-parametrised ensembles of 3x3 stochastic matrices as the
clusters are interpolated into one another.  The published run's seed was never recorded,
so this script re-runs the design with a seed that IS recorded, and the thesis states the
result as produced here rather than as published.

    ternary.pdf     the four base Dirichlet densities on the simplex
    rand_index.pdf  mean adjusted Rand index against the interpolation parameter t,
                    one panel per distance, that distance in the foreground

The stochastic matrix distance comes from ``procposets.distance.smd_rows`` -- the same code
path the rest of the thesis uses -- checked against the closed form of Result 3,
``2 sqrt(sum_i arccos^2 sum_j sqrt(P1_ij P2_ij))``, at import time.  The clustering and the
adjusted Rand index are implemented here in numpy so the figure depends on nothing beyond
this package and matplotlib.

    uv run --extra viz python examples/ch4_clustering_figures.py --outdir <thesis>/figures

Defaults reproduce the thesis figures exactly: --steps 21 --repetitions 20
--cluster-size 15 --seed 20260907.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from procposets.distance import smd_rows

# Sec. IV: a reference cluster and three displaced ones.
ALPHAS = np.array([[8.0, 8.0, 8.0], [8.0, 2.0, 2.0], [2.0, 8.0, 2.0], [2.0, 2.0, 8.0]])
STATES = ["0", "1", "2"]
PANEL_TITLES = ["a: reference $\\alpha_0$", "b: $\\alpha_1$", "c: $\\alpha_2$", "d: $\\alpha_3$"]


# --------------------------------------------------------------------------- distances


def as_rows(P: np.ndarray) -> dict[str, dict[str, float]]:
    """A 3x3 stochastic matrix as the row dictionary ``smd_rows`` consumes."""
    return {STATES[i]: {STATES[j]: float(v) for j, v in enumerate(row)} for i, row in enumerate(P)}


def stochastic_matrix_distance(P1: np.ndarray, P2: np.ndarray) -> float:
    """Result 3, evaluated by the package rather than by a local reimplementation."""
    return smd_rows((as_rows(P1), STATES), (as_rows(P2), STATES))[0]


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Solve ``pi P = pi`` as a linear system (accurate for slowly mixing chains)."""
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


def symmetrised_kl_rate(P1: np.ndarray, P2: np.ndarray) -> float:
    """Sum of the two stationary-weighted Kullback-Leibler divergence rates."""
    total = 0.0
    for A, B in ((P1, P2), (P2, P1)):
        pi = stationary_distribution(A)
        total += float((pi[:, None] * A * np.log(A / B)).sum())
    return total


# Panel names follow the chapter's figure.  Total variation is L1/2 and Frobenius is L2;
# both are monotone rescalings of the L1- and L2-induced distances, so the clustering --
# and hence the Rand index -- is unchanged by the choice.
DISTANCES = {
    "(Symmetrised) KL divergence": symmetrised_kl_rate,
    "Frobenius": lambda P1, P2: float(np.sqrt(((P1 - P2) ** 2).sum())),
    "Total variation": lambda P1, P2: float(np.abs(P1 - P2).sum() / 2.0),
    "Stochastic matrix": stochastic_matrix_distance,
}


def _check_smd_closed_form() -> None:
    rng = np.random.default_rng(0)
    P1, P2 = rng.dirichlet([4, 2, 2], size=3), rng.dirichlet([2, 4, 2], size=3)
    bc = np.sqrt(P1 * P2).sum(axis=1)
    closed = 2.0 * float(np.sqrt((np.arccos(np.clip(bc, 0.0, 1.0)) ** 2).sum()))
    if not math.isclose(closed, stochastic_matrix_distance(P1, P2), rel_tol=1e-12):
        raise AssertionError("smd_rows disagrees with Result 3's closed form")


# ------------------------------------------------------------------- clustering + index


def average_linkage(D: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage agglomerative clustering from a distance matrix alone.

    Lance-Williams update, merging until ``k`` clusters remain.  Naive and O(n^3), which is
    ample at the sixty matrices per repetition this experiment uses, and keeps the example
    free of a scipy dependency.
    """
    n = D.shape[0]
    members = {i: [i] for i in range(n)}
    dist = D.astype(float).copy()
    np.fill_diagonal(dist, np.inf)
    active = list(range(n))
    while len(active) > k:
        sub = dist[np.ix_(active, active)]
        flat = int(np.argmin(sub))
        i, j = active[flat // len(active)], active[flat % len(active)]
        ni, nj = len(members[i]), len(members[j])
        for m in active:
            if m in (i, j):
                continue
            merged = (ni * dist[i, m] + nj * dist[j, m]) / (ni + nj)
            dist[i, m] = dist[m, i] = merged
        members[i] = members[i] + members[j]
        active.remove(j)
    labels = np.empty(n, dtype=int)
    for label, root in enumerate(active):
        labels[members[root]] = label
    return labels


def adjusted_rand_index(truth: np.ndarray, labels: np.ndarray) -> float:
    """Hubert-Arabie form, 2(ad - bc) / [(a+b)(b+d) + (a+c)(c+d)] -- the chapter's Eq. (4.8)."""
    N = len(truth)
    rows, cols = np.unique(truth), np.unique(labels)
    n = np.array([[np.sum((truth == u) & (labels == v)) for v in cols] for u in rows], float)
    t = float((n ** 2).sum())
    r = float((n.sum(axis=1) ** 2).sum())
    c = float((n.sum(axis=0) ** 2).sum())
    a = (t - N) / 2.0
    b = (r - t) / 2.0
    cc = (c - t) / 2.0
    d = (t - r - c + N * N) / 2.0
    denominator = (a + b) * (b + d) + (a + cc) * (cc + d)
    return 0.0 if denominator == 0 else 2.0 * (a * d - b * cc) / denominator


# ------------------------------------------------------------------------------- the run


def sample_cluster(rng, alpha, n_members, n_states=3):
    """``n_members`` stochastic matrices, every row drawn from ``Dirichlet(alpha)``."""
    return rng.dirichlet(alpha, size=(n_members, n_states))


def distance_matrix(matrices, distance):
    n = len(matrices)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = distance(matrices[i], matrices[j])
    return D


def run(steps=21, repetitions=20, cluster_size=15, seed=20260907):
    """Algorithm 4.1, returning ``(t_values, {distance name: mean ARI per t})``."""
    rng = np.random.default_rng(seed)
    k = len(ALPHAS)
    truth = np.repeat(np.arange(k), cluster_size)
    t_values = np.linspace(0.0, 1.0, steps)
    scores = {name: np.zeros(steps) for name in DISTANCES}

    for step, t in enumerate(t_values):
        alphas_t = (1.0 - t) * ALPHAS + t * ALPHAS[0]
        for _ in range(repetitions):
            matrices = [m for alpha in alphas_t for m in sample_cluster(rng, alpha, cluster_size)]
            for name, distance in DISTANCES.items():
                labels = average_linkage(distance_matrix(matrices, distance), k)
                scores[name][step] += adjusted_rand_index(truth, labels)
        for values in scores.values():
            values[step] /= repetitions
        print(
            f"t={t:5.3f}  " + "  ".join(f"{n}={scores[n][step]:6.3f}" for n in DISTANCES),
            flush=True,
        )
    return t_values, scores


# ----------------------------------------------------------------------------- the plots


def _barycentric(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simplex points to plane coordinates, with (1,0,0) at the left of the base."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (math.sqrt(3.0) / 2.0) * points[:, 2]
    return x, y


def plot_ternary(path: str, seed: int, draws: int = 40000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.2), constrained_layout=True)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    for ax, alpha, title in zip(axes.ravel(), ALPHAS, PANEL_TITLES):
        x, y = _barycentric(rng.dirichlet(alpha, size=draws))
        ax.hexbin(x, y, gridsize=42, cmap="viridis", mincnt=1, linewidths=0.0)
        ax.plot(*np.vstack([corners, corners[:1]]).T, color="0.2", linewidth=0.8)
        ax.set_title(f"{title} = ({alpha[0]:.0f}, {alpha[1]:.0f}, {alpha[2]:.0f})", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.savefig(path)
    print(f"wrote {path}")


def plot_rand_index(path: str, t_values, scores) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for ax, (label, (name, values)) in zip(axes.ravel(), zip("abcd", scores.items())):
        for other, other_values in scores.items():
            if other != name:
                ax.plot(t_values, other_values, color="0.75", linewidth=1)
        ax.plot(t_values, values, color="tab:blue", linewidth=1.4)
        ax.set_title(f"{label}: {name}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.0, 1.0)
    fig.supxlabel("iteration parameter $t$")
    fig.supylabel("mean adjusted Rand index")
    fig.savefig(path)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=21, help="values of t in [0, 1]")
    parser.add_argument("--repetitions", type=int, default=20, help="runs averaged per t")
    parser.add_argument("--cluster-size", type=int, default=15, help="matrices per cluster")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--outdir", default=".", help="where the two PDFs are written")
    args = parser.parse_args()

    _check_smd_closed_form()
    os.makedirs(args.outdir, exist_ok=True)
    plot_ternary(os.path.join(args.outdir, "ternary.pdf"), args.seed)
    t_values, scores = run(args.steps, args.repetitions, args.cluster_size, args.seed)
    plot_rand_index(os.path.join(args.outdir, "rand_index.pdf"), t_values, scores)


if __name__ == "__main__":
    main()
