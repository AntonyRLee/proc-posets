"""String-diagram visualisers
(matplotlib string diagrams + gamma1->gamma2 DAGs;
generator-cospan boundary ports are (src, type, tgt) triples).

Two products for a model (a weighted set of normal-form posets):
  signature_catalogue(model, ...)  -- the generator-cospan signature Sigma: EACH activity generator
                                      drawn as its own cospan, stacked vertically, with its input and
                                      output legs labelled by their (src, omega, tgt) port triples.
                                      Concurrency shows as a generator with several out-legs (a split);
                                      a synchronising activity as one with several in-legs (a merge).
  render_morphisms(model, ...)     -- the gamma1 -> gamma2 morphisms the model allows: each normal
                                      form as a left-to-right occurrence DAG.

matplotlib only (no graphviz binary needed).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

from ..distance import _augment, smd
from ..matrix import END, START, build
from ..poset import Poset

# palette (centralized in viz/palette.py)
from .palette import BOUNDARY_GREY as _BOUNDARY  # gamma1 / gamma2 boundary points
from .palette import COTTON_BLUE as _COTTON  # activities / wires
from .palette import OLD_BLACK as _BLACK
_BW, _BH = 1.35, 0.66


def _box(ax, x, y, label, edge, face="#ffffff", w=_BW, h=_BH):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x, y, f"${label}$", ha="center", va="center", fontsize=11, color=_BLACK, zorder=3)


def _covers(P: Poset):
    less = P.less
    return {(u, v) for (u, v) in less
            if not any((u, w) in less and (w, v) in less for w in P.elements)}


def _tex(label: str) -> str:
    return {"g1": "\\gamma_1", "g2": "\\gamma_2"}.get(label, label)


def _port(src: str, tgt: str) -> str:
    return f"$({_tex(src)},\\,\\omega,\\,{_tex(tgt)})$"


# --- signature: generator cospans -------------------------------------------

def extract_generators(model):
    """The generator-cospan signature of the model: a sorted set of
    (label, left_ports, right_ports), where a port is a (src, tgt) pair and the
    context is read from the causal poset bounded by gamma_1/gamma_2. A source
    activity gets a left leg from gamma_1; a sink gets a right leg to gamma_2.
    Distinct contexts of the same activity are distinct generators (so a choice
    yields several single-leg generators; concurrency yields multi-leg ones)."""
    gens = set()
    for P, _ in model:
        cov = _covers(P)
        pred: dict[int, list[int]] = {e: [] for e in P.elements}
        succ: dict[int, list[int]] = {e: [] for e in P.elements}
        for (u, v) in cov:
            succ[u].append(v)
            pred[v].append(u)
        for e in P.elements:
            x = P.labels[e]
            left = tuple(sorted((P.labels[p], x) for p in pred[e])) or (("g1", x),)
            right = tuple(sorted((x, P.labels[s]) for s in succ[e])) or ((x, "g2"),)
            gens.add((x, left, right))
    return sorted(gens)


def signature_catalogue(model, path, title="signature $\\Sigma$"):
    gens = extract_generators(model)
    bw, leg, step, gap = 1.0, 1.25, 0.55, 0.85
    heights = [max(len(L), len(R), 1) * step + 0.3 for _, L, R in gens]
    total = sum(heights) + gap * max(len(gens) - 1, 0)
    fig, ax = plt.subplots(figsize=(6.6, max(2.2, 0.62 * total + 1.2)))

    y = 0.0
    for (x, L, R), h in zip(gens, heights):
        cy = y - h / 2

        def ys(k):
            return [cy + ((k - 1) / 2 - i) * step for i in range(k)]

        # in-legs (left) -- fan into the box's left edge
        for (s, t), yy in zip(L, ys(len(L))):
            ax.plot([-bw / 2 - leg, -bw / 2], [yy, cy], color=_COTTON, lw=1.2, zorder=1)
            ax.plot(-bw / 2 - leg, yy, "o", ms=4, color=_COTTON, zorder=2)
            ax.text(-bw / 2 - leg - 0.12, yy, _port(s, t), ha="right", va="center", fontsize=8)
        # out-legs (right) -- fan out of the box's right edge
        for (s, t), yy in zip(R, ys(len(R))):
            ax.plot([bw / 2, bw / 2 + leg], [cy, yy], color=_COTTON, lw=1.2, zorder=1)
            ax.plot(bw / 2 + leg, yy, "o", ms=4, color=_COTTON, zorder=2)
            ax.text(bw / 2 + leg + 0.12, yy, _port(s, t), ha="left", va="center", fontsize=8)
        _box(ax, 0.0, cy, x, _COTTON, w=bw, h=h)
        ax.text(0.0, cy + h / 2 + 0.08, f"$g_{{{x}}}$", ha="center", va="bottom", fontsize=9, color=_BLACK)
        y -= (h + gap)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(y - 0.2, 0.7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- gamma1 -> gamma2 morphisms ---------------------------------------------

def _levels(P: Poset):
    cov = _covers(P)
    preds: dict[int, list[int]] = {e: [] for e in P.elements}
    for (u, v) in cov:
        preds[v].append(u)
    memo: dict[int, int] = {}

    def lvl(v):
        if v not in memo:
            memo[v] = 0 if not preds[v] else 1 + max(lvl(u) for u in preds[v])
        return memo[v]

    return {e: lvl(e) for e in P.elements}, cov


def draw_morphism(ax, P: Poset, title=None):
    lv, cov = _levels(P)
    maxlv = max(lv.values(), default=0)
    by_level: dict[int, list[int]] = {}
    for e, l in lv.items():
        by_level.setdefault(l, []).append(e)
    pos = {}
    for l, es in by_level.items():
        es.sort(key=lambda e: P.labels[e])
        for k, e in enumerate(es):
            pos[e] = (l, (len(es) - 1) / 2 - k)

    def xy(e):
        x, y = pos[e]
        return x * (_BW + 0.9), y * (_BH + 0.5)

    def edge(p, q):
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=_BLACK, lw=1.2, shrinkA=10, shrinkB=10))

    g1xy = (-1.0 * (_BW + 0.9), 0.0)
    g2xy = ((maxlv + 1.0) * (_BW + 0.9), 0.0)
    minimals = [e for e in P.elements if lv[e] == 0]
    maximals = [e for e in P.elements if not any((e, w) in cov for w in P.elements)]
    for e in minimals:
        edge(g1xy, xy(e))
    for (u, v) in cov:
        edge(xy(u), xy(v))
    for e in maximals:
        edge(xy(e), g2xy)
    for e in P.elements:
        _box(ax, *xy(e), P.labels[e], _COTTON)
    for (cx, cy), lab in [(g1xy, "\\gamma_1"), (g2xy, "\\gamma_2")]:
        ax.add_patch(Circle((cx, cy), 0.13, color=_BOUNDARY, zorder=3))
        ax.text(cx, cy - 0.42, f"${lab}$", ha="center", va="top", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    ax.margins(0.12)


def render_morphisms(model, path, title=None):
    """One panel per normal form the model allows, titled with its weight."""
    tot = sum(w for _, w in model)
    order = sorted(model, key=lambda pw: -pw[1])
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.6), squeeze=False)
    for ax, (P, w) in zip(axes[0], order):
        draw_morphism(ax, P, title=f"$p={w / tot:.2f}$")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- block-transition matrix (on a common state space) ----------------------

def _short(s: str) -> str:
    if s.startswith("N{"):
        import re
        return "N{" + "".join(sorted(set(re.findall(r"[a-z0-9]", s)))) + "}"
    return s


def common_state_space(models, context_depth: int = 1):
    """Ordered union of block states across `models` (START first, END last) so every model's
    matrix is drawn over the SAME rows/columns and is directly comparable."""
    s: set[str] = set()
    for m in models:
        _, st = build(m, context_depth)
        s |= st
    return [START] + sorted(x for x in s if x not in (START, END)) + [END]


def _matrix_on(model, states, context_depth: int = 1):
    m, _ = build(model, context_depth)
    idx = {s: i for i, s in enumerate(states)}
    A = np.zeros((len(states), len(states)))
    for s, row in _augment(m, states).items():   # same normalisation as the SMD (NORMALISATION default)
        for t, p in row.items():
            if t in idx:
                A[idx[s], idx[t]] = p
    return A


def draw_matrix(model, states, path, title=None, context_depth: int = 1):
    """Heatmap of the model's block-transition stochastic matrix P over the given (common) states."""
    A = _matrix_on(model, states, context_depth)
    n = len(states)
    labels = [_short(s) for s in states]
    fig, ax = plt.subplots(figsize=(0.62 * n + 1.8, 0.62 * n + 1.4))
    im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    for i in range(n):
        for j in range(n):
            if A[i, j] > 1e-9:
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if A[i, j] > 0.5 else _BLACK)
    if title:
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- clustering figure: pairwise SMD distance matrix ------------------------

def cluster_heatmap(models, labels, names, path, title="pairwise SMD (models ordered by group)"):
    order = sorted(range(len(models)), key=lambda i: labels[i])
    ms = [models[i] for i in order]
    ls = [labels[i] for i in order]
    n = len(ms)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = smd(ms[i], ms[j])[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(D, cmap="viridis")
    bounds = [k for k in range(1, n) if ls[k] != ls[k - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="w", lw=1.0)
        ax.axvline(b - 0.5, color="w", lw=1.0)
    centres, start = [], 0
    for b in bounds + [n]:
        centres.append((start + b - 1) / 2)
        start = b
    ax.set_xticks(centres); ax.set_yticks(centres)
    ax.set_xticklabels(names, fontsize=9); ax.set_yticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="SMD")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
