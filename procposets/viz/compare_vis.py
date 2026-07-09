"""Render a :class:`cpm.cospan.signature_compare.ComparisonReport` as an
"at-a-glance" matrix: rows = canonical generators, columns = notations, each cell
the generator's N-linear binding parameters, colour-coded against the reference.

This is the comparison twin of :func:`cpm.vis.catalogue` (which draws *one*
signature's generator cospans). Here every notation's parameterised generators are
laid out on the **same rows** -- so master vs OCCN vs OCPN is read down a column and
the agreement/disagreement off the colour. Modular: a column per notation, however
many are passed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..cospan.signature_compare import ComparisonReport  # noqa: E402

# status -> (fill, edge) colours
_STATUS_COLOUR = {
    "ref": ("#eef1f5", "#9aa6b2"),     # reference baseline (neutral)
    "match": ("#d7f0d2", "#5a9e4e"),   # green: rederived, params agree
    "diff": ("#fde2b8", "#d8881a"),    # amber: present, params differ
    "absent": ("#f3f3f3", "#cfcfcf"),  # grey: generator missing here
    "novel": ("#cfe2f7", "#3b78c2"),   # blue: present, absent in reference
}
_VERDICT_COLOUR = {"match": "#5a9e4e", "param-diff": "#d8881a", "partial": "#7a7a7a"}


def comparison_matrix(report: ComparisonReport, path: str | None = None, *, title: str | None = None):
    """Draw the comparison matrix; save to ``path`` (``.svg``/``.png``) if given and
    return the Figure."""
    cols = list(report.notations)
    rows = list(report.rows)
    n_cols = len(cols)

    # cell texts up front, so column/label widths can be sized to the content (no overlap)
    cell_txt = {(i, n): ("—" if cell.profile is None else cell.profile.render())
                for i, r in enumerate(rows) for n, cell in r.cells}
    max_label = max((len(r.key.label) for r in rows), default=6)
    lbl_indent = max(1.7, 0.12 * max_label + 0.35)    # arity starts clear of the longest bold label
    max_arity = max((len(r.key.arity_str()) for r in rows), default=8)
    lbl_w = lbl_indent + 0.10 * max_arity + 0.3       # label column width, content-sized
    max_cell_chars = max((len(t) for t in cell_txt.values()), default=4)
    cell_w = max(2.6, 0.40 + 0.105 * max_cell_chars)  # cell width, content-sized
    cell_h = 0.66
    head_h = 0.74
    foot_h = 1.9
    has_boundary = any(r.key.boundary for r in rows)
    total_w = lbl_w + n_cols * cell_w
    total_h = head_h + len(rows) * cell_h + foot_h + (cell_h if has_boundary else 0)

    fig, ax = plt.subplots(figsize=(total_w * 0.64, total_h * 0.64))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.axis("off")

    def cell_fontsize(txt: str) -> float:
        return 7.6 if len(txt) <= 18 else (6.6 if len(txt) <= 28 else 5.8)

    # header row
    ax.text(0.12, head_h / 2, "generator", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(lbl_indent, head_h / 2, "(in → out)", ha="left", va="center", fontsize=8, color="#555")
    for j, n in enumerate(cols):
        x = lbl_w + j * cell_w
        is_ref = n == report.reference
        ax.add_patch(mpatches.Rectangle((x, 0), cell_w, head_h, facecolor="#33414f" if is_ref else "#5b6b7a",
                                        edgecolor="white"))
        ax.text(x + cell_w / 2, head_h / 2, n + ("  (ref)" if is_ref else ""), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    y = head_h

    boundary_drawn = False
    for i, r in enumerate(rows):
        if r.key.boundary and not boundary_drawn:
            ax.add_patch(mpatches.Rectangle((0, y), total_w, cell_h, facecolor="#e8e8ee", edgecolor="white"))
            ax.text(0.12, y + cell_h / 2, "boundary generators  (start/end — adapter-specific encoding)",
                    ha="left", va="center", fontsize=8, style="italic", color="#555")
            y += cell_h
            boundary_drawn = True

        vcol = _VERDICT_COLOUR[r.verdict()]
        ax.add_patch(mpatches.Rectangle((0, y), lbl_w, cell_h, facecolor="white", edgecolor="#dddddd"))
        ax.text(0.12, y + cell_h / 2, r.key.label, ha="left", va="center", fontsize=8.5,
                fontweight="bold", color=vcol)
        ax.text(lbl_indent, y + cell_h / 2, r.key.arity_str(), ha="left", va="center",
                fontsize=7.3, color="#444")

        for j, (n, cell) in enumerate(r.cells):
            x = lbl_w + j * cell_w
            fill, edge = _STATUS_COLOUR[cell.status]
            ax.add_patch(mpatches.Rectangle((x, y), cell_w, cell_h, facecolor=fill, edgecolor=edge, lw=1.0))
            txt = cell_txt[(i, n)]
            ax.text(x + cell_w / 2, y + cell_h / 2, txt, ha="center", va="center",
                    fontsize=cell_fontsize(txt), color="#222" if cell.profile is not None else "#aaa")
        y += cell_h

    # footer: colour legend (one row, evenly spaced) + per-notation summary
    y += 0.3
    legend = [
        ("match", "rederived — agree"),
        ("diff", "present — params differ"),
        ("absent", "generator absent"),
        ("novel", "present — absent in ref"),
    ]
    slot = total_w / len(legend)
    for k, (status, desc) in enumerate(legend):
        x = 0.12 + k * slot
        fill, edge = _STATUS_COLOUR[status]
        ax.add_patch(mpatches.Rectangle((x, y), 0.42, cell_h * 0.72, facecolor=fill, edgecolor=edge))
        ax.text(x + 0.56, y + cell_h * 0.36, desc, ha="left", va="center", fontsize=7.4)
    y += cell_h + 0.15

    pn = report.per_notation()
    detail = "      ".join(
        f"{n}: {t['match']} rederived / {t['diff']} param-diff / {t['absent']} absent / {t['novel']} novel"
        for n, t in pn.items()
    )
    ax.text(0.12, y + 0.2, f"vs reference {report.reference} —   {detail}",
            ha="left", va="center", fontsize=8, color="#333", fontweight="bold")
    ax.text(0.12, y + 0.2 + cell_h,
            "↓ input leg, ↑ output leg; ranges are objects-per-firing (§32); "
            "boundary generators encode start/end per-adapter",
            ha="left", va="center", fontsize=7.2, color="#777")

    ax.set_title(title or f"signature comparison — {n_cols} notations, {len(rows)} canonical generators",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig
