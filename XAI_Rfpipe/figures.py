"""
XAI_Rfpipe.figures - figures built from stage artefacts, not from memory.

Only figures whose content is an experimental result belong here. The study-area
and methodology figures are cartography and schematics; they live with the
manuscript sources and are not reproducible from these CSVs.

Fig. 5, the signed lag-feature attribution map, is a result: every cell is a
number this pipeline computed, so the figure is regenerated from
07_attributions.csv rather than kept as an image whose provenance has to be
taken on trust.

Reading the map
---------------
Rows are the annual descriptors entering the feature matrix, columns the lagged
hydrological years. Panels (a) and (b) give normalised signed attributions from
the LIME-style local surrogate and from Shapley sampling, averaged over models
and targets. Panel (c) keeps only the cells where the two methods agree in
direction, which is the honest summary: where they disagree there is no
attribution to report.

Three things the map depends on, each stated because it changes how a cell reads
-------------------------------------------------------------------------------
* A blank cell means no attribution, never "not computed". This holds only while
  `run_xai` samples Shapley over every feature (`shapley_top_k=None`). With a
  top-k restriction most of panel (b) is empty for the second reason, and the
  figure silently lies. `build_fig5` refuses to draw rather than allow that.
* Integrated Gradients is absent because it needs a model gradient and this
  pipeline has no deep-learning dependency. Permutation importance is computed
  but is a magnitude, not a direction, so it cannot carry a signed map. The two
  methods shown are the two that are both signed and model-agnostic.
* Each target contributes its own autoregressive lags under different names
  (D_KL_lag1 and so on). These are pooled into one row, since across targets
  they play the same role.

Diverging scale: two hues with a neutral midpoint, never a rainbow. Zero is
exactly neutral, so a cell's hue reads as direction and its intensity as weight.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
INK, INK2, RULE = "#15171a", "#4b5056", "#c9c7c0"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#eceae5"

CMAP = LinearSegmentedColormap.from_list(
    "shift_div", [BLUE, "#9dc0ea", NEUTRAL, "#f5b89c", ORANGE], N=256)

# Coverage below this fraction means the Shapley panel is mostly holes, and a
# blank cell can no longer be read as "no attribution".
MIN_SHAPLEY_COVERAGE = 0.95


def build_fig5(out: Path, targets=("D_KL", "D_B", "W1"), stations=None,
               stem="Fig5_xai_heatmap", strict=True) -> dict:
    """Signed lag-feature attribution map from 07_attributions.csv.

    `stations` defaults to every station present. With `strict`, incomplete
    Shapley coverage raises instead of producing a misleading panel.
    """
    out = Path(out)
    src = out / "07_attributions.csv"
    if not src.exists():
        return dict(written=[], reason="07_attributions.csv not found")

    a = pd.read_csv(src)
    if stations:
        a = a[a["station"].isin(stations)]
    if a.empty:
        return dict(written=[], reason="no attribution rows")

    cov = float(a["shapley"].notna().mean())
    if cov < MIN_SHAPLEY_COVERAGE:
        msg = (f"Shapley coverage is {cov:.1%} of cells. Panel (b) would be mostly "
               f"empty, and an empty cell would mean 'not computed' rather than "
               f"'no attribution'. Rerun the xai stage with shapley_top_k=None.")
        if strict:
            raise ValueError(msg)
        return dict(written=[], reason=msg, shapley_coverage=cov)

    lag = a["feature"].str.extract(r"_lag(\d+)$")[0].astype(float)
    base = a["feature"].str.replace(r"_lag\d+$", "", regex=True)
    a = a.assign(lag=lag,
                 row=np.where(base.isin(set(targets)), "shift target (own lag)", base))
    a = a.dropna(subset=["lag"])
    a["lag"] = a["lag"].astype(int)

    def grid(col):
        g = a.pivot_table(index="row", columns="lag", values=col, aggfunc="mean")
        m = np.nanmax(np.abs(g.to_numpy()))
        return g / m if m and np.isfinite(m) else g

    G_lime, G_shap = grid("lime"), grid("shapley")
    G_shap = G_shap.reindex(index=G_lime.index, columns=G_lime.columns)
    order = (G_lime.abs().mean(axis=1).fillna(0)
             + G_shap.abs().mean(axis=1).fillna(0)).sort_values(ascending=False).index
    G_lime, G_shap = G_lime.loc[order], G_shap.loc[order]

    agree = np.sign(G_lime.to_numpy()) == np.sign(G_shap.to_numpy())
    G_agree = pd.DataFrame(
        np.where(agree, np.sign(G_lime.to_numpy())
                 * np.minimum(np.abs(G_lime.to_numpy()), np.abs(G_shap.to_numpy())),
                 np.nan),
        index=G_lime.index, columns=G_lime.columns)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.9), dpi=600, facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.14, left=0.235, right=0.90,
                                              top=0.93, bottom=0.10))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    titles = ["a   LIME surrogate", "b   Shapley sampling", "c   Direction agreed"]

    for ax, G, title in zip(axes, [G_lime, G_shap, G_agree], titles):
        ax.set_facecolor("#f4f2ee")
        ax.imshow(G.to_numpy(), cmap=CMAP, norm=norm, aspect="auto")
        ax.set_xticks(range(G.shape[1]))
        ax.set_xticklabels([f"$t-{c}$" for c in G.columns], fontsize=8.8, color=INK2)
        ax.set_title(title, fontsize=9.8, color=INK, loc="left", pad=8,
                     fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_color(RULE); sp.set_linewidth(0.6)
        # a surface-coloured gap between cells, so adjacent fills never touch
        ax.set_xticks(np.arange(-0.5, G.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, G.shape[0], 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        ax.set_yticks(range(len(G_lime.index)))
        ax.set_yticklabels([])

    axes[0].set_yticklabels(G_lime.index, fontsize=7, color=INK)
    n_agree = int(np.nansum(~np.isnan(G_agree.to_numpy())))
    n_cell = int(G_agree.size)
    axes[2].set_xlabel(f"{n_agree} of {n_cell} cells agree", fontsize=8.6,
                       color=INK2, labelpad=7)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
                      fraction=0.028, pad=0.015, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["$-1$\nlowers", "0", "$+1$\nraises"], fontsize=8.6,
                          color=INK2)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(length=0)

    written = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        # Suppress the embedded creation timestamp, so that rebuilding the
        # figure from the same artefacts gives a byte-identical file and a
        # reproduction check is not defeated by the clock.
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(p, dpi=600, facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.05, metadata=meta)
        written.append(str(p))
    plt.close(fig)

    return dict(written=written, shapley_coverage=cov, rows=int(G_lime.shape[0]),
                lags=[int(c) for c in G_lime.columns],
                cells_agreeing=n_agree, cells=n_cell,
                agreement_fraction=round(n_agree / n_cell, 4) if n_cell else None)


# ──────────────────────────────────── the resolvability figure (methodology)
def build_resolvability_figure(out: Path, designs=None, stem="Fig_resolvability") -> dict:
    """Why a family win count needs a candidate-aware null, in one figure.

    Panels a and b put each run's observed count against two nulls: the fair
    coin that the statistic is usually read against, and the exchangeability
    null of eq. (1), whose mean is set by how many candidates each family
    brought. Panel c is the design view - at what number of blocks either
    family claim first becomes attainable - and is computable before any model
    is fit.

    `designs` is a list of (label, k_a, k_b, n_blocks, observed_wins); the
    default is this study's two runs. Nothing here reads the record, so the
    figure is a property of the designs, not of the rainfall.
    """
    from scipy import stats as _st
    from . import resolvability as _rv

    out = Path(out)
    designs = designs or [
        ("a   Original run\n     5 neural vs 8 conventional", 5, 8, 9, 6),
        ("b   Reimplementation\n     3 neural vs 11 conventional", 3, 11, 9, 0)]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), dpi=600, facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.34, left=0.065, right=0.985,
                                              top=0.80, bottom=0.20))

    for ax, (label, ka, kb, n, obs) in zip(axes, designs):
        p0 = _rv.family_win_probability(ka, kb)
        x = np.arange(n + 1)
        aware = _st.binom.pmf(x, n, p0)
        naive = _st.binom.pmf(x, n, 0.5)
        ax.set_facecolor(SURFACE)
        ax.bar(x, aware, width=0.62, color=BLUE, alpha=0.9, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
        ax.step(np.append(x, n + 1) - 0.5, np.append(naive, naive[-1]),
                where="post", color=ORANGE, lw=1.6, zorder=4)
        top = max(aware.max(), naive.max())
        # a capped rule, not axvline: a full-height line would strike through
        # the p-value annotations in the band above the distribution
        ax.vlines(obs, 0, top * 1.06, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=5)
        ax.annotate(f"observed {obs}", xy=(obs, 0), xytext=(4, 4),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=7.2, color=INK,
                    bbox=dict(fc=SURFACE, ec="none", pad=1.0))
        t = _rv.family_win_test(obs, n, ka, kb)
        ax.annotate(f"candidate-aware $p$ = {t['p_two_sided']:.2f}",
                    xy=(0.03, 0.99), xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.2, color=BLUE)
        ax.annotate(f"fair coin $p$ = {t['naive_p_two_sided']:.3f}",
                    xy=(0.03, 0.87), xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.2, color=ORANGE)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(x[::2])
        ax.set_xlabel("blocks won by the neural family", fontsize=7.6, color=INK2)
        ax.set_title(label, fontsize=8.2, color=INK, loc="left", pad=7,
                     fontweight="bold", linespacing=1.5)
        ax.tick_params(labelsize=7.2, length=2.5, color=RULE, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
    axes[0].set_ylabel("probability under the null", fontsize=7.6, color=INK2)

    # panel c: when either claim first becomes attainable, by design
    c = axes[2]
    c.set_facecolor(SURFACE)
    ns = np.arange(2, 41)
    for (lab, ka, kb, col, ls) in [("3 vs 11", 3, 11, BLUE, "-"),
                                   ("5 vs 8", 5, 8, ORANGE, (0, (4, 2)))]:
        hi, lo = [], []
        for n in ns:
            d = _rv.detectable_win_counts(int(n), ka, kb)
            hi.append(d["min_significant_wins"] if d["min_significant_wins"] is not None else np.nan)
            lo.append(d["max_significant_losses"] if d["max_significant_losses"] is not None else np.nan)
        c.plot(ns, np.array(hi) / ns, color=col, lw=1.8, ls=ls, zorder=4)
        c.plot(ns, np.array(lo) / ns, color=col, lw=1.8, ls=ls, zorder=4, alpha=0.55)
        c.annotate(lab, xy=(ns[-1], hi[-1] / ns[-1]), xytext=(-1, 6),
                   textcoords="offset points", ha="right", fontsize=7.2, color=col)
    c.axvline(9, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=5)
    c.annotate("9 blocks here", xy=(9, 1.02), xytext=(4, 0),
               textcoords="offset points", ha="left", va="top", fontsize=7.2,
               color=INK)
    c.annotate("to claim “better”", xy=(19.5, 0.63), fontsize=7.0,
               color=INK2, ha="left", va="bottom")
    c.annotate("to claim “worse”", xy=(19.5, 0.20), fontsize=7.0,
               color=INK2, ha="left", va="top")
    c.set_xlim(2, 40)
    c.set_ylim(0, 1.05)
    c.set_xlabel("blocks in the design", fontsize=7.6, color=INK2)
    c.set_ylabel("win share needed at $\\alpha=0.05$", fontsize=7.6, color=INK2)
    c.set_title("c   What a design can resolve", fontsize=8.2, color=INK,
                loc="left", pad=7, fontweight="bold")
    c.tick_params(labelsize=7.2, length=2.5, color=RULE, colors=INK2)
    c.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)
    c.set_axisbelow(True)
    for sp in ("top", "right"):
        c.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        c.spines[sp].set_color(RULE)

    written = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=600, facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.04,
                    metadata={"CreationDate": None} if ext == "pdf" else {})
        written.append(str(p))
    plt.close(fig)
    return dict(written=written, designs=[d[0].split("\n")[0] for d in designs])


def build_error_rate_figure(out: Path, stem="Fig_error_rate", alpha=0.05) -> dict:
    """How often the fair-coin reading rejects a true null, by design.

    Candidate share is an ordered quantity, so the lines take a single-hue
    sequential ramp rather than categorical colours; the balanced design and
    the demonstration's own design are called out against it. Nothing here
    reads any data: every value is an exact finite sum over the support of W.
    """
    from . import resolvability as _rv

    out = Path(out)
    ns = np.unique(np.round(np.geomspace(5, 200, 28)).astype(int))
    # (pool, label, colour, linewidth, x at which to place the label). Three of
    # the six saturate at 1.0, so they are labelled where they are still apart
    # rather than stacked on top of each other at the right edge.
    POOLS = [((1, 1), "1 vs 1", "#9aa0a6", 1.9, None),
             ((2, 3), "2 vs 3", "#a8c6ef", 1.7, None),
             ((5, 8), "5 vs 8", "#6ea3e0", 1.7, None),
             ((3, 11), "3 vs 11", ORANGE, 2.3, 24),
             ((1, 9), "1 vs 9", "#2f6fc4", 1.7, 11),
             ((1, 18), "1 vs 18", "#1b4a8a", 1.7, 7)]

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=600, facecolor=SURFACE,
                           gridspec_kw=dict(left=0.085, right=0.80, top=0.90,
                                            bottom=0.155))
    ax.set_facecolor(SURFACE)
    ax.axhline(alpha, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"nominal $\\alpha$ = {alpha}", xy=(5.2, alpha), xytext=(0, 5),
                textcoords="offset points", fontsize=7.4, color=INK2, va="bottom")

    for (ka, kb), lab, col, lw, at_x in POOLS:
        y = np.array([_rv.fair_coin_error_rate(int(n), ka, kb, alpha)["fair_coin_type_I"]
                      for n in ns])
        ax.plot(ns, y, color=col, lw=lw, zorder=4, solid_capstyle="round")
        bold = (lab == "3 vs 11")
        if at_x is None:
            ax.annotate(lab, xy=(ns[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7.6, color=col,
                        va="center", fontweight="bold" if bold else "normal")
        else:
            i = int(np.argmin(np.abs(ns - at_x)))
            ax.annotate(lab, xy=(ns[i], y[i]), xytext=(4, -11),
                        textcoords="offset points", fontsize=7.6, color=col,
                        ha="left", va="top",
                        fontweight="bold" if bold else "normal",
                        bbox=dict(fc=SURFACE, ec="none", pad=1.2))

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(5, 215)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("blocks in the design", fontsize=8.4, color=INK2)
    ax.set_ylabel("probability of rejecting\na true null", fontsize=8.4,
                  color=INK2, linespacing=1.5)
    ax.set_title("What the fair-coin reading does when no model is better than any other",
                 fontsize=9.2, color=INK, loc="left", pad=9, fontweight="bold")
    ax.tick_params(labelsize=7.6, length=2.5, color=RULE, colors=INK2)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)

    written = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=600, facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.04,
                    metadata={"CreationDate": None} if ext == "pdf" else {})
        written.append(str(p))
    plt.close(fig)
    return dict(written=written, designs=[p[1] for p in POOLS])


def build_power_figure(out: Path, stem="Fig_power", k_a=3, k_b=11, alpha=0.05) -> dict:
    """Power against the alternative, for several block counts.

    Attainability is a property of the design; power needs an alternative.
    Plotting them together is the point of the figure: the vertical rule marks
    the null, and the distance from it to a usable power level is the effect a
    design of that size can actually find.
    """
    from . import robustness as _rb
    from .resolvability import family_win_probability

    out = Path(out)
    p0 = family_win_probability(k_a, k_b)
    fig, ax = plt.subplots(figsize=(7.0, 3.3), dpi=600, facecolor=SURFACE,
                           gridspec_kw=dict(left=0.085, right=0.82, top=0.90,
                                            bottom=0.17))
    ax.set_facecolor(SURFACE)
    # labels are placed where the curves are still apart, not at the right
    # edge where all three saturate
    for n, col, lw, at in [(9, ORANGE, 2.3, 0.62), (19, "#2f6fc4", 1.8, 0.47),
                           (50, "#a8c6ef", 1.8, 0.33)]:
        d = _rb.power_curve(n, k_a, k_b, alpha)
        ax.plot(d["alt_p"], d["power"], color=col, lw=lw, zorder=4)
        i = int((d["alt_p"] - at).abs().idxmin())
        ax.annotate(f"$N={n}$", xy=(d["alt_p"].iloc[i], d["power"].iloc[i]),
                    xytext=(5, -10), textcoords="offset points", fontsize=7.6,
                    color=col, ha="left", va="top",
                    fontweight="bold" if n == 9 else "normal",
                    bbox=dict(fc=SURFACE, ec="none", pad=1.2))
    # the flat left tail of the N=9 curve IS the attainability result; it is
    # explained in the caption rather than annotated across three curves
    ax.axvline(p0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.annotate(f"null $p_0$ = {p0:.3f}", xy=(p0, 1.02), xytext=(4, 0),
                textcoords="offset points", fontsize=7.4, color=INK2, va="top")
    ax.axhline(0.8, color=RULE, lw=1.0, zorder=2)
    ax.annotate("80% power", xy=(0.02, 0.81), fontsize=7.2, color=INK2,
                va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("alternative win probability $p_1$", fontsize=8.4, color=INK2)
    ax.set_ylabel("power", fontsize=8.4, color=INK2)
    ax.set_title(f"Power of the candidate-aware test, {k_a} versus {k_b} candidates",
                 fontsize=9.2, color=INK, loc="left", pad=9, fontweight="bold")
    ax.tick_params(labelsize=7.6, length=2.5, color=RULE, colors=INK2)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    written = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=600, facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.04,
                    metadata={"CreationDate": None} if ext == "pdf" else {})
        written.append(str(p))
    plt.close(fig)
    return dict(written=written)
