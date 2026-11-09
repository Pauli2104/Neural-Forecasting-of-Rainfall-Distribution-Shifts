"""
XAI_Rfpipe.proxy - do the shift targets track documented flooding?

The flood catalogue is the only external reference in the study, so this stage is
the one external check on whether the divergence targets mean anything
hydrologically. It is deliberately conservative:

* discrimination is measured by AUROC of each annual indicator against the binary
  flood-year label, pooled across stations and also per station;
* significance comes from a permutation test on the labels, not from a parametric
  approximation, because 129 station-years with ~30% positives is small;
* raw descriptors (RX1day, PRCPTOT and the rest) are scored on exactly the same
  footing as the divergence targets. If a raw descriptor discriminates better
  than a shift target, that is the finding and it is reported as such.

The catalogue's spatial support is wider than the grid cell that supplies the
predictors, so the result is a weak external check, not a validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def auroc(score, label) -> float:
    """Rank-based AUROC (equivalent to the Mann-Whitney U statistic)."""
    s = np.asarray(score, float); y = np.asarray(label, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def permutation_p(score, label, n_perm=10000, seed=0) -> tuple[float, float]:
    """Two-sided permutation p-value for AUROC against 0.5."""
    rng = np.random.default_rng(seed)
    obs = auroc(score, label)
    if not np.isfinite(obs):
        return float("nan"), float("nan")
    y = np.asarray(label, int).copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(y)
        null[i] = auroc(score, y)
    p = float((np.abs(null - 0.5) >= abs(obs - 0.5)).mean())
    return obs, p


def bootstrap_auroc_ci(score, label, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(score, float); y = np.asarray(label, int)
    n = len(s)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(auroc(s[idx], y[idx]))
    if len(draws) < 50:
        return float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def build_panel(targets: pd.DataFrame, desc: pd.DataFrame,
                labels: pd.DataFrame) -> pd.DataFrame:
    """One row per station-year carrying every candidate indicator and the label."""
    p = targets.merge(desc.drop(columns=["n_days"], errors="ignore"),
                      on=["station", "hydro_year"], how="inner")
    return p.merge(labels, on=["station", "hydro_year"], how="inner")


def discriminate(panel: pd.DataFrame, indicators=None, n_perm=10000,
                 n_boot=2000, seed=0) -> pd.DataFrame:
    """AUROC per indicator, pooled and per station, with permutation p and CI."""
    skip = {"station", "hydro_year", "flood", "n_days"}
    indicators = indicators or [c for c in panel.columns
                                if c not in skip and panel[c].dtype.kind in "fi"]
    rows = []
    for ind in indicators:
        for scope, sub in [("pooled", panel)] + list(panel.groupby("station")):
            s = sub[ind].to_numpy(float)
            y = sub["flood"].to_numpy(int)
            if np.unique(y).size < 2 or np.isfinite(s).sum() < 8:
                continue
            a, p = permutation_p(s, y, n_perm if scope == "pooled" else n_perm // 5,
                                 seed)
            lo, hi = bootstrap_auroc_ci(s, y, n_boot if scope == "pooled"
                                        else n_boot // 4, 0.05, seed)
            rows.append(dict(indicator=ind, scope=scope, n=int(len(y)),
                             n_flood_years=int(y.sum()), auroc=a,
                             auroc_lo=lo, auroc_hi=hi, p_perm=p,
                             kind=("target" if ind in ("D_KL", "D_B", "W1", "EMD")
                                   else "descriptor")))
    out = pd.DataFrame(rows)
    if len(out):
        from .statistics import holm
        m = out["scope"] == "pooled"
        out.loc[m, "p_holm"] = holm(out.loc[m, "p_perm"])
        out["significant"] = out.get("p_holm", out["p_perm"]) < 0.05
    return out.sort_values(["scope", "auroc"], ascending=[True, False])


def best_vs_targets(disc: pd.DataFrame) -> dict:
    """The headline comparison: does any raw descriptor beat every shift target?"""
    p = disc[disc["scope"] == "pooled"]
    if p.empty:
        return {}
    tg = p[p["kind"] == "target"].sort_values("auroc", ascending=False)
    ds = p[p["kind"] == "descriptor"].sort_values("auroc", ascending=False)
    if tg.empty or ds.empty:
        return {}
    bt, bd = tg.iloc[0], ds.iloc[0]
    return dict(best_target=bt["indicator"], best_target_auroc=float(bt["auroc"]),
                best_target_p=float(bt["p_perm"]),
                best_descriptor=bd["indicator"],
                best_descriptor_auroc=float(bd["auroc"]),
                best_descriptor_p=float(bd["p_perm"]),
                descriptor_beats_all_targets=bool(bd["auroc"] > tg["auroc"].max()),
                margin=float(bd["auroc"] - bt["auroc"]))
