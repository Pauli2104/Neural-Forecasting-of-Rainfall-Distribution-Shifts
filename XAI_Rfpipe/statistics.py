"""
XAI_Rfpipe.statistics - the comparison protocol.

Omnibus first, post-hoc only if the omnibus rejects. Effect sizes accompany every
p-value, and multiplicity is controlled explicitly. The one function here that
does name per-block winners, `family_win_counts`, counts its own denominator and
returns the critical difference beside the count, so the count cannot be quoted
without the reason it is unreliable.

Protocol
--------
1. Friedman test over models ranked within each station x target block.
2. Iman-Davenport F correction, which is less conservative than Friedman's
   chi-square for small numbers of blocks.
3. Nemenyi critical difference for all-pairs comparison (Demsar 2006).
4. Wilcoxon signed-rank for named pairwise contrasts, with rank-biserial effect
   size and Holm correction.
5. Tukey HSD across algorithm families on the raw metric.
6. Per-block family win counts, reported with their denominator derived from the
   data and with the critical difference attached.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy import stats

# Nemenyi critical values q_alpha for alpha = 0.05, infinite degrees of freedom,
# indexed by the number of compared methods (Demsar 2006, Table 5).
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
       15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def block_matrix(res: pd.DataFrame, metric="test_MASE", agg="median") -> pd.DataFrame:
    """Blocks (station x target) by models, aggregated across seeds."""
    d = res.dropna(subset=[metric]).copy()
    d["block"] = d["station"] + " / " + d["target"]
    m = d.pivot_table(index="block", columns="model", values=metric, aggfunc=agg)
    return m.dropna(axis=1, how="any")


def friedman(mat: pd.DataFrame) -> dict:
    """Friedman omnibus plus the Iman-Davenport correction."""
    k = mat.shape[1]
    n = mat.shape[0]
    if k < 3 or n < 2:
        return dict(error="need >=3 models and >=2 blocks", k=k, n=n)
    chi2, p = stats.friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    # Iman-Davenport
    denom = n * (k - 1) - chi2
    F = ((n - 1) * chi2 / denom) if denom > 0 else np.inf
    p_F = float(stats.f.sf(F, k - 1, (k - 1) * (n - 1))) if np.isfinite(F) else 0.0
    ranks = mat.rank(axis=1).mean(axis=0)
    return dict(n_blocks=int(n), k_models=int(k),
                friedman_chi2=float(chi2), friedman_p=float(p),
                iman_davenport_F=float(F), iman_davenport_p=p_F,
                df1=int(k - 1), df2=int((k - 1) * (n - 1)),
                mean_ranks=ranks.sort_values().to_dict())


def nemenyi(mat: pd.DataFrame, alpha=0.05) -> tuple[float, pd.DataFrame]:
    """Critical difference and the all-pairs rank-difference matrix."""
    k, n = mat.shape[1], mat.shape[0]
    q = Q05.get(k)
    if q is None or n < 2:
        return float("nan"), pd.DataFrame()
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    ranks = mat.rank(axis=1).mean(axis=0)
    rows = []
    for a, b in itertools.combinations(ranks.index, 2):
        diff = abs(ranks[a] - ranks[b])
        rows.append(dict(model_a=a, model_b=b, rank_a=ranks[a], rank_b=ranks[b],
                         rank_diff=diff, significant=bool(diff > cd)))
    return float(cd), pd.DataFrame(rows).sort_values("rank_diff", ascending=False)


def rank_biserial(x, y) -> float:
    """Matched-pairs rank-biserial correlation: the Wilcoxon effect size."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum(); rm = r[d < 0].sum()
    return float((rp - rm) / r.sum())


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def pairwise_wilcoxon(mat: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    """Signed-rank contrasts, with effect size and Holm correction.

    With `reference`, every model is compared against that one; otherwise all
    pairs are compared.
    """
    cols = list(mat.columns)
    pairs = ([(reference, c) for c in cols if c != reference] if reference
             else list(itertools.combinations(cols, 2)))
    rows = []
    for a, b in pairs:
        x, y = mat[a].to_numpy(), mat[b].to_numpy()
        if len(x) < 3 or np.allclose(x, y):
            continue
        try:
            W, p = stats.wilcoxon(x, y)
        except ValueError:
            continue
        rows.append(dict(model_a=a, model_b=b, n=len(x), W=float(W), p=float(p),
                         median_diff=float(np.median(x - y)),
                         rank_biserial=rank_biserial(x, y)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        out["significant_holm"] = out["p_holm"] < 0.05
    return out.sort_values("p") if len(out) else out


def family_tukey(res: pd.DataFrame, metric="test_MASE", by="model") -> pd.DataFrame:
    """Tukey HSD across groups on the raw metric."""
    d = res.dropna(subset=[metric])
    groups = [g[metric].to_numpy(float) for _, g in d.groupby(by) if len(g) > 1]
    names = [k for k, g in d.groupby(by) if len(g) > 1]
    if len(groups) < 3:
        return pd.DataFrame()
    r = stats.tukey_hsd(*groups)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lo, hi = r.confidence_interval().low[i, j], r.confidence_interval().high[i, j]
            rows.append(dict(group_a=names[i], group_b=names[j],
                             statistic=float(r.statistic[i, j]),
                             p=float(r.pvalue[i, j]), ci_lo=float(lo), ci_hi=float(hi),
                             significant=bool(r.pvalue[i, j] < 0.05)))
    return pd.DataFrame(rows).sort_values("p")


def mannwhitney_family(res: pd.DataFrame, metric="test_MASE") -> dict:
    """Neural versus conventional, as an unpaired contrast."""
    a = res.loc[res["family"] == "neural", metric].dropna().to_numpy()
    b = res.loc[res["family"] == "conventional", metric].dropna().to_numpy()
    if len(a) < 3 or len(b) < 3:
        return {}
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    cles = U / (len(a) * len(b))              # common-language effect size
    return dict(n_neural=len(a), n_conventional=len(b),
                median_neural=float(np.median(a)),
                median_conventional=float(np.median(b)),
                U=float(U), p=float(p), cles_neural_better=float(1 - cles))


def family_win_counts(res: pd.DataFrame, metric="test_MASE", agg="median",
                      neural: set[str] | None = None,
                      alt_conventional: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict]:
    """Per-block best-neural versus best-conventional, with its own health warning.

    This statistic is reported because readers ask for it, not because it is
    sound. Three precautions are built in, each answering a way the count has
    gone wrong in this project before.

    1. The denominator is COUNTED from the block matrix, never assumed. The
       design has len(stations) x len(targets) blocks; writing "of twelve" for
       a three-station, three-target design is an arithmetic error, and it is
       the error that produced two of the three conflicting counts this file
       exists to prevent.
    2. A model whose family assignment is arguable - a flattened MLP that is
       neural by construction and a conventional baseline in behaviour - is
       counted BOTH ways via `alt_conventional`, rather than one way silently.
    3. The count is reported next to the Nemenyi critical difference. When
       several models sit inside one CD of the best mean rank, which of them
       happens to win a given block is noise, and `resolvable` is False. Quote
       the count only together with that flag.

    Returns (per-block table, summary dict).
    """
    d = res.dropna(subset=[metric]).copy()
    if "family" not in d.columns and neural is None:
        raise ValueError("need a `family` column in res or an explicit `neural` set")
    mat = block_matrix(d, metric=metric, agg=agg)
    if mat.empty:
        return pd.DataFrame(), {}

    if neural is None:
        fam = (d.drop_duplicates("model").set_index("model")["family"].to_dict())
        neural = {m for m, f in fam.items() if f == "neural"}
    neural = {m for m in neural if m in mat.columns}

    def _tally(neural_set):
        conv = [c for c in mat.columns if c not in neural_set]
        neu = [c for c in mat.columns if c in neural_set]
        if not neu or not conv:
            return pd.DataFrame(), {}
        rows = []
        for blk, r in mat.iterrows():
            bn, bc = r[neu].min(), r[conv].min()
            rows.append(dict(
                block=blk,
                best_neural_model=r[neu].idxmin(), best_neural=float(bn),
                best_conventional_model=r[conv].idxmin(), best_conventional=float(bc),
                winner=("neural" if bn < bc else "conventional" if bc < bn else "tie"),
                margin=float(bc - bn)))
        t = pd.DataFrame(rows)
        return t, dict(n_blocks=int(len(t)),
                       neural_wins=int((t["winner"] == "neural").sum()),
                       conventional_wins=int((t["winner"] == "conventional").sum()),
                       ties=int((t["winner"] == "tie").sum()))

    per_block, counts = _tally(neural)
    if per_block.empty:
        return per_block, {}

    cd, _ = nemenyi(mat)
    ranks = mat.rank(axis=1).mean(axis=0)
    within = int((ranks - ranks.min() <= cd).sum()) if np.isfinite(cd) else -1

    summary = dict(
        metric=metric, aggregation=agg,
        n_stations=int(d["station"].nunique()), n_targets=int(d["target"].nunique()),
        **counts,
        neural_models=sorted(neural),
        critical_difference=float(cd),
        n_models_within_cd_of_best=within,
        resolvable=bool(within == 1),
        caveat=("the leading models are separated by less than one critical "
                "difference, so a per-block win count reflects noise, not ordering"
                if within != 1 else
                "the best model is separated from the rest by more than one "
                "critical difference"))

    moved = [m for m in alt_conventional if m in neural]
    absent = [m for m in alt_conventional if m not in neural]
    if moved:
        _, alt = _tally(neural - set(moved))
        summary["alternate_classification"] = dict(
            moved_to_conventional=moved, **{k: v for k, v in alt.items()
                                            if k != "n_blocks"})
    if absent:
        # Recorded rather than ignored: a reclassification that silently matches
        # nothing looks exactly like one that was applied and changed nothing.
        summary["alternate_classification_skipped"] = dict(
            not_in_this_model_zoo=absent)
    return per_block, summary


def kendall_trends(desc: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Monotonic trend per station per descriptor (Mann-Kendall via Kendall tau)."""
    columns = columns or ["PRCPTOT", "wet_days", "SDII", "RX1day", "CDD", "CWD",
                          "R10mm", "R20mm", "R95p"]
    rows = []
    for st, g in desc.groupby("station"):
        x = g["hydro_year"].to_numpy(float)
        for c in columns:
            if c not in g:
                continue
            y = g[c].to_numpy(float)
            ok = np.isfinite(y)
            if ok.sum() < 8:
                continue
            tau, p = stats.kendalltau(x[ok], y[ok])
            sl = stats.theilslopes(y[ok], x[ok])[0]
            rows.append(dict(station=st, descriptor=c, n=int(ok.sum()),
                             kendall_tau=float(tau), p=float(p),
                             theil_sen_per_decade=float(sl * 10)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        out["significant_holm"] = out["p_holm"] < 0.05
    return out.sort_values("p")
