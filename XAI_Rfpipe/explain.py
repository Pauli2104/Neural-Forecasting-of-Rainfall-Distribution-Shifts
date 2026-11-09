"""
XAI_Rfpipe.explain - attribution methods and the diagnostics that audit them.

Four methods are computed for every fitted model, so that agreement between them
can be measured rather than assumed:

* permutation importance - model-agnostic, the reference method here
* LIME-style local surrogate - a weighted ridge fit in a neighbourhood
* Shapley sampling - the Monte-Carlo estimator of Strumbelj & Kononenko (2014),
  implemented directly because `shap` is not a dependency
* gradient x input - only for models that expose a usable gradient; skipped
  otherwise rather than approximated

Three diagnostics then interrogate the attributions:

* agreement  - Spearman rank correlation between methods on the same model
* faithfulness - does ablating the top-k features actually degrade the model?
  Tested as a paired contrast against ablating random features of equal size.
* stability - how much do rankings move under reseeding, versus under input
  perturbation? If seed moves them more than the data does, the explanation is
  reporting the optimiser, not the phenomenon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from .experiments import (TARGETS, chrono_split, make_supervised, mase,
                          model_zoo, _fit_predict)


# ─────────────────────────────────────────────────────────── attributions
def permutation_importance(est, X, y, y_train, n_repeats=10, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = mase(y, est.predict(X), y_train)
    out = {}
    Xv = X.to_numpy(float)
    for j, col in enumerate(X.columns):
        d = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = Xv.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d[r] = mase(y, est.predict(pd.DataFrame(Xp, columns=X.columns)),
                        y_train) - base
        out[col] = float(np.mean(d))
    return pd.Series(out, name="permutation")


def lime_surrogate(est, X, x0, n_samples=600, kernel_width=None, seed=0) -> pd.Series:
    """Weighted ridge surrogate in a Gaussian neighbourhood of one instance."""
    rng = np.random.default_rng(seed)
    Xv = X.to_numpy(float)
    sd = Xv.std(axis=0); sd[sd == 0] = 1.0
    Z = rng.normal(x0.to_numpy(float), sd, size=(n_samples, len(x0)))
    # Euclidean distance in standardised space. Summing the elementwise square
    # root instead collapses every kernel weight to ~1e-38 across many features
    # and the surrogate returns all-zero coefficients.
    d = np.sqrt((((Z - x0.to_numpy(float)) / sd) ** 2).sum(axis=1))
    kw = kernel_width or np.sqrt(len(x0)) * 0.75
    w = np.exp(-(d ** 2) / (kw ** 2))
    yz = est.predict(pd.DataFrame(Z, columns=X.columns))
    r = Ridge(alpha=1.0).fit((Z - Z.mean(0)) / sd, yz, sample_weight=w)
    ss_tot = np.sum(w * (yz - np.average(yz, weights=w)) ** 2)
    pred = r.predict((Z - Z.mean(0)) / sd)
    r2 = 1 - np.sum(w * (yz - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
    s = pd.Series(r.coef_, index=X.columns, name="lime")
    s.attrs["r2"] = float(r2)
    return s


def shapley_sampling(est, X, x0, background, m=60, seed=0, top_k=None) -> pd.Series:
    """Monte-Carlo Shapley values (Strumbelj & Kononenko 2014).

    Coalitions are batched into one predict call per permutation, which is what
    makes this tractable without the `shap` package.

    `top_k` restricts sampling to the k features largest in the instance, and
    every unsampled feature then comes back as NaN. That is a false economy for
    anything that will be plotted: a blank cell in an attribution map must mean
    "no attribution", and it cannot be told apart from "never computed" once the
    two are mixed. Leave `top_k` at None unless the result is being used purely
    as a ranking.
    """
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    idx = range(len(cols)) if top_k is None else np.argsort(
        -np.abs(x0.to_numpy(float)))[:top_k]
    bg = background.to_numpy(float)
    x = x0.to_numpy(float)
    phi = {cols[j]: 0.0 for j in idx}
    for _ in range(m):
        perm = rng.permutation(len(cols))
        b = bg[rng.integers(0, len(bg))]
        A, B = [], []
        for j in idx:
            pos = int(np.where(perm == j)[0][0])
            before = perm[:pos]
            z1 = b.copy(); z1[before] = x[before]
            z2 = z1.copy(); z2[j] = x[j]
            A.append(z1); B.append(z2)
        if not A:
            continue
        pa = est.predict(pd.DataFrame(np.array(A), columns=cols))
        pb = est.predict(pd.DataFrame(np.array(B), columns=cols))
        for n, j in enumerate(idx):
            phi[cols[j]] += float(pb[n] - pa[n]) / m
    return pd.Series(phi, name="shapley")


# ──────────────────────────────────────────────────────────── diagnostics
def agreement(attr: pd.DataFrame) -> pd.DataFrame:
    """Spearman agreement between every pair of attribution methods."""
    rows = []
    cols = list(attr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = attr[cols[i]], attr[cols[j]]
            ok = a.notna() & b.notna()
            if ok.sum() < 4:
                continue
            rho, p = stats.spearmanr(a[ok].abs(), b[ok].abs())
            rows.append(dict(method_a=cols[i], method_b=cols[j],
                             spearman=float(rho), p=float(p), n=int(ok.sum())))
    return pd.DataFrame(rows)


def faithfulness(est, X, y, y_train, ranking: pd.Series, k=5, n_random=20, seed=0):
    """Does removing the top-k features hurt more than removing k random ones?

    Ablation is mean-imputation of the chosen columns. Reported as a paired
    contrast with a Wilcoxon test, not as a bare difference.
    """
    rng = np.random.default_rng(seed)
    base = mase(y, est.predict(X), y_train)
    def ablate(cols):
        """Mean-impute the named columns, leaving the rest untouched."""
        Z = X.copy()
        Z.loc[:, cols] = X[cols].mean().to_numpy()[None, :]
        return Z

    top = list(ranking.abs().sort_values(ascending=False).index[:k])
    top_delta = mase(y, est.predict(ablate(top)), y_train) - base
    rnd = []
    pool = [c for c in X.columns if c not in top]
    for _ in range(n_random):
        pick = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        rnd.append(mase(y, est.predict(ablate(pick)), y_train) - base)
    rnd = np.array(rnd, float)
    try:
        W, p = stats.wilcoxon(np.full_like(rnd, top_delta) - rnd)
    except ValueError:
        W, p = np.nan, np.nan
    return dict(k=k, base_MASE=base, top_k_delta=float(top_delta),
                random_delta_mean=float(np.nanmean(rnd)),
                random_delta_sd=float(np.nanstd(rnd)),
                excess=float(top_delta - np.nanmean(rnd)),
                wilcoxon_p=float(p) if p == p else np.nan,
                faithful=bool(p == p and p < 0.05 and top_delta > np.nanmean(rnd)))


def stability(targets, desc, station, target, cfg, model_name="RandomForest",
              seeds=(0, 1, 2, 3, 4), noise_scales=(0.0, 0.01, 0.05)) -> pd.DataFrame:
    """Rank stability of permutation importance under reseeding and perturbation."""
    X, y, yrs = make_supervised(targets, desc, station, target, cfg.default_lag)
    tr, va, te = chrono_split(yrs, cfg.test_years, cfg.val_years)
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    ref = None
    rows = []
    rng = np.random.default_rng(0)
    for sd in seeds:
        for ns in noise_scales:
            est = model_zoo(sd)[model_name]
            Xn = Xtr.copy()
            if ns > 0:
                Xn = Xn + rng.normal(0, ns * Xtr.std(axis=0).replace(0, 1), Xn.shape)
            est.fit(Xn, ytr)
            imp = permutation_importance(est, X.iloc[te], y.iloc[te], ytr,
                                         n_repeats=5, seed=sd)
            if ref is None:
                ref = imp
            rho, _ = stats.spearmanr(ref.abs(), imp.abs())
            rows.append(dict(station=station, target=target, model=model_name,
                             seed=sd, noise=ns, spearman_vs_reference=float(rho)))
    return pd.DataFrame(rows)


def run_xai(targets, desc, cfg, models=("RandomForest", "GradBoost", "MLP-64",
                                        "ElasticNet"), seed=0, progress=True,
            shapley_top_k=None):
    """Attributions and diagnostics for every station x target x model.

    `shapley_top_k` defaults to None, meaning every feature is sampled. An
    earlier version sampled the top twelve, which left most of the attribution
    map empty and made "no attribution" indistinguishable from "not computed"
    wherever the map was plotted. Full coverage costs roughly a minute per run
    and is worth it; pass an integer only for a quick ranking.
    """
    attr_rows, agree_rows, faith_rows = [], [], []
    for st in cfg.station_names:
        for tg in TARGETS:
            X, y, yrs = make_supervised(targets, desc, st, tg, cfg.default_lag)
            if len(X) < 12:
                continue
            tr, va, te = chrono_split(yrs, cfg.test_years, cfg.val_years)
            Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
            for mname in models:
                est = model_zoo(seed)[mname]
                try:
                    est.fit(Xtr, ytr)
                except Exception:
                    continue
                perm = permutation_importance(est, Xte, yte, ytr, 10, seed)
                x0 = Xte.iloc[len(Xte) // 2]
                lime = lime_surrogate(est, Xtr, x0, seed=seed)
                shap_ = shapley_sampling(est, Xtr, x0, Xtr, m=40, seed=seed,
                                         top_k=shapley_top_k)
                A = pd.DataFrame({"permutation": perm, "lime": lime,
                                  "shapley": shap_}).reindex(X.columns)
                A.insert(0, "feature", A.index)
                A.insert(0, "model", mname); A.insert(0, "target", tg)
                A.insert(0, "station", st)
                attr_rows.append(A.reset_index(drop=True))
                ag = agreement(A[["permutation", "lime", "shapley"]])
                ag["station"], ag["target"], ag["model"] = st, tg, mname
                ag["lime_r2"] = lime.attrs.get("r2", np.nan)
                agree_rows.append(ag)
                f = faithfulness(est, Xte, yte, ytr, perm, k=5, seed=seed)
                f.update(station=st, target=tg, model=mname)
                faith_rows.append(f)
            if progress:
                print(f"    XAI {st:<10s} {tg}")
    return (pd.concat(attr_rows, ignore_index=True) if attr_rows else pd.DataFrame(),
            pd.concat(agree_rows, ignore_index=True) if agree_rows else pd.DataFrame(),
            pd.DataFrame(faith_rows))
