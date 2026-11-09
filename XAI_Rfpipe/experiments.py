"""
XAI_Rfpipe.experiments - supervised framing, model zoo, evaluation.

Leakage
-------
The targets are built from a moving window of `win_years` hydrological years, so
two consecutive target values share `win_years - 1` years of daily data. That is
genuine overlap, and it is why every split here is strictly chronological and why
`leakage_report` quantifies the shared support rather than leaving it implicit.
A random split of these rows would let a model see most of a test year's own
window during training; the pipeline never does that.

Uncertainty
-----------
No point estimate is reported alone. Every metric carries a bootstrap confidence
interval and a margin of error, every model is run across seeds and summarised by
quantiles, and interval forecasts are calibrated by split conformal prediction so
that nominal and empirical coverage can be compared.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGETS = ["D_KL", "D_B", "W1"]          # EMD omitted: identical to W1 (Prop. 1)
NEURAL = {"MLP-64", "MLP-128-64", "MLP-32-16"}


# ──────────────────────────────────────────────────────── supervised framing
def make_supervised(targets: pd.DataFrame, desc: pd.DataFrame, station: str,
                    target: str, lag: int, use_descriptors: bool = True
                    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Rows: one per predictable hydrological year at one station.

    Features are `lag` lagged values of the target plus, optionally, the same
    lagged annual descriptors. Only information available strictly before the
    target year enters the feature matrix.
    """
    t = targets[targets["station"] == station].sort_values("hydro_year")
    d = desc[desc["station"] == station].sort_values("hydro_year")
    df = t.merge(d.drop(columns=["station"]), on="hydro_year", how="left")
    dcols = [c for c in d.columns if c not in ("station", "hydro_year", "n_days")]

    X, y, yrs = [], [], []
    vals = df[target].to_numpy(float)
    dmat = df[dcols].to_numpy(float) if use_descriptors else None
    years = df["hydro_year"].to_numpy(int)
    for i in range(lag, len(df)):
        row = list(vals[i - lag:i][::-1])                 # t-1 … t-lag
        names = [f"{target}_lag{k}" for k in range(1, lag + 1)]
        if use_descriptors:
            for k in range(1, lag + 1):
                row += list(dmat[i - k])
                names += [f"{c}_lag{k}" for c in dcols]
        X.append(row); y.append(vals[i]); yrs.append(years[i])
    cols = names
    return pd.DataFrame(X, columns=cols), pd.Series(y, name=target), np.array(yrs)


def chrono_split(years: np.ndarray, n_test: int, n_val: int):
    """Strictly chronological train / validation / test indices."""
    n = len(years)
    n_test = min(n_test, max(1, n // 4))
    n_val = min(n_val, max(1, n // 5))
    te = np.arange(n - n_test, n)
    va = np.arange(n - n_test - n_val, n - n_test)
    tr = np.arange(0, n - n_test - n_val)
    return tr, va, te


def leakage_report(years: np.ndarray, tr, te, win_years: int) -> dict:
    """How much daily data a test row's window shares with any training row's.

    Each target at year h summarises the daily record of years
    [h-win_years+1 … h]. Overlap is measured in shared hydrological years.
    """
    def span(h):
        return set(range(h - win_years + 1, h + 1))
    tr_years = set().union(*[span(years[i]) for i in tr]) if len(tr) else set()
    shared = [len(span(years[i]) & tr_years) / win_years for i in te]
    return dict(win_years=win_years,
                mean_shared_fraction=float(np.mean(shared)) if shared else 0.0,
                max_shared_fraction=float(np.max(shared)) if shared else 0.0,
                n_test_rows_with_overlap=int(sum(s > 0 for s in shared)),
                n_test_rows=int(len(te)),
                gap_years=int(years[te[0]] - years[tr[-1]]) if len(tr) and len(te) else 0)


# ─────────────────────────────────────────────────────────────── model zoo
def model_zoo(seed: int) -> dict:
    """Conventional baselines and neural forecasters, one construction point."""
    def pipe(est):
        return Pipeline([("sc", StandardScaler()), ("m", est)])
    return {
        "Mean":        DummyRegressor(strategy="mean"),
        "Persistence": "persistence",                    # handled specially
        "Linear":      pipe(LinearRegression()),
        "Ridge":       pipe(Ridge(alpha=1.0, random_state=seed)),
        "ElasticNet":  pipe(ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=seed,
                                       max_iter=20000)),
        "kNN-5":       pipe(KNeighborsRegressor(n_neighbors=5)),
        "SVR-RBF":     pipe(SVR(C=1.0, epsilon=0.05)),
        "DecisionTree": DecisionTreeRegressor(random_state=seed, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed,
                                              n_jobs=1, min_samples_leaf=2),
        "ExtraTrees":  ExtraTreesRegressor(n_estimators=300, random_state=seed,
                                           n_jobs=1, min_samples_leaf=2),
        "GradBoost":   GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                                 max_depth=2, learning_rate=0.05),
        "MLP-64":      pipe(MLPRegressor(hidden_layer_sizes=(64,), random_state=seed,
                                         max_iter=3000, early_stopping=False)),
        "MLP-128-64":  pipe(MLPRegressor(hidden_layer_sizes=(128, 64), random_state=seed,
                                         max_iter=3000, early_stopping=False)),
        "MLP-32-16":   pipe(MLPRegressor(hidden_layer_sizes=(32, 16), random_state=seed,
                                         max_iter=3000, early_stopping=False)),
    }


def _fit_predict(name, est, Xtr, ytr, Xte, lag_col):
    if est == "persistence":
        return Xte[lag_col].to_numpy(float)
    est.fit(Xtr, ytr)
    return np.asarray(est.predict(Xte), dtype=float)


# ──────────────────────────────────────────────────────────────── metrics
def mase(y_true, y_pred, y_train) -> float:
    """Mean absolute scaled error, scaled by the in-sample naive-1 error."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    scale = np.mean(np.abs(np.diff(np.asarray(y_train, float))))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def metric_block(y_true, y_pred, y_train) -> dict:
    e = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return dict(MASE=mase(y_true, y_pred, y_train),
                MAE=float(np.mean(np.abs(e))),
                RMSE=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)))


def bootstrap_ci(y_true, y_pred, y_train, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap over test residual pairs, for every metric at once."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n < 3:
        return {}
    keys = ("MASE", "MAE", "RMSE", "bias")
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metric_block(y_true[idx], y_pred[idx], y_train)
        for k in keys:
            draws[k][b] = m[k]
    out = {}
    for k in keys:
        d = draws[k][np.isfinite(draws[k])]
        if len(d) < 10:
            continue
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        out[f"{k}_lo"] = float(lo)
        out[f"{k}_hi"] = float(hi)
        out[f"{k}_moe"] = float((hi - lo) / 2)
        out[f"{k}_se"] = float(d.std(ddof=1))
    return out


def conformal_interval(y_cal, p_cal, p_test, alpha=0.10):
    """Split-conformal prediction interval from calibration residuals."""
    r = np.abs(np.asarray(y_cal, float) - np.asarray(p_cal, float))
    n = len(r)
    if n == 0:
        return None, None, float("nan")
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    q = float(np.sort(r)[max(k, 0)])
    return p_test - q, p_test + q, q


# ──────────────────────────────────────────────────────────── the experiment
def run_block(targets, desc, station, target, cfg, lag=None, seed=0,
              use_descriptors=True, models=None):
    """One station x target block, all models, one seed."""
    lag = lag or cfg.default_lag
    X, y, yrs = make_supervised(targets, desc, station, target, lag, use_descriptors)
    if len(X) < 12:
        return pd.DataFrame(), {}
    tr, va, te = chrono_split(yrs, cfg.test_years, cfg.val_years)
    leak = leakage_report(yrs, tr, te, cfg.win_years)
    lag_col = f"{target}_lag1"
    zoo = models or model_zoo(seed)
    rows = []
    for name, est in zoo.items():
        try:
            p_te = _fit_predict(name, est, X.iloc[tr], y.iloc[tr], X.iloc[te], lag_col)
            p_va = _fit_predict(name, est, X.iloc[tr], y.iloc[tr], X.iloc[va], lag_col)
            p_tr = _fit_predict(name, est, X.iloc[tr], y.iloc[tr], X.iloc[tr], lag_col)
        except Exception as exc:                      # a model that cannot fit
            rows.append(dict(station=station, target=target, model=name, seed=seed,
                             lag=lag, error=str(exc)[:80]))
            continue
        rec = dict(station=station, target=target, model=name, seed=seed, lag=lag,
                   family="neural" if name in NEURAL else "conventional",
                   n_train=len(tr), n_val=len(va), n_test=len(te))
        for split, yy, pp in (("train", y.iloc[tr], p_tr), ("val", y.iloc[va], p_va),
                              ("test", y.iloc[te], p_te)):
            for k, v in metric_block(yy, pp, y.iloc[tr]).items():
                rec[f"{split}_{k}"] = v
        rec.update({f"test_{k}": v for k, v in
                    bootstrap_ci(y.iloc[te], p_te, y.iloc[tr],
                                 cfg.n_boot, 0.05, seed).items()})
        lo, hi, q = conformal_interval(y.iloc[va], p_va, p_te, cfg.conformal_alpha)
        if lo is not None:
            inside = ((y.iloc[te].to_numpy() >= lo) & (y.iloc[te].to_numpy() <= hi))
            rec["conformal_q"] = q
            rec["conformal_nominal"] = 1 - cfg.conformal_alpha
            rec["conformal_empirical"] = float(inside.mean())
            rec["conformal_width"] = float(2 * q)
        rows.append(rec)
    return pd.DataFrame(rows), leak


def run_all(targets, desc, cfg, seeds=None, lag=None, use_descriptors=True,
            progress=True):
    """Every station x target block, across seeds."""
    seeds = seeds if seeds is not None else cfg.seeds
    frames, leaks = [], []
    for st in cfg.station_names:
        for tg in TARGETS:
            for sd in seeds:
                df, leak = run_block(targets, desc, st, tg, cfg, lag, sd,
                                     use_descriptors)
                if len(df):
                    frames.append(df)
                if leak and sd == seeds[0]:
                    leaks.append(dict(station=st, target=tg, **leak))
            if progress:
                print(f"    {st:<10s} {tg:<5s} done")
    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return res, pd.DataFrame(leaks)


def seed_summary(res: pd.DataFrame, metric="test_MASE") -> pd.DataFrame:
    """Across-seed quantiles - the alternative to reporting a single best run."""
    g = res.dropna(subset=[metric]).groupby(["station", "target", "model", "family"])
    out = g[metric].agg(
        n_seeds="size", mean="mean", sd="std", min="min",
        q05=lambda s: s.quantile(0.05), q25=lambda s: s.quantile(0.25),
        median="median", q75=lambda s: s.quantile(0.75),
        q95=lambda s: s.quantile(0.95), max="max").reset_index()
    out["range"] = out["max"] - out["min"]
    out["optimism"] = out["mean"] - out["min"]       # cost of reporting best-of
    return out.sort_values(["station", "target", "mean"])
