"""
XAI_Rfpipe.sensitivity - how much do undocumented design choices move the result?

Three factors are swept independently, each holding the others at the default:

* lag window     - how many previous years enter the feature matrix
* feature set    - target lags only, versus target lags plus annual descriptors
* random seed    - the same configuration refitted

The comparison that matters is not the absolute error but the spread each factor
induces, measured on the same scale. If reseeding moves MASE further than the
choice of model family does, then a paper that reports one run per model is
reporting noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .experiments import TARGETS, run_block, seed_summary


def sweep_lag(targets, desc, cfg, seeds=(0, 1, 2), progress=True) -> pd.DataFrame:
    out = []
    for lag in cfg.lag_windows:
        for st in cfg.station_names:
            for tg in TARGETS:
                for sd in seeds:
                    df, _ = run_block(targets, desc, st, tg, cfg, lag=lag, seed=sd)
                    if len(df):
                        out.append(df)
        if progress:
            print(f"    lag={lag} done")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sweep_features(targets, desc, cfg, seeds=(0, 1, 2), progress=True) -> pd.DataFrame:
    out = []
    for use_desc in (True, False):
        for st in cfg.station_names:
            for tg in TARGETS:
                for sd in seeds:
                    df, _ = run_block(targets, desc, st, tg, cfg, seed=sd,
                                      use_descriptors=use_desc)
                    if len(df):
                        df["feature_set"] = "target+descriptors" if use_desc else "target-only"
                        out.append(df)
        if progress:
            print(f"    feature_set={'full' if use_desc else 'ablated'} done")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def factor_effects(res_seed, res_lag, res_feat, metric="test_MASE") -> pd.DataFrame:
    """Spread in the metric attributable to each design factor, on one scale.

    For each factor the statistic is the mean over station x target x model of the
    within-cell range of the metric as that factor varies.
    """
    rows = []

    def spread(df, factor):
        if df is None or not len(df) or factor not in df:
            return None
        g = df.dropna(subset=[metric]).groupby(["station", "target", "model"])[metric]
        rng = g.agg(lambda s: s.max() - s.min())
        return rng

    for name, df, factor in (("random seed", res_seed, "seed"),
                             ("lag window", res_lag, "lag"),
                             ("feature set", res_feat, "feature_set")):
        r = spread(df, factor)
        if r is None or r.empty:
            continue
        rows.append(dict(factor=name, n_cells=int(len(r)),
                         mean_range=float(r.mean()), median_range=float(r.median()),
                         q90_range=float(r.quantile(0.9)), max_range=float(r.max())))
    out = pd.DataFrame(rows)
    if len(out) > 1:
        out["relative_to_smallest"] = out["mean_range"] / out["mean_range"].min()
    return out.sort_values("mean_range", ascending=False) if len(out) else out


def best_of_optimism(res_seed, metric="test_MASE") -> pd.DataFrame:
    """What reporting the best seed buys you, per model - the selective-reporting cost."""
    s = seed_summary(res_seed, metric)
    s = s[s["n_seeds"] > 1].copy()
    s["optimism_pct"] = 100 * s["optimism"] / s["mean"].replace(0, np.nan)
    return s.sort_values("optimism", ascending=False)
