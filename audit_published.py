#!/usr/bin/env python3
"""
Apply the resolvability audit to comparisons this pipeline did not produce.

Two cases, for two different reasons.

  original run   The thirteen-model run reported in the accompanying paper's
                 Table 5, read from data/published/original_run_block_matrix.csv.
                 Included so that every audited number in the paper comes from a
                 file, including the ones belonging to the earlier run.

  published      Two comparisons from the literature, audited from their
                 PUBLISHED SUMMARY STATISTICS alone - w, N, K_A, K_B - which is
                 all the binomial route needs. No data of theirs is used, and
                 none is required.

    python audit_published.py [--out results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from XAI_Rfpipe import resolvability as rv

NEURAL = ["TFT", "TCN", "LSTM", "GRU", "MLP"]
ARGUABLE = "MLP"        # neural by construction, a flat baseline in behaviour


def original_run(out: Path) -> dict:
    src = Path(__file__).parent / "data/published/original_run_block_matrix.csv"
    mat = pd.read_csv(src, comment="#", index_col="block")
    rows, detail = [], {}
    for label, neural in [("MLP as neural", NEURAL),
                          ("MLP as conventional", [c for c in NEURAL if c != ARGUABLE])]:
        a = [c for c in mat.columns if c in neural]
        b = [c for c in mat.columns if c not in a]
        wins = int((mat[a].min(axis=1) < mat[b].min(axis=1)).sum())
        t = rv.family_win_test(wins, len(mat), len(a), len(b))
        d = rv.design_report(mat.shape[1], mat.shape[0], len(a), len(b))
        p = rv.family_win_permutation(mat, a, n_perm=20000, seed=0)
        rows.append(dict(run=f"original, {label}", k_a=len(a), k_b=len(b),
                         n_blocks=len(mat), wins=wins,
                         null_p=round(t["null_p"], 4),
                         p_candidate_aware=round(t["p_two_sided"], 4),
                         p_fair_coin=round(t["naive_p_two_sided"], 4),
                         min_significant_wins=d["min_significant_wins"],
                         max_significant_losses=d["max_significant_losses"],
                         p_permutation=round(p["p_two_sided"], 4)))
        detail[label] = dict(design=d, win_count_test=t, permutation_test=p)
    pd.DataFrame(rows).to_csv(out / "10_original_run_audit.csv", index=False)
    (out / "10_original_run_audit.json").write_text(json.dumps(detail, indent=2,
                                                               default=str))
    return detail


def published(out: Path) -> pd.DataFrame:
    # (label, wins, n_blocks, k_a, k_b, source)
    CASES = [
        ("Arsenault et al. 2023: LSTM vs per-basin best of 18, 86% of 148 basins",
         127, 148, 1, 18, "doi:10.5194/hess-27-139-2023"),
        ("Kratzert et al. 2019: one model vs one ensemble, 2 of 447 basins",
         2, 447, 1, 1, "doi:10.5194/hess-23-5089-2019"),
    ]
    rows = []
    for label, w, n, ka, kb, src in CASES:
        t = rv.family_win_test(w, n, ka, kb)
        rows.append(dict(comparison=label, source=src, wins=w, n_blocks=n,
                         k_a=ka, k_b=kb, null_p=round(t["null_p"], 4),
                         expected_wins=t["expected_wins"],
                         p_candidate_aware=t["p_two_sided"],
                         p_fair_coin=t["naive_p_two_sided"],
                         verdict_changes=t["verdict_changes_under_correct_null"]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "10_published_comparisons.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    d = original_run(o)
    for k, v in d.items():
        t, g = v["win_count_test"], v["design"]
        print(f"original run, {k}: {t['wins']}/{t['n_blocks']} wins, "
              f"p0={t['null_p']:.4f}, p={t['p_two_sided']:.4f} "
              f"(fair coin {t['naive_p_two_sided']:.4f}); "
              f"could ever be significant at >={g['min_significant_wins']} wins "
              f"or <={g['max_significant_losses']}")
    print()
    print(published(o).to_string(index=False))
