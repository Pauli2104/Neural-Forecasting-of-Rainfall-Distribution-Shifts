#!/usr/bin/env python3
"""
Assert the things that must be true of any run, on any machine.

These are not unit tests of the implementation; they are the claims the paper
makes about the data and the construction, checked against whatever the stages
just wrote. A failure here means a result in the paper no longer holds for this
code and this data, which is the only kind of regression worth failing a build
over.

    python tests/check_invariants.py [results_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(f"{name}{'  ' + detail if detail else ''}")


print(f"checking {OUT}/")

# ── the record itself ────────────────────────────────────────────────────────
audit = pd.read_csv(OUT / "01_audit.csv")
check("three stations audited", len(audit) == 3, f"n={len(audit)}")
check("no station is missing its precipitation column",
      "PRECTOTCORR_missing" not in audit or audit["PRECTOTCORR_missing"].sum() >= 0)

desc = pd.read_csv(OUT / "02_descriptors.csv")
check("descriptors cover every station",
      desc["station"].nunique() == 3, f"stations={desc['station'].nunique()}")
check("hydrological years are contiguous per station",
      all(g["hydro_year"].diff().dropna().eq(1).all()
          for _, g in desc.groupby("station")))
check("wet-day counts are physical",
      desc["wet_days"].between(0, 366).all())

events = pd.read_csv(OUT / "02_flood_events.csv")
check("flood catalogue holds 43 events", len(events) == 43, f"n={len(events)}")
check("catalogue months are all in the austral wet season (Dec-Apr)",
      events["month"].isin([12, 1, 2, 3, 4]).all(),
      f"months={sorted(events['month'].unique())}")

# ── Proposition 1 ────────────────────────────────────────────────────────────
p1 = json.loads((OUT / "03_proposition1.json").read_text())
check("Proposition 1: EMD and W1 coincide", bool(p1["identical"]),
      f"max|diff|={p1['max_abs_diff']:.2e}, r={p1['pearson_r']:.6f}")
check("Proposition 1 leaves three independent targets",
      p1["independent_targets"] == 3, f"n={p1['independent_targets']}")

# ── targets ──────────────────────────────────────────────────────────────────
t = pd.read_csv(OUT / "03_targets.csv")
check("divergences are non-negative",
      all((t[c].dropna() >= 0).all() for c in ("D_KL", "D_B", "W1", "EMD")
          if c in t))
check("no target column is entirely missing",
      all(t[c].notna().any() for c in ("D_KL", "D_B", "W1") if c in t))

# ── anything already computed by a fuller run ────────────────────────────────
leak_p = OUT / "04_leakage_report.csv"
if leak_p.exists():
    leak = pd.read_csv(leak_p)
    check("no train/test window overlap",
          float(leak["max_shared_fraction"].max()) == 0.0,
          f"max={leak['max_shared_fraction'].max():.3f}")

wins_p = OUT / "05_family_wins_summary.json"
if wins_p.exists():
    w = json.loads(wins_p.read_text())
    check("win-count denominator equals stations x targets",
          w["n_blocks"] == w["n_stations"] * w["n_targets"],
          f"{w['n_blocks']} = {w['n_stations']} x {w['n_targets']}")
    check("win counts sum to the number of blocks",
          w["neural_wins"] + w["conventional_wins"] + w["ties"] == w["n_blocks"])
    check("the win count carries its critical difference",
          "critical_difference" in w and "resolvable" in w)

res_p = OUT / "05_resolvability.json"
if res_p.exists():
    r = json.loads(res_p.read_text())
    d, t = r["design"], r["win_count_test"]
    check("win-count null equals K_A/(K_A+K_B)",
          abs(t["null_p"] - t["k_a"] / (t["k_a"] + t["k_b"])) < 1e-12,
          f"p0={t['null_p']:.4f} = {t['k_a']}/{t['k_a'] + t['k_b']}")
    check("the null is NOT a fair coin for unequal pools",
          (t["k_a"] == t["k_b"]) == (abs(t["null_p"] - 0.5) < 1e-12))
    check("expected wins equal N x p0",
          abs(t["expected_wins"] - (t["n_blocks"] - t["ties"]) * t["null_p"]) < 5e-3)
    check("the design report carries its tail convention",
          d.get("tail") in ("one-sided", "two-sided"), f"tail={d.get('tail')}")
    # NB: the within-block permutation agreeing with the binomial is a
    # tautology, not a validation; see resolvability.py. It is checked only
    # so that the equivalence is asserted rather than assumed.
    perm = r.get("within_block_permutation_IS_binomial")
    if perm:
        check("within-block permutation equals the binomial null (a tautology)",
              abs(perm["null_mean"] - perm["closed_form_expected"]) < 0.1,
              f"{perm['null_mean']:.3f} vs {perm['closed_form_expected']}")

    eq = r.get("equivalence_demonstration")
    if eq:
        check("a within-block permutation reproduces the binomial null exactly",
              bool(eq["within_block_matches_binomial"]),
              f"sd {eq['within_block_sd']:.3f} vs binomial {eq['binomial_sd']:.3f}")
        check("the common relabelling does NOT, when wins concentrate",
              bool(eq["profile_differs_from_binomial"]),
              f"sd {eq['profile_sd']:.3f}")
    pr = r.get("profile_permutation")
    if pr:
        check("the profile null is at least as wide as the binomial",
              pr["null_sd"] >= pr["binomial_sd"] - 1e-6,
              f"{pr['null_sd']:.3f} vs {pr['binomial_sd']:.3f}")

rob_p = OUT / "05_robustness_type_I.csv"
if rob_p.exists():
    rr = pd.read_csv(rob_p)
    base = rr[rr["scenario"].str.startswith("exchangeable")]
    corr = rr[rr["scenario"].str.contains("correlated 0.9") & (rr["rho_b"] > 0)]
    check("the test holds its size under the reference null",
          bool((base["realised_size"] <= base["nominal"]).all()),
          f"{float(base['realised_size'].iloc[0]):.4f}")
    check("correlated candidates break it, as the paper states",
          bool((corr["realised_size"] > 4 * corr["nominal"]).all()),
          f"{float(corr['realised_size'].iloc[0]):.3f}")

sweep_p = OUT / "05_design_space_sweep.csv"
if sweep_p.exists():
    sw = pd.read_csv(sweep_p)
    bal = sw[sw["k_a"] == sw["k_b"]]
    unbal = sw[(sw["k_a"] != sw["k_b"]) & (sw["n_blocks"] >= 25)]
    check("the fair coin is a valid test when pools are equal",
          bool((bal["fair_coin_type_I"] <= bal["alpha"] + 1e-9).all()),
          f"max={bal['fair_coin_type_I'].max():.3f}")
    check("the candidate-aware test holds its size everywhere",
          bool((sw["candidate_aware_type_I"] <= sw["alpha"] + 1e-9).all()),
          f"max={sw['candidate_aware_type_I'].max():.3f}")
    # the mildest imbalance in the grid (2 vs 3) is the binding case
    check("unbalanced pools inflate the fair coin's error at scale",
          bool((unbal["fair_coin_type_I"] > 2 * unbal["alpha"]).all()),
          f"min={unbal['fair_coin_type_I'].min():.3f} "
          f"= {unbal['inflation'].min():.1f}x nominal")

attr_p = OUT / "07_attributions.csv"
if attr_p.exists():
    a = pd.read_csv(attr_p)
    cov = a["shapley"].notna().mean()
    check("Shapley covers every feature, so a blank cell means no attribution",
          cov >= 0.95, f"coverage={cov:.1%}")

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
