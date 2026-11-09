#!/usr/bin/env python3
"""
Run the full experimental pipeline for

  "Best of how many? Candidate-count bias in environmental
   model benchmarking"

Every stage writes its artefacts to --out and can be run on its own; later stages
read the earlier stages' CSVs from disk, so a long run can be resumed.

    python run_pipeline.py --stages all
    python run_pipeline.py --stages data,targets,models --quick
    python run_pipeline.py --stages stats,proxy          # reuses results/ on disk

Stages
    data        load, audit, descriptors
    targets     divergence targets + Proposition 1 check + trend tests
    models      all models x stations x targets x seeds, with uncertainty
    stats       Friedman / Iman-Davenport / Nemenyi / Wilcoxon / Tukey
    sensitivity lag-window, feature-set and seed sweeps
    xai         attributions and their agreement, faithfulness and stability,
                and the signed attribution map (Fig. 5) built from them
    proxy       discrimination of documented flood years
    report      summary tables, LaTeX fragments and a run manifest

--quick trims seeds, bootstrap and permutation counts for a smoke test. It does
not change any definition, so quick results are directionally comparable but
should not be quoted.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from XAI_Rfpipe import core, experiments as ex, explain, figures, proxy, resolvability as rv, robustness as rob, sensitivity
from XAI_Rfpipe import statistics as st

ALL = ["data", "targets", "models", "stats", "sensitivity", "xai", "proxy",
       "postproc", "report"]


def save(df, out: Path, name: str):
    if df is None or (hasattr(df, "empty") and df.empty):
        print(f"      (nothing to write for {name})")
        return
    p = out / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"      -> {p.name}  ({len(df)} rows)")


def savej(obj, out: Path, name: str):
    (out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))
    print(f"      -> {name}.json")


def load(out: Path, name: str):
    p = out / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    stages = ALL if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = core.Config(root=root, out=out)
    if args.quick:
        cfg.seeds = (0, 1, 2)
        cfg.n_boot = 200
    if args.seeds:
        cfg.seeds = tuple(range(args.seeds))
    n_perm = 1000 if args.quick else 10000

    manifest = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__,
                    quick=args.quick, seeds=list(cfg.seeds), stages=stages)
    t_all = time.time()

    power = floods = desc = targets = None

    # ─────────────────────────────────────────────────────────────── data
    if "data" in stages:
        print("[data] loading and auditing")
        power, floods = core.load_all(cfg)
        a = core.audit(power, floods)
        save(a, out, "01_audit")
        desc = core.descriptors(power)
        save(desc, out, "02_descriptors")
        save(floods, out, "02_flood_events")
        manifest["n_daily_rows"] = int(len(power))
        manifest["n_flood_events"] = int(len(floods))
        manifest["flood_events_by_station"] = floods.groupby("station").size().to_dict()
        print(a.to_string(index=False))

    # ───────────────────────────────────────────────────────────── targets
    if "targets" in stages:
        print("[targets] building divergence targets")
        if power is None:
            power, floods = core.load_all(cfg)
        if desc is None:
            desc = load(out, "02_descriptors") or core.descriptors(power)
        targets = core.build_targets(power, cfg)
        save(targets, out, "03_targets")
        prop1 = core.verify_prop1(targets)
        savej(prop1, out, "03_proposition1")
        print(f"      Proposition 1: EMD == W1 -> {prop1['identical']} "
              f"(max |diff| {prop1['max_abs_diff']:.2e}); "
              f"{prop1['independent_targets']} independent targets")
        labels = core.flood_labels(floods, targets)
        save(labels, out, "03_flood_labels")
        save(st.kendall_trends(desc), out, "03_descriptor_trends")

    if targets is None:
        targets = load(out, "03_targets")
    if desc is None:
        desc = load(out, "02_descriptors")

    # ────────────────────────────────────────────────────────────── models
    if "models" in stages:
        print(f"[models] {len(cfg.station_names)} stations x {len(ex.TARGETS)} targets "
              f"x {len(cfg.seeds)} seeds")
        t0 = time.time()
        res, leak = ex.run_all(targets, desc, cfg)
        save(res, out, "04_model_results")
        save(leak, out, "04_leakage_report")
        save(ex.seed_summary(res), out, "04_seed_summary")
        manifest["models_seconds"] = round(time.time() - t0, 1)
        if len(leak):
            print(f"      window overlap between train and test: "
                  f"max {leak['max_shared_fraction'].max():.3f}")

    res = load(out, "04_model_results")

    # ─────────────────────────────────────────────────────────────── stats
    if "stats" in stages and res is not None:
        print("[stats] omnibus, post-hoc and contrasts")
        mat = st.block_matrix(res)
        save(mat.reset_index(), out, "05_block_matrix")
        fr = st.friedman(mat)
        savej(fr, out, "05_friedman")
        cd, nem = st.nemenyi(mat)
        savej({"critical_difference": cd}, out, "05_nemenyi_cd")
        save(nem, out, "05_nemenyi_pairs")
        best = min(fr.get("mean_ranks", {}), key=fr.get("mean_ranks", {}).get) \
            if fr.get("mean_ranks") else None
        if best:
            save(st.pairwise_wilcoxon(mat, reference=best), out, "05_wilcoxon_vs_best")
            manifest["best_mean_rank_model"] = best
        save(st.family_tukey(res), out, "05_tukey_models")
        savej(st.mannwhitney_family(res), out, "05_neural_vs_conventional")
        # Reported because readers ask for it, with the denominator counted from
        # the data and the critical difference attached. See family_win_counts.
        # "MLP-flat" belongs to the manuscript's model zoo, not this one; it is
        # named so that a run against that zoo reports both classifications, and
        # recorded as skipped here rather than passing silently.
        wins, wsum = st.family_win_counts(res, alt_conventional=("MLP-flat",))
        save(wins, out, "05_family_wins")
        savej(wsum, out, "05_family_wins_summary")
        # What the design could resolve, before any model was fit, and whether
        # the win count means anything once candidate counts are accounted for.
        design, wtest, wperm, wpool = rv.run_audit(res, n_perm=20000, seed=0)
        # The common-relabelling randomisation is the one that responds to
        # dependence; the within-block version is kept only because it is
        # provably equal to the binomial null and readers may try it.
        mat_a = rv.block_matrix_and_family(res)
        wprof = rv.family_win_profile_permutation(mat_a[0], mat_a[1],
                                                  n_perm=40000, seed=0)
        savej({"design": design, "win_count_test": wtest,
               "within_block_permutation_IS_binomial": wperm,
               "profile_permutation": wprof,
               "equivalence_demonstration":
                   rv.within_block_permutation_is_binomial()},
              out, "05_resolvability")
        save(wpool, out, "05_pool_sensitivity")
        # What the fair-coin reading does across the space of designs. Exact,
        # and touches no data: it is a property of the statistic.
        sweep = rv.design_space_sweep()
        save(sweep, out, "05_design_space_sweep")
        save(rob.type_I_under_departures(n_sim=4000), out, "05_robustness_type_I")
        save(rob.power_curve(9, 3, 11), out, "05_power_curve")
        f_err = figures.build_error_rate_figure(out)
        if f_err.get("written"):
            bad = sweep[(sweep.k_a != sweep.k_b) & (sweep.n_blocks >= 25)]
            print(f"      fair-coin type-I error across designs: "
                  f"{sweep.fair_coin_type_I.min():.3f} to "
                  f"{sweep.fair_coin_type_I.max():.3f}; "
                  f"median {bad.fair_coin_type_I.median():.2f} for unbalanced "
                  f"pools at N>=25")
        if wtest:
            print(f"      win count {wtest['wins']}/{wtest['n_blocks']}: "
                  f"expected {wtest['expected_wins']} under the candidate-aware "
                  f"null p0={wtest['null_p']:.3f} (not 0.5); "
                  f"p={wtest['p_two_sided']:.3f} vs naive p={wtest['naive_p_two_sided']:.3f}"
                  + ("  <-- VERDICT FLIPS" if wtest['verdict_changes_under_correct_null'] else ""))
            print(f"      design could only ever call a family claim with "
                  f">={design['min_significant_wins']} wins or "
                  f"<={design['max_significant_losses']} wins")
        if wsum:
            print(f"      best neural beats best conventional in "
                  f"{wsum['neural_wins']} of {wsum['n_blocks']} blocks "
                  f"({wsum['n_stations']} stations x {wsum['n_targets']} targets); "
                  f"{wsum['n_models_within_cd_of_best']} models lie within one CD "
                  f"of the best, so resolvable={wsum['resolvable']}")
        print(f"      Friedman p={fr.get('friedman_p'):.3g}, "
              f"Iman-Davenport p={fr.get('iman_davenport_p'):.3g}, CD={cd:.3f}")

    # ───────────────────────────────────────────────────────── sensitivity
    if "sensitivity" in stages:
        print("[sensitivity] sweeping design factors")
        sd3 = cfg.seeds[:3]
        r_lag = sensitivity.sweep_lag(targets, desc, cfg, seeds=sd3)
        save(r_lag, out, "06_sweep_lag")
        r_feat = sensitivity.sweep_features(targets, desc, cfg, seeds=sd3)
        save(r_feat, out, "06_sweep_features")
        eff = sensitivity.factor_effects(res, r_lag, r_feat)
        save(eff, out, "06_factor_effects")
        save(sensitivity.best_of_optimism(res), out, "06_best_of_optimism")
        if len(eff):
            print(eff.to_string(index=False))

    # ───────────────────────────────────────────────────────────────── xai
    if "xai" in stages:
        print("[xai] attributions and diagnostics")
        attr, agree, faith = explain.run_xai(targets, desc, cfg)
        save(attr, out, "07_attributions")
        save(agree, out, "07_attribution_agreement")
        save(faith, out, "07_faithfulness")
        stab = pd.concat([explain.stability(targets, desc, s, t, cfg)
                          for s in cfg.station_names for t in ex.TARGETS[:1]],
                         ignore_index=True)
        save(stab, out, "07_stability")
        if len(agree):
            print(f"      cross-method agreement: median rho "
                  f"{agree['spearman'].median():.3f}")
        if len(faith):
            print(f"      faithful attributions: {int(faith['faithful'].sum())}"
                  f"/{len(faith)} model-blocks")
        f5 = figures.build_fig5(out, targets=ex.TARGETS, strict=False)
        if f5.get("written"):
            print(f"      -> Fig5_xai_heatmap.png/.pdf  "
                  f"(Shapley coverage {f5['shapley_coverage']:.0%}, "
                  f"direction agreed in {f5['cells_agreeing']}/{f5['cells']} cells)")
        else:
            print(f"      (Fig. 5 not drawn: {f5.get('reason')})")

    # ─────────────────────────────────────────────────────────────── proxy
    if "proxy" in stages:
        print("[proxy] flood-year discrimination")
        labels = load(out, "03_flood_labels")
        if labels is None:
            power, floods = core.load_all(cfg)
            labels = core.flood_labels(floods, targets)
        panel = proxy.build_panel(targets, desc, labels)
        save(panel, out, "08_proxy_panel")
        disc = proxy.discriminate(panel, n_perm=n_perm, n_boot=cfg.n_boot)
        save(disc, out, "08_discrimination")
        head = proxy.best_vs_targets(disc)
        savej(head, out, "08_headline")
        if head:
            print(f"      best target     {head['best_target']:<10s} "
                  f"AUROC {head['best_target_auroc']:.3f} (p={head['best_target_p']:.3f})")
            print(f"      best descriptor {head['best_descriptor']:<10s} "
                  f"AUROC {head['best_descriptor_auroc']:.3f} "
                  f"(p={head['best_descriptor_p']:.3f})")

    # ────────────────────────────────────────────── post-processing ablation
    if "postproc" in stages:
        print("[postproc] leave-one-year-out ablation of post-processing components")
        from XAI_Rfpipe import postproc as pp
        if power is None:
            power, floods = core.load_all(cfg)
        n_mem = 20 if args.quick else 50
        pres, _ = pp.run_ablation(power, cfg, n_members=n_mem)
        save(pres, out, "09_postproc_raw")
        for metric in ("CRPS", "variogram", "acf1_err", "Brier", "CDD_err"):
            save(pp.summarise(pres, metric), out, f"09_postproc_summary_{metric}")
        save(pp.test_components(pres, "CRPS"), out, "09_postproc_tests_CRPS")
        save(pp.test_components(pres, "variogram"), out, "09_postproc_tests_variogram")
        save(pp.test_components(pres, "acf1_err"), out, "09_postproc_tests_acf1")
        fr, cd, pairs = pp.friedman_configs(pres, "CRPS")
        savej({**fr, "nemenyi_cd": cd}, out, "09_postproc_friedman")
        save(pairs, out, "09_postproc_nemenyi")
        manifest["postproc_members"] = n_mem
        for metric in ("CRPS", "acf1_err"):
            s = pp.summarise(pres, metric)
            best = s.iloc[0]
            print(f"      best by {metric:<9s}: {best['config']:<16s} "
                  f"{best['mean']:.4f}  (full = "
                  f"{float(s.loc[s['config'] == 'full', 'mean'].iloc[0]):.4f})")

    # ────────────────────────────────────────────────────────────── report
    if "report" in stages:
        print("[report] assembling summary")
        from XAI_Rfpipe.report import build_report
        build_report(out, cfg)

    manifest["elapsed_seconds"] = round(time.time() - t_all, 1)
    savej(manifest, out, "00_manifest")
    print(f"\ndone in {manifest['elapsed_seconds']}s -> {out}/")


if __name__ == "__main__":
    main()
