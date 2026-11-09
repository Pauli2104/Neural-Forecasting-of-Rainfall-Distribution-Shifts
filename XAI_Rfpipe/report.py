"""
XAI_Rfpipe.report - turn the stage artefacts into tables the paper can use.

Produces a plain-text summary for reading, LaTeX fragments for pasting, and a
findings file that states the things a reviewer will check first. Nothing here
recomputes anything: it reads what the stages wrote, so the report cannot drift
from the results.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read(out: Path, name: str):
    p = out / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _readj(out: Path, name: str):
    p = out / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _latex(df: pd.DataFrame, caption: str, label: str, floatfmt="%.3f") -> str:
    body = df.to_latex(index=False, escape=True, float_format=lambda v: floatfmt % v,
                       column_format="l" * 1 + "r" * (df.shape[1] - 1))
    # Springer tables carry no vertical rules
    body = body.replace("\\begin{tabular}", "\\begin{tabular}")
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{body}\\end{{table}}\n")


def build_report(out: Path, cfg) -> None:
    out = Path(out)
    tex = []
    lines = ["RAINFALL-SHIFT PIPELINE - RUN SUMMARY", "=" * 60, ""]

    audit = _read(out, "01_audit")
    if audit is not None:
        lines += ["DATA", audit.to_string(index=False), ""]
        tex.append(_latex(audit, "Integrity audit of the daily record and the "
                                 "documented flood catalogue", "tab:audit"))

    p1 = _readj(out, "03_proposition1")
    if p1:
        lines += ["PROPOSITION 1 (EMD = W1 on a unit-spaced ordered support)",
                  f"  n = {p1['n']},  Pearson r = {p1['pearson_r']:.10f},"
                  f"  max |diff| = {p1['max_abs_diff']:.3e}",
                  f"  verdict: {'identical' if p1['identical'] else 'NOT identical'}"
                  f" -> {p1['independent_targets']} independent targets", ""]

    leak = _read(out, "04_leakage_report")
    if leak is not None and len(leak):
        lines += ["LEAKAGE (window overlap between train and test rows)",
                  f"  mean shared fraction {leak['mean_shared_fraction'].mean():.3f},"
                  f"  max {leak['max_shared_fraction'].max():.3f}",
                  f"  test rows with any overlap: "
                  f"{int(leak['n_test_rows_with_overlap'].sum())}"
                  f" of {int(leak['n_test_rows'].sum())}", ""]

    seeds = _read(out, "04_seed_summary")
    if seeds is not None and len(seeds):
        top = (seeds.sort_values("median").groupby(["station", "target"]).head(3)
               [["station", "target", "model", "family", "median", "q05", "q95",
                 "range", "optimism"]])
        lines += ["BEST THREE MODELS PER BLOCK (median test MASE across seeds)",
                  top.round(3).to_string(index=False), ""]
        tex.append(_latex(top.round(3),
                          "Test MASE by model, summarised across seeds. "
                          "Optimism is the gap between the mean and the best seed",
                          "tab:seeds"))

    fr = _readj(out, "05_friedman")
    cd = _readj(out, "05_nemenyi_cd")
    if fr:
        lines += ["MODEL COMPARISON",
                  f"  Friedman chi2 = {fr.get('friedman_chi2', float('nan')):.3f},"
                  f" p = {fr.get('friedman_p', float('nan')):.3g}",
                  f"  Iman-Davenport F = {fr.get('iman_davenport_F', float('nan')):.3f},"
                  f" p = {fr.get('iman_davenport_p', float('nan')):.3g}"]
        if cd:
            lines.append(f"  Nemenyi critical difference = "
                         f"{cd['critical_difference']:.3f}")
        mr = fr.get("mean_ranks", {})
        if mr:
            lines.append("  mean ranks: " + ", ".join(
                f"{k} {v:.2f}" for k, v in sorted(mr.items(), key=lambda kv: kv[1])))
        lines.append("")

    wins = _read(out, "05_family_wins")
    wsum = _readj(out, "05_family_wins_summary")
    if wins is not None and len(wins) and wsum:
        lines += ["PER-BLOCK WIN COUNT (best neural vs best conventional)",
                  wins[["block", "best_neural_model", "best_neural",
                        "best_conventional_model", "best_conventional",
                        "winner"]].round(3).to_string(index=False),
                  f"  neural wins {wsum['neural_wins']} of {wsum['n_blocks']} blocks"
                  f"  ({wsum['n_stations']} stations x {wsum['n_targets']} targets"
                  f" - the denominator is counted, not assumed)"]
        alt = wsum.get("alternate_classification")
        if alt:
            lines.append(f"  with {', '.join(alt['moved_to_conventional'])} counted as "
                         f"conventional: {alt['neural_wins']} of {wsum['n_blocks']}")
        lines += [f"  critical difference {wsum['critical_difference']:.3f}; "
                  f"{wsum['n_models_within_cd_of_best']} models lie within one CD "
                  f"of the best mean rank",
                  f"  RESOLVABLE: {wsum['resolvable']} - {wsum['caveat']}",
                  "  quote this count only together with the line above.", ""]
        tex.append(_latex(wins[["block", "best_neural", "best_conventional",
                                "winner"]].round(3),
                          f"Best neural versus best conventional model in each of the "
                          f"{wsum['n_blocks']} station--target blocks. MASE, lower is "
                          f"better. The ordering is not resolvable at this sample size: "
                          f"{wsum['n_models_within_cd_of_best']} models lie within one "
                          f"critical difference of the best mean rank",
                          "tab:winloss"))

    nvc = _readj(out, "05_neural_vs_conventional")
    if nvc:
        lines += ["NEURAL VERSUS CONVENTIONAL",
                  f"  median MASE neural {nvc['median_neural']:.3f} vs "
                  f"conventional {nvc['median_conventional']:.3f}",
                  f"  Mann-Whitney p = {nvc['p']:.3g}; P(neural better) = "
                  f"{nvc['cles_neural_better']:.3f}", ""]

    eff = _read(out, "06_factor_effects")
    if eff is not None and len(eff):
        lines += ["DESIGN FACTORS (spread in test MASE induced by each)",
                  eff.round(3).to_string(index=False), ""]
        tex.append(_latex(eff.round(3),
                          "Spread in test MASE attributable to each design factor, "
                          "measured as the mean within-cell range",
                          "tab:factors"))

    ag = _read(out, "07_attribution_agreement")
    fa = _read(out, "07_faithfulness")
    if ag is not None and len(ag):
        lines += ["EXPLANATIONS",
                  f"  cross-method Spearman: median {ag['spearman'].median():.3f}, "
                  f"range {ag['spearman'].min():.3f} to {ag['spearman'].max():.3f}"]
        if fa is not None and len(fa):
            lines.append(f"  faithful (top-k ablation beats random, p<0.05): "
                         f"{int(fa['faithful'].sum())}/{len(fa)} model-blocks")
        stab = _read(out, "07_stability")
        if stab is not None and len(stab):
            bys = stab.groupby("noise")["spearman_vs_reference"].median()
            lines.append("  rank stability by input-noise scale: " +
                         ", ".join(f"{k:g}->{v:.3f}" for k, v in bys.items()))
        lines.append("")

    disc = _read(out, "08_discrimination")
    head = _readj(out, "08_headline")
    if disc is not None and len(disc):
        pooled = disc[disc["scope"] == "pooled"].head(10)
        lines += ["FLOOD-YEAR DISCRIMINATION (pooled, top 10 by AUROC)",
                  pooled[["indicator", "kind", "auroc", "auroc_lo", "auroc_hi",
                          "p_perm"]].round(3).to_string(index=False), ""]
        tex.append(_latex(pooled[["indicator", "kind", "auroc", "auroc_lo",
                                  "auroc_hi", "p_perm"]].round(3),
                          "Discrimination of documented flood years by annual "
                          "indicator, pooled over stations", "tab:proxy"))
    if head:
        lines += ["HEADLINE",
                  f"  best shift target : {head['best_target']} "
                  f"AUROC {head['best_target_auroc']:.3f} (p={head['best_target_p']:.3f})",
                  f"  best raw descriptor: {head['best_descriptor']} "
                  f"AUROC {head['best_descriptor_auroc']:.3f} "
                  f"(p={head['best_descriptor_p']:.3f})",
                  f"  a raw descriptor outperforms every shift target: "
                  f"{head['descriptor_beats_all_targets']}", ""]

    pp_crps = _read(out, "09_postproc_summary_CRPS")
    pp_acf = _read(out, "09_postproc_summary_acf1_err")
    pp_t = _read(out, "09_postproc_tests_CRPS")
    if pp_crps is not None and len(pp_crps):
        lines += ["POST-PROCESSING ABLATION (leave-one-year-out)",
                  "  by CRPS (marginal accuracy):",
                  pp_crps[["config", "mean", "delta_vs_full",
                           "pct_vs_full"]].round(4).to_string(index=False)]
        if pp_acf is not None and len(pp_acf):
            lines += ["  by lag-1 autocorrelation error (temporal structure):",
                      pp_acf[["config", "mean", "delta_vs_full",
                              "pct_vs_full"]].round(4).to_string(index=False)]
        if pp_t is not None and len(pp_t):
            keep = pp_t[pp_t["component"] != "-"]
            lines += ["  components whose removal significantly worsens CRPS "
                      "(Wilcoxon, Holm):",
                      keep[["component", "median_delta", "p_holm",
                            "worse_than_full"]].round(4).to_string(index=False)]
        lines += ["  note: a component that costs CRPS may still earn its place on "
                  "structure;", "        the Schaake shuffle is the clear case.", ""]
        tex.append(_latex(pp_crps.round(4),
                          "Leave-one-year-out ablation of daily post-processing "
                          "components, scored by CRPS", "tab:postproc"))

    (out / "SUMMARY.txt").write_text("\n".join(lines))
    (out / "tables.tex").write_text("\n".join(tex))
    print(f"      -> SUMMARY.txt ({len(lines)} lines), tables.tex ({len(tex)} tables)")
