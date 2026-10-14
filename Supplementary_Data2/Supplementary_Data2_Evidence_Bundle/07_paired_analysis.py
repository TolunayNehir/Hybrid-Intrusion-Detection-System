"""
Outputs:
  - results/paired_analysis.json: paired differences, Wilcoxon signed-rank and
    paired t-test between (fusion equal-weight vs RNN) and (calibrated vs
    equal-weight fusion), explicitly caveated for n=5 low power.
  - results/per_seed_confusion_matrices.csv: long-format per-seed, per-model
    confusion matrices (supplementary transparency table).
  - results/vqc_per_seed_table.csv: VQC's per-seed metrics in isolation, to
    make the bimodal trainability failure visible rather than only reporting
    the aggregate mean+-std.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


# Wilcoxon result trustworthy at this sample size, and it is reported for
# transparency only, not as evidence.
WILCOXON_ZERO_METHOD = "pratt"

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    with open(RESULTS_DIR / "metrics_per_seed.json") as f:
        per_seed = json.load(f)
    with open(RESULTS_DIR / "fusion_per_seed.json") as f:
        fusion_per_seed = json.load(f)

    # ---- Paired seed-level comparison: fusion (equal-weight) vs RNN -----------
    rnn_f1 = np.array([m["macro_f1"] for m in per_seed["rnn"]])
    fusion_eq_f1 = np.array([m["macro_f1"] for m in fusion_per_seed["equal_weight"]])
    fusion_cal_f1 = np.array([m["macro_f1"] for m in fusion_per_seed["calibrated"]])

    diff_fusion_vs_rnn = fusion_eq_f1 - rnn_f1
    diff_cal_vs_eq = fusion_cal_f1 - fusion_eq_f1

    def paired_report(diff, label):
        n = len(diff)
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        ties = int((diff == 0).sum())
        n_nonzero = n - ties
        t_stat, t_p = stats.ttest_rel(diff, np.zeros(n))
        try:
            w_stat, w_p = stats.wilcoxon(diff, zero_method=WILCOXON_ZERO_METHOD)
            wilcoxon_reliable = n_nonzero >= 5  # rule of thumb: too few non-tied pairs -> not meaningful
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
            wilcoxon_reliable = False
        return {
            "label": label,
            "per_seed_diff": diff.tolist(),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if n > 1 else 0.0,  # sample SD, post-audit fix
            "n_seeds": n,
            "wins": wins, "losses": losses, "ties": ties, "n_nonzero_pairs": n_nonzero,
            "paired_ttest_stat": float(t_stat), "paired_ttest_p": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
            "wilcoxon_zero_method": WILCOXON_ZERO_METHOD,
            "scipy_version": scipy.__version__,
            "wilcoxon_reliable": wilcoxon_reliable,
            "caveat": (
                f"n={n} seeds gives very low statistical power; a paired test at this "
                f"sample size cannot reliably distinguish a small true effect from noise. "
                f"Reported for transparency, not as proof of a significant difference."
                + ("" if wilcoxon_reliable else
                   f" With only {n_nonzero} non-tied pair(s) out of {n}, the Wilcoxon "
                   f"result above is not statistically meaningful regardless of "
                   f"zero_method convention or scipy version -- report the paired "
                   f"t-test as the primary (also low-power) statistic for this "
                   f"comparison and treat the Wilcoxon number as informational only.")
            ),
        }

    fusion_vs_rnn = paired_report(diff_fusion_vs_rnn, "fusion(equal-weight) - RNN")
    cal_vs_eq = paired_report(diff_cal_vs_eq, "fusion(calibrated) - fusion(equal-weight)")

    print(f"(scipy {scipy.__version__}, wilcoxon zero_method={WILCOXON_ZERO_METHOD})")
    print("=== fusion(equal-weight) vs RNN, per-seed macro-F1 diff ===")
    print(f"  per-seed diffs: {[round(d,4) for d in diff_fusion_vs_rnn]}")
    print(f"  wins/losses/ties: {fusion_vs_rnn['wins']}/{fusion_vs_rnn['losses']}/{fusion_vs_rnn['ties']}")
    print(f"  paired t-test p={fusion_vs_rnn['paired_ttest_p']:.3f}  "
          f"Wilcoxon p={fusion_vs_rnn['wilcoxon_p']:.3f} (reliable={fusion_vs_rnn['wilcoxon_reliable']})")
    print()
    print("=== fusion(calibrated) vs fusion(equal-weight), per-seed macro-F1 diff ===")
    print(f"  per-seed diffs: {[round(d,4) for d in diff_cal_vs_eq]}")
    print(f"  wins/losses/ties: {cal_vs_eq['wins']}/{cal_vs_eq['losses']}/{cal_vs_eq['ties']}")
    print(f"  paired t-test p={cal_vs_eq['paired_ttest_p']:.3f}  "
          f"Wilcoxon p={cal_vs_eq['wilcoxon_p']:.3f} (reliable={cal_vs_eq['wilcoxon_reliable']})")

    with open(RESULTS_DIR / "paired_analysis.json", "w") as f:
        json.dump({"fusion_vs_rnn": fusion_vs_rnn, "calibrated_vs_equal_weight": cal_vs_eq}, f, indent=2)

    # ---- Per-seed confusion matrices (supplementary transparency table) ------
    rows = []
    for model_name, seeds_list in per_seed.items():
        for m in seeds_list:
            cm = m["confusion_matrix"]
            rows.append({
                "model": model_name, "seed": m.get("seed", "N/A (deterministic)"),
                "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
                "macro_f1": m["macro_f1"], "mcc": m["mcc"], "roc_auc": m["roc_auc"],
                "threshold": m["threshold"],
            })
    for fusion_name, seeds_list in fusion_per_seed.items():
        if fusion_name == "leave_one_out":
            continue
        for m in seeds_list:
            cm = m["confusion_matrix"]
            rows.append({
                "model": f"fusion_{fusion_name}", "seed": m["seed"],
                "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
                "macro_f1": m["macro_f1"], "mcc": m["mcc"], "roc_auc": m["roc_auc"],
                "threshold": m["threshold"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "per_seed_confusion_matrices.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'per_seed_confusion_matrices.csv'} ({len(df)} rows)")

    # ---- VQC per-seed table in isolation (bimodal trainability disclosure) ---
    vqc_rows = df[df.model == "vqc"][["seed", "macro_f1", "mcc", "roc_auc", "threshold", "tn", "fp", "fn", "tp"]]
    vqc_rows.to_csv(RESULTS_DIR / "vqc_per_seed_table.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'vqc_per_seed_table.csv'}")
    print(vqc_rows.to_string(index=False))


if __name__ == "__main__":
    main()
