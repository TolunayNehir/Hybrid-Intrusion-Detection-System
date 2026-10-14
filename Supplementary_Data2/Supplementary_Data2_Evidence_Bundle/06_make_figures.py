"""
Generate figures for the manuscript from results/*.json.
No numbers are hand-typed; everything is read from the experiment outputs.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = PROJECT_ROOT.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

MODEL_LABELS = {
    "cnn": "CNN", "rnn": "RNN", "vqc": "VQC (re-uploading)",
    "quantumai_deploy": "QuantumAI (deployed, Fisher/PCA-linear)",
    "quantumai_circ": "QuantumAI (circuit-only, diagnostic)",
    "classical8d": "Classical-8D (logistic reg.)", "cosine": "Cosine/kNN",
}


def fig_confusion_matrices():
    with open(RESULTS_DIR / "metrics_summary.json") as f:
        summary = json.load(f)
    models = ["cnn", "rnn", "quantumai_deploy", "quantumai_circ", "vqc", "classical8d", "cosine"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes = axes.flatten()
    for ax, name in zip(axes, models):
        cm = summary[name]["confusion_matrix_sum"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > mat.max() / 2 else "black", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        n_seeds = summary[name]["n_seeds"]
        title = MODEL_LABELS[name] + (f"\n(summed over {n_seeds} seeds)" if n_seeds > 1 else "\n(deterministic)")
        ax.set_title(title, fontsize=8)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrices.pdf")
    plt.close()
    print("Wrote confusion_matrices.pdf")


def fig_roc_pr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    seed = 0
    curves = [
        ("cnn", f"cnn_seed{seed}_test.npz", "prob"),
        ("rnn", f"rnn_seed{seed}_test.npz", "prob"),
        ("vqc", f"vqc_seed{seed}_test.npz", "prob"),
        ("quantumai_deploy", f"quantumai_seed{seed}_test.npz", "p_deploy"),
        ("classical8d", "classical8d_test.npz", "prob"),
        ("cosine", "cosine_test.npz", "prob"),
    ]
    for name, fname, key in curves:
        d = np.load(PRED_DIR / fname)
        y, p = d["y"], d[key]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, label=MODEL_LABELS[name], linewidth=1.3)
        ax2.plot(rec, prec, label=MODEL_LABELS[name], linewidth=1.3)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC (seed 0, locked $D_{test}$)")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall (seed 0, locked $D_{test}$)")
    ax1.legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_pr_curves.pdf")
    plt.close()
    print("Wrote roc_pr_curves.pdf")


def fig_latency():
    with open(RESULTS_DIR / "latency_benchmark.json") as f:
        lat = json.load(f)
    names = ["cosine_single", "quantumai_deploy_single", "rnn_single", "cnn_single",
              "quantumai_circuit_single", "vqc_single", "full_pipeline_single"]
    labels = ["Cosine/kNN", "QuantumAI\n(deployed)", "RNN", "CNN",
              "QuantumAI\n(circuit, diagnostic)", "VQC\n(comparator)", "Full fused\npipeline (4 agents)"]
    means = [lat[n]["mean_ms"] for n in names]
    stds = [lat[n]["std_ms"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0"] * (len(names) - 1) + ["#C44E52"]
    ax.bar(labels, means, yerr=stds, capsize=3, color=colors)
    ax.set_ylabel("Latency per event (ms)")
    ax.set_title(f"Single-event inference latency\n({lat['environment']['cpu_model']}, CPU only)")
    ax.set_yscale("log")
    plt.xticks(rotation=20, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "latency_benchmark.pdf")
    plt.close()
    print("Wrote latency_benchmark.pdf")


def fig_fusion_ablation():
    with open(RESULTS_DIR / "fusion_ablation.json") as f:
        fab = json.load(f)
    labels = ["Equal-weight\n(4 agents)", "Logistic-stacked\n(D_val-fit)", "Drop\nQuantumAI",
              "Drop\nCosine", "Drop\nRNN", "Drop\nCNN"]
    keys = [("equal_weight", None), ("calibrated", None), ("leave_one_out", "quantumai"),
            ("leave_one_out", "cosine"), ("leave_one_out", "rnn"), ("leave_one_out", "cnn")]
    means, stds = [], []
    for top, sub in keys:
        d = fab[top][sub] if sub else fab[top]
        means.append(d["macro_f1"]["mean"])
        stds.append(d["macro_f1"]["std"])

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#55A868", "#55A868"] + ["#DD8452"] * 4
    ax.bar(labels, means, yerr=stds, capsize=3, color=colors)
    ax.set_ylabel("Macro-F1 (locked $D_{test}$, mean $\\pm$ std over 5 seeds)")
    ax.set_title("Fusion ablation: equal-weight vs. logistic-stacked vs. leave-one-out")
    ax.set_ylim(0.85, 1.0)
    plt.xticks(rotation=15, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fusion_ablation.pdf")
    plt.close()
    print("Wrote fusion_ablation.pdf")


def fig_split_diagram():
    with open(RESULTS_DIR / "split_summary.json") as f:
        summ = json.load(f)
    sizes = summ["partition_sizes"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    parts = ["train", "validation", "test"]
    vals = [sizes[p] for p in parts]
    bars = ax.bar(parts, vals, color=["#4C72B0", "#DD8452", "#55A868"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Number of events")
    ax.set_title(
        f"Locked dataset split (n={summ['n_rows_total']}) after group-aware\n"
        f"deduplication: {summ['n_rows_reassigned']} rows reassigned across "
        f"{summ['n_multi_partition_groups_before_fix']} duplicate groups"
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "split_diagram.pdf")
    plt.close()
    print("Wrote split_diagram.pdf")


if __name__ == "__main__":
    fig_confusion_matrices()
    fig_roc_pr()
    fig_latency()
    fig_fusion_ablation()
    fig_split_diagram()
    print("\nAll figures written to", FIG_DIR)
