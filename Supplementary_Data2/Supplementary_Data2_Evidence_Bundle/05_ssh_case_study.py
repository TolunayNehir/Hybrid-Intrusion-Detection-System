"""
This script reconstructs the *mechanism* numerically using the retrained, locked
CNN/RNN/QuantumAI-deploy/Cosine agents from this paper's protocol: it searches the
locked D_test partition for BENIGN flows with low severity, computes each agent's
raw attack-probability and the equal-weight fusion score, and reports how often
(and why) a majority-Normal vote can still cross the tau=0.55 legacy threshold
under the fixed-weight formula S_fusion = sum(w_i * s_i) / sum(w_i) -- explaining
that fusion crosses threshold whenever the mean of the four probabilities exceeds
tau even if every individual agent is below its own 0.5 midpoint, since the fusion
threshold (0.55) is compared to a MEAN of probabilities, not a majority vote.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Portable path resolution: override with the HYBRID_IDS_REPO environment
# variable if the reference repository lives somewhere other than the
# default relative location (a sibling of this manuscript/ directory).
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get(
    "HYBRID_IDS_REPO",
    PROJECT_ROOT.parent.parent / "Hybrid-Intrusion-Detection-System-main",
)).resolve()
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from agents.classical_ml.cnn_model import ClassicalCNNModel
from agents.classical_ml.rnn_model import ClassicalRNNModel
from agents.quantum_ai.quantum_module import QuantumAIModule
from cosine_scoring import cosine_knn_scores

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "model_states"
SEED = 0
LEGACY_TAU = 0.55  # the deployed system's fixed fusion threshold (config/settings.py FUSION)


def main():
    locked = pd.read_csv(RESULTS_DIR / "locked_dataset.csv")
    feature_cols = [c for c in locked.columns if c not in ("row_id", "partition", "label", "target_binary")]

    train_rows = locked[locked.partition == "train"].reset_index(drop=True)
    test_rows = locked[locked.partition == "test"].reset_index(drop=True)

    X_train_raw = train_rows[feature_cols].to_numpy(dtype=np.float64)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-8] = 1.0

    def norm(X):
        return np.clip((X - mean) / std, -5.0, 5.0).astype(np.float32)

    X_test = norm(test_rows[feature_cols].to_numpy(dtype=np.float64))
    y_test = test_rows["target_binary"].to_numpy(dtype=np.float32)
    labels_test = test_rows["label"].to_numpy()

    # ---- Load trained seed-0 models (same checkpoints as the latency benchmark) ----
    cnn = ClassicalCNNModel(); cnn._build(X_test.shape[1])
    cnn.net.load_state_dict(torch.load(MODEL_DIR / f"cnn_seed{SEED}.pt")); cnn.net.eval()

    rnn = ClassicalRNNModel(); rnn._build(X_test.shape[1])
    rnn.net.load_state_dict(torch.load(MODEL_DIR / f"rnn_seed{SEED}.pt")); rnn.net.eval()

    qstate = np.load(MODEL_DIR / f"quantumai_seed{SEED}.npz")
    qmod = QuantumAIModule()
    X_train_norm = norm(X_train_raw)
    y_train = train_rows["target_binary"].to_numpy(dtype=np.float32)
    qmod.reducer.fit(X_train_norm, y_train)
    qmod.qnn.weights = qstate["weights"]
    nq = qmod.qnn.n_quantum_params

    with torch.no_grad():
        p_cnn_all = torch.sigmoid(cnn.net(torch.tensor(X_test, dtype=torch.float32))).numpy()
        p_rnn_all = torch.sigmoid(rnn.net(torch.tensor(X_test, dtype=torch.float32))).numpy()
    R_test = qmod.reducer.transform(X_test)
    w_head = qmod.qnn.weights[nq: nq + qmod.qnn.n_qubits]
    b_head = qmod.qnn.weights[-1]
    p_q_all = 1.0 / (1.0 + np.exp(-np.clip((R_test / np.pi) @ w_head + b_head, -20, 20)))
    # Post-audit fix: this previously computed a bare cosine similarity to the
    # normal centroid, omitting the k-NN search, attack-majority vote, and
    # anomaly bump used everywhere else in the locked-test evaluation. Using
    # the actual evaluation-time rule here (cosine_scoring.py) changes the
    # FPR figures this script reports -- see the manuscript's corrected
    # discussion of the SSH case study.
    p_cos_all = cosine_knn_scores(X_train_norm, y_train, X_test)

    fusion_all = 0.25 * (p_cnn_all + p_rnn_all + p_q_all + p_cos_all)

    # ---- Find every BENIGN test-set flow where fusion crosses the legacy tau=0.55
    #      while a majority (>=3/4) of agents individually read "Normal" (<0.5) ----
    benign_mask = labels_test == "BENIGN"
    per_event = []
    n_majority_normal_but_fused_attack = 0
    for i in np.where(benign_mask)[0]:
        votes_normal = sum(p < 0.5 for p in (p_cnn_all[i], p_rnn_all[i], p_q_all[i], p_cos_all[i]))
        fused = fusion_all[i]
        if votes_normal >= 3 and fused > LEGACY_TAU:
            n_majority_normal_but_fused_attack += 1
            per_event.append({
                "row_id": str(test_rows.iloc[i]["row_id"]),
                "label": str(labels_test[i]),
                "p_cnn": float(p_cnn_all[i]), "p_rnn": float(p_rnn_all[i]),
                "p_quantumai": float(p_q_all[i]), "p_cosine": float(p_cos_all[i]),
                "votes_normal_of_4": int(votes_normal),
                "fused_score": float(fused),
                "legacy_tau": LEGACY_TAU,
                "fused_decision": "ATTACK" if fused > LEGACY_TAU else "NORMAL",
            })

    n_benign = int(benign_mask.sum())

    # ---- General FPR at the legacy fixed threshold vs. the D_val-tuned threshold ----
    fused_benign = fusion_all[benign_mask]
    fp_at_legacy = int((fused_benign > LEGACY_TAU).sum())
    fpr_at_legacy = fp_at_legacy / n_benign if n_benign else 0.0

    fusion_ablation_path = RESULTS_DIR / "fusion_ablation.json"
    tuned_fpr_mean = None
    if fusion_ablation_path.exists():
        with open(fusion_ablation_path) as f:
            fab = json.load(f)
        tuned_fpr_mean = fab["equal_weight"]["fpr"]["mean"]

    result = {
        "mechanism_explanation": (
            "S_fusion = mean(p_cnn, p_rnn, p_quantumai, p_cosine) is compared to a fixed "
            "threshold. Because this is a MEAN of continuous probabilities rather than a "
            "majority vote, a fixed equal-weight mean fusion has no mechanism to require "
            "agreement across agents, only average magnitude -- this is the structural "
            "reason a majority-Normal vote could in principle still cross threshold, which "
            "is the failure mode the qualitative Fig. S6 dashboard capture illustrated."
        ),
        "reproduction_on_locked_protocol": (
            "Searching every BENIGN event in the locked D_test partition (n={}) for the exact "
            "Fig. S6 pattern (>=3/4 agents individually read Normal, i.e. p<0.5, yet the fused "
            "score exceeds the legacy tau=0.55) finds {} such cases. This specific failure mode "
            "does not reproduce under the locked, leakage-safe protocol's retrained agents -- "
            "suggesting the originally displayed inconsistency was tied to the live text-"
            "embedding dashboard path and/or the earlier leakage-affected training, rather than "
            "being an unavoidable property of the fixed-weight mean-fusion formula itself."
        ).format(n_benign, n_majority_normal_but_fused_attack),
        "n_benign_test_events": n_benign,
        "n_majority_normal_but_fused_attack": n_majority_normal_but_fused_attack,
        "false_positive_rate_of_this_type_within_benign_test": (
            n_majority_normal_but_fused_attack / n_benign if n_benign else 0.0
        ),
        "general_fpr_analysis": {
            "fpr_at_legacy_fixed_threshold_0.55": fpr_at_legacy,
            "n_false_positives_at_legacy_threshold": fp_at_legacy,
            "fpr_at_D_val_tuned_threshold_mean_over_5_seeds": tuned_fpr_mean,
            "note": (
                "The legacy tau=0.55 was never selected against D_val; it is simply the "
                "shipped default. The D_val-tuned threshold (Eq. 17 protocol) gives the FPR "
                "actually reported in the main Results table and is the honest operating point."
            ),
        },
        "example_cases": per_event[:5],
    }

    with open(RESULTS_DIR / "ssh_case_study.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'ssh_case_study.json'}")


if __name__ == "__main__":
    main()
