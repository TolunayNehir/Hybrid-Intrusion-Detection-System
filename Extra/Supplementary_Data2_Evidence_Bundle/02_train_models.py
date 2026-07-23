"""
All hyperparameters below are the *documented* defaults from
Hybrid-Intrusion-Detection-System-main/config/settings.py.

Models trained (5 seeds each unless noted):
  - CNN            : agents/classical_ml/cnn_model.py (_CNNNet), direct numeric-feature mode
  - RNN             : agents/classical_ml/rnn_model.py (_RNNNet), non-temporal feature-grouped
  - QuantumAI       : agents/quantum_ai/quantum_module.py (DimensionReducerNormalizer + QiskitQMLP)
                      -> reports BOTH p_circ (diagnostic circuit-only) and p_deploy (deployed
                      Fisher/PCA-linear decision), never conflated
  - Classical-8D    : logistic regression on the SAME q=8 Fisher/PCA features QuantumAI uses
                      (deterministic -- run once, reported as such)
  - VQC comparator  : manuscript/experiments/vqc_model.py -- genuine re-uploading circuit,
                      decision from its own readout qubit, no classical-linear shortcut
  - Cosine/kNN      : agents/similarity/cosine_analyzer.py logic, reimplemented inline here
                      (deterministic, no seed needed)

Outputs (per seed, per model): predicted probabilities on train/val/test to
manuscript/experiments/results/predictions/<model>_seed<seed>.npz
"""
import sys
import json
import os
import time
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from agents.classical_ml.cnn_model import ClassicalCNNModel
from agents.classical_ml.rnn_model import ClassicalRNNModel
from agents.quantum_ai.quantum_module import QuantumAIModule
from sklearn.linear_model import LogisticRegression

from vqc_model import ReuploadingVQC
from cosine_scoring import cosine_knn_scores

RESULTS_DIR = Path(__file__).parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "model_states"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def load_locked_data():
    df = pd.read_csv(RESULTS_DIR / "locked_dataset.csv")
    feature_cols = [c for c in df.columns if c not in ("row_id", "partition", "label", "target_binary")]

    train = df[df.partition == "train"].reset_index(drop=True)
    val = df[df.partition == "validation"].reset_index(drop=True)
    test = df[df.partition == "test"].reset_index(drop=True)

    X_train_raw = train[feature_cols].to_numpy(dtype=np.float64)
    X_val_raw = val[feature_cols].to_numpy(dtype=np.float64)
    X_test_raw = test[feature_cols].to_numpy(dtype=np.float64)

    y_train = train["target_binary"].to_numpy(dtype=np.float32)
    y_val = val["target_binary"].to_numpy(dtype=np.float32)
    y_test = test["target_binary"].to_numpy(dtype=np.float32)

    # Z-score normalize (fit on train only), clip to [-5, 5] -- same convention as
    # utils/dataset_analyzer.py's build_feature_matrix.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-8] = 1.0

    def norm(X):
        return np.clip((X - mean) / std, -5.0, 5.0).astype(np.float32)

    return {
        "X_train": norm(X_train_raw), "y_train": y_train,
        "X_val": norm(X_val_raw), "y_val": y_val,
        "X_test": norm(X_test_raw), "y_test": y_test,
        "feature_mean": mean, "feature_std": std,
        "feature_cols": feature_cols,
    }


# Cosine / kNN + centroid agent (deterministic) -- the canonical scoring
# function now lives in cosine_scoring.py (imported above) so the latency
# benchmark and SSH case study cannot silently drift from this evaluation
# path again (see cosine_scoring.py's module docstring for why).


def main():
    data = load_locked_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"Locked splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}  "
          f"n_features={X_train.shape[1]}")

    timing_log = {}

    # ---- Cosine/kNN (deterministic, single run) ---------------------------------
    t0 = time.perf_counter()
    for split_name, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        scores = cosine_knn_scores(X_train, y_train, X)
        np.savez(PRED_DIR / f"cosine_{split_name}.npz", prob=scores,
                 y=data[f"y_{split_name}" if split_name != "val" else "y_val"])
    timing_log["cosine"] = time.perf_counter() - t0
    print(f"Cosine/kNN done ({timing_log['cosine']:.2f}s)")

    # ---- Classical-8D logistic regression (deterministic) -----------------------
    # Uses the SAME Fisher+PCA q=8 reducer the QuantumAI head consumes -- fit once
    # (on seed 0's reducer, since the reducer fit itself is deterministic given
    # fixed X_train, y_train and involves no random init beyond a degenerate
    # fallback branch that isn't triggered here).
    qmod_for_reducer = QuantumAIModule()
    qmod_for_reducer.reducer.fit(X_train, y_train)
    R_train = qmod_for_reducer.reducer.transform(X_train)
    R_val = qmod_for_reducer.reducer.transform(X_val)
    R_test = qmod_for_reducer.reducer.transform(X_test)
    np.savez(RESULTS_DIR / "fisher_pca_features.npz",
             R_train=R_train, R_val=R_val, R_test=R_test)

    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(R_train, y_train)
    for split_name, R, y in (("train", R_train, y_train), ("val", R_val, y_val), ("test", R_test, y_test)):
        prob = clf.predict_proba(R)[:, 1]
        np.savez(PRED_DIR / f"classical8d_{split_name}.npz", prob=prob, y=y)
    timing_log["classical8d"] = time.perf_counter() - t0
    print(f"Classical-8D logistic regression done ({timing_log['classical8d']:.2f}s) "
          f"[deterministic -- single run, not repeated over seeds]")

    # ---- Seeded models: CNN, RNN, QuantumAI, VQC ---------------------------------
    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- CNN ---
        t0 = time.perf_counter()
        cnn = ClassicalCNNModel()
        cnn.train(X_train, y_train, batch_size=32)
        cnn.net.eval()
        with torch.no_grad():
            for split_name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
                logits = cnn.net(torch.tensor(X, dtype=torch.float32))
                prob = torch.sigmoid(logits).numpy()
                np.savez(PRED_DIR / f"cnn_seed{seed}_{split_name}.npz", prob=prob, y=y)
        torch.save(cnn.net.state_dict(), MODEL_DIR / f"cnn_seed{seed}.pt")
        dt_cnn = time.perf_counter() - t0
        print(f"  CNN trained ({dt_cnn:.2f}s)")

        # --- RNN ---
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.perf_counter()
        rnn = ClassicalRNNModel()
        rnn.train(X_train, y_train, batch_size=32)
        rnn.net.eval()
        with torch.no_grad():
            for split_name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
                logits = rnn.net(torch.tensor(X, dtype=torch.float32))
                prob = torch.sigmoid(logits).numpy()
                np.savez(PRED_DIR / f"rnn_seed{seed}_{split_name}.npz", prob=prob, y=y)
        torch.save(rnn.net.state_dict(), MODEL_DIR / f"rnn_seed{seed}.pt")
        dt_rnn = time.perf_counter() - t0
        print(f"  RNN trained ({dt_rnn:.2f}s)")

        # --- QuantumAI (QMLP): p_circ (diagnostic) + p_deploy (deployed) ---
        t0 = time.perf_counter()
        qmod = QuantumAIModule()
        nq = qmod.qnn.n_quantum_params
        rng = np.random.RandomState(seed)
        qmod.qnn.weights = np.concatenate([
            rng.uniform(-0.2, 0.2, nq),
            rng.randn(qmod.qnn.n_qubits) * 0.05,
            np.zeros(1),
        ])
        np.random.seed(seed)  # controls Phase-2 SPSA stochasticity (minibatch idx + perturbation sign)
        qmod.train(X_train, y_train)

        for split_name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
            R = qmod.reducer.transform(X)
            # p_deploy: linear head directly on Fisher/PCA features (Eq. 11 of Clean draft)
            w_head = qmod.qnn.weights[nq: nq + qmod.qnn.n_qubits]
            b_head = qmod.qnn.weights[-1]
            fisher_feats = (R / np.pi).astype(np.float64)
            logit = fisher_feats @ w_head + b_head
            p_deploy = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
            # p_circ: full statevector circuit forward pass (diagnostic only)
            p_circ = qmod.qnn.forward_batch(R)
            np.savez(PRED_DIR / f"quantumai_seed{seed}_{split_name}.npz",
                     p_deploy=p_deploy, p_circ=p_circ, y=y, threshold=qmod.threshold)
        np.savez(MODEL_DIR / f"quantumai_seed{seed}.npz", weights=qmod.qnn.weights, threshold=qmod.threshold)
        dt_q = time.perf_counter() - t0
        print(f"  QuantumAI (QMLP) trained ({dt_q:.2f}s), threshold={qmod.threshold:.4f}")

        # --- VQC comparator (genuine circuit-driven decision) ---
        t0 = time.perf_counter()
        vqc = ReuploadingVQC(n_qubits=8, n_layers=4)
        best_loss = vqc.train(R_train, y_train, seed=seed)
        for split_name, R, y in (("train", R_train, y_train), ("val", R_val, y_val), ("test", R_test, y_test)):
            p_vqc = vqc.forward_batch(R)
            np.savez(PRED_DIR / f"vqc_seed{seed}_{split_name}.npz", prob=p_vqc, y=y)
        np.savez(MODEL_DIR / f"vqc_seed{seed}.npz", weights=vqc.weights)
        dt_vqc = time.perf_counter() - t0
        print(f"  VQC comparator trained ({dt_vqc:.2f}s), final_loss={best_loss:.4f}")

        timing_log[f"seed{seed}"] = {
            "cnn": dt_cnn, "rnn": dt_rnn, "quantumai": dt_q, "vqc": dt_vqc,
        }

    with open(RESULTS_DIR / "training_timing.json", "w") as f:
        json.dump(timing_log, f, indent=2)

    print("\nAll models trained. Predictions written to results/predictions/, "
          "model states written to results/model_states/.")


if __name__ == "__main__":
    main()
