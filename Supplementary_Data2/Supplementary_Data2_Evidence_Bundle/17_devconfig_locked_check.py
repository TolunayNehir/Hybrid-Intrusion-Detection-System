#!/usr/bin/env python3
"""
17_devconfig_locked_check.py
============================
Configuration-transfer check: transfer the development-stage auto-tuned
quantum winner configuration (config/arch_config_asfound_on_disk.json:
n_qubits=4, n_hidden_layers=3, max_iterations=80, lr=0.05, perturbation=0.01
-- the configuration behind the development record's best validation
accuracies: Phase-I 0.959, best of 18 exploratory runs on the separate
generated corpus, and the apparent Phase-II/III 1.000 on the
duplicate-overlapped development split) to the UNCHANGED locked protocol:

  - locked, duplicate-safe split from results/locked_dataset.csv
  - z-score normalization fitted on D_train only, clip [-5, 5]
  - identical weight-initialization convention as 02_train_models.py
  - five seeds (0-4)
  - decision threshold swept on D_val only (macro-F1), D_test touched once
  - a matched Classical-4D logistic baseline on the same q=4 Fisher/PCA
    features (mirror of the Classical-8D construction in 02_train_models.py)

Also run with max_iterations=100 (the Phase-II/III winner record's setting).

Output: results_devconfig_check/devconfig_locked_results.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "vendored_repo"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import QUANTUM_AI  # vendored_repo/config/settings.py

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = PROJECT_ROOT / "results_devconfig_check"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

# The auto-tuned development winner exactly as found on disk
# (config/arch_config_asfound_on_disk.json, "quantum" block).
DEV_CONFIG = {
    "n_qubits": 4,
    "n_layers": 3,
    "n_hidden_layers": 3,
    "max_iterations": 80,
    "learning_rate": 0.05,
    "perturbation": 0.01,
}


def load_locked_data():
    """Identical to 02_train_models.py::load_locked_data."""
    df = pd.read_csv(RESULTS_DIR / "locked_dataset.csv")
    feature_cols = [c for c in df.columns
                    if c not in ("row_id", "partition", "label", "target_binary")]
    train = df[df.partition == "train"].reset_index(drop=True)
    val = df[df.partition == "validation"].reset_index(drop=True)
    test = df[df.partition == "test"].reset_index(drop=True)
    X_train_raw = train[feature_cols].to_numpy(dtype=np.float64)
    X_val_raw = val[feature_cols].to_numpy(dtype=np.float64)
    X_test_raw = test[feature_cols].to_numpy(dtype=np.float64)
    y_train = train["target_binary"].to_numpy(dtype=np.float32)
    y_val = val["target_binary"].to_numpy(dtype=np.float32)
    y_test = test["target_binary"].to_numpy(dtype=np.float32)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-8] = 1.0

    def norm(X):
        return np.clip((X - mean) / std, -5.0, 5.0).astype(np.float32)

    return (norm(X_train_raw), y_train, norm(X_val_raw), y_val,
            norm(X_test_raw), y_test)


def best_threshold(y_val, p_val):
    """Identical protocol to 03_evaluate.py::best_threshold."""
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.arange(0.01, 1.00, 0.01):
        pred = (p_val >= tau).astype(int)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), float(f1)
    return best_tau, best_f1


def test_metrics(y_test, p_test, tau):
    pred = (p_test >= tau).astype(int)
    return {
        "threshold": float(tau),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def run_quantum(max_iterations, X_train, y_train, X_val, y_val, X_test, y_test):
    # Patch the vendored settings dict IN PLACE so QuantumAIModule and
    # QiskitQMLP.train() both see the development configuration.
    QUANTUM_AI.update(DEV_CONFIG)
    QUANTUM_AI["max_iterations"] = max_iterations

    from agents.quantum_ai.quantum_module import QuantumAIModule

    per_seed = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        np.random.seed(seed)
        qmod = QuantumAIModule()
        nq = qmod.qnn.n_quantum_params
        rng = np.random.RandomState(seed)
        # Same init convention as 02_train_models.py
        qmod.qnn.weights = np.concatenate([
            rng.uniform(-0.2, 0.2, nq),
            rng.randn(qmod.qnn.n_qubits) * 0.05,
            np.zeros(1),
        ])
        np.random.seed(seed)  # Phase-2 SPSA stochasticity
        qmod.train(X_train, y_train)

        rec = {"seed": seed}
        probs = {}
        for split, X in (("val", X_val), ("test", X_test)):
            R = qmod.reducer.transform(X)
            w_head = qmod.qnn.weights[nq: nq + qmod.qnn.n_qubits]
            b_head = qmod.qnn.weights[-1]
            fisher = (R / np.pi).astype(np.float64)
            logit = fisher @ w_head + b_head
            probs[("deploy", split)] = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
            probs[("circ", split)] = qmod.qnn.forward_batch(R)

        for path in ("deploy", "circ"):
            tau, val_f1 = best_threshold(y_val, probs[(path, "val")])
            m = test_metrics(y_test, probs[(path, "test")], tau)
            m["val_macro_f1_at_tau"] = val_f1
            m["val_accuracy_at_tau"] = float(accuracy_score(
                y_val, (probs[(path, "val")] >= tau).astype(int)))
            rec[path] = m
        rec["train_seconds"] = round(time.perf_counter() - t0, 2)
        per_seed.append(rec)
        print(f"  seed {seed}: deploy test mF1={rec['deploy']['macro_f1']:.4f} "
              f"(val acc@tau={rec['deploy']['val_accuracy_at_tau']:.4f}) | "
              f"circ test mF1={rec['circ']['macro_f1']:.4f} "
              f"[{rec['train_seconds']}s]", flush=True)
    return per_seed


def run_classical_4d(X_train, y_train, X_val, y_val, X_test, y_test):
    """Classical-4D: logistic regression on the SAME q=4 Fisher/PCA features
    (mirror of the Classical-8D construction in 02_train_models.py)."""
    QUANTUM_AI.update(DEV_CONFIG)
    from agents.quantum_ai.quantum_module import QuantumAIModule
    qmod = QuantumAIModule()
    qmod.reducer.fit(X_train, y_train)
    R_train = qmod.reducer.transform(X_train)
    R_val = qmod.reducer.transform(X_val)
    R_test = qmod.reducer.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(R_train, y_train)
    p_val = clf.predict_proba(R_val)[:, 1]
    p_test = clf.predict_proba(R_test)[:, 1]
    tau, val_f1 = best_threshold(y_val, p_val)
    m = test_metrics(y_test, p_test, tau)
    m["val_macro_f1_at_tau"] = val_f1
    return m


def agg(per_seed, path, key):
    v = np.array([r[path][key] for r in per_seed])
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "per_seed": [float(x) for x in v]}


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_locked_data()
    print(f"Locked splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    out = {"dev_config": dict(DEV_CONFIG)}

    print("\n=== Dev-winner config, max_iterations=80 (cache as found on disk) ===")
    ps80 = run_quantum(80, X_train, y_train, X_val, y_val, X_test, y_test)
    out["quantum_iters80"] = {
        "per_seed": ps80,
        "deploy_macro_f1": agg(ps80, "deploy", "macro_f1"),
        "deploy_mcc": agg(ps80, "deploy", "mcc"),
        "deploy_accuracy": agg(ps80, "deploy", "accuracy"),
        "circ_macro_f1": agg(ps80, "circ", "macro_f1"),
        "circ_mcc": agg(ps80, "circ", "mcc"),
    }

    print("\n=== Dev-winner config, max_iterations=100 (Phase-II/III record) ===")
    ps100 = run_quantum(100, X_train, y_train, X_val, y_val, X_test, y_test)
    out["quantum_iters100"] = {
        "per_seed": ps100,
        "deploy_macro_f1": agg(ps100, "deploy", "macro_f1"),
        "circ_macro_f1": agg(ps100, "circ", "macro_f1"),
        "circ_mcc": agg(ps100, "circ", "mcc"),
    }

    print("\n=== Classical-4D logistic baseline (same q=4 Fisher/PCA features) ===")
    c4 = run_classical_4d(X_train, y_train, X_val, y_val, X_test, y_test)
    out["classical_4d"] = c4
    print(f"  Classical-4D: test mF1={c4['macro_f1']:.4f} MCC={c4['mcc']:.4f} "
          f"acc={c4['accuracy']:.4f}")

    with open(OUT_DIR / "devconfig_locked_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'devconfig_locked_results.json'}")

    d80 = out["quantum_iters80"]["deploy_macro_f1"]
    c80 = out["quantum_iters80"]["circ_macro_f1"]
    print("\n================ SUMMARY ================")
    print(f"deploy (q=4,L=3,80it):  mF1 {d80['mean']:.3f} ± {d80['std']:.3f}")
    print(f"circ   (q=4,L=3,80it):  mF1 {c80['mean']:.3f} ± {c80['std']:.3f}")
    print(f"Classical-4D:           mF1 {c4['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
