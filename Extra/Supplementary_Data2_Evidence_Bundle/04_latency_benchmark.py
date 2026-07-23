"""
Calls each trained agent's inference path DIRECTLY (bypassing web_app.py's
/api/analyze, which computes 4 of the 5 agents twice per request. 
Measured on local machine's CPU only; no GPU is used. 
Reports single-event latency (mean/std/p95 over repeated single-sample
calls) and batch-of-100 throughput, per agent and for the full fused pipeline.
"""
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import qiskit
import sklearn

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
from vqc_model import ReuploadingVQC
from cosine_scoring import cosine_knn_score_single

RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "model_states"
SEED = 0  # benchmark uses the seed-0 trained checkpoints; latency does not depend on seed
N_WARMUP = 20     # discarded before recording, to exclude first-call/lazy-init overhead
N_SINGLE = 200    # timed single-event calls per session (post-warmup)
N_SESSIONS = 5    # independent repeated timing sessions per agent (parallels the paper's n=5 seed protocol)
N_BATCH = 100     # batch size for throughput


def get_cpu_info():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        model = [l for l in out.splitlines() if l.strip().startswith("Model name")][0].split(":", 1)[1].strip()
        n_cpus = [l for l in out.splitlines() if l.strip().startswith("CPU(s):")][0].split(":", 1)[1].strip()
    except Exception:
        model, n_cpus = platform.processor(), "unknown"
    return {
        "cpu_model": model, "n_cpus": n_cpus, "platform": platform.platform(), "device": "CPU only, no GPU",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "qiskit_version": qiskit.__version__, "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "warmup_calls_discarded_per_session": N_WARMUP,
        "n_sessions_per_agent": N_SESSIONS,
    }


def _time_one_session(fn, n, n_warmup):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return np.array(times)


def time_calls(fn, n, n_warmup=N_WARMUP, n_sessions=N_SESSIONS, raw_out_path=None):
    """Repeats the timed measurement over n_sessions independent sessions (fresh
    warmup discarded each time) and reports mean +- sample std *across
    session-level means* -- the between-session variability that actually
    matters if someone reruns this benchmark, rather than the within-session
    call-to-call jitter of a single 200-call sample."""
    session_arrays = [_time_one_session(fn, n, n_warmup) for _ in range(n_sessions)]
    session_means = np.array([a.mean() for a in session_arrays])
    pooled = np.concatenate(session_arrays)
    if raw_out_path is not None:
        np.save(raw_out_path, pooled)
    return {
        "mean_ms": float(session_means.mean()),
        "std_ms": float(session_means.std(ddof=1)),  # sample SD across sessions, not within-session call jitter
        "session_means_ms": [float(m) for m in session_means],
        "p50_ms": float(np.percentile(pooled, 50)), "p95_ms": float(np.percentile(pooled, 95)),
        "p99_ms": float(np.percentile(pooled, 99)),
        "min_ms": float(pooled.min()), "max_ms": float(pooled.max()),
        "n_warmup_discarded_per_session": n_warmup, "n_recorded_per_session": n, "n_sessions": n_sessions,
    }


def main():
    data = np.load(RESULTS_DIR / "fisher_pca_features.npz")
    df_test_features = None
    import pandas as pd
    locked = pd.read_csv(RESULTS_DIR / "locked_dataset.csv")
    feature_cols = [c for c in locked.columns if c not in ("row_id", "partition", "label", "target_binary")]
    test_rows = locked[locked.partition == "test"].reset_index(drop=True)
    X_test_raw = test_rows[feature_cols].to_numpy(dtype=np.float64)

    train_rows = locked[locked.partition == "train"].reset_index(drop=True)
    X_train_raw = train_rows[feature_cols].to_numpy(dtype=np.float64)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-8] = 1.0
    X_test = np.clip((X_test_raw - mean) / std, -5.0, 5.0).astype(np.float32)

    rng = np.random.RandomState(123)
    single_idx = rng.randint(0, len(X_test), size=N_SINGLE)
    batch_idx = rng.randint(0, len(X_test), size=N_BATCH)

    results = {"environment": get_cpu_info(), "n_single_calls": N_SINGLE, "batch_size": N_BATCH}

    # ---- CNN ----
    cnn = ClassicalCNNModel()
    cnn._build(X_test.shape[1])
    cnn.net.load_state_dict(torch.load(MODEL_DIR / f"cnn_seed{SEED}.pt"))
    cnn.net.eval()

    def cnn_single():
        with torch.no_grad():
            x = torch.tensor(X_test[single_idx[cnn_single.i % N_SINGLE]], dtype=torch.float32).unsqueeze(0)
            _ = torch.sigmoid(cnn.net(x))
        cnn_single.i += 1
    cnn_single.i = 0
    results["cnn_single"] = time_calls(cnn_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_cnn_single.npy")

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch.sigmoid(cnn.net(torch.tensor(X_test[batch_idx], dtype=torch.float32)))
    results["cnn_batch100_ms"] = (time.perf_counter() - t0) * 1000.0

    # ---- RNN ----
    rnn = ClassicalRNNModel()
    rnn._build(X_test.shape[1])
    rnn.net.load_state_dict(torch.load(MODEL_DIR / f"rnn_seed{SEED}.pt"))
    rnn.net.eval()

    def rnn_single():
        with torch.no_grad():
            x = torch.tensor(X_test[single_idx[rnn_single.i % N_SINGLE]], dtype=torch.float32).unsqueeze(0)
            _ = torch.sigmoid(rnn.net(x))
        rnn_single.i += 1
    rnn_single.i = 0
    results["rnn_single"] = time_calls(rnn_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_rnn_single.npy")

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch.sigmoid(rnn.net(torch.tensor(X_test[batch_idx], dtype=torch.float32)))
    results["rnn_batch100_ms"] = (time.perf_counter() - t0) * 1000.0

    # ---- QuantumAI (deployed Fisher/PCA-linear path -- the actual production path) ----
    qstate = np.load(MODEL_DIR / f"quantumai_seed{SEED}.npz")
    qmod = QuantumAIModule()
    X_train_norm_for_reducer = np.clip((X_train_raw - mean) / std, -5, 5).astype(np.float32)
    y_train_for_reducer = train_rows["target_binary"].to_numpy(dtype=np.float32)
    # Supervised Fisher+PCA fit is deterministic given fixed (X_train, y_train) --
    # reproduces exactly the projection used during training in 02_train_models.py.
    qmod.reducer.fit(X_train_norm_for_reducer, y_train_for_reducer)
    qmod.qnn.weights = qstate["weights"]
    nq = qmod.qnn.n_quantum_params

    def quantum_deploy_single():
        i = single_idx[quantum_deploy_single.i % N_SINGLE]
        r = qmod.reducer.transform(X_test[i:i+1])[0]
        w = qmod.qnn.weights[nq: nq + qmod.qnn.n_qubits]
        b = qmod.qnn.weights[-1]
        _ = 1.0 / (1.0 + np.exp(-np.clip((r / np.pi) @ w + b, -20, 20)))
        quantum_deploy_single.i += 1
    quantum_deploy_single.i = 0
    results["quantumai_deploy_single"] = time_calls(quantum_deploy_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_quantumai_deploy_single.npy")

    def quantum_circ_single():
        i = single_idx[quantum_circ_single.i % N_SINGLE]
        r = qmod.reducer.transform(X_test[i:i+1])[0]
        _ = qmod.qnn.forward(r)
        quantum_circ_single.i += 1
    quantum_circ_single.i = 0
    results["quantumai_circuit_single"] = time_calls(quantum_circ_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_quantumai_circuit_single.npy")

    t0 = time.perf_counter()
    r_batch = qmod.reducer.transform(X_test[batch_idx])
    _ = qmod.qnn.forward_batch(r_batch)
    results["quantumai_circuit_batch100_ms"] = (time.perf_counter() - t0) * 1000.0

    # ---- VQC comparator ----
    vstate = np.load(MODEL_DIR / f"vqc_seed{SEED}.npz")
    vqc = ReuploadingVQC(n_qubits=8, n_layers=4)
    vqc.weights = vstate["weights"]

    def vqc_single():
        i = single_idx[vqc_single.i % N_SINGLE]
        r = qmod.reducer.transform(X_test[i:i+1])[0]
        _ = vqc.forward_batch(r.reshape(1, -1))
        vqc_single.i += 1
    vqc_single.i = 0
    results["vqc_single"] = time_calls(vqc_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_vqc_single.npy")

    # ---- Cosine/kNN ----
    # Post-audit fix: this previously computed a bare cosine similarity to the
    # normal centroid with no k-NN search, no attack-majority vote, and no
    # anomaly bump -- a simplified stand-in for the actual evaluation-time
    # scoring rule, not a measurement of it. Both timed closures below now
    # call the single, canonical cosine_knn_score_single (cosine_scoring.py),
    # identical to what 02_train_models.py / 03_evaluate.py actually use.
    normal_mask = train_rows["target_binary"].to_numpy() < 0.5
    X_train_norm = np.clip((X_train_raw - mean) / std, -5, 5).astype(np.float32)
    y_train_norm = train_rows["target_binary"].to_numpy(dtype=np.float32)

    def cosine_single():
        i = single_idx[cosine_single.i % N_SINGLE]
        _ = cosine_knn_score_single(X_train_norm, y_train_norm, X_test[i])
        cosine_single.i += 1
    cosine_single.i = 0
    results["cosine_single"] = time_calls(cosine_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_cosine_single.npy")

    # ---- Full fused pipeline (4 agents + fusion arithmetic), single event ----
    def full_pipeline_single():
        i = single_idx[full_pipeline_single.i % N_SINGLE]
        x = X_test[i:i+1]
        with torch.no_grad():
            p_cnn = torch.sigmoid(cnn.net(torch.tensor(x, dtype=torch.float32))).item()
            p_rnn = torch.sigmoid(rnn.net(torch.tensor(x, dtype=torch.float32))).item()
        r = qmod.reducer.transform(x)[0]
        w = qmod.qnn.weights[nq: nq + qmod.qnn.n_qubits]
        b = qmod.qnn.weights[-1]
        p_q = float(1.0 / (1.0 + np.exp(-np.clip((r / np.pi) @ w + b, -20, 20))))
        p_cos = cosine_knn_score_single(X_train_norm, y_train_norm, x[0])
        _ = 0.25 * (p_cnn + p_rnn + p_q + p_cos)
        full_pipeline_single.i += 1
    full_pipeline_single.i = 0
    results["full_pipeline_single"] = time_calls(full_pipeline_single, N_SINGLE, raw_out_path=RESULTS_DIR / "raw_latency_full_pipeline_single.npy")
    results["full_pipeline_throughput_events_per_sec"] = 1000.0 / results["full_pipeline_single"]["mean_ms"]

    with open(RESULTS_DIR / "latency_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'latency_benchmark.json'}")


if __name__ == "__main__":
    main()
