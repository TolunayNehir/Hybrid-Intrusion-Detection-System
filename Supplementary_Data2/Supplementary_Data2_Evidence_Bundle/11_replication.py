"""
Replication of the locked protocol on two additional public benchmarks:
NSL-KDD and UNSW-NB15.

Protocol is IDENTICAL to the primary CICIDS2017-derived experiment:
  - moderate corpus (n=1771), numeric features only, identifiers excluded
  - exact-feature-fingerprint dedupe, group-aware immutable 3-way split
    (proportions matched to the primary corpus: 74/13/13), zero cross-partition
    fingerprints asserted programmatically
  - all agents retrained with the FROZEN hyperparameters from config/settings.py
    (no re-tuning on the new datasets -- a configuration-transfer replication)
  - thresholds selected on D_val only (macro-F1 sweep), D_test touched once
    per model per seed; 5 seeds; fusion equal-weight + logistic-stacked;
    paired significance fusion vs best single agent

Outputs: experiments/results_replication/<dataset>/
"""
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
REPO = Path(os.environ["HYBRID_IDS_REPO"]).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(P))

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score)
from scipy import stats

from agents.classical_ml.cnn_model import ClassicalCNNModel
from agents.classical_ml.rnn_model import ClassicalRNNModel
from agents.quantum_ai.quantum_module import QuantumAIModule
from vqc_model import ReuploadingVQC
from cosine_scoring import cosine_knn_scores

DATA = P / "replication_data"
OUT = P / "results_replication"
SEEDS = [0, 1, 2, 3, 4]
SPLIT_FRACS = (0.74, 0.13, 0.13)  # matches primary corpus 1311/232/228
SAMPLE_SEED = 42

NSL_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"]
NSL_FAMILY = {}
for a in ["back","land","neptune","pod","smurf","teardrop"]: NSL_FAMILY[a] = "DoS"
for a in ["ipsweep","nmap","portsweep","satan"]: NSL_FAMILY[a] = "Probe"
for a in ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]: NSL_FAMILY[a] = "R2L"
for a in ["buffer_overflow","loadmodule","perl","rootkit"]: NSL_FAMILY[a] = "U2R"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_nslkdd():
    df = pd.read_csv(DATA / "KDDTrain+.txt", header=None, names=NSL_COLS)
    df["family"] = df["label"].map(lambda l: "Normal" if l == "normal" else NSL_FAMILY.get(l, "Other"))
    df = df[df["family"] != "Other"]
    rng = np.random.RandomState(SAMPLE_SEED)
    quota = {"Normal": 750, "DoS": 334, "Probe": 333, "R2L": 333, "U2R": 21}
    parts = []
    for fam, n in quota.items():
        sub = df[df["family"] == fam]
        idx = rng.choice(sub.index.values, size=min(n, len(sub)), replace=False)
        parts.append(df.loc[idx])
    corpus = pd.concat(parts).reset_index(drop=True)
    feat_cols = [c for c in NSL_COLS if c not in ("protocol_type", "service", "flag", "label", "difficulty")]
    X = corpus[feat_cols].to_numpy(np.float64)
    y = (corpus["family"] != "Normal").astype(np.float32).to_numpy()
    fam = corpus["family"].to_numpy()
    return X, y, fam, feat_cols, {"source_file": "KDDTrain+.txt", "sha256": sha256(DATA / "KDDTrain+.txt"),
                                  "quota": quota, "n_features": len(feat_cols),
                                  "categorical_dropped": ["protocol_type", "service", "flag"]}


def build_unsw():
    df = pd.read_csv(DATA / "UNSW_NB15_training-set.csv")
    drop = ["id", "proto", "service", "state", "attack_cat", "label"]
    feat_cols = [c for c in df.columns if c not in drop]
    rng = np.random.RandomState(SAMPLE_SEED)
    quota = {"Normal": 750, "Exploits": 334, "Fuzzers": 333, "Reconnaissance": 333, "Worms": 21}
    parts = []
    for fam, n in quota.items():
        sub = df[df["attack_cat"] == fam]
        idx = rng.choice(sub.index.values, size=min(n, len(sub)), replace=False)
        parts.append(df.loc[idx])
    corpus = pd.concat(parts).reset_index(drop=True)
    X = corpus[feat_cols].to_numpy(np.float64)
    y = corpus["label"].astype(np.float32).to_numpy()
    fam = corpus["attack_cat"].to_numpy()
    return X, y, fam, feat_cols, {"source_file": "UNSW_NB15_training-set.csv",
                                  "sha256": sha256(DATA / "UNSW_NB15_training-set.csv"),
                                  "quota": quota, "n_features": len(feat_cols),
                                  "categorical_dropped": ["proto", "service", "state"]}


def locked_split(X, y, fam):
    """Fingerprint-dedupe group-aware split, mirroring 01_build_locked_split.py."""
    n = len(X)
    rng = np.random.RandomState(SAMPLE_SEED)
    order = rng.permutation(n)
    n_tr = int(round(SPLIT_FRACS[0] * n)); n_va = int(round(SPLIT_FRACS[1] * n))
    part = np.empty(n, dtype=object)
    part[order[:n_tr]] = "train"; part[order[n_tr:n_tr + n_va]] = "validation"
    part[order[n_tr + n_va:]] = "test"
    # fingerprint on exact model-facing vector (rounded 6 decimals)
    fps = np.array([hashlib.sha256(np.round(x, 6).tobytes()).hexdigest() for x in X])
    groups = {}
    for i, f in enumerate(fps): groups.setdefault(f, []).append(i)
    n_multi, n_moved = 0, 0
    prio = {"train": 0, "validation": 1, "test": 2}
    for f, idxs in groups.items():
        parts_here = set(part[i] for i in idxs)
        if len(parts_here) > 1:
            n_multi += 1
            counts = {}
            for i in idxs: counts[part[i]] = counts.get(part[i], 0) + 1
            best = sorted(counts.items(), key=lambda kv: (-kv[1], prio[kv[0]]))[0][0]
            for i in idxs:
                if part[i] != best: part[i] = best; n_moved += 1
    # assert zero cross-partition
    for f, idxs in groups.items():
        assert len(set(part[i] for i in idxs)) == 1
    return part, {"n_fingerprint_groups": len(groups), "n_multi_partition_groups": n_multi,
                  "n_rows_reassigned": n_moved,
                  "partition_sizes": {p: int((part == p).sum()) for p in ("train", "validation", "test")}}


def macroF1_threshold(prob_val, yval):
    best_t, best = 0.5, -1.0
    for t in np.arange(0.01, 1.0, 0.01):
        f = f1_score(yval, (prob_val > t).astype(int), average="macro")
        if f > best: best, best_t = f, t
    return best_t


def metrics(y, prob, t):
    yh = (prob > t).astype(int)
    out = {"macro_f1": f1_score(y, yh, average="macro"),
           "precision": precision_score(y, yh, zero_division=0),
           "recall": recall_score(y, yh, zero_division=0),
           "mcc": matthews_corrcoef(y, yh)}
    try: out["roc_auc"] = roc_auc_score(y, prob)
    except Exception: out["roc_auc"] = float("nan")
    return out


def run_dataset(name, build):
    print(f"\n################ {name} ################")
    X_raw, y, fam, feat_cols, manifest = build()
    part, split_info = locked_split(X_raw, y, fam)
    tr, va, te = part == "train", part == "validation", part == "test"
    print(f"n={len(X_raw)} feats={X_raw.shape[1]} split={split_info['partition_sizes']} "
          f"multi-groups={split_info['n_multi_partition_groups']} moved={split_info['n_rows_reassigned']}")
    mean = X_raw[tr].mean(0); std = X_raw[tr].std(0); std[std < 1e-8] = 1.0
    Z = np.clip((X_raw - mean) / std, -5.0, 5.0).astype(np.float32)
    Xtr, Xva, Xte = Z[tr], Z[va], Z[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    outdir = OUT / name; outdir.mkdir(parents=True, exist_ok=True)
    preds = {}  # (agent, seed or None) -> {'val':..., 'test':...}

    t0 = time.perf_counter()
    for split_name, Xs in (("val", Xva), ("test", Xte)):
        preds.setdefault(("cosine", None), {})[split_name] = cosine_knn_scores(Xtr, ytr, Xs)
    # Fisher/PCA q=8 reducer (train-only fit) + Classical-8D
    qref = QuantumAIModule(); qref.reducer.fit(Xtr, ytr)
    Rtr, Rva, Rte = qref.reducer.transform(Xtr), qref.reducer.transform(Xva), qref.reducer.transform(Xte)
    clf = LogisticRegression(max_iter=1000, random_state=0).fit(Rtr, ytr)
    preds[("classical8d", None)] = {"val": clf.predict_proba(Rva)[:, 1], "test": clf.predict_proba(Rte)[:, 1]}
    print(f"deterministic agents done ({time.perf_counter()-t0:.1f}s)")

    for seed in SEEDS:
        ts = time.perf_counter()
        torch.manual_seed(seed); np.random.seed(seed)
        cnn = ClassicalCNNModel(); cnn.train(Xtr, ytr, batch_size=32); cnn.net.eval()
        with torch.no_grad():
            preds[("cnn", seed)] = {s: torch.sigmoid(cnn.net(torch.tensor(Xs))).numpy()
                                    for s, Xs in (("val", Xva), ("test", Xte))}
        torch.manual_seed(seed); np.random.seed(seed)
        rnn = ClassicalRNNModel(); rnn.train(Xtr, ytr, batch_size=32); rnn.net.eval()
        with torch.no_grad():
            preds[("rnn", seed)] = {s: torch.sigmoid(rnn.net(torch.tensor(Xs))).numpy()
                                    for s, Xs in (("val", Xva), ("test", Xte))}
        # QuantumAI (deployed + circuit-only), same init convention as 02_train_models.py
        qmod = QuantumAIModule()
        nq = qmod.qnn.n_quantum_params
        rng = np.random.RandomState(seed)
        qmod.qnn.weights = np.concatenate([rng.uniform(-0.2, 0.2, nq),
                                           rng.randn(qmod.qnn.n_qubits) * 0.05, np.zeros(1)])
        np.random.seed(seed)
        qmod.train(Xtr, ytr)
        for s, Xs in (("val", Xva), ("test", Xte)):
            R = qmod.reducer.transform(Xs)
            w = qmod.qnn.weights[nq:nq + qmod.qnn.n_qubits]; b = qmod.qnn.weights[-1]
            logit = (R / np.pi) @ w + b
            preds.setdefault(("quantumai_deploy", seed), {})[s] = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
            preds.setdefault(("quantumai_circ", seed), {})[s] = qmod.qnn.forward_batch(R)
        vqc = ReuploadingVQC(n_qubits=8, n_layers=4); vqc.train(Rtr, ytr, seed=seed)
        preds[("vqc", seed)] = {"val": vqc.forward_batch(Rva), "test": vqc.forward_batch(Rte)}
        print(f"  seed {seed} done ({time.perf_counter()-ts:.1f}s)")

    # ---- evaluation ----
    def agent_prob(agent, seed, split):
        key = (agent, seed if (agent, seed) in preds else None)
        return np.asarray(preds[key][split], dtype=np.float64)

    summary, per_seed = {}, {}
    seeded = ["cnn", "rnn", "quantumai_deploy", "quantumai_circ", "vqc"]
    determ = ["classical8d", "cosine"]
    for agent in seeded + determ:
        rows = []
        for seed in (SEEDS if agent in seeded else [None]):
            pv, pt = agent_prob(agent, seed, "val"), agent_prob(agent, seed, "test")
            t = macroF1_threshold(pv, yva)
            m = metrics(yte, pt, t); m["threshold"] = t; m["seed"] = seed
            rows.append(m)
        per_seed[agent] = rows
        summary[agent] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                              "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
                          for k in ("macro_f1", "precision", "recall", "mcc", "roc_auc")}

    # fusion (equal weight over the 4 deployed agents) + logistic stacker (val-fit)
    fusion_rows, stacked_rows, fus_diffs = [], [], []
    best_single = max(("cnn", "rnn"), key=lambda a: summary[a]["macro_f1"]["mean"])
    for seed in SEEDS:
        pv = np.mean([agent_prob(a, seed, "val") for a in ("quantumai_deploy", "cosine", "rnn", "cnn")], 0)
        pt = np.mean([agent_prob(a, seed, "test") for a in ("quantumai_deploy", "cosine", "rnn", "cnn")], 0)
        t = macroF1_threshold(pv, yva)
        m = metrics(yte, pt, t); fusion_rows.append(m)
        # stacked
        Vs = np.column_stack([agent_prob(a, seed, "val") for a in ("quantumai_deploy", "cosine", "rnn", "cnn")])
        Ts = np.column_stack([agent_prob(a, seed, "test") for a in ("quantumai_deploy", "cosine", "rnn", "cnn")])
        st = LogisticRegression(max_iter=1000, random_state=0).fit(Vs, yva.astype(int))
        pv2, pt2 = st.predict_proba(Vs)[:, 1], st.predict_proba(Ts)[:, 1]
        stacked_rows.append(metrics(yte, pt2, macroF1_threshold(pv2, yva)))
        # paired diff vs best single
        pbs = agent_prob(best_single, seed, "test")
        tbs = macroF1_threshold(agent_prob(best_single, seed, "val"), yva)
        fus_diffs.append(m["macro_f1"] - metrics(yte, pbs, tbs)["macro_f1"])
    for label, rows in (("fusion_equal", fusion_rows), ("fusion_stacked", stacked_rows)):
        summary[label] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                              "std": float(np.std([r[k] for r in rows], ddof=1))}
                          for k in ("macro_f1", "precision", "recall", "mcc", "roc_auc")}
    d = np.array(fus_diffs)
    tt = stats.ttest_rel(d, np.zeros_like(d))
    try: wz = stats.wilcoxon(d, zero_method="pratt")
    except Exception: wz = None
    paired = {"fusion_minus_best_single": {"best_single": best_single,
              "per_seed_diff": [float(x) for x in d], "mean_diff": float(d.mean()),
              "ttest_p": float(tt.pvalue), "wilcoxon_p": float(wz.pvalue) if wz else None}}

    result = {"dataset": name, "manifest": manifest, "split": split_info,
              "summary": summary, "per_seed": per_seed, "paired": paired,
              "environment": {"torch": torch.__version__, "python": sys.version.split()[0],
                              "os": "Windows 10", "frozen_hparams": "config/settings.py (no re-tuning)"}}
    with open(outdir / "replication_results.json", "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"--- {name} summary (macro-F1 mean+/-std) ---")
    for a in seeded + determ + ["fusion_equal", "fusion_stacked"]:
        s = summary[a]["macro_f1"]; print(f"  {a:18s} {s['mean']:.3f} +/- {s['std']:.3f}")
    print("  paired fusion-vs-%s: mean diff %.4f, t-test p=%.3f" %
          (best_single, d.mean(), tt.pvalue))
    return result


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    r1 = run_dataset("nslkdd", build_nslkdd)
    r2 = run_dataset("unsw", build_unsw)
    print("\nAll replication runs complete.")
