"""
Quantum-path improvement study (three stages, locked protocol throughout):

  Stage A :
    A1. Circuit path + marginal-trained head: logistic head trained (D_train only)
        on the QMLP circuit's per-qubit marginals instead of borrowing the
        feature-trained head. Uses the as-trained QMLP quantum parameters
        (archived weights for the primary corpus; retrained for NSL-KDD/UNSW-NB15).
    A2. VQC + quantile angle scaling: per-dimension rank/quantile transform
        (fit on D_train) -> [0, pi], replacing min-max, then the paper's
        re-uploading VQC retrained as usual.

  Stage B : architecture search, VALIDATION-ONLY selection on the
    primary corpus (seeds 0-2, quantile scaling, pure single-qubit readout):
    reup-L1, reup-L2, reup-L4 (paper's), qcnn-pool.

  Stage C (combined): winning architecture + quantile scaling +
    logistic head on all 8 marginals, trained from scratch, 5 seeds, evaluated
    once per seed on the locked D_test of all three corpora.

Decision thresholds always selected on D_val only; D_test touched once per
model per seed. No test-set information is used anywhere in selection.
"""
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
REPO = Path(os.environ["HYBRID_IDS_REPO"]).resolve()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(P))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from scipy import stats

from agents.quantum_ai.quantum_module import QuantumAIModule
from qarch import make_arch

OUT = P / "results_quantum_improved"; OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
SEARCH_SEEDS = [0, 1, 2]
ARCHS = ["reup-L1", "reup-L2", "reup-L4", "qcnn-pool"]
SAMPLE_SEED = 42
SPLIT_FRACS = (0.74, 0.13, 0.13)
DATA = P / "replication_data"

def log(*a):
    print(*a, flush=True)

# ---------------- corpora (identical to 11_replication.py) ----------------
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
for a in ["back","land","neptune","pod","smurf","teardrop"]: NSL_FAMILY[a]="DoS"
for a in ["ipsweep","nmap","portsweep","satan"]: NSL_FAMILY[a]="Probe"
for a in ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]: NSL_FAMILY[a]="R2L"
for a in ["buffer_overflow","loadmodule","perl","rootkit"]: NSL_FAMILY[a]="U2R"

def _sample(df, fam_col, quota):
    rng = np.random.RandomState(SAMPLE_SEED); parts=[]
    for fam,n in quota.items():
        sub = df[df[fam_col]==fam]
        idx = rng.choice(sub.index.values, size=min(n,len(sub)), replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts).reset_index(drop=True)

def build_nslkdd():
    df = pd.read_csv(DATA/"KDDTrain+.txt", header=None, names=NSL_COLS)
    df["family"] = df["label"].map(lambda l: "Normal" if l=="normal" else NSL_FAMILY.get(l,"Other"))
    df = df[df["family"]!="Other"]
    corpus = _sample(df, "family", {"Normal":750,"DoS":334,"Probe":333,"R2L":333,"U2R":21})
    fcols = [c for c in NSL_COLS if c not in ("protocol_type","service","flag","label","difficulty")]
    return corpus[fcols].to_numpy(np.float64), (corpus["family"]!="Normal").astype(np.float32).to_numpy()

def build_unsw():
    df = pd.read_csv(DATA/"UNSW_NB15_training-set.csv")
    drop = ["id","proto","service","state","attack_cat","label"]
    fcols = [c for c in df.columns if c not in drop]
    corpus = _sample(df, "attack_cat", {"Normal":750,"Exploits":334,"Fuzzers":333,"Reconnaissance":333,"Worms":21})
    return corpus[fcols].to_numpy(np.float64), corpus["label"].astype(np.float32).to_numpy()

def locked_split(X):
    n = len(X); rng = np.random.RandomState(SAMPLE_SEED)
    order = rng.permutation(n)
    n_tr = int(round(SPLIT_FRACS[0]*n)); n_va = int(round(SPLIT_FRACS[1]*n))
    part = np.empty(n, dtype=object)
    part[order[:n_tr]]="train"; part[order[n_tr:n_tr+n_va]]="validation"; part[order[n_tr+n_va:]]="test"
    fps = np.array([hashlib.sha256(np.round(x,6).tobytes()).hexdigest() for x in X])
    groups={}
    for i,f in enumerate(fps): groups.setdefault(f,[]).append(i)
    prio={"train":0,"validation":1,"test":2}
    for f,idxs in groups.items():
        if len(set(part[i] for i in idxs))>1:
            counts={}
            for i in idxs: counts[part[i]]=counts.get(part[i],0)+1
            best=sorted(counts.items(), key=lambda kv:(-kv[1],prio[kv[0]]))[0][0]
            for i in idxs: part[i]=best
    for f,idxs in groups.items():
        assert len(set(part[i] for i in idxs))==1
    return part

def zscore(Xtr_raw, X_raw):
    m=Xtr_raw.mean(0); s=Xtr_raw.std(0); s[s<1e-8]=1.0
    return np.clip((X_raw-m)/s,-5,5)

def prep_dataset(name):
    """Return dict with z-scored X splits, y splits, and the QMLP reducer angles."""
    if name == "cicids":
        df = pd.read_csv(P/"results"/"locked_dataset.csv")
        fcols=[c for c in df.columns if c not in ("row_id","partition","label","target_binary")]
        part = df["partition"].to_numpy()
        X_raw = df[fcols].to_numpy(np.float64); y = df["target_binary"].to_numpy(np.float32)
        part = np.where(part=="validation","validation",part)
    else:
        X_raw, y = (build_nslkdd() if name=="nslkdd" else build_unsw())
        part = locked_split(X_raw)
    tr,va,te = part=="train", part=="validation", part=="test"
    Xz = zscore(X_raw[tr], X_raw)
    return {"Xtr":Xz[tr].astype(np.float32),"Xva":Xz[va].astype(np.float32),"Xte":Xz[te].astype(np.float32),
            "ytr":y[tr],"yva":y[va],"yte":y[te]}

def macroF1_threshold(pv,yv):
    best_t,best=0.5,-1
    for t in np.arange(0.01,1.0,0.01):
        f=f1_score(yv,(pv>t).astype(int),average="macro")
        if f>best: best,best_t=f,t
    return best_t

def evaluate(yte,pt,t):
    yh=(pt>t).astype(int)
    out={"macro_f1":f1_score(yte,yh,average="macro"),"mcc":matthews_corrcoef(yte,yh)}
    try: out["roc_auc"]=roc_auc_score(yte,pt)
    except Exception: out["roc_auc"]=float("nan")
    return out

def agg(rows,k):
    v=[r[k] for r in rows]
    return {"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)) if len(v)>1 else 0.0}

def quantile_angles(Atr, Ava, Ate):
    """Per-dim rank/quantile transform fit on train -> [0,pi]."""
    qt = QuantileTransformer(n_quantiles=min(1000,len(Atr)), output_distribution="uniform",
                             subsample=int(1e9), random_state=0)
    qt.fit(Atr)
    return (np.pi*qt.transform(Atr), np.pi*qt.transform(Ava), np.pi*qt.transform(Ate))


def get_qmlp_and_angles(name, d, seed):
    """Trained QMLP (quantum params) + reducer angles for all splits.
    Primary corpus: archived weights + archived Fisher/PCA angles (faithful to paper).
    NSL/UNSW: retrain exactly as in 11_replication.py."""
    if name == "cicids":
        R = np.load(P/"results"/"fisher_pca_features.npz")
        Atr,Ava,Ate = R["R_train"],R["R_val"],R["R_test"]
        w = np.load(P/"results"/"model_states"/f"quantumai_seed{seed}.npz")["weights"]
        qmod = QuantumAIModule(); qmod.qnn.weights = w
        return qmod, Atr, Ava, Ate
    qmod = QuantumAIModule()
    nq = qmod.qnn.n_quantum_params
    rng = np.random.RandomState(seed)
    qmod.qnn.weights = np.concatenate([rng.uniform(-0.2,0.2,nq),
                                       rng.randn(qmod.qnn.n_qubits)*0.05, np.zeros(1)])
    np.random.seed(seed)
    qmod.train(d["Xtr"], d["ytr"])
    return qmod, qmod.reducer.transform(d["Xtr"]), qmod.reducer.transform(d["Xva"]), qmod.reducer.transform(d["Xte"])


def qmlp_marginals(qmod, A):
    nq = qmod.qnn.n_quantum_params
    qp = qmod.qnn.weights[:nq]
    return np.array([qmod.qnn._statevector_marginals(a, qp) for a in A])


def main():
    results = {"stageA": {}, "stageB": {}, "stageC": {}}
    datasets = ["cicids", "nslkdd", "unsw"]
    data = {n: prep_dataset(n) for n in datasets}
    for n in datasets:
        log(f"[prep] {n}: train={len(data[n]['Xtr'])} val={len(data[n]['Xva'])} test={len(data[n]['Xte'])}")

    # ================= STAGE A =================
    for name in datasets:
        d = data[name]
        a1_rows, a2_rows = [], []
        angles_cache = {}
        for seed in SEEDS:
            t0=time.perf_counter()
            qmod, Atr, Ava, Ate = get_qmlp_and_angles(name, d, seed)
            angles_cache[seed] = (Atr, Ava, Ate)
            # A1: marginal-trained head (original min-max angles, as-trained circuit params)
            Mtr, Mva, Mte = qmlp_marginals(qmod,Atr), qmlp_marginals(qmod,Ava), qmlp_marginals(qmod,Ate)
            lr = LogisticRegression(max_iter=1000, random_state=0).fit(Mtr, d["ytr"].astype(int))
            pv, pt = lr.predict_proba(Mva)[:,1], lr.predict_proba(Mte)[:,1]
            a1_rows.append(evaluate(d["yte"], pt, macroF1_threshold(pv, d["yva"])))
            # A2: VQC (reup-L4) + quantile scaling
            Qtr,Qva,Qte = quantile_angles(Atr,Ava,Ate)
            v = make_arch("reup-L4"); v.train(Qtr, d["ytr"], seed=seed)
            pv, pt = v.forward_batch(Qva), v.forward_batch(Qte)
            a2_rows.append(evaluate(d["yte"], pt, macroF1_threshold(pv, d["yva"])))
            log(f"[A] {name} seed{seed}: circ+head F1={a1_rows[-1]['macro_f1']:.3f} "
                f"vqc+quant F1={a2_rows[-1]['macro_f1']:.3f} ({time.perf_counter()-t0:.0f}s)")
        results["stageA"][name] = {
            "circuit_marginal_head": {k:agg(a1_rows,k) for k in ("macro_f1","mcc","roc_auc")},
            "vqc_quantile":          {k:agg(a2_rows,k) for k in ("macro_f1","mcc","roc_auc")},
            "per_seed": {"circuit_marginal_head":a1_rows, "vqc_quantile":a2_rows}}
        data[name]["angles"] = angles_cache
        json.dump(results, open(OUT/"quantum_improvements.json","w"), indent=1, default=float)

    # ================= STAGE B (primary corpus, VALIDATION-only selection) ==========
    d = data["cicids"]
    for arch in ARCHS:
        rows=[]
        for seed in SEARCH_SEEDS:
            t0=time.perf_counter()
            Atr,Ava,Ate = d["angles"][seed]
            Qtr,Qva,_ = quantile_angles(Atr,Ava,Ate)
            m = make_arch(arch); m.train(Qtr, d["ytr"], seed=seed)
            pv = m.forward_batch(Qva)
            t = macroF1_threshold(pv, d["yva"])
            rows.append({"val_macro_f1": f1_score(d["yva"], (pv>t).astype(int), average="macro"),
                         "n_params": m.n_params})
            log(f"[B] {arch} seed{seed}: VAL F1={rows[-1]['val_macro_f1']:.3f} ({time.perf_counter()-t0:.0f}s)")
        results["stageB"][arch] = {"val_macro_f1_mean": float(np.mean([r["val_macro_f1"] for r in rows])),
                                   "val_macro_f1_std": float(np.std([r["val_macro_f1"] for r in rows],ddof=1)),
                                   "n_params": rows[0]["n_params"], "per_seed": rows}
        json.dump(results, open(OUT/"quantum_improvements.json","w"), indent=1, default=float)
    winner = max(results["stageB"], key=lambda a: results["stageB"][a]["val_macro_f1_mean"])
    results["stageB"]["winner"] = winner
    log(f"[B] WINNER (validation-only selection): {winner}")

    # ================= STAGE C: combined 1+2+5 =================
    for name in datasets:
        d = data[name]
        rows=[]
        for seed in SEEDS:
            t0=time.perf_counter()
            Atr,Ava,Ate = d["angles"][seed]
            Qtr,Qva,Qte = quantile_angles(Atr,Ava,Ate)
            m = make_arch(winner); m.train(Qtr, d["ytr"], seed=seed)
            Mtr,Mva,Mte = m.marginals_batch(Qtr), m.marginals_batch(Qva), m.marginals_batch(Qte)
            lr = LogisticRegression(max_iter=1000, random_state=0).fit(Mtr, d["ytr"].astype(int))
            pv,pt = lr.predict_proba(Mva)[:,1], lr.predict_proba(Mte)[:,1]
            rows.append(evaluate(d["yte"], pt, macroF1_threshold(pv, d["yva"])))
            log(f"[C] {name} seed{seed}: combined F1={rows[-1]['macro_f1']:.3f} ({time.perf_counter()-t0:.0f}s)")
        results["stageC"][name] = {"combined": {k:agg(rows,k) for k in ("macro_f1","mcc","roc_auc")},
                                   "per_seed": rows, "architecture": winner}
        json.dump(results, open(OUT/"quantum_improvements.json","w"), indent=1, default=float)

    log("ALL STAGES DONE")


if __name__ == "__main__":
    main()
