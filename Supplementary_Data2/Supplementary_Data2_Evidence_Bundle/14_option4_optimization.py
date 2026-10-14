"""
Option-4 (optimization) study for the quantum-path improvements.

  D1. Learning-rate screening (primary corpus, seeds 0-2, VALIDATION only):
      reup-L4 and qcnn-pool at lr in {0.005, 0.02}, 300 iters, vs the known
      lr=0.01 results of the architecture search.
  D2. Depth-rescue test (primary corpus, seeds 0-2, VALIDATION only):
      reup-L2 and reup-L4 with the enhanced optimizer (3 restarts x 600
      iterations, patience 80, best screened lr), restart selected on
      validation macro-F1. Answers whether the depth collapse is
      budget-limited or persists under a stronger optimization protocol.
  D3. Combined model + enhanced optimizer (all 3 corpora, 5 seeds):
      qcnn-pool + quantile encoding + marginal-trained head, with 3 restarts
      per seed selected on validation macro-F1 of the full hybrid pipeline;
      locked D_test evaluated once per seed. Paired per-seed comparison
      against the standard-optimizer combined model of stage C.

All selection (lr, restart) uses validation data only.
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

OUT = P / "results_quantum_improved"
SEEDS = [0, 1, 2, 3, 4]
SCREEN_SEEDS = [0, 1, 2]
RESTARTS = 3
ENH_ITERS = 600
ENH_PATIENCE = 80
SAMPLE_SEED = 42
SPLIT_FRACS = (0.74, 0.13, 0.13)
DATA = P / "replication_data"

def log(*a): print(*a, flush=True)

# ---------- corpora (identical to 11/12) ----------
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

def build(name):
    if name=="nslkdd":
        df = pd.read_csv(DATA/"KDDTrain+.txt", header=None, names=NSL_COLS)
        df["family"] = df["label"].map(lambda l: "Normal" if l=="normal" else NSL_FAMILY.get(l,"Other"))
        df = df[df["family"]!="Other"]
        corpus = _sample(df,"family",{"Normal":750,"DoS":334,"Probe":333,"R2L":333,"U2R":21})
        fcols=[c for c in NSL_COLS if c not in ("protocol_type","service","flag","label","difficulty")]
        return corpus[fcols].to_numpy(np.float64), (corpus["family"]!="Normal").astype(np.float32).to_numpy()
    df = pd.read_csv(DATA/"UNSW_NB15_training-set.csv")
    drop=["id","proto","service","state","attack_cat","label"]
    fcols=[c for c in df.columns if c not in drop]
    corpus = _sample(df,"attack_cat",{"Normal":750,"Exploits":334,"Fuzzers":333,"Reconnaissance":333,"Worms":21})
    return corpus[fcols].to_numpy(np.float64), corpus["label"].astype(np.float32).to_numpy()

def locked_split(X):
    n=len(X); rng=np.random.RandomState(SAMPLE_SEED); order=rng.permutation(n)
    n_tr=int(round(SPLIT_FRACS[0]*n)); n_va=int(round(SPLIT_FRACS[1]*n))
    part=np.empty(n,dtype=object)
    part[order[:n_tr]]="train"; part[order[n_tr:n_tr+n_va]]="validation"; part[order[n_tr+n_va:]]="test"
    fps=np.array([hashlib.sha256(np.round(x,6).tobytes()).hexdigest() for x in X])
    groups={}
    for i,f in enumerate(fps): groups.setdefault(f,[]).append(i)
    prio={"train":0,"validation":1,"test":2}
    for f,idxs in groups.items():
        if len(set(part[i] for i in idxs))>1:
            counts={}
            for i in idxs: counts[part[i]]=counts.get(part[i],0)+1
            best=sorted(counts.items(), key=lambda kv:(-kv[1],prio[kv[0]]))[0][0]
            for i in idxs: part[i]=best
    return part

def quantile_pi(Atr,Ava,Ate):
    qt=QuantileTransformer(n_quantiles=min(1000,len(Atr)),output_distribution="uniform",
                           subsample=int(1e9),random_state=0).fit(Atr)
    return np.pi*qt.transform(Atr), np.pi*qt.transform(Ava), np.pi*qt.transform(Ate)

def prep(name):
    if name=="cicids":
        df=pd.read_csv(P/"results"/"locked_dataset.csv")
        fcols=[c for c in df.columns if c not in ("row_id","partition","label","target_binary")]
        part=df["partition"].to_numpy(); X_raw=df[fcols].to_numpy(np.float64)
        y=df["target_binary"].to_numpy(np.float32)
        R=np.load(P/"results"/"fisher_pca_features.npz")
        Atr,Ava,Ate=R["R_train"],R["R_val"],R["R_test"]
    else:
        X_raw,y=build(name); part=locked_split(X_raw)
    tr,va,te = part=="train", part=="validation", part=="test"
    if name!="cicids":
        m=X_raw[tr].mean(0); s=X_raw[tr].std(0); s[s<1e-8]=1.0
        Xz=np.clip((X_raw-m)/s,-5,5)
        qref=QuantumAIModule(); qref.reducer.fit(Xz[tr], y[tr])
        Atr,Ava,Ate = qref.reducer.transform(Xz[tr]), qref.reducer.transform(Xz[va]), qref.reducer.transform(Xz[te])
    Qtr,Qva,Qte = quantile_pi(Atr,Ava,Ate)
    return {"Qtr":Qtr,"Qva":Qva,"Qte":Qte,"ytr":y[tr],"yva":y[va],"yte":y[te]}

def thr(pv,yv):
    best_t,best=0.5,-1
    for t in np.arange(0.01,1.0,0.01):
        f=f1_score(yv,(pv>t).astype(int),average="macro")
        if f>best: best,best_t=f,t
    return best_t

def val_f1(pv,yv):
    t=thr(pv,yv); return f1_score(yv,(pv>t).astype(int),average="macro")

def evaluate(yte,pt,t):
    yh=(pt>t).astype(int)
    out={"macro_f1":f1_score(yte,yh,average="macro"),"mcc":matthews_corrcoef(yte,yh)}
    try: out["roc_auc"]=roc_auc_score(yte,pt)
    except Exception: out["roc_auc"]=float("nan")
    return out

def agg(rows,k):
    v=[r[k] for r in rows]
    return {"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)) if len(v)>1 else 0.0}


def main():
    res={"D1_lr_screen":{},"D2_depth_rescue":{},"D3_combined_opt4":{}}
    prev=json.load(open(OUT/"quantum_improvements.json"))
    data={n:prep(n) for n in ("cicids","nslkdd","unsw")}
    d0=data["cicids"]
    log("[prep] done")

    # ---------- D1: LR screening ----------
    for arch in ("reup-L4","qcnn-pool"):
        res["D1_lr_screen"][arch]={"0.01":{"val_mean":prev["stageB"][arch]["val_macro_f1_mean"],"source":"stageB"}}
        for lr in (0.005,0.02):
            vals=[]
            for seed in SCREEN_SEEDS:
                t0=time.perf_counter()
                m=make_arch(arch); m.train(d0["Qtr"],d0["ytr"],seed=seed,lr=lr)
                vals.append(val_f1(m.forward_batch(d0["Qva"]),d0["yva"]))
                log(f"[D1] {arch} lr={lr} seed{seed}: VAL F1={vals[-1]:.3f} ({time.perf_counter()-t0:.0f}s)")
            res["D1_lr_screen"][arch][str(lr)]={"val_mean":float(np.mean(vals)),"per_seed":vals}
        best_lr=max(res["D1_lr_screen"][arch], key=lambda k: res["D1_lr_screen"][arch][k]["val_mean"])
        res["D1_lr_screen"][arch]["best_lr"]=best_lr
        log(f"[D1] {arch} best lr = {best_lr}")
        json.dump(res,open(OUT/"option4_results.json","w"),indent=1,default=float)

    lr_l4=float(res["D1_lr_screen"]["reup-L4"]["best_lr"])
    lr_qc=float(res["D1_lr_screen"]["qcnn-pool"]["best_lr"])

    # ---------- D2: depth rescue (enhanced optimizer, val-only) ----------
    for arch in ("reup-L2","reup-L4"):
        vals=[]
        for seed in SCREEN_SEEDS:
            t0=time.perf_counter()
            best_v=-1
            for r in range(RESTARTS):
                m=make_arch(arch)
                m.train(d0["Qtr"],d0["ytr"],seed=10*seed+r,max_iter=ENH_ITERS,
                        patience=ENH_PATIENCE,lr=lr_l4)
                v=val_f1(m.forward_batch(d0["Qva"]),d0["yva"])
                if v>best_v: best_v=v
            vals.append(best_v)
            log(f"[D2] {arch} seed{seed}: best-of-{RESTARTS} VAL F1={best_v:.3f} ({time.perf_counter()-t0:.0f}s)")
        res["D2_depth_rescue"][arch]={"val_mean":float(np.mean(vals)),
                                      "val_std":float(np.std(vals,ddof=1)),"per_seed":vals,
                                      "standard_val_mean":prev["stageB"].get(arch,{}).get("val_macro_f1_mean")}
        json.dump(res,open(OUT/"option4_results.json","w"),indent=1,default=float)

    # ---------- D3: combined + enhanced optimizer ----------
    for name in ("cicids","nslkdd","unsw"):
        d=data[name]; rows=[]; diffs=[]
        prevC=prev["stageC"][name]["per_seed"]
        for seed in SEEDS:
            t0=time.perf_counter()
            best=(None,-1,None)  # (model, val_f1, (pv,pt))
            for r in range(RESTARTS):
                m=make_arch("qcnn-pool")
                m.train(d["Qtr"],d["ytr"],seed=10*seed+r,max_iter=ENH_ITERS,
                        patience=ENH_PATIENCE,lr=lr_qc)
                Mtr=m.marginals_batch(d["Qtr"]); Mva=m.marginals_batch(d["Qva"])
                head=LogisticRegression(max_iter=1000,random_state=0).fit(Mtr,d["ytr"].astype(int))
                pv=head.predict_proba(Mva)[:,1]
                v=val_f1(pv,d["yva"])
                if v>best[1]:
                    Mte=m.marginals_batch(d["Qte"])
                    best=(m, v, (pv, head.predict_proba(Mte)[:,1]))
            pv,pt=best[2]
            met=evaluate(d["yte"],pt,thr(pv,d["yva"]))
            rows.append(met)
            diffs.append(met["macro_f1"]-prevC[seed]["macro_f1"])
            log(f"[D3] {name} seed{seed}: combined-opt4 F1={met['macro_f1']:.3f} "
                f"(delta vs stageC {diffs[-1]:+.3f}) ({time.perf_counter()-t0:.0f}s)")
        dif=np.array(diffs)
        tt=stats.ttest_rel(dif,np.zeros_like(dif))
        res["D3_combined_opt4"][name]={
            "combined_opt4":{k:agg(rows,k) for k in ("macro_f1","mcc","roc_auc")},
            "per_seed":rows,
            "paired_vs_stageC":{"per_seed_diff":[float(x) for x in dif],
                                "mean_diff":float(dif.mean()),"ttest_p":float(tt.pvalue)}}
        json.dump(res,open(OUT/"option4_results.json","w"),indent=1,default=float)

    log("OPTION4 DONE")


if __name__ == "__main__":
    main()
