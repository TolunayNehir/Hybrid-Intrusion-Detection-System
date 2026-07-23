"""
V7 additions:

  T. Strong tabular baselines (XGBoost, LightGBM, CatBoost, Random Forest,
     RBF-SVM, sigmoid-calibrated logistic regression on the full features),
     default configurations (no per-dataset tuning), locked protocol,
     5 seeds where the model is seed-sensitive, all three corpora.
  M. Multiclass attack-family analysis on the primary corpus (6 classes):
     per-class precision/recall/F1 on the locked test partition.
  S. Shot-noise robustness of the combined QCNN-pool hybrid: per-qubit
     marginals estimated from finite measurement shots (binomial sampling)
     instead of exact statevector values, at 256/1024/4096 shots.

Thresholds (binary) selected on D_val only; D_test touched once per model
per seed. Multiclass uses argmax (no threshold).
"""
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
REPO = Path(os.environ["HYBRID_IDS_REPO"]).resolve()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(P))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
                             precision_recall_fscore_support)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from agents.quantum_ai.quantum_module import QuantumAIModule
from qarch import make_arch

OUT = P / "results_final_additions"; OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
SAMPLE_SEED = 42
SPLIT_FRACS = (0.74, 0.13, 0.13)
DATA = P / "replication_data"

def log(*a): print(*a, flush=True)

# ---- corpora builders (identical to 11/12/14) ----
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

def prep(name):
    if name=="cicids":
        df=pd.read_csv(P/"results"/"locked_dataset.csv")
        fcols=[c for c in df.columns if c not in ("row_id","partition","label","target_binary")]
        part=df["partition"].to_numpy(); X_raw=df[fcols].to_numpy(np.float64)
        y=df["target_binary"].to_numpy(np.float32); lab=df["label"].to_numpy()
    else:
        X_raw,y=build(name); part=locked_split(X_raw); lab=None
    tr,va,te = part=="train", part=="validation", part=="test"
    m=X_raw[tr].mean(0); s=X_raw[tr].std(0); s[s<1e-8]=1.0
    Xz=np.clip((X_raw-m)/s,-5,5)
    out={"Xtr":Xz[tr],"Xva":Xz[va],"Xte":Xz[te],"ytr":y[tr],"yva":y[va],"yte":y[te]}
    if lab is not None:
        out["ltr"],out["lva"],out["lte"]=lab[tr],lab[va],lab[te]
    return out

def thr(pv,yv):
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

def tab_models(seed):
    return {
        "xgboost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     subsample=0.9, colsample_bytree=0.9, random_state=seed,
                                     eval_metric="logloss", verbosity=0, n_jobs=4),
        "lightgbm": lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbosity=-1, n_jobs=4),
        "catboost": CatBoostClassifier(iterations=300, random_seed=seed, verbose=0,
                                       allow_writing_files=False, thread_count=4),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=4),
        "rbf_svm": SVC(kernel="rbf", probability=True, random_state=seed),
        "calibrated_lr": CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, random_state=seed), method="sigmoid", cv=5),
    }

def main():
    res={"tabular":{},"multiclass":{},"shot_noise":{}}
    data={n:prep(n) for n in ("cicids","nslkdd","unsw")}
    log("[prep] done")

    # ================= T: tabular baselines =================
    for name in ("cicids","nslkdd","unsw"):
        d=data[name]; res["tabular"][name]={}
        for mname in ("xgboost","lightgbm","catboost","random_forest","rbf_svm","calibrated_lr"):
            rows=[]
            for seed in SEEDS:
                m=tab_models(seed)[mname]
                m.fit(d["Xtr"], d["ytr"].astype(int))
                pv=m.predict_proba(d["Xva"])[:,1]; pt=m.predict_proba(d["Xte"])[:,1]
                rows.append(evaluate(d["yte"],pt,thr(pv,d["yva"])))
            res["tabular"][name][mname]={k:agg(rows,k) for k in ("macro_f1","mcc","roc_auc")}
            s=res["tabular"][name][mname]["macro_f1"]
            log(f"[T] {name} {mname}: F1 {s['mean']:.3f}+/-{s['std']:.3f}")
        json.dump(res,open(OUT/"final_additions.json","w"),indent=1,default=float)

    # ================= M: multiclass (primary corpus, 6 classes) =================
    d=data["cicids"]
    classes=sorted(set(d["ltr"]))
    log("[M] classes:", classes)
    cls_idx={c:i for i,c in enumerate(classes)}
    ytr=np.array([cls_idx[c] for c in d["ltr"]]); yte=np.array([cls_idx[c] for c in d["lte"]])
    mc_models={
        "xgboost": xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,random_state=0,
                                     eval_metric="mlogloss",verbosity=0,n_jobs=4),
        "lightgbm": lgb.LGBMClassifier(n_estimators=300,random_state=0,verbosity=-1,n_jobs=4),
        "random_forest": RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=4),
        "multinomial_lr": LogisticRegression(max_iter=2000,random_state=0),
    }
    for mname,m in mc_models.items():
        m.fit(d["Xtr"],ytr)
        yh=m.predict(d["Xte"])
        p,r,f,s=precision_recall_fscore_support(yte,yh,labels=list(range(len(classes))),zero_division=0)
        res["multiclass"][mname]={
            "macro_f1":float(f1_score(yte,yh,average="macro")),
            "per_class":{classes[i]:{"precision":float(p[i]),"recall":float(r[i]),
                                     "f1":float(f[i]),"support":int(s[i])} for i in range(len(classes))}}
        log(f"[M] {mname}: macro-F1 {res['multiclass'][mname]['macro_f1']:.3f}")
    json.dump(res,open(OUT/"final_additions.json","w"),indent=1,default=float)

    # ================= S: shot-noise robustness (combined QCNN-pool hybrid, primary) ==========
    R=np.load(P/"results"/"fisher_pca_features.npz")
    Atr,Ava,Ate=R["R_train"],R["R_val"],R["R_test"]
    qt=QuantileTransformer(n_quantiles=min(1000,len(Atr)),output_distribution="uniform",
                           subsample=int(1e9),random_state=0).fit(Atr)
    Qtr,Qva,Qte=np.pi*qt.transform(Atr),np.pi*qt.transform(Ava),np.pi*qt.transform(Ate)
    d=data["cicids"]
    SHOTS=[256,1024,4096,None]  # None = exact statevector
    rows_by_shots={sh:[] for sh in SHOTS}
    for seed in SEEDS:
        t0=time.perf_counter()
        m=make_arch("qcnn-pool"); m.train(Qtr,d["ytr"],seed=seed)
        Mtr=m.marginals_batch(Qtr); Mva=m.marginals_batch(Qva); Mte=m.marginals_batch(Qte)
        head=LogisticRegression(max_iter=1000,random_state=0).fit(Mtr,d["ytr"].astype(int))
        rng=np.random.RandomState(1000+seed)
        for sh in SHOTS:
            def noisy(M):
                if sh is None: return M
                return rng.binomial(sh,np.clip(M,0,1))/float(sh)
            pv=head.predict_proba(noisy(Mva))[:,1]; pt=head.predict_proba(noisy(Mte))[:,1]
            rows_by_shots[sh].append(evaluate(d["yte"],pt,thr(pv,d["yva"])))
        log(f"[S] seed{seed} done ({time.perf_counter()-t0:.0f}s)")
    for sh in SHOTS:
        key="exact" if sh is None else str(sh)
        res["shot_noise"][key]={k:agg(rows_by_shots[sh],k) for k in ("macro_f1","mcc")}
        s=res["shot_noise"][key]["macro_f1"]
        log(f"[S] shots={key}: F1 {s['mean']:.3f}+/-{s['std']:.3f}")
    json.dump(res,open(OUT/"final_additions.json","w"),indent=1,default=float)
    log("FINAL ADDITIONS DONE")

if __name__=="__main__":
    main()
