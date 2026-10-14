"""
Phase-VI additions computed from ARCHIVED per-seed predictions (no retraining):
  - Probability calibration: Brier score + Expected Calibration Error (ECE),
    uncalibrated vs Platt-scaled (Platt fit on D_val only). (PhaseVI 2.9, 4.4)
  - Reliability diagram figure. (PhaseVI 4.4)
  - McNemar's paired test on locked-D_test binary predictions. (PhaseVI 2.18)
  - 95% bootstrap confidence intervals for macro-F1 and MCC. (PhaseVI 2.18, 4.4)
  - Validation threshold-sensitivity curve. (PhaseVI 2.10)
  - CNN/RNN train-validation loss-curve figure (from 08_train_curves.py). (2.12)

Reads results/predictions/*.npz (archived); writes results_phaseVI/*.json and
figures_phaseVI/*.pdf. Canonical results/ is never modified.
"""
import os, json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
PRED = P / "results" / "predictions"
OUT = P / "results_phaseVI"; OUT.mkdir(exist_ok=True)
FIG = P.parent / "figures_phaseVI"; FIG.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
rng = np.random.RandomState(12345)


def load_prob(agent, split, seed=None, field="prob"):
    if seed is None:
        d = np.load(PRED / f"{agent}_{split}.npz")
    else:
        d = np.load(PRED / f"{agent}_seed{seed}_{split}.npz")
    return d[field], d["y"]


def seed_avg_prob(agent, split, field="prob"):
    ys = None; ps = []
    for s in SEEDS:
        p, y = load_prob(agent, split, s, field)
        ps.append(p); ys = y
    return np.mean(ps, axis=0), ys


def get_prob(agent, split, seed=None):
    """Unified accessor returning (prob, y) for the deployed decision of each agent."""
    if agent == "quantumai":  # deployed decision
        if seed is None:
            ps = []; y = None
            for s in SEEDS:
                d = np.load(PRED / f"quantumai_seed{s}_{split}.npz"); ps.append(d["p_deploy"]); y = d["y"]
            return np.mean(ps, 0), y
        d = np.load(PRED / f"quantumai_seed{seed}_{split}.npz"); return d["p_deploy"], d["y"]
    if agent in ("cnn", "rnn", "vqc"):
        if seed is None: return seed_avg_prob(agent, split)
        return load_prob(agent, split, seed)
    if agent in ("cosine", "classical8d"):
        return load_prob(agent, split, None)
    raise ValueError(agent)


AGENTS = ["cnn", "rnn", "quantumai", "vqc", "classical8d", "cosine"]


def best_threshold(pval, yval):
    best_t, best = 0.5, -1
    for t in np.arange(0.01, 1.0, 0.01):
        f = f1_score(yval, (pval > t).astype(int), average="macro")
        if f > best: best, best_t = f, t
    return best_t


def fusion_prob(split, seed=None):
    parts = [get_prob(a, split, seed)[0] for a in ("quantumai", "cosine", "rnn", "cnn")]
    _, y = get_prob("cnn", split, seed)
    return np.mean(parts, 0), y


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0; N = len(y)
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1]) if i > 0 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0: continue
        conf = p[m].mean(); acc = y[m].mean()
        e += (m.sum() / N) * abs(acc - conf)
    return e


def reliability_points(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); xs, ys = [], []
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1]) if i > 0 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0: continue
        xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


# ---------- 1. Calibration (Brier + ECE), uncalibrated vs Platt ----------
calib = {}
recs = list(AGENTS) + ["fusion"]
for a in recs:
    if a == "fusion":
        pv, yv = fusion_prob("val"); pt, yt = fusion_prob("test")
    else:
        pv, yv = get_prob(a, "val"); pt, yt = get_prob(a, "test")
    # uncalibrated
    b0 = brier_score_loss(yt, np.clip(pt, 0, 1)); e0 = ece(yt, np.clip(pt, 0, 1))
    # Platt scaling: fit logistic on val logits -> test
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    eps = 1e-6; lv = np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps)))
    lt = np.log(np.clip(pt, eps, 1 - eps) / (1 - np.clip(pt, eps, 1 - eps)))
    platt.fit(lv.reshape(-1, 1), yv.astype(int))
    pt_cal = platt.predict_proba(lt.reshape(-1, 1))[:, 1]
    b1 = brier_score_loss(yt, pt_cal); e1 = ece(yt, pt_cal)
    calib[a] = {"brier_uncal": b0, "ece_uncal": e0, "brier_platt": b1, "ece_platt": e1}
    print(f"  calib {a:12s} Brier {b0:.3f}->{b1:.3f}  ECE {e0:.3f}->{e1:.3f}")

# reliability diagram for a representative subset
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
for a, col in [("cnn", "#1f77b4"), ("rnn", "#ff7f0e"), ("quantumai", "#2ca02c"), ("fusion", "#d62728")]:
    if a == "fusion":
        pt, yt = fusion_prob("test")
    else:
        pt, yt = get_prob(a, "test")
    xs, ys = reliability_points(yt, np.clip(pt, 0, 1), 10)
    lbl = {"quantumai": "QuantumAI (deployed)", "fusion": "Fusion (equal-weight)"}.get(a, a.upper())
    ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.4, label=lbl)
ax.set_xlabel("Mean predicted probability (test)"); ax.set_ylabel("Empirical attack fraction")
ax.set_title("Reliability diagram (locked $D_{\\mathrm{test}}$, seed-averaged)")
ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "reliability_diagram.pdf"); plt.close(fig)

# ---------- 2. McNemar's test on locked-D_test (seed 0 representative) ----------
def preds_at_val_thr(a, seed=0):
    pv, yv = get_prob(a, "val", seed); pt, yt = get_prob(a, "test", seed)
    t = best_threshold(pv, yv)
    return (pt > t).astype(int), yt

def mcnemar(a, b, seed=0):
    if a == "fusion":
        pv, yv = fusion_prob("val", seed); pt, yt = fusion_prob("test", seed); ta = best_threshold(pv, yv); pa = (pt > ta).astype(int)
    else:
        pa, yt = preds_at_val_thr(a, seed)
    if b == "fusion":
        pv, yv = fusion_prob("val", seed); pt, _ = fusion_prob("test", seed); tb = best_threshold(pv, yv); pb = (pt > tb).astype(int)
    else:
        pb, yt = preds_at_val_thr(b, seed)
    ca = (pa == yt); cb = (pb == yt)
    n01 = int(np.sum(~ca & cb)); n10 = int(np.sum(ca & ~cb))
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "stat": 0.0, "p": 1.0, "note": "no discordant pairs"}
    # exact binomial McNemar (two-sided)
    k = min(n01, n10)
    p = float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))
    stat = (abs(n01 - n10) - 1) ** 2 / n  # continuity-corrected chi-square (reported alongside)
    return {"n01": n01, "n10": n10, "chi2_cc": float(stat), "p_exact": p}

mc = {
    "fusion_vs_rnn": mcnemar("fusion", "rnn"),
    "cnn_vs_rnn": mcnemar("cnn", "rnn"),
    "quantumai_vs_classical8d": mcnemar("quantumai", "classical8d"),
    "rnn_vs_quantumai": mcnemar("rnn", "quantumai"),
}
for k, v in mc.items(): print(f"  McNemar {k}: {v}")

# ---------- 3. Bootstrap 95% CI for macro-F1 and MCC (seed-averaged prob) ----------
def boot_ci(a, B=2000):
    if a == "fusion":
        pv, yv = fusion_prob("val"); pt, yt = fusion_prob("test")
    else:
        pv, yv = get_prob(a, "val"); pt, yt = get_prob(a, "test")
    t = best_threshold(pv, yv); yhat = (pt > t).astype(int)
    N = len(yt); f1s = []; mccs = []
    idx0 = np.arange(N)
    for _ in range(B):
        bi = rng.randint(0, N, N)
        yt_b, yh_b = yt[bi], yhat[bi]
        if len(np.unique(yt_b)) < 2: continue
        f1s.append(f1_score(yt_b, yh_b, average="macro"))
        mccs.append(matthews_corrcoef(yt_b, yh_b))
    f1s = np.array(f1s); mccs = np.array(mccs)
    return {"macro_f1": float(f1_score(yt, yhat, average="macro")),
            "macro_f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "mcc": float(matthews_corrcoef(yt, yhat)),
            "mcc_ci": [float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))]}

boot = {a: boot_ci(a) for a in AGENTS + ["fusion"]}
for a, v in boot.items():
    print(f"  bootCI {a:12s} F1 {v['macro_f1']:.3f} [{v['macro_f1_ci'][0]:.3f},{v['macro_f1_ci'][1]:.3f}]")

# ---------- 3b. Per-seed 95% t-CI (matches the main mean +/- std table) ----------
T975_DF4 = 2.776  # t_{0.975, df=4}
def perseed_ci(a):
    f1s, mccs = [], []
    for s in SEEDS:
        pv, yv = get_prob(a, "val", s); pt, yt = get_prob(a, "test", s)
        t = best_threshold(pv, yv); yh = (pt > t).astype(int)
        f1s.append(f1_score(yt, yh, average="macro")); mccs.append(matthews_corrcoef(yt, yh))
    f1s = np.array(f1s); mccs = np.array(mccs)
    def ci(x):
        m = x.mean(); sd = x.std(ddof=1); h = T975_DF4 * sd / np.sqrt(len(x))
        return float(m), float(sd), [float(m - h), float(m + h)]
    mf, sf, cf = ci(f1s); mm, sm, cm = ci(mccs)
    return {"per_seed_macro_f1": [float(v) for v in f1s], "macro_f1_mean": mf, "macro_f1_std": sf, "macro_f1_ci95": cf,
            "mcc_mean": mm, "mcc_std": sm, "mcc_ci95": cm}
seed_ci = {a: perseed_ci(a) for a in ("cnn", "rnn", "quantumai", "vqc")}
for a, v in seed_ci.items():
    print(f"  seedCI {a:10s} F1 {v['macro_f1_mean']:.3f}+/-{v['macro_f1_std']:.3f} CI95 [{v['macro_f1_ci95'][0]:.3f},{v['macro_f1_ci95'][1]:.3f}]")

# ---------- 4. Validation threshold-sensitivity curve ----------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ts = np.arange(0.01, 1.0, 0.01)
for a, col in [("cnn", "#1f77b4"), ("rnn", "#ff7f0e"), ("quantumai", "#2ca02c"), ("fusion", "#d62728")]:
    if a == "fusion":
        pv, yv = fusion_prob("val")
    else:
        pv, yv = get_prob(a, "val")
    fs = [f1_score(yv, (pv > t).astype(int), average="macro") for t in ts]
    lbl = {"quantumai": "QuantumAI (deployed)", "fusion": "Fusion (equal-weight)"}.get(a, a.upper())
    ax.plot(ts, fs, "-", color=col, lw=1.5, label=lbl)
    bt = ts[int(np.argmax(fs))]; ax.axvline(bt, color=col, ls=":", lw=0.8, alpha=0.6)
ax.set_xlabel("Decision threshold $\\tau$ (applied to $D_{\\mathrm{val}}$)")
ax.set_ylabel("Validation macro-$F_1$")
ax.set_title("Validation threshold sensitivity (dotted = selected $\\hat\\tau$)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 1)
fig.tight_layout(); fig.savefig(FIG / "threshold_sensitivity.pdf"); plt.close(fig)

# ---------- 5. Train/validation loss curves ----------
tc_path = OUT / "train_curves.json"
if tc_path.exists():
    tc = json.load(open(tc_path))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, m, title in [(axes[0], "cnn", "CNN"), (axes[1], "rnn", "RNN")]:
        tr = np.array(tc[m]["train"]); va = np.array(tc[m]["val"])
        ep = np.arange(1, tr.shape[1] + 1)
        ax.plot(ep, tr.mean(0), "-", color="#1f77b4", label="train loss")
        ax.fill_between(ep, tr.mean(0) - tr.std(0), tr.mean(0) + tr.std(0), color="#1f77b4", alpha=0.2)
        ax.plot(ep, va.mean(0), "-", color="#d62728", label="validation loss")
        ax.fill_between(ep, va.mean(0) - va.std(0), va.mean(0) + va.std(0), color="#d62728", alpha=0.2)
        ax.set_title(f"{title} (mean $\\pm$ std over 5 seeds)"); ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "train_val_loss_curves.pdf"); plt.close(fig)

# ---------- write JSON ----------
res = {"calibration": calib, "mcnemar": mc, "bootstrap_ci": boot, "per_seed_ci": seed_ci,
       "train_curves_final": {m: {"mean": tc[m]["test_macro_f1_mean"], "std": tc[m]["test_macro_f1_std"]}
                              for m in ("cnn", "rnn")} if tc_path.exists() else {}}
json.dump(res, open(OUT / "phaseVI_stats.json", "w"), indent=1)
print("\nWrote", OUT / "phaseVI_stats.json")
print("Figures ->", FIG)
