"""
For every model/seed, sweep the decision threshold on D_val only, pick
tau_hat = argmax_tau MacroF1(D_val; tau), then evaluate ONCE on the frozen D_test
using that tau. D_test is never used for any selection decision.

The optional LLM/rule-based agent is excluded from this quantitative locked
evaluation: it depends on an external GPT-4o API call (or a keyword heuristic
fallback) that is not fixed by this reproducible protocol, so it cannot be
frozen/repeated the way the other four agents can. Fusion here is therefore over
the four empirically-evaluable agents (QuantumAI-deploy, Cosine/kNN, RNN, CNN),
equal-weighted at 0.25 each (renormalized from the deployed system's 0.20 x 5).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = Path(__file__).parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
SEEDS = [0, 1, 2, 3, 4]
THRESH_GRID = np.arange(0.01, 1.00, 0.01)


def best_threshold(y_val, p_val):
    best_t, best_f1 = 0.5, -1.0
    for t in THRESH_GRID:
        pred = (p_val > t).astype(int)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def full_metrics(y_true, p_prob, tau):
    pred = (p_prob > tau).astype(int)
    y_true = y_true.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    out = {
        "threshold": float(tau),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "attack_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, p_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
    }
    return out


def aggregate_seeds(per_seed_metrics):
    # Post-audit fix: use sample standard deviation (ddof=1, Bessel's
    # correction), the standard convention for "mean +/- SD" over a finite
    # sample of n=5 stochastic seeds, rather than numpy's default population
    # SD (ddof=0). The seed count (5) is >1 everywhere this is called, so
    # ddof=1 is always well-defined here.
    keys = ["precision", "recall", "macro_f1", "weighted_f1", "attack_f1",
            "fpr", "fnr", "mcc", "roc_auc", "pr_auc"]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=np.float64)
        agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1))}
    # sum confusion matrices for a representative aggregate table
    cm_sum = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for m in per_seed_metrics:
        for kk in cm_sum:
            cm_sum[kk] += m["confusion_matrix"][kk]
    agg["confusion_matrix_sum"] = cm_sum
    agg["thresholds"] = [m["threshold"] for m in per_seed_metrics]
    agg["n_seeds"] = len(per_seed_metrics)
    return agg


def eval_seeded_model(model_name, prob_key="prob"):
    per_seed = []
    for seed in SEEDS:
        val = np.load(PRED_DIR / f"{model_name}_seed{seed}_val.npz")
        test = np.load(PRED_DIR / f"{model_name}_seed{seed}_test.npz")
        tau, _ = best_threshold(val["y"], val[prob_key])
        m = full_metrics(test["y"], test[prob_key], tau)
        m["seed"] = seed
        per_seed.append(m)
    return per_seed, aggregate_seeds(per_seed)


def eval_deterministic_model(model_name, prob_key="prob"):
    val = np.load(PRED_DIR / f"{model_name}_val.npz")
    test = np.load(PRED_DIR / f"{model_name}_test.npz")
    tau, _ = best_threshold(val["y"], val[prob_key])
    m = full_metrics(test["y"], test[prob_key], tau)
    return [m], {k: {"mean": v, "std": 0.0} for k, v in
                 {kk: m[kk] for kk in ("precision", "recall", "macro_f1", "weighted_f1",
                                       "attack_f1", "fpr", "fnr", "mcc", "roc_auc", "pr_auc")}.items()} | {
        "confusion_matrix_sum": m["confusion_matrix"], "thresholds": [m["threshold"]], "n_seeds": 1,
    }


def main():
    all_results = {}
    per_seed_all = {}

    print("=== Seeded models (CNN, RNN, VQC, QuantumAI-deploy, QuantumAI-circ) ===")
    for name, key in (("cnn", "prob"), ("rnn", "prob"), ("vqc", "prob")):
        per_seed, agg = eval_seeded_model(name, key)
        all_results[name] = agg
        per_seed_all[name] = per_seed
        print(f"{name}: macro_f1={agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f}  "
              f"MCC={agg['mcc']['mean']:.3f}±{agg['mcc']['std']:.3f}")

    for name, key in (("quantumai", "p_deploy"), ("quantumai", "p_circ")):
        tag = "quantumai_deploy" if key == "p_deploy" else "quantumai_circ"
        per_seed = []
        for seed in SEEDS:
            val = np.load(PRED_DIR / f"quantumai_seed{seed}_val.npz")
            test = np.load(PRED_DIR / f"quantumai_seed{seed}_test.npz")
            tau, _ = best_threshold(val["y"], val[key])
            m = full_metrics(test["y"], test[key], tau)
            m["seed"] = seed
            per_seed.append(m)
        agg = aggregate_seeds(per_seed)
        all_results[tag] = agg
        per_seed_all[tag] = per_seed
        print(f"{tag}: macro_f1={agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f}  "
              f"MCC={agg['mcc']['mean']:.3f}±{agg['mcc']['std']:.3f}")

    print("\n=== Deterministic models (Classical-8D, Cosine/kNN) ===")
    for name in ("classical8d", "cosine"):
        per_seed, agg = eval_deterministic_model(name)
        all_results[name] = agg
        per_seed_all[name] = per_seed
        print(f"{name}: macro_f1={agg['macro_f1']['mean']:.3f}  MCC={agg['mcc']['mean']:.3f}  "
              f"(deterministic, single run)")

    # ---------------------------------------------------------------------
    # Fusion: equal-weight (4-agent), calibrated (Platt/logistic on D_val),
    # and leave-one-out ablation. Computed per seed (CNN/RNN/QuantumAI-deploy
    # vary by seed; Cosine is fixed).
    # ---------------------------------------------------------------------
    print("\n=== Fusion (equal-weight / calibrated / leave-one-out ablation) ===")
    cosine_val = np.load(PRED_DIR / "cosine_val.npz")
    cosine_test = np.load(PRED_DIR / "cosine_test.npz")

    fusion_per_seed = {"equal_weight": [], "calibrated": []}
    loo_per_seed = {name: [] for name in ("quantumai", "cosine", "rnn", "cnn")}

    for seed in SEEDS:
        cnn_val = np.load(PRED_DIR / f"cnn_seed{seed}_val.npz")
        cnn_test = np.load(PRED_DIR / f"cnn_seed{seed}_test.npz")
        rnn_val = np.load(PRED_DIR / f"rnn_seed{seed}_val.npz")
        rnn_test = np.load(PRED_DIR / f"rnn_seed{seed}_test.npz")
        q_val = np.load(PRED_DIR / f"quantumai_seed{seed}_val.npz")
        q_test = np.load(PRED_DIR / f"quantumai_seed{seed}_test.npz")

        y_val = cnn_val["y"]
        y_test = cnn_test["y"]
        assert np.array_equal(y_val, cosine_val["y"]) and np.array_equal(y_val, rnn_val["y"])

        agents_val = {
            "quantumai": q_val["p_deploy"], "cosine": cosine_val["prob"],
            "rnn": rnn_val["prob"], "cnn": cnn_val["prob"],
        }
        agents_test = {
            "quantumai": q_test["p_deploy"], "cosine": cosine_test["prob"],
            "rnn": rnn_test["prob"], "cnn": cnn_test["prob"],
        }

        # --- Equal-weight fusion (0.25 each) ---
        w_eq = {k: 0.25 for k in agents_val}
        fusion_val_eq = sum(w_eq[k] * agents_val[k] for k in agents_val)
        fusion_test_eq = sum(w_eq[k] * agents_test[k] for k in agents_test)
        tau_eq, _ = best_threshold(y_val, fusion_val_eq)
        m_eq = full_metrics(y_test, fusion_test_eq, tau_eq)
        m_eq["seed"] = seed
        fusion_per_seed["equal_weight"].append(m_eq)

        # --- Calibrated fusion: logistic regression over the 4 raw agent
        #     probabilities, fit on D_val only, applied once to D_test ---
        X_val_stack = np.column_stack([agents_val[k] for k in ("quantumai", "cosine", "rnn", "cnn")])
        X_test_stack = np.column_stack([agents_test[k] for k in ("quantumai", "cosine", "rnn", "cnn")])
        calibrator = LogisticRegression(max_iter=1000, random_state=seed)
        calibrator.fit(X_val_stack, y_val)
        fusion_val_cal = calibrator.predict_proba(X_val_stack)[:, 1]
        fusion_test_cal = calibrator.predict_proba(X_test_stack)[:, 1]
        tau_cal, _ = best_threshold(y_val, fusion_val_cal)
        m_cal = full_metrics(y_test, fusion_test_cal, tau_cal)
        m_cal["seed"] = seed
        m_cal["calibrator_coef"] = calibrator.coef_[0].tolist()
        m_cal["calibrator_intercept"] = float(calibrator.intercept_[0])
        fusion_per_seed["calibrated"].append(m_cal)

        # --- Leave-one-out ablation (equal-weight over remaining 3) ---
        for drop in loo_per_seed:
            remaining = [k for k in agents_val if k != drop]
            w = 1.0 / len(remaining)
            fv = sum(w * agents_val[k] for k in remaining)
            ft = sum(w * agents_test[k] for k in remaining)
            tau_d, _ = best_threshold(y_val, fv)
            m_d = full_metrics(y_test, ft, tau_d)
            m_d["seed"] = seed
            m_d["dropped_agent"] = drop
            loo_per_seed[drop].append(m_d)

    fusion_agg = {name: aggregate_seeds(ms) for name, ms in fusion_per_seed.items()}
    loo_agg = {name: aggregate_seeds(ms) for name, ms in loo_per_seed.items()}

    for name, agg in fusion_agg.items():
        print(f"fusion[{name}]: macro_f1={agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f}  "
              f"MCC={agg['mcc']['mean']:.3f}±{agg['mcc']['std']:.3f}")
    for name, agg in loo_agg.items():
        print(f"fusion[drop={name}]: macro_f1={agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f}")

    # ---------------------------------------------------------------------
    # Write everything out.
    # ---------------------------------------------------------------------
    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(RESULTS_DIR / "metrics_per_seed.json", "w") as f:
        json.dump(per_seed_all, f, indent=2)
    with open(RESULTS_DIR / "fusion_ablation.json", "w") as f:
        json.dump({"equal_weight": fusion_agg["equal_weight"], "calibrated": fusion_agg["calibrated"],
                    "leave_one_out": loo_agg}, f, indent=2)
    with open(RESULTS_DIR / "fusion_per_seed.json", "w") as f:
        json.dump({"equal_weight": fusion_per_seed["equal_weight"],
                    "calibrated": fusion_per_seed["calibrated"],
                    "leave_one_out": loo_per_seed}, f, indent=2)

    print(f"\nWrote metrics_summary.json, metrics_per_seed.json, fusion_ablation.json, fusion_per_seed.json")


if __name__ == "__main__":
    main()
