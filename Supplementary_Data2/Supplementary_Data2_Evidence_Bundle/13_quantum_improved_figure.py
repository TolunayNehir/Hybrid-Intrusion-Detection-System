"""Comparison figure for the quantum-path improvement study.
Labels use the manuscript's Roman-numeral scheme:
  (i) marginal-trained readout, (ii) quantile encoding,
  (iii) QCNN-pool architecture, (iv) enhanced optimization."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
qi = json.load(open(P / "results_quantum_improved" / "quantum_improvements.json"))
o4 = json.load(open(P / "results_quantum_improved" / "option4_results.json"))
rep = {n: json.load(open(P / "results_replication" / n / "replication_results.json"))
       for n in ("nslkdd", "unsw")}
prim = json.load(open(P / "results" / "metrics_summary.json"))

def g(d):
    return d["mean"] if isinstance(d, dict) else d

DS = [("cicids", "CICIDS2017"), ("nslkdd", "NSL-KDD"), ("unsw", "UNSW-NB15")]

def baseline(ds, agent):
    if ds == "cicids":
        return g(prim[agent]["macro_f1"])
    return rep[ds]["summary"][agent]["macro_f1"]["mean"]

series = {
    "Circuit-only (original)": [baseline(d, "quantumai_circ") for d, _ in DS],
    "Circuit + marginal-trained head (i)": [qi["stageA"][d]["circuit_marginal_head"]["macro_f1"]["mean"] for d, _ in DS],
    "VQC (original)": [baseline(d, "vqc") for d, _ in DS],
    "VQC + quantile encoding (ii)": [qi["stageA"][d]["vqc_quantile"]["macro_f1"]["mean"] for d, _ in DS],
    "Combined (i)+(ii)+(iii)": [qi["stageC"][d]["combined"]["macro_f1"]["mean"] for d, _ in DS],
    "Combined (i)+(ii)+(iii)+(iv)": [o4["D3_combined_opt4"][d]["combined_opt4"]["macro_f1"]["mean"] for d, _ in DS],
}
cls8 = [baseline(d, "classical8d") for d, _ in DS]
dep = [baseline(d, "quantumai_deploy") for d, _ in DS]

fig, ax = plt.subplots(figsize=(9.2, 4.6))
x = np.arange(len(DS)); W = 0.125
colors = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d", "#74c476", "#238b45"]
for i, (lbl, vals) in enumerate(series.items()):
    ax.bar(x + (i - 2.5) * W, vals, W * 0.9, label=lbl, color=colors[i])
for xi, (c8, dp) in enumerate(zip(cls8, dep)):
    ax.hlines(c8, xi - 0.44, xi + 0.44, color="k", ls="--", lw=1.2)
    ax.hlines(dp, xi - 0.44, xi + 0.44, color="gray", ls=":", lw=1.2)
ax.plot([], [], "k--", lw=1.2, label="Classical-8D (same features)")
ax.plot([], [], color="gray", ls=":", lw=1.2, label="QuantumAI deployed (classical head)")
ax.set_xticks(x); ax.set_xticklabels([n for _, n in DS])
ax.set_ylabel("Locked-test macro-$F_1$ (mean over 5 seeds)")
ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.3)
ax.set_title("Quantum decision paths: original vs improvements (i)--(iv)")
ax.legend(fontsize=7.2, ncol=2, loc="lower right")
fig.tight_layout()
out = P.parent / "figures_phaseVI" / "quantum_improvements.pdf"
fig.savefig(out)
print("wrote", out)
for lbl, vals in series.items():
    print(f"  {lbl:38s}", ["%.3f" % v for v in vals])
