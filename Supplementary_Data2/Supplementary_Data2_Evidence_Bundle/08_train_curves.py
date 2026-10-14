"""
Phase-VI addition: CNN and RNN train/validation loss curves (PhaseVI 2.12).

Re-runs the SAME training recipe as 02_train_models.py (same _CNNNet/_RNNNet
classes, same settings.py hyperparameters, same seeds, same z-score+clip
preprocessing), but additionally records per-epoch TRAIN and VALIDATION loss so
that training dynamics can be plotted. Also records the final locked-D_test
macro-F1 per seed as a reproducibility check against the archived tables.

Outputs (kept separate from the archived results/ dir so the canonical numbers
are never overwritten):
  experiments/results_phaseVI/train_curves.json
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ["HYBRID_IDS_REPO"]).resolve()
sys.path.insert(0, str(REPO_ROOT))

from agents.classical_ml.cnn_model import _CNNNet
from agents.classical_ml.rnn_model import _RNNNet
from config.settings import CLASSICAL_ML

RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "results_phaseVI"
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def load_locked():
    df = pd.read_csv(RESULTS / "locked_dataset.csv")
    fcols = [c for c in df.columns if c not in ("row_id", "partition", "label", "target_binary")]
    tr = df[df.partition == "train"].reset_index(drop=True)
    va = df[df.partition == "validation"].reset_index(drop=True)
    te = df[df.partition == "test"].reset_index(drop=True)
    Xtr = tr[fcols].to_numpy(np.float64); Xva = va[fcols].to_numpy(np.float64); Xte = te[fcols].to_numpy(np.float64)
    ytr = tr.target_binary.to_numpy(np.float32); yva = va.target_binary.to_numpy(np.float32); yte = te.target_binary.to_numpy(np.float32)
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-8] = 1.0
    norm = lambda X: np.clip((X - mean) / std, -5.0, 5.0).astype(np.float32)
    return norm(Xtr), ytr, norm(Xva), yva, norm(Xte), yte


def macro_f1_at_val_threshold(logits_val, yval, logits_test, ytest):
    pv = 1/(1+np.exp(-logits_val)); pt = 1/(1+np.exp(-logits_test))
    best_t, best = 0.5, -1
    for t in np.arange(0.01, 1.0, 0.01):
        f = f1_score(yval, (pv > t).astype(int), average="macro")
        if f > best: best, best_t = f, t
    return f1_score(ytest, (pt > best_t).astype(int), average="macro")


def train_cnn(Xtr, ytr, Xva, yva, Xte, yte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = CLASSICAL_ML["cnn"]
    net = _CNNNet(input_dim=Xtr.shape[1], n_conv_layers=cfg["n_conv_layers"], conv1_out=cfg["conv1_out"],
                  conv2_out=cfg["conv2_out"], fc_hidden=cfg["fc_hidden"], kernel_size=cfg["kernel_size"], dropout=cfg["dropout"])
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True, drop_last=False)
    n_pos = max(float(ytr.sum()), 1.0); n_neg = max(float(len(ytr) - ytr.sum()), 1.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg/n_pos]))
    opt = optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)
    tr_hist, va_hist = [], []
    for epoch in range(1, cfg["epochs"]+1):
        net.train(); tot = 0.0; n = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(net(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*len(xb); n += len(xb)
        tr_hist.append(tot/max(n, 1))
        net.eval()
        with torch.no_grad():
            va_hist.append(float(crit(net(Xva_t), yva_t).item()))
    net.eval()
    with torch.no_grad():
        lv = net(Xva_t).numpy(); lt = net(torch.tensor(Xte)).numpy()
    return tr_hist, va_hist, macro_f1_at_val_threshold(lv, yva, lt, yte)


def train_rnn(Xtr, ytr, Xva, yva, Xte, yte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = CLASSICAL_ML["rnn"]
    net = _RNNNet(input_dim=Xtr.shape[1], hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    n_neg = float((ytr == 0).sum()); n_pos = float((ytr == 1).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg/max(n_pos, 1)]))
    opt = optim.Adam(net.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=max(cfg["epochs"]//3, 10), gamma=0.5)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)
    tr_hist, va_hist = [], []
    for epoch in range(1, cfg["epochs"]+1):
        net.train(); tot = 0.0; n = 0
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(net(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0); opt.step()
            tot += loss.item()*len(xb); n += len(xb)
        sched.step(); tr_hist.append(tot/max(n, 1))
        net.eval()
        with torch.no_grad():
            va_hist.append(float(crit(net(Xva_t), yva_t).item()))
    net.eval()
    with torch.no_grad():
        lv = net(Xva_t).numpy(); lt = net(torch.tensor(Xte)).numpy()
    return tr_hist, va_hist, macro_f1_at_val_threshold(lv, yva, lt, yte)


def main():
    Xtr, ytr, Xva, yva, Xte, yte = load_locked()
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)} feat={Xtr.shape[1]}")
    out = {"cnn": {"train": [], "val": [], "test_macro_f1": []},
           "rnn": {"train": [], "val": [], "test_macro_f1": []}}
    for s in SEEDS:
        tr, va, f1 = train_cnn(Xtr, ytr, Xva, yva, Xte, yte, s)
        out["cnn"]["train"].append(tr); out["cnn"]["val"].append(va); out["cnn"]["test_macro_f1"].append(f1)
        print(f"  CNN seed{s}: test macroF1={f1:.4f}")
        tr, va, f1 = train_rnn(Xtr, ytr, Xva, yva, Xte, yte, s)
        out["rnn"]["train"].append(tr); out["rnn"]["val"].append(va); out["rnn"]["test_macro_f1"].append(f1)
        print(f"  RNN seed{s}: test macroF1={f1:.4f}")
    for m in ("cnn", "rnn"):
        arr = np.array(out[m]["test_macro_f1"])
        out[m]["test_macro_f1_mean"] = float(arr.mean()); out[m]["test_macro_f1_std"] = float(arr.std(ddof=1))
    out["torch_version"] = torch.__version__
    with open(OUT / "train_curves.json", "w") as f:
        json.dump(out, f, indent=1)
    print("CNN fresh mean macroF1 = %.4f (archived 0.951)" % out["cnn"]["test_macro_f1_mean"])
    print("RNN fresh mean macroF1 = %.4f (archived 0.954)" % out["rnn"]["test_macro_f1_mean"])
    print("wrote", OUT / "train_curves.json")


if __name__ == "__main__":
    main()
