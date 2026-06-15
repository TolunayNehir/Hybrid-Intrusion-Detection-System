"""
Classical CNN Model Agent  (Traditional CNN)
=============================================
Part of the Classical ML path in the General System Diagram:
  Reference Embedding Data Lake → CNN Model → CNN Classification Output → Decision Fusion Layer

Architecture (Traditional):
  Conv1d(1 → 16,  kernel=3, padding=1) → ReLU → MaxPool1d(2)
  Conv1d(16 → 32, kernel=3, padding=1) → ReLU → MaxPool1d(2)
  Flatten  (32 × 32 = 1 024 for input_dim=128)
  Linear(1024 → 64)  → ReLU
  Linear(64  → 1)    → raw logit

Spatial progression (input_dim=128):
  128 → MaxPool → 64 → MaxPool → 32
  Flat size: 32 channels × 32 length = 1 024

Training:
  Adam (lr=0.001) + BCEWithLogitsLoss (pos_weight for class imbalance)
  Inference: CNN probability blended with k-NN + centroid correction
"""

import numpy as np
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_models import PreprocessedData, EmbeddingVector, ClassificationResult
from utils.logger import get_logger
from config.settings import CLASSICAL_ML

logger = get_logger("CNN_Classical")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Network Definition  (Traditional CNN)
# ─────────────────────────────────────────────────────────────

class _CNNNet(nn.Module):
    """
    Traditional 1-D CNN for binary intrusion detection.

    Input : (batch, input_dim)  — flat embedding vector (default 128-dim)
    Output: (batch,)            — raw logit (BCEWithLogitsLoss-compatible)

    Layers:
      Conv1d: 1  → 16  (kernel=3, padding=1) → ReLU → MaxPool1d(2)
      Conv1d: 16 → 32  (kernel=3, padding=1) → ReLU → MaxPool1d(2)
      Flatten  → 1 024  (for input_dim=128)
      Linear: 1024 → 64  → ReLU
      Linear:   64 → 1
    """

    def __init__(self, input_dim: int = 128,
                 n_conv_layers: int = 2,
                 conv1_out: int = 16, conv2_out: int = 32,
                 fc_hidden: int = 64, kernel_size: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        n_conv_layers = max(1, int(n_conv_layers))

        # Build conv blocks dynamically
        # Layer 0 : 1        → conv1_out
        # Layer 1 : conv1_out → conv2_out
        # Layer i≥2: prev_out → min(prev_out*2, 512)
        blocks = []
        in_ch  = 1
        out_ch = conv1_out
        for i in range(n_conv_layers):
            if i == 0:
                out_ch = conv1_out
            elif i == 1:
                in_ch  = conv1_out
                out_ch = conv2_out
            else:
                in_ch  = out_ch
                out_ch = min(out_ch * 2, 512)
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        flat_size = self._compute_flat_size(input_dim)

        head_layers = [nn.Linear(flat_size, fc_hidden), nn.ReLU()]
        if dropout > 0.0:
            head_layers.append(nn.Dropout(p=dropout))
        head_layers.append(nn.Linear(fc_hidden, 1))
        self.head = nn.Sequential(*head_layers)

    def _compute_flat_size(self, input_dim: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out   = self.conv(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)    # (B, D) → (B, 1, D)
        x = self.conv(x)       # (B, 32, D/4)
        x = x.flatten(1)       # (B, flat_size)
        return self.head(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Public Agent Class
# ─────────────────────────────────────────────────────────────

class ClassicalCNNModel:
    """
    Classical CNN (Traditional) — simple 2-conv architecture, k-NN corrected.

    Two inference modes:
      "embedding" — 128-dim semantic embedding vectors (startup default)
      "direct"    — raw numeric CSV features (after CSV upload)
    """

    def __init__(self):
        cfg = CLASSICAL_ML["cnn"]
        self.lr:            float = cfg["learning_rate"]
        self.epochs:        int   = cfg["epochs"]
        self.n_conv_layers: int   = cfg.get("n_conv_layers",  2)
        self.conv1_out:     int   = cfg.get("conv1_out",     16)
        self.conv2_out:     int   = cfg.get("conv2_out",     32)
        self.fc_hidden:     int   = cfg.get("fc_hidden",     64)
        self.kernel_size:   int   = cfg.get("kernel_size",    3)
        self.dropout:       float = cfg.get("dropout",       0.0)

        self.net:        Optional[_CNNNet] = None
        self.input_dim:  Optional[int]    = None
        self.is_trained: bool             = False

        self.training_mode:   str           = "embedding"
        self.feature_columns: List[str]     = []
        self.feature_mean:    Optional[np.ndarray] = None
        self.feature_std:     Optional[np.ndarray] = None

        self.train_X:           Optional[np.ndarray] = None
        self.train_y:           Optional[np.ndarray] = None
        self._attack_centroid:  Optional[np.ndarray] = None
        self._normal_centroid:  Optional[np.ndarray] = None

        logger.info(
            f"ClassicalCNN initialized | device={DEVICE} | "
            f"n_conv_layers={self.n_conv_layers} | "
            f"conv=[1→{self.conv1_out}→{self.conv2_out}…] | "
            f"fc={self.fc_hidden} | kernel={self.kernel_size} | "
            f"dropout={self.dropout} | lr={self.lr} | epochs={self.epochs}"
        )

    def _build(self, input_dim: int):
        self.input_dim = input_dim
        self.net = _CNNNet(
            input_dim=input_dim,
            n_conv_layers=self.n_conv_layers,
            conv1_out=self.conv1_out,
            conv2_out=self.conv2_out,
            fc_hidden=self.fc_hidden,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in self.net.parameters())
        logger.info(
            f"_CNNNet built | input_dim={input_dim} | n_conv_layers={self.n_conv_layers} | "
            f"params={n_params:,} | conv1={self.conv1_out} conv2={self.conv2_out} | "
            f"kernel={self.kernel_size} | dropout={self.dropout} | fc={self.fc_hidden}"
        )

    # ── Training ──────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """
        Train Traditional CNN:
          Adam (lr=0.001) + BCEWithLogitsLoss with class-imbalance pos_weight
        """
        self._build(X.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=min(batch_size, len(X)),
                             shuffle=True, drop_last=False)

        n_pos      = max(float(y.sum()), 1.0)
        n_neg      = max(float(len(y) - y.sum()), 1.0)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        logger.info(
            f"Training ClassicalCNN (Traditional): {len(X)} samples | {self.epochs} epochs | "
            f"pos_weight={pos_weight.item():.2f}"
        )

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            correct    = 0
            total      = 0

            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(xb)
                with torch.no_grad():
                    preds   = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == yb).sum().item()
                    total   += len(xb)

            if epoch % 10 == 0 or epoch == self.epochs:
                avg_loss = total_loss / max(total, 1)
                acc      = correct / max(total, 1)
                lr_now   = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  CNN Epoch {epoch:>3}/{self.epochs}: "
                    f"loss={avg_loss:.4f}  acc={acc:.4f}  lr={lr_now:.6f}"
                )

        self.net.eval()
        self.is_trained = True

        self.train_X = X.copy()
        self.train_y = y.astype(np.float32).copy()
        self._compute_centroids()

        logger.info("ClassicalCNN (Traditional) training complete")

    # ── k-NN + Centroid helpers ────────────────────────────────

    def _compute_centroids(self):
        if self.train_X is None or len(self.train_X) == 0:
            return
        attack_mask = self.train_y > 0.5
        normal_mask = ~attack_mask.astype(bool)
        if attack_mask.any():
            c = self.train_X[attack_mask].mean(axis=0)
            n = np.linalg.norm(c)
            self._attack_centroid = c / (n + 1e-8)
        if normal_mask.any():
            c = self.train_X[normal_mask].mean(axis=0)
            n = np.linalg.norm(c)
            self._normal_centroid = c / (n + 1e-8)

    def _centroid_score(self, vec: np.ndarray) -> float:
        ac = self._attack_centroid
        nc = self._normal_centroid
        if ac is None or nc is None:
            return 0.5
        vec_n = vec / (np.linalg.norm(vec) + 1e-8)
        sa = float(vec_n @ ac)
        sn = float(vec_n @ nc)
        return float(1.0 / (1.0 + np.exp(-(sa - sn) * 5.0)))

    def _knn_score(self, vec: np.ndarray, k: int = 7) -> float:
        if self.train_X is None or len(self.train_X) == 0:
            return 0.5
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-8:
            return float(self.train_y.mean())
        train_norms = np.linalg.norm(self.train_X, axis=1) + 1e-8
        sims  = (self.train_X @ vec) / (train_norms * vec_norm)
        top_k = np.argsort(sims)[-min(k, len(sims)):]
        knn_vote      = float(self.train_y[top_k].mean())
        centroid_vote = self._centroid_score(vec)
        return 0.35 * knn_vote + 0.65 * centroid_vote

    def _knn_scores_batch(self, vecs: np.ndarray, k: int = 7) -> np.ndarray:
        if self.train_X is None or len(self.train_X) == 0:
            return np.full(len(vecs), 0.5)
        vec_norms   = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        train_norms = np.linalg.norm(self.train_X, axis=1, keepdims=True) + 1e-8
        vecs_n  = vecs / vec_norms
        train_n = self.train_X / train_norms
        sims    = vecs_n @ train_n.T
        k_eff   = min(k, sims.shape[1])
        top_k   = np.argsort(sims, axis=1)[:, -k_eff:]
        knn_votes = np.array([self.train_y[idx].mean() for idx in top_k])
        ac = self._attack_centroid
        nc = self._normal_centroid
        if ac is not None and nc is not None:
            sa = vecs_n @ ac
            sn = vecs_n @ nc
            centroid_votes = 1.0 / (1.0 + np.exp(-(sa - sn) * 5.0))
        else:
            centroid_votes = np.full(len(vecs), 0.5)
        return 0.35 * knn_votes + 0.65 * centroid_votes

    @staticmethod
    def _adaptive_blend(cnn_p: float, knn_p: float) -> float:
        return 0.50 * cnn_p + 0.50 * knn_p

    # ── Inference ─────────────────────────────────────────────

    def _predict_prob(self, vector: np.ndarray) -> float:
        self.net.eval()
        with torch.no_grad():
            t     = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logit = self.net(t).item()
            return float(torch.sigmoid(torch.tensor(logit)).item())

    def _direct_vector(self, preprocessed: PreprocessedData) -> np.ndarray:
        vec = np.array(preprocessed.features, dtype=np.float32).flatten()
        if self.input_dim is not None:
            if len(vec) < self.input_dim:
                vec = np.pad(vec, (0, self.input_dim - len(vec)))
            else:
                vec = vec[:self.input_dim]
        if self.feature_mean is not None and self.feature_std is not None:
            std = np.where(self.feature_std > 0, self.feature_std, 1.0)
            vec = (vec - self.feature_mean) / std
        return vec

    # ── Public API ────────────────────────────────────────────

    def predict(self,
                embedding: EmbeddingVector,
                preprocessed: PreprocessedData) -> ClassificationResult:
        if not self.is_trained:
            return ClassificationResult(
                agent_name="CNN_Classical",
                event_id=preprocessed.event_id,
                prediction=0, confidence=0.5, label="Normal",
                details={"status": "untrained", "model": "pytorch_cnn_traditional"},
            )

        vec      = self._direct_vector(preprocessed) if self.training_mode == "direct" \
                   else embedding.vector
        cnn_prob = self._predict_prob(vec)

        if self.training_mode == "embedding" and self.train_X is not None:
            knn_prob = self._knn_score(vec)
            prob     = self._adaptive_blend(cnn_prob, knn_prob)
        else:
            prob     = cnn_prob
            knn_prob = None

        prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else (1.0 - prob)
        label      = (
            preprocessed.metadata.get("attack_type", "Intrusion")
            if prediction == 1 else "Normal"
        )

        details = {
            "cnn_probability":   float(cnn_prob),
            "final_probability": float(prob),
            "model_type":        "pytorch_cnn_traditional",
            "architecture":      "Conv1d(1→16) → Conv1d(16→32) → Flatten → Linear(1024→64) → Linear(64→1)",
            "device":            str(DEVICE),
            "inference_mode":    self.training_mode,
        }
        if knn_prob is not None:
            details["knn_probability"] = float(knn_prob)

        return ClassificationResult(
            agent_name="CNN_Classical",
            event_id=preprocessed.event_id,
            prediction=prediction,
            confidence=float(confidence),
            label=label,
            details=details,
        )

    def predict_batch(self,
                      embeddings: List[EmbeddingVector],
                      preprocessed_batch: List[PreprocessedData]
                      ) -> List[ClassificationResult]:
        if not self.is_trained:
            return [self.predict(e, p) for e, p in zip(embeddings, preprocessed_batch)]

        self.net.eval()
        # In "direct" mode use preprocessed numeric features (same space as
        # training), otherwise fall back to word2vec embedding vectors.
        if self.training_mode == "direct":
            vecs = np.stack([self._direct_vector(p) for p in preprocessed_batch], axis=0)
        else:
            vecs = np.stack([e.vector for e in embeddings], axis=0)

        with torch.no_grad():
            t         = torch.tensor(vecs, dtype=torch.float32).to(DEVICE)
            logits    = self.net(t)
            cnn_probs = torch.sigmoid(logits).cpu().numpy()

        if self.training_mode == "embedding" and self.train_X is not None:
            knn_probs   = self._knn_scores_batch(vecs)
            final_probs = np.array([
                self._adaptive_blend(float(cp), float(kp))
                for cp, kp in zip(cnn_probs, knn_probs)
            ])
        else:
            knn_probs   = None
            final_probs = cnn_probs

        results: List[ClassificationResult] = []
        for i, (prob, emb, prep) in enumerate(
                zip(final_probs, embeddings, preprocessed_batch)):
            prediction = int(prob > 0.5)
            confidence = float(prob) if prediction == 1 else float(1.0 - prob)
            label      = (prep.metadata.get("attack_type", "Intrusion")
                          if prediction == 1 else "Normal")
            det = {
                "cnn_probability":   float(cnn_probs[i]),
                "final_probability": float(prob),
                "model_type":        "pytorch_cnn_traditional",
                "device":            str(DEVICE),
            }
            if knn_probs is not None:
                det["knn_probability"] = float(knn_probs[i])
            results.append(ClassificationResult(
                agent_name="CNN_Classical",
                event_id=prep.event_id,
                prediction=prediction,
                confidence=confidence,
                label=label,
                details=det,
            ))

        logger.info(
            f"CNN_Classical (Traditional) predicted {len(results)} events: "
            f"{sum(r.prediction for r in results)} intrusions"
        )
        return results

    def save_model(self) -> bool:
        from utils.model_persistence import save_classical_cnn, ensure_model_dirs
        ensure_model_dirs()
        return save_classical_cnn(self)

    def load_model(self) -> bool:
        from utils.model_persistence import load_classical_cnn
        return load_classical_cnn(self)
