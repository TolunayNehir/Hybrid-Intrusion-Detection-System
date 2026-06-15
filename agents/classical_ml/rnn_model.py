"""
Classical RNN Model Agent  (PyTorch Implementation)
====================================================
Part of the Classical ML path in the General System Diagram:
  Reference Embedding Data Lake → RNN Model → RNN Classification Output → Decision Fusion Layer

Architecture:
  Embedding → nn.RNN (multi-layer, bidirectional option) → Linear → Sigmoid

Each embedding vector is treated as a sequence of *feature-wise* timesteps so
the RNN can capture ordered dependencies inside the feature vector.
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

logger = get_logger("RNN_Classical")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# PyTorch Network Definition
# ─────────────────────────────────────────────────────────────

class _RNNNet(nn.Module):
    """
    Multi-layer Elman RNN for binary classification.

    Input  : (batch, input_dim)  — flat embedding vector
    The vector is reshaped to (batch, n_groups=8, group_size=input_dim//8)
    to keep the sequence short (≤8 steps) and avoid gradient vanishing
    that occurs when every feature is its own time-step (seq=128, size=1).

    Output : (batch,)  — sigmoid probability.
    """

    def __init__(self, input_dim: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()

        self.input_dim   = input_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Group features into ≤8 time steps.  Find the largest divisor of
        # input_dim that is ≤8 (guarantees exact reshape, no padding needed).
        n_groups = min(8, input_dim)
        while input_dim % n_groups != 0 and n_groups > 1:
            n_groups -= 1
        self._seq_len       = n_groups
        self._feat_per_step = input_dim // n_groups

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=self._feat_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            nonlinearity="tanh",
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(hidden_size // 2, 16), 1),
        )  # Raw logit — Sigmoid applied externally during inference

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_dim)
        returns: (batch,)
        """
        # Reshape flat vector → (batch, seq_len, feat_per_step)
        x   = x.reshape(x.shape[0], self._seq_len, self._feat_per_step)
        out, _ = self.rnn(x)           # (batch, seq, hidden)
        last   = out[:, -1, :]         # (batch, hidden) — last timestep
        return self.head(last).squeeze(-1)   # (batch,)


# ─────────────────────────────────────────────────────────────
# Public Agent Class
# ─────────────────────────────────────────────────────────────

class ClassicalRNNModel:
    """
    Classical RNN Model for intrusion detection.

    Receives embedding vectors from the Reference Embedding Data Lake.
    Outputs ClassificationResult objects for the Decision Fusion Layer.

    Two training modes:
      "embedding" — trained on 128-dim semantic embedding vectors (default)
      "direct"    — trained on raw numeric CSV features via DatasetAnalyzer
    """

    def __init__(self):
        cfg = CLASSICAL_ML["rnn"]
        self.hidden_size: int  = cfg["hidden_size"]
        self.num_layers: int   = cfg["num_layers"]
        self.dropout: float    = cfg["dropout"]
        self.lr: float         = cfg["learning_rate"]
        self.epochs: int       = cfg["epochs"]

        self.net: Optional[_RNNNet] = None
        self.input_dim: Optional[int] = None
        self.is_trained: bool = False

        # Direct-mode metadata (set when trained via DatasetAnalyzer)
        self.training_mode: str = "embedding"
        self.feature_columns: List[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std:  Optional[np.ndarray] = None

        logger.info(
            f"ClassicalRNN (PyTorch) initialized | device={DEVICE} | "
            f"hidden={self.hidden_size} | layers={self.num_layers} | "
            f"dropout={self.dropout} | lr={self.lr} | epochs={self.epochs}"
        )

    # ------------------------------------------------------------------
    def _build(self, input_dim: int):
        """Instantiate the network and move it to the target device."""
        self.input_dim = input_dim
        self.net = _RNNNet(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(DEVICE)
        logger.info(
            f"_RNNNet built | input_dim={input_dim} | "
            f"params={sum(p.numel() for p in self.net.parameters()):,}"
        )

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """
        Train the RNN on reference embedding data.

        Parameters
        ----------
        X : np.ndarray  shape (N, input_dim)
        y : np.ndarray  shape (N,)  binary labels {0, 1}
        batch_size : int
        """
        self._build(X.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Class-weighted loss to handle Normal/Attack imbalance
        n_neg = float((y == 0).sum())   # Normal count
        n_pos = float((y == 1).sum())   # Attack count
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Gentle LR warmdown: halve only twice over the full run
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(self.epochs // 3, 10), gamma=0.5
        )

        self.net.train()
        logger.info(
            f"Training ClassicalRNN (PyTorch): {len(X)} samples | {self.epochs} epochs"
            f" | pos_weight={pos_weight.item():.2f}"
        )

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            correct    = 0
            total      = 0

            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                optimizer.zero_grad()
                preds = self.net(xb)
                loss  = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(xb)
                correct    += ((preds > 0.5).float() == yb).sum().item()
                total      += len(xb)

            scheduler.step()

            if epoch % 5 == 0 or epoch == self.epochs:
                avg_loss = total_loss / total
                acc      = correct / total
                logger.info(
                    f"  RNN Epoch {epoch:>3}/{self.epochs}: "
                    f"loss={avg_loss:.4f}  acc={acc:.4f}"
                )

        self.net.eval()
        self.is_trained = True
        logger.info("ClassicalRNN (PyTorch) training complete")

    # ------------------------------------------------------------------
    def _predict_prob(self, vector: np.ndarray) -> float:
        """Return the sigmoid probability for a single embedding vector."""
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logit = self.net(t).item()
            return float(torch.sigmoid(torch.tensor(logit)).item())

    # ------------------------------------------------------------------
    def _direct_vector(self, preprocessed: PreprocessedData) -> np.ndarray:
        """
        Build a correctly-sized input vector for direct-mode inference.
        Uses preprocessed.features, padded/truncated to match self.input_dim,
        then applies the stored z-score normalization.
        """
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

    def predict(self,
                embedding: EmbeddingVector,
                preprocessed: PreprocessedData) -> ClassificationResult:
        """
        Predict intrusion probability for a single event.

        Parameters
        ----------
        embedding    : EmbeddingVector
        preprocessed : PreprocessedData

        Returns
        -------
        ClassificationResult
        """
        if not self.is_trained:
            return ClassificationResult(
                agent_name="RNN_Classical",
                event_id=preprocessed.event_id,
                prediction=0,
                confidence=0.5,
                label="Normal",
                details={"status": "untrained", "model": "pytorch_rnn"},
            )

        # Direct mode: use preprocessed numeric features (not embedding vector)
        if self.training_mode == "direct":
            vec = self._direct_vector(preprocessed)
        else:
            vec = embedding.vector

        prob       = self._predict_prob(vec)
        prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else (1.0 - prob)
        label      = (
            preprocessed.metadata.get("attack_type", "Intrusion")
            if prediction == 1 else "Normal"
        )

        return ClassificationResult(
            agent_name="RNN_Classical",
            event_id=preprocessed.event_id,
            prediction=prediction,
            confidence=float(confidence),
            label=label,
            details={
                "rnn_probability": float(prob),
                "model_type": "pytorch_rnn",
                "device": str(DEVICE),
                "inference_mode": self.training_mode,
            },
        )

    # ------------------------------------------------------------------
    def predict_batch(self,
                      embeddings: List[EmbeddingVector],
                      preprocessed_batch: List[PreprocessedData]
                      ) -> List[ClassificationResult]:
        """Batch prediction — runs a single forward pass for efficiency."""
        if not self.is_trained:
            return [self.predict(e, p) for e, p in zip(embeddings, preprocessed_batch)]

        self.net.eval()
        # In "direct" mode use preprocessed numeric features (consistent with
        # training), otherwise fall back to word2vec embedding vectors.
        if self.training_mode == "direct":
            vectors = np.stack([self._direct_vector(p) for p in preprocessed_batch], axis=0)
        else:
            vectors = np.stack([e.vector for e in embeddings], axis=0)
        with torch.no_grad():
            t      = torch.tensor(vectors, dtype=torch.float32).to(DEVICE)
            logits = self.net(t)                        # (N,) raw logits
            probs  = torch.sigmoid(logits).cpu().numpy()  # (N,)

        results: List[ClassificationResult] = []
        for prob, emb, prep in zip(probs, embeddings, preprocessed_batch):
            prediction = int(prob > 0.5)
            confidence = float(prob) if prediction == 1 else float(1.0 - prob)
            label = (
                prep.metadata.get("attack_type", "Intrusion")
                if prediction == 1 else "Normal"
            )
            results.append(ClassificationResult(
                agent_name="RNN_Classical",
                event_id=prep.event_id,
                prediction=prediction,
                confidence=confidence,
                label=label,
                details={
                    "rnn_probability": float(prob),
                    "model_type": "pytorch_rnn",
                    "device": str(DEVICE),
                },
            ))

        n_intrusions = sum(r.prediction for r in results)
        logger.info(
            f"RNN_Classical predicted {len(results)} events: "
            f"{n_intrusions} intrusions"
        )
        return results

    # ------------------------------------------------------------------
    def save_model(self) -> bool:
        """Save PyTorch RNN model checkpoint."""
        from utils.model_persistence import save_classical_rnn, ensure_model_dirs
        ensure_model_dirs()
        return save_classical_rnn(self)

    def load_model(self) -> bool:
        """Load PyTorch RNN model from checkpoint."""
        from utils.model_persistence import load_classical_rnn
        return load_classical_rnn(self)
