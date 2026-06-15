"""
Model Persistence & Checkpoint Management
============================================
Provides save/load utilities for all model components:
  - Quantum AI (QNN parameters, dimension reducer, RNN bridge)
  - Classical RNN & CNN (weights, biases, architecture)
  - Cosine Similarity (reference vectors, centroids)
  - Embedding Engine (vocabulary)
  - Reference Data Lake (embeddings, labels)

Features:
  - JSON metadata with versioning, training date, performance metrics
  - Pickle-based model state serialization
  - Auto-detection of existing checkpoints
  - Model versioning support
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from utils.logger import get_logger
from config.settings import MODEL_DIR

logger = get_logger("ModelPersistence")

# ─── Version ───
PERSISTENCE_VERSION = "1.0.0"

# ─── Subdirectory structure ───
MODEL_SUBDIRS = {
    "quantum": os.path.join(MODEL_DIR, "quantum"),
    "classical_rnn": os.path.join(MODEL_DIR, "classical", "rnn"),
    "classical_cnn": os.path.join(MODEL_DIR, "classical", "cnn"),
    "similarity": os.path.join(MODEL_DIR, "similarity"),
    "embeddings": os.path.join(MODEL_DIR, "embeddings"),
    "datalake": os.path.join(MODEL_DIR, "datalake"),
}


def ensure_model_dirs():
    """Create all model directories if they don't exist."""
    for name, path in MODEL_SUBDIRS.items():
        os.makedirs(path, exist_ok=True)
    logger.info(f"Model directories ensured under {MODEL_DIR}")


def _save_metadata(directory: str, model_name: str, metadata: Dict[str, Any]):
    """Save model metadata as JSON."""
    meta_path = os.path.join(directory, f"{model_name}_metadata.json")
    metadata["persistence_version"] = PERSISTENCE_VERSION
    metadata["saved_at"] = datetime.now().isoformat()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved: {meta_path}")


def _load_metadata(directory: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Load model metadata from JSON."""
    meta_path = os.path.join(directory, f"{model_name}_metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def _save_state(directory: str, model_name: str, state: Dict[str, Any]):
    """Save model state as pickle."""
    state_path = os.path.join(directory, f"{model_name}_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"State saved: {state_path}")


def _load_state(directory: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Load model state from pickle."""
    state_path = os.path.join(directory, f"{model_name}_state.pkl")
    if not os.path.exists(state_path):
        return None
    with open(state_path, "rb") as f:
        return pickle.load(f)


def checkpoint_exists(model_name: str) -> bool:
    """Check if a checkpoint exists for the given model."""
    subdir_map = {
        "quantum_ai": "quantum",
        "classical_rnn": "classical_rnn",
        "classical_cnn": "classical_cnn",
        "cosine_similarity": "similarity",
        "embedding_engine": "embeddings",
        "data_lake": "datalake",
    }
    subdir_key = subdir_map.get(model_name)
    if subdir_key is None:
        return False
    directory = MODEL_SUBDIRS[subdir_key]
    state_path = os.path.join(directory, f"{model_name}_state.pkl")
    return os.path.exists(state_path)


def get_all_checkpoint_status() -> Dict[str, bool]:
    """Get checkpoint availability for all models."""
    models = ["quantum_ai", "classical_rnn", "classical_cnn",
              "cosine_similarity", "embedding_engine", "data_lake"]
    return {m: checkpoint_exists(m) for m in models}


def get_checkpoint_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata info for a specific model checkpoint."""
    subdir_map = {
        "quantum_ai": "quantum",
        "classical_rnn": "classical_rnn",
        "classical_cnn": "classical_cnn",
        "cosine_similarity": "similarity",
        "embedding_engine": "embeddings",
        "data_lake": "datalake",
    }
    subdir_key = subdir_map.get(model_name)
    if subdir_key is None:
        return None
    return _load_metadata(MODEL_SUBDIRS[subdir_key], model_name)


# ═══════════════════════════════════════════════
# Quantum AI Save/Load
# ═══════════════════════════════════════════════

def save_quantum_ai(module) -> bool:
    """Save QuantumAIModule state."""
    try:
        directory = MODEL_SUBDIRS["quantum"]
        os.makedirs(directory, exist_ok=True)

        bridge = module.rnn_bridge   # QuantumGRUBridge (via property alias)
        # Build bridge state dict — supports both old RNN and new GRU bridges
        bridge_state: dict = {"bridge_hidden_size": bridge.hidden_size}
        if hasattr(bridge, "Wz_ih"):
            # GRU bridge (v2)
            bridge_state.update({
                "gru_Wz_ih": bridge.Wz_ih.copy(), "gru_Wz_hh": bridge.Wz_hh.copy(),
                "gru_bz":    bridge.bz.copy(),
                "gru_Wr_ih": bridge.Wr_ih.copy(), "gru_Wr_hh": bridge.Wr_hh.copy(),
                "gru_br":    bridge.br.copy(),
                "gru_Wn_ih": bridge.Wn_ih.copy(), "gru_Wn_hh": bridge.Wn_hh.copy(),
                "gru_bn":    bridge.bn.copy(),
                "gru_W_out": bridge.W_out.copy(),
                "gru_b_out": bridge.b_out.copy(),
            })
        else:
            # Legacy RNN bridge (v1) — save what exists
            bridge_state.update({
                "rnn_W_ih":  bridge.W_ih.copy(),
                "rnn_W_hh":  bridge.W_hh.copy(),
                "rnn_W_out": bridge.W_out.copy(),
                "rnn_b_h":   bridge.b_h.copy(),
                "rnn_b_out": bridge.b_out.copy(),
            })

        reducer_pm = (module.reducer.projection_matrix.copy()
                      if module.reducer.projection_matrix is not None else None)
        reducer_dm = getattr(module.reducer, "_data_mean", None)
        reducer_ds = getattr(module.reducer, "_data_std",  None)

        # v4: save Adam optimiser state for warm-restart
        adam_m = getattr(module, "adam_m", None)
        adam_v = getattr(module, "adam_v", None)
        adam_t = getattr(module, "adam_t", 0)

        # v4: save training-time normalisation statistics (min/max fix)
        reducer_train_min = getattr(module.reducer, "_train_min", None)
        reducer_train_max = getattr(module.reducer, "_train_max", None)

        state = {
            "qnn_thetas":                module.qnn.thetas.copy(),
            "qnn_n_qubits":              module.qnn.n_qubits,
            "qnn_n_layers":              module.qnn.n_layers,
            "reducer_projection_matrix": reducer_pm,
            "reducer_n_qubits":          module.reducer.n_qubits,
            "reducer_data_mean":         reducer_dm.copy() if reducer_dm is not None else None,
            "reducer_data_std":          reducer_ds.copy() if reducer_ds is not None else None,
            "reducer_train_min":         reducer_train_min.copy() if reducer_train_min is not None else None,
            "reducer_train_max":         reducer_train_max.copy() if reducer_train_max is not None else None,
            "optimizer_iteration":       module.optimizer.iteration,
            "is_trained":                module.is_trained,
            "adam_m":                    adam_m.copy() if adam_m is not None else None,
            "adam_v":                    adam_v.copy() if adam_v is not None else None,
            "adam_t":                    adam_t,
            **bridge_state,
        }
        _save_state(directory, "quantum_ai", state)

        # Capture input feature dim from projection matrix (for future compat checks)
        input_feat_dim = (reducer_pm.shape[0]
                          if reducer_pm is not None else None)

        metadata = {
            "model_name":          "quantum_ai",
            "version":             "5.0",
            "n_qubits":            module.qnn.n_qubits,
            "n_layers":            module.qnn.n_layers,
            "n_params":            module.qnn.n_params,
            "optimizer":           "Adam_SPSA",
            "optimizer_iterations": module.optimizer.iteration,
            "adam_t":              adam_t,
            "is_trained":          module.is_trained,
            "train_accuracy":      getattr(module, "train_accuracy", None),
            "dataset_accuracy":    getattr(module, "dataset_accuracy", None),
            "input_feature_dim":   input_feat_dim,
        }
        _save_metadata(directory, "quantum_ai", metadata)
        logger.info("✅ QuantumAI model saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save QuantumAI: {e}")
        return False


def load_quantum_ai(module) -> bool:
    """Load QuantumAIModule state from checkpoint."""
    try:
        directory = MODEL_SUBDIRS["quantum"]

        # ── Version check: reject checkpoints incompatible with QMLP v5 ──
        meta_early = _load_metadata(directory, "quantum_ai")
        if meta_early is not None:
            ckpt_ver = meta_early.get("version", "0.0")
            ckpt_opt = meta_early.get("optimizer", "")
            if not ckpt_ver.startswith("5"):
                logger.warning(
                    f"⚠️ Quantum checkpoint version '{ckpt_ver}' "
                    f"(optimizer: {ckpt_opt}) is incompatible with QMLP v5 "
                    f"(requires version 5.x / Adam_SPSA). Discarding checkpoint — "
                    f"please retrain the Quantum model."
                )
                return False

        state = _load_state(directory, "quantum_ai")
        if state is None:
            return False

        # Validate theta shape before applying (architecture may have changed)
        loaded_thetas = state["qnn_thetas"]
        if loaded_thetas.shape == module.qnn.thetas.shape:
            module.qnn.thetas = loaded_thetas
        else:
            logger.warning(
                f"QNN theta shape mismatch: checkpoint={loaded_thetas.shape}, "
                f"current={module.qnn.thetas.shape}. Keeping fresh random thetas."
            )

        # Validate projection matrix shape
        pm = state.get("reducer_projection_matrix")
        if pm is not None and pm.shape[-1] == module.reducer.n_qubits:
            module.reducer.projection_matrix = pm
        elif pm is not None:
            logger.warning(
                f"Reducer projection matrix shape mismatch: checkpoint={pm.shape}, "
                f"n_qubits={module.reducer.n_qubits}. Discarding stale projection."
            )

        if state.get("reducer_data_mean") is not None:
            module.reducer._data_mean = state["reducer_data_mean"]
        if state.get("reducer_data_std") is not None:
            module.reducer._data_std  = state["reducer_data_std"]
        # v4: restore training-statistics normalisation
        if state.get("reducer_train_min") is not None:
            module.reducer._train_min = state["reducer_train_min"]
        if state.get("reducer_train_max") is not None:
            module.reducer._train_max = state["reducer_train_max"]

        bridge = module.rnn_bridge   # QuantumGRUBridge via property alias
        if "gru_Wz_ih" in state and hasattr(bridge, "Wz_ih"):
            # GRU bridge (v2)
            bridge.Wz_ih  = state["gru_Wz_ih"];  bridge.Wz_hh = state["gru_Wz_hh"]
            bridge.bz     = state["gru_bz"]
            bridge.Wr_ih  = state["gru_Wr_ih"];  bridge.Wr_hh = state["gru_Wr_hh"]
            bridge.br     = state["gru_br"]
            bridge.Wn_ih  = state["gru_Wn_ih"];  bridge.Wn_hh = state["gru_Wn_hh"]
            bridge.bn     = state["gru_bn"]
            bridge.W_out  = state["gru_W_out"];  bridge.b_out = state["gru_b_out"]
        elif "rnn_W_ih" in state and hasattr(bridge, "W_ih"):
            # Legacy RNN bridge (v1 checkpoint on v1 module — no-op for v2)
            bridge.W_ih   = state["rnn_W_ih"];   bridge.W_hh  = state["rnn_W_hh"]
            bridge.W_out  = state["rnn_W_out"];  bridge.b_h   = state["rnn_b_h"]
            bridge.b_out  = state["rnn_b_out"]

        module.optimizer.iteration = state["optimizer_iteration"]
        module.is_trained = state["is_trained"]

        # v4: restore Adam optimiser state (warm-restart support)
        if state.get("adam_m") is not None and hasattr(module, "adam_m"):
            loaded_m = np.array(state["adam_m"], dtype=np.float64)
            loaded_v = np.array(state["adam_v"], dtype=np.float64)
            if loaded_m.shape == module.adam_m.shape:
                module.adam_m = loaded_m
                module.adam_v = loaded_v
                module.adam_t = int(state.get("adam_t", 0))
            else:
                logger.warning(
                    f"Adam state shape mismatch: checkpoint={loaded_m.shape}, "
                    f"current={module.adam_m.shape}. Starting Adam from scratch."
                )

        meta = _load_metadata(directory, "quantum_ai")
        module.train_accuracy   = meta.get("train_accuracy",   None)
        module.dataset_accuracy = meta.get("dataset_accuracy", None)
        logger.info(
            f"✅ QuantumAI loaded from checkpoint "
            f"(v={meta.get('version','?')}, saved: {meta.get('saved_at', 'unknown')})"
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load QuantumAI: {e}")
        return False


# ═══════════════════════════════════════════════
# Classical RNN Save/Load
# ═══════════════════════════════════════════════

def save_classical_rnn(model) -> bool:
    """Save ClassicalRNNModel (PyTorch) state."""
    try:
        import torch
        directory = MODEL_SUBDIRS["classical_rnn"]
        os.makedirs(directory, exist_ok=True)

        state = {
            "net_state_dict":  model.net.state_dict() if model.net is not None else None,
            "input_dim":       model.input_dim,
            "hidden_size":     model.hidden_size,
            "num_layers":      model.num_layers,
            "dropout":         model.dropout,
            "is_trained":      model.is_trained,
            # Direct-mode extras
            "training_mode":   getattr(model, "training_mode", "embedding"),
            "feature_columns": getattr(model, "feature_columns", []),
            "feature_mean":    getattr(model, "feature_mean", None),
            "feature_std":     getattr(model, "feature_std",  None),
        }
        _save_state(directory, "classical_rnn", state)

        metadata = {
            "model_name":      "classical_rnn",
            "version":         "2.1",
            "backend":         "pytorch",
            "hidden_size":     model.hidden_size,
            "num_layers":      model.num_layers,
            "input_dim":       model.input_dim,
            "epochs":          model.epochs,
            "is_trained":      model.is_trained,
            "training_mode":   getattr(model, "training_mode", "embedding"),
            "n_features":      len(getattr(model, "feature_columns", [])),
        }
        _save_metadata(directory, "classical_rnn", metadata)
        logger.info("✅ Classical RNN model (PyTorch) saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save Classical RNN: {e}")
        return False


def load_classical_rnn(model) -> bool:
    """Load ClassicalRNNModel (PyTorch) state from checkpoint."""
    try:
        import torch
        directory = MODEL_SUBDIRS["classical_rnn"]
        state = _load_state(directory, "classical_rnn")
        if state is None:
            return False

        # Legacy numpy checkpoint — skip rather than crash
        if "layers" in state:
            logger.warning("⚠️  Legacy numpy RNN checkpoint detected — skipping load. "
                           "Retrain to create a PyTorch checkpoint.")
            return False

        input_dim = state.get("input_dim")
        if input_dim is None or state.get("net_state_dict") is None:
            return False

        from agents.classical_ml.rnn_model import _RNNNet
        model._build(input_dim)
        model.net.load_state_dict(state["net_state_dict"])
        model.net.eval()
        model.is_trained = state["is_trained"]

        # Restore direct-mode metadata if present
        model.training_mode   = state.get("training_mode",   "embedding")
        model.feature_columns = state.get("feature_columns", [])
        model.feature_mean    = state.get("feature_mean",    None)
        model.feature_std     = state.get("feature_std",     None)

        meta = _load_metadata(directory, "classical_rnn")
        logger.info(
            f"✅ Classical RNN (PyTorch) loaded from checkpoint "
            f"(mode={model.training_mode}, saved: {meta.get('saved_at', 'unknown')})"
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Classical RNN: {e}")
        return False


# ═══════════════════════════════════════════════
# Classical CNN Save/Load
# ═══════════════════════════════════════════════

def save_classical_cnn(model) -> bool:
    """Save ClassicalCNNModel (PyTorch) state."""
    try:
        import torch
        directory = MODEL_SUBDIRS["classical_cnn"]
        os.makedirs(directory, exist_ok=True)

        state = {
            "net_state_dict":  model.net.state_dict() if model.net is not None else None,
            "input_dim":       model.input_dim,
            "n_conv_layers":   getattr(model, "n_conv_layers",  2),
            "conv1_out":       getattr(model, "conv1_out",     16),
            "conv2_out":       getattr(model, "conv2_out",     32),
            "fc_hidden":       getattr(model, "fc_hidden",     64),
            "kernel_size":     getattr(model, "kernel_size",    3),
            "dropout":         getattr(model, "dropout",       0.0),
            "epochs":          model.epochs,
            "is_trained":      model.is_trained,
            # Direct-mode extras
            "training_mode":   getattr(model, "training_mode", "embedding"),
            "feature_columns": getattr(model, "feature_columns", []),
            "feature_mean":    getattr(model, "feature_mean", None),
            "feature_std":     getattr(model, "feature_std",  None),
            # k-NN correction data
            "train_X":         getattr(model, "train_X", None),
            "train_y":         getattr(model, "train_y", None),
        }
        _save_state(directory, "classical_cnn", state)

        metadata = {
            "model_name":    "classical_cnn",
            "version":       "3.0",
            "backend":       "pytorch",
            "n_conv_layers": getattr(model, "n_conv_layers",  2),
            "conv1_out":     getattr(model, "conv1_out",     16),
            "conv2_out":     getattr(model, "conv2_out",     32),
            "fc_hidden":     getattr(model, "fc_hidden",     64),
            "kernel_size":   getattr(model, "kernel_size",    3),
            "dropout":       getattr(model, "dropout",       0.0),
            "input_dim":     model.input_dim,
            "epochs":        model.epochs,
            "is_trained":    model.is_trained,
            "training_mode": getattr(model, "training_mode", "embedding"),
            "n_features":    len(getattr(model, "feature_columns", [])),
        }
        _save_metadata(directory, "classical_cnn", metadata)
        logger.info("✅ Classical CNN model (PyTorch) saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save Classical CNN: {e}")
        return False


def load_classical_cnn(model) -> bool:
    """Load ClassicalCNNModel (PyTorch) state from checkpoint."""
    try:
        import torch
        directory = MODEL_SUBDIRS["classical_cnn"]
        state = _load_state(directory, "classical_cnn")
        if state is None:
            return False

        # Legacy numpy checkpoint — skip rather than crash
        if "conv_layers" in state:
            logger.warning("⚠️  Legacy numpy CNN checkpoint detected — skipping load. "
                           "Retrain to create a PyTorch checkpoint.")
            return False

        input_dim = state.get("input_dim")
        if input_dim is None or state.get("net_state_dict") is None:
            return False

        from agents.classical_ml.cnn_model import _CNNNet
        model._build(input_dim)
        model.net.load_state_dict(state["net_state_dict"])
        model.net.eval()
        model.is_trained = state["is_trained"]

        # Restore direct-mode metadata if present
        model.training_mode   = state.get("training_mode",   "embedding")
        model.feature_columns = state.get("feature_columns", [])
        model.feature_mean    = state.get("feature_mean",    None)
        model.feature_std     = state.get("feature_std",     None)
        # Restore k-NN correction data
        model.train_X = state.get("train_X", None)
        model.train_y = state.get("train_y", None)
        # Recompute centroids from restored training data
        if hasattr(model, "_compute_centroids"):
            model._compute_centroids()

        meta = _load_metadata(directory, "classical_cnn")
        logger.info(
            f"✅ Classical CNN (PyTorch) loaded from checkpoint "
            f"(mode={model.training_mode}, saved: {meta.get('saved_at', 'unknown')})"
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Classical CNN: {e}")
        return False


# ═══════════════════════════════════════════════
# Cosine Similarity Save/Load
# ═══════════════════════════════════════════════

def save_cosine_similarity(analyzer) -> bool:
    """Save CosineSimilarityAnalyzer state."""
    try:
        directory = MODEL_SUBDIRS["similarity"]
        os.makedirs(directory, exist_ok=True)

        state = {
            "normal_centroids": analyzer.normal_centroids.copy() if analyzer.normal_centroids is not None else None,
            "reference_vectors": [v.copy() for v in analyzer.reference_vectors],
            "reference_labels": list(analyzer.reference_labels),
            "threshold": analyzer.threshold,
            "top_k": analyzer.top_k,
        }
        _save_state(directory, "cosine_similarity", state)

        n_normal = sum(1 for l in analyzer.reference_labels if l == "Normal")
        n_attack = len(analyzer.reference_labels) - n_normal
        metadata = {
            "model_name": "cosine_similarity",
            "version": "1.0",
            "n_reference_vectors": len(analyzer.reference_vectors),
            "n_normal": n_normal,
            "n_attack": n_attack,
            "threshold": analyzer.threshold,
            "top_k": analyzer.top_k,
        }
        _save_metadata(directory, "cosine_similarity", metadata)
        logger.info("✅ Cosine Similarity model saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save Cosine Similarity: {e}")
        return False


def load_cosine_similarity(analyzer) -> bool:
    """Load CosineSimilarityAnalyzer state from checkpoint."""
    try:
        directory = MODEL_SUBDIRS["similarity"]
        state = _load_state(directory, "cosine_similarity")
        if state is None:
            return False

        analyzer.normal_centroids = state["normal_centroids"]
        analyzer.reference_vectors = state["reference_vectors"]
        analyzer.reference_labels = state["reference_labels"]
        analyzer.threshold = state["threshold"]
        analyzer.top_k = state["top_k"]

        meta = _load_metadata(directory, "cosine_similarity")
        logger.info(f"✅ Cosine Similarity loaded from checkpoint (saved: {meta.get('saved_at', 'unknown')})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Cosine Similarity: {e}")
        return False


# ═══════════════════════════════════════════════
# Embedding Engine Save/Load
# ═══════════════════════════════════════════════

def save_embedding_engine(engine) -> bool:
    """Save EmbeddingEngine vocabulary."""
    try:
        directory = MODEL_SUBDIRS["embeddings"]
        os.makedirs(directory, exist_ok=True)

        state = {
            "vocabulary": {k: v.copy() for k, v in engine.vocabulary.items()},
            "dimension": engine.dimension,
            "method": engine.method,
        }
        _save_state(directory, "embedding_engine", state)

        metadata = {
            "model_name": "embedding_engine",
            "version": "1.0",
            "vocabulary_size": len(engine.vocabulary),
            "dimension": engine.dimension,
            "method": engine.method,
        }
        _save_metadata(directory, "embedding_engine", metadata)
        logger.info("✅ Embedding Engine saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save Embedding Engine: {e}")
        return False


def load_embedding_engine(engine) -> bool:
    """Load EmbeddingEngine vocabulary from checkpoint."""
    try:
        directory = MODEL_SUBDIRS["embeddings"]
        state = _load_state(directory, "embedding_engine")
        if state is None:
            return False

        engine.vocabulary = state["vocabulary"]
        engine.dimension = state["dimension"]
        engine.method = state["method"]

        meta = _load_metadata(directory, "embedding_engine")
        logger.info(f"✅ Embedding Engine loaded from checkpoint "
                     f"(vocab={len(engine.vocabulary)}, saved: {meta.get('saved_at', 'unknown')})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Embedding Engine: {e}")
        return False


# ═══════════════════════════════════════════════
# Reference Data Lake Save/Load
# ═══════════════════════════════════════════════

def save_data_lake(data_lake) -> bool:
    """Save ReferenceEmbeddingDataLake state."""
    try:
        directory = MODEL_SUBDIRS["datalake"]
        os.makedirs(directory, exist_ok=True)

        # Serialize embedding vectors
        embeddings_data = []
        for emb in data_lake.embeddings:
            embeddings_data.append({
                "event_id": emb.event_id,
                "vector": emb.vector.copy(),
                "source_type": emb.source_type,
                "timestamp": emb.timestamp,
            })

        state = {
            "embeddings": embeddings_data,
            "labels": list(data_lake.labels),
            "binary_labels": list(data_lake.binary_labels),
        }
        _save_state(directory, "data_lake", state)

        n_attacks = sum(data_lake.binary_labels)
        metadata = {
            "model_name": "data_lake",
            "version": "1.0",
            "total_embeddings": len(data_lake.embeddings),
            "n_attacks": n_attacks,
            "n_normal": len(data_lake.binary_labels) - n_attacks,
            "label_distribution": {},
        }
        from collections import Counter
        label_counts = Counter(data_lake.labels)
        metadata["label_distribution"] = dict(label_counts)
        _save_metadata(directory, "data_lake", metadata)
        logger.info("✅ Data Lake saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save Data Lake: {e}")
        return False


def load_data_lake(data_lake) -> bool:
    """Load ReferenceEmbeddingDataLake state from checkpoint."""
    try:
        directory = MODEL_SUBDIRS["datalake"]
        state = _load_state(directory, "data_lake")
        if state is None:
            return False

        from utils.data_models import EmbeddingVector

        data_lake.embeddings = []
        for ed in state["embeddings"]:
            data_lake.embeddings.append(EmbeddingVector(
                event_id=ed["event_id"],
                vector=ed["vector"],
                source_type=ed["source_type"],
                timestamp=ed["timestamp"],
            ))
        data_lake.labels = state["labels"]
        data_lake.binary_labels = state["binary_labels"]

        meta = _load_metadata(directory, "data_lake")
        logger.info(f"✅ Data Lake loaded from checkpoint "
                     f"({len(data_lake.embeddings)} embeddings, saved: {meta.get('saved_at', 'unknown')})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Data Lake: {e}")
        return False


# ═══════════════════════════════════════════════
# Convenience: Save/Load All
# ═══════════════════════════════════════════════

def save_all_models(quantum_ai, rnn_classical, cnn_classical,
                    cosine_analyzer, embedding_engine, data_lake) -> Dict[str, bool]:
    """Save all model checkpoints."""
    ensure_model_dirs()
    results = {
        "quantum_ai": save_quantum_ai(quantum_ai),
        "classical_rnn": save_classical_rnn(rnn_classical),
        "classical_cnn": save_classical_cnn(cnn_classical),
        "cosine_similarity": save_cosine_similarity(cosine_analyzer),
        "embedding_engine": save_embedding_engine(embedding_engine),
        "data_lake": save_data_lake(data_lake),
    }
    success = sum(v for v in results.values())
    logger.info(f"Save complete: {success}/{len(results)} models saved successfully")
    return results


def load_all_models(quantum_ai, rnn_classical, cnn_classical,
                    cosine_analyzer, embedding_engine, data_lake) -> Dict[str, bool]:
    """Load all model checkpoints."""
    results = {
        "quantum_ai": load_quantum_ai(quantum_ai),
        "classical_rnn": load_classical_rnn(rnn_classical),
        "classical_cnn": load_classical_cnn(cnn_classical),
        "cosine_similarity": load_cosine_similarity(cosine_analyzer),
        "embedding_engine": load_embedding_engine(embedding_engine),
        "data_lake": load_data_lake(data_lake),
    }
    success = sum(v for v in results.values())
    logger.info(f"Load complete: {success}/{len(results)} models loaded from checkpoints")
    return results
