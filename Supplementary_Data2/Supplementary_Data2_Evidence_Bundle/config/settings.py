"""
Hybrid Intrusion Detection System - Configuration Settings
Based on: General System Diagram (Fig. 1) from the academic paper
"""

import os

# ─── Project Paths ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Data Source Settings ───
DATA_SOURCES = {
    "ips_alerts":    {"type": "IPS",      "priority": "high"},
    "siem_events":   {"type": "SIEM",     "priority": "high"},
    "firewall_logs": {"type": "Firewall", "priority": "medium"},
    "soar_events":   {"type": "SOAR",     "priority": "medium"},
    "ids_alerts":    {"type": "IDS",      "priority": "high"},
}

# ─── Preprocessing Settings ───
PREPROCESSING = {
    "max_features": 64,
    "normalization": "minmax",
    "handle_missing": "mean",
    "feature_extraction_method": "statistical",
}

# ─── Embedding Settings ───
EMBEDDING = {
    "dimension": 128,
    "method": "word2vec",
    "batch_size": 32,
}

# ─── Quantum AI Module Settings ───
# v4: Adam + parameter shift rule, dual Ry+Rz encoding, ring+cross entanglement
# n_params = 2 × n_qubits × n_layers = 48  (doubled vs v3)
QUANTUM_AI = {
    # Circuit geometry
    "n_qubits":        8,
    "n_layers":        4,          # hidden quantum layers (QMLP depth)
    "n_hidden_layers": 4,          # alias used by QiskitQMLP
    # Optimizer
    "optimizer":       "Adam_SPSA",
    "max_iterations":  300,
    "learning_rate":   0.01,
    "adam_beta1":      0.9,
    "adam_beta2":      0.999,
    "perturbation":    0.25,
    "threshold":       0.5,
}

# ─── Classical ML Settings ───
# CNN: Traditional 2-conv architecture (1→16→32 → Flatten → 1024→64→1)
# RNN: unchanged
CLASSICAL_ML = {
    "rnn": {
        "hidden_size":    128,
        "num_layers":     2,
        "dropout":        0.25,
        "epochs":         40,
        "learning_rate":  0.001,
    },
    "cnn": {
        "epochs":         40,
        "learning_rate":  0.001,
        "n_conv_layers":  2,
        "conv1_out":      16,
        "conv2_out":      32,
        "fc_hidden":      64,
        "kernel_size":    3,
        "dropout":        0.0,
    },
}

# ─── Architecture Presets ───
# All previous & current architectures available as named presets.
ARCH_PRESETS = {
    "cnn": {
        "traditional_v5": {
            "label":          "Traditional CNN v5 (Mevcut)",
            "is_current":     True,
            "n_conv_layers":  2,
            "conv1_out":      16,
            "conv2_out":      32,
            "fc_hidden":      64,
            "kernel_size":    3,
            "dropout":        0.0,
            "epochs":         40,
            "learning_rate":  0.001,
            "desc": "2 Conv katmanı [1→16→32], FC 64, kernel 3 — aktif mimari",
        },
        "deep_v4": {
            "label":          "Deep CNN v4 (Önceki)",
            "n_conv_layers":  3,
            "conv1_out":      32,
            "conv2_out":      64,
            "fc_hidden":      128,
            "kernel_size":    3,
            "dropout":        0.3,
            "epochs":         40,
            "learning_rate":  0.001,
            "desc": "3 Conv katmanı [1→32→64→128], FC 128, dropout 0.3 — önceki derin mimari",
        },
        "lightweight": {
            "label":          "Lightweight CNN",
            "n_conv_layers":  1,
            "conv1_out":      8,
            "conv2_out":      16,
            "fc_hidden":      32,
            "kernel_size":    3,
            "dropout":        0.0,
            "epochs":         30,
            "learning_rate":  0.001,
            "desc": "1 Conv katmanı [1→8], FC 32 — en hızlı eğitim",
        },
        "wide_kernel": {
            "label":          "Wide-Kernel CNN",
            "n_conv_layers":  2,
            "conv1_out":      32,
            "conv2_out":      64,
            "fc_hidden":      256,
            "kernel_size":    5,
            "dropout":        0.2,
            "epochs":         50,
            "learning_rate":  0.0005,
            "desc": "2 Conv katmanı, geniş alıcı alan (kernel=5), büyük FC 256",
        },
        "very_deep": {
            "label":          "Very Deep CNN",
            "n_conv_layers":  4,
            "conv1_out":      16,
            "conv2_out":      32,
            "fc_hidden":      128,
            "kernel_size":    3,
            "dropout":        0.3,
            "epochs":         60,
            "learning_rate":  0.0005,
            "desc": "4 Conv katmanı [1→16→32→64→128], FC 128 — en derin mimari",
        },
    },
    "rnn": {
        "compact_v5": {
            "label":         "Compact RNN v5 (Mevcut)",
            "is_current":    True,
            "hidden_size":   128,
            "num_layers":    2,
            "dropout":       0.25,
            "epochs":        40,
            "learning_rate": 0.001,
            "desc": "2 katmanlı Elman RNN, hidden=128 — aktif mimari",
        },
        "deep_rnn": {
            "label":         "Deep RNN",
            "hidden_size":   256,
            "num_layers":    3,
            "dropout":       0.3,
            "epochs":        50,
            "learning_rate": 0.0005,
            "desc": "3 katmanlı daha derin RNN, hidden=256 — karmaşık diziler için",
        },
        "minimal_rnn": {
            "label":         "Minimal RNN",
            "hidden_size":   64,
            "num_layers":    1,
            "dropout":       0.0,
            "epochs":        30,
            "learning_rate": 0.001,
            "desc": "Tek katman, hidden=64 — en hızlı eğitim",
        },
        "large_rnn": {
            "label":         "Large RNN",
            "hidden_size":   512,
            "num_layers":    2,
            "dropout":       0.4,
            "epochs":        40,
            "learning_rate": 0.0005,
            "desc": "Büyük gizli boyut (512), güçlü düzenlileştirme",
        },
    },
    "quantum": {
        "qmlp_v5": {
            "label":           "QMLP v5 (Mevcut)",
            "is_current":      True,
            "n_qubits":        8,
            "n_hidden_layers": 4,
            "max_iterations":  300,
            "learning_rate":   0.01,
            "perturbation":    0.25,
            "desc": "8 qubit, 4 gizli katman, Adam-SPSA — aktif mimari (73 param)",
        },
        "vqc_v4": {
            "label":           "VQC v4 (Önceki)",
            "n_qubits":        4,
            "n_hidden_layers": 3,
            "max_iterations":  200,
            "learning_rate":   0.01,
            "perturbation":    0.25,
            "desc": "Önceki VQC stili: 4 qubit, 3 katman, veri yeniden yükleme (48 param)",
        },
        "deep_qmlp": {
            "label":           "Deep QMLP",
            "n_qubits":        8,
            "n_hidden_layers": 6,
            "max_iterations":  400,
            "learning_rate":   0.005,
            "perturbation":    0.2,
            "desc": "8 qubit, 6 gizli katman, daha fazla SPSA adımı (97 param)",
        },
        "minimal_qmlp": {
            "label":           "Minimal QMLP",
            "n_qubits":        4,
            "n_hidden_layers": 2,
            "max_iterations":  150,
            "learning_rate":   0.02,
            "perturbation":    0.3,
            "desc": "4 qubit, 2 katman — en hızlı kuantum eğitimi (21 param)",
        },
    },
}

# ─── Cosine Similarity Settings ───
SIMILARITY = {
    "anomaly_threshold": 0.7,
    "top_k_matches":     5,
}

# ─── LLM Analysis Settings ───
LLM_ANALYSIS = {
    "model_type":           "rule_based",
    "context_window":       512,
    "confidence_threshold": 0.6,
}

# ─── Decision Fusion Settings ───
FUSION = {
    "method": "weighted_average",
    "weights": {
        "quantum_ai":        0.20,
        "cosine_similarity": 0.20,
        "rnn_classical":     0.20,
        "cnn_classical":     0.20,
        "llm_analysis":      0.20,
    },
    "intrusion_threshold": 0.55,
}

# ─── Dashboard & Alerts Settings ───
DASHBOARD = {
    "refresh_interval":   5,
    "max_alerts_display": 100,
    "severity_levels":    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
}

# ─── Feedback Loop Settings ───
FEEDBACK = {
    "update_interval":   100,
    "retrain_threshold": 0.1,
    "targets":           ["embeddings", "data_lake"],
}

# ─── Model Persistence Settings ───
MODEL_PERSISTENCE = {
    "enabled":               True,
    "auto_save_after_training": True,
    "auto_load_on_startup":  True,
    "checkpoint_dir":        MODEL_DIR,
}

# ─── Logging ───
LOGGING = {
    "level":  "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
}
