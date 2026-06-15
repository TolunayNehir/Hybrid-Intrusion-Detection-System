"""
Dataset Schema Analyzer
========================
Auto-detects CSV dataset format and extracts feature matrices for CNN/RNN training.
Supports: KDD Cup 99, NSL-KDD, CICIDS 2017/2018, and any generic numeric dataset.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from utils.logger import get_logger

logger = get_logger("DatasetAnalyzer")

# ─── Known format signatures ──────────────────────────────────────────────────

_KDD_COLS = {
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
}

_CICIDS_KEYWORDS = {
    "flow duration", "total fwd packets", "total backward packets",
    "fwd packet length", "bwd packet length", "flow bytes/s",
    "flow packets/s", "flow iat mean", "fwd iat total",
}

# ─── Comprehensive label → binary map ────────────────────────────────────────

LABEL_TO_BINARY: Dict[str, int] = {
    # Benign / normal → 0
    "normal": 0, "benign": 0, "legitimate": 0, "safe": 0, "legit": 0,
    "0": 0, "0.0": 0, "false": 0, "no": 0,
    # Generic attack → 1
    "attack": 1, "anomaly": 1, "intrusion": 1, "malicious": 1, "suspicious": 1,
    "1": 1, "1.0": 1, "true": 1, "yes": 1,
    # KDD Cup 99 / NSL-KDD specific labels
    "dos": 1, "probe": 1, "r2l": 1, "u2r": 1,
    "neptune": 1, "smurf": 1, "back": 1, "teardrop": 1, "pod": 1, "land": 1,
    "portsweep": 1, "ipsweep": 1, "satan": 1, "nmap": 1,
    "warezclient": 1, "warezmaster": 1, "guess_passwd": 1, "ftp_write": 1,
    "imap": 1, "multihop": 1, "phf": 1, "spy": 1, "rootkit": 1,
    "buffer_overflow": 1, "loadmodule": 1, "perl": 1,
    "httptunnel": 1, "named": 1, "sendmail": 1,
    "snmpgetattack": 1, "snmpguess": 1, "worm": 1,
    "xlock": 1, "xsnoop": 1, "xterm": 1,
    # CICIDS 2017 / 2018 labels
    "ddos": 1, "dos hulk": 1, "dos goldeneye": 1, "dos slowloris": 1,
    "dos slowhttptest": 1, "heartbleed": 1,
    "web attack \u2013 brute force": 1, "web attack \u2013 xss": 1,
    "web attack \u2013 sql injection": 1, "infiltration": 1,
    "portscan": 1, "botnet": 1, "ftp-patator": 1, "ssh-patator": 1,
    # Other common labels
    "brute force": 1, "brute_force": 1, "sql injection": 1, "sqli": 1,
    "xss": 1, "port scan": 1, "scanning": 1, "flood": 1,
}


def label_to_binary(val: str) -> int:
    """Convert any label value to 0 (normal) or 1 (attack)."""
    v = str(val).strip().lower()
    if v in LABEL_TO_BINARY:
        return LABEL_TO_BINARY[v]
    try:
        return 1 if float(v) > 0.5 else 0
    except (ValueError, TypeError):
        pass
    for key, bval in LABEL_TO_BINARY.items():
        if bval == 1 and len(key) > 3 and key in v:
            return 1
    if any(w in v for w in ("normal", "benign", "legit", "safe")):
        return 0
    return 0


# ─── Main class ───────────────────────────────────────────────────────────────

class DatasetAnalyzer:
    """
    Analyzes CSV structure, auto-detects format, and extracts a normalized
    feature matrix (X, y) ready for CNN/RNN training.
    """

    def __init__(self):
        self.schema: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std:  Optional[np.ndarray] = None

    # ── Schema detection ──────────────────────────────────────────────────────

    def analyze(self,
                headers: List[str],
                rows: List[Dict],
                label_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Inspect the dataset and return a schema dict.

        Parameters
        ----------
        headers   : column names from the CSV
        rows      : list of row dicts (from csv.DictReader)
        label_col : column name holding the target label (may be None)

        Returns
        -------
        schema dict with keys:
          format, n_rows, n_cols, numeric_cols, categorical_cols,
          label_col, label_distribution, n_numeric_features, col_stats
        """
        n = len(rows)
        probe = rows[:min(200, n)]

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        col_stats: Dict[str, Any] = {}

        for col in headers:
            if col == label_col:
                continue
            sample_vals = [str(r.get(col, "")).strip() for r in probe]
            ok = 0
            for v in sample_vals:
                try:
                    float(v)
                    ok += 1
                except ValueError:
                    pass
            if len(sample_vals) == 0 or ok / len(sample_vals) >= 0.7:
                numeric_cols.append(col)
                nums = []
                for r in rows:
                    try:
                        nums.append(float(str(r.get(col, 0)).strip()))
                    except Exception:
                        nums.append(0.0)
                nums_arr = np.array(nums, dtype=np.float64)
                nums_arr = np.nan_to_num(nums_arr, nan=0.0, posinf=1e9, neginf=-1e9)
                col_stats[col] = {
                    "min":  float(nums_arr.min()),
                    "max":  float(nums_arr.max()),
                    "mean": float(nums_arr.mean()),
                    "std":  float(nums_arr.std()),
                    "unique": int(np.unique(nums_arr).size),
                }
            else:
                categorical_cols.append(col)

        # Remove constant numeric columns (zero variance → useless)
        numeric_cols = [c for c in numeric_cols
                        if col_stats.get(c, {}).get("std", 0) > 1e-8]

        # Detect dataset format
        headers_lower = {h.strip().lower() for h in headers}
        if len(_KDD_COLS & headers_lower) >= 20:
            fmt = "KDD Cup 99 / NSL-KDD"
        elif any(kw in " ".join(headers_lower) for kw in _CICIDS_KEYWORDS):
            fmt = "CICIDS 2017/2018"
        elif len(numeric_cols) >= 10:
            fmt = "Generic Numeric"
        elif len(numeric_cols) >= 3:
            fmt = "Mixed"
        else:
            fmt = "Text-Rich / Categorical"

        # Label distribution
        label_dist: Dict[str, int] = {}
        if label_col:
            for r in rows:
                v = str(r.get(label_col, "")).strip()
                label_dist[v] = label_dist.get(v, 0) + 1

        self.schema = {
            "format":             fmt,
            "n_rows":             n,
            "n_cols":             len(headers),
            "numeric_cols":       numeric_cols,
            "categorical_cols":   categorical_cols,
            "label_col":          label_col,
            "label_distribution": label_dist,
            "n_numeric_features": len(numeric_cols),
            "col_stats":          col_stats,
        }

        logger.info(
            f"Dataset analyzed | format={fmt} | rows={n} | "
            f"numeric={len(numeric_cols)} | categorical={len(categorical_cols)}"
        )
        return self.schema

    # ── Feature extraction ────────────────────────────────────────────────────

    def _encode_categorical(self, col: str, values: List[str]) -> List[float]:
        """Ordinal encoding for a single categorical column."""
        unique = sorted(set(str(v).strip().lower() for v in values))
        denom = max(len(unique) - 1, 1)
        mapping = {v: i / denom for i, v in enumerate(unique)}
        return [mapping.get(str(v).strip().lower(), 0.0) for v in values]

    def build_feature_matrix(
        self,
        rows: List[Dict],
        label_col: Optional[str],
        use_categorical: bool = True,
        max_features: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build a normalized (X, y) pair from the CSV rows.

        Parameters
        ----------
        rows          : list of row dicts
        label_col     : label column name
        use_categorical: include encoded categorical columns
        max_features  : cap on total feature count

        Returns
        -------
        X    : np.ndarray  (N, F)  — z-score normalized, clipped to [-5, 5]
        y    : np.ndarray  (N,)    — binary labels
        info : dict                — feature metadata (for saving with model)
        """
        if not self.schema:
            raise RuntimeError("Call analyze() first")

        numeric_cols    = self.schema["numeric_cols"]
        categorical_cols = self.schema["categorical_cols"] if use_categorical else []
        n = len(rows)

        # How many numeric / categorical columns to use
        n_num = min(len(numeric_cols), max_features)
        n_cat = min(len(categorical_cols), max(0, max_features - n_num) // 2)

        sel_num = numeric_cols[:n_num]
        sel_cat = categorical_cols[:n_cat]

        parts: List[np.ndarray] = []
        names: List[str] = []

        # Numeric block
        if sel_num:
            X_num = np.zeros((n, len(sel_num)), dtype=np.float32)
            for j, col in enumerate(sel_num):
                for i, row in enumerate(rows):
                    try:
                        X_num[i, j] = float(str(row.get(col, 0)).strip())
                    except Exception:
                        X_num[i, j] = 0.0
            X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)
            parts.append(X_num)
            names.extend(sel_num)

        # Categorical block
        for col in sel_cat:
            vals = [str(row.get(col, "")).strip() for row in rows]
            enc  = self._encode_categorical(col, vals)
            parts.append(np.array(enc, dtype=np.float32).reshape(-1, 1))
            names.append(col)

        if not parts:
            logger.warning("No features extracted — using zero placeholder")
            X = np.zeros((n, 1), dtype=np.float32)
            names = ["placeholder"]
        else:
            X = np.hstack(parts)

        # Z-score normalization per column
        self.feature_mean = X.mean(axis=0)
        self.feature_std  = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        X = (X - self.feature_mean) / self.feature_std
        X = np.clip(X, -5.0, 5.0).astype(np.float32)

        self.feature_columns = names

        # Labels
        if label_col:
            y = np.array(
                [label_to_binary(r.get(label_col, "0")) for r in rows],
                dtype=np.float32,
            )
        else:
            y = np.zeros(n, dtype=np.float32)

        n_attack = int(y.sum())
        n_normal = n - n_attack

        info = {
            "n_samples":       n,
            "n_features":      X.shape[1],
            "feature_columns": names,
            "n_numeric":       len(sel_num),
            "n_categorical":   len(sel_cat),
            "n_attack":        n_attack,
            "n_normal":        n_normal,
            "feature_mean":    self.feature_mean.tolist(),
            "feature_std":     self.feature_std.tolist(),
            "dataset_format":  self.schema.get("format", "unknown"),
        }

        logger.info(
            f"Feature matrix: shape={X.shape} | "
            f"attacks={n_attack} | normal={n_normal}"
        )
        return X, y, info

    # ── Inference transform ───────────────────────────────────────────────────

    def transform(self, rows: List[Dict]) -> np.ndarray:
        """
        Apply fitted scaler to new rows for inference.
        Uses only columns from self.feature_columns.
        """
        if not self.feature_columns:
            raise RuntimeError("Call build_feature_matrix() first")
        n = len(rows)
        X = np.zeros((n, len(self.feature_columns)), dtype=np.float32)
        for j, col in enumerate(self.feature_columns):
            for i, row in enumerate(rows):
                try:
                    X[i, j] = float(str(row.get(col, 0)).strip())
                except Exception:
                    X[i, j] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X = (X - self.feature_mean) / self.feature_std
        X = np.clip(X, -5.0, 5.0).astype(np.float32)
        return X

    # ── Helper: check column overlap ─────────────────────────────────────────

    def columns_overlap(self, available_headers: List[str]) -> float:
        """
        Return the fraction of training feature_columns present in available_headers.
        Used to decide whether to use direct-mode inference.
        """
        if not self.feature_columns:
            return 0.0
        avail = {h.strip().lower() for h in available_headers}
        matched = sum(1 for c in self.feature_columns if c.strip().lower() in avail)
        return matched / len(self.feature_columns)
