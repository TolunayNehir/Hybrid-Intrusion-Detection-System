"""
Single source of truth for the cosine/k-NN + centroid anomaly score.

"""
import numpy as np


def cosine_knn_scores(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray,
                       k: int = 5, anomaly_bump: float = 0.2) -> np.ndarray:
    """Batch cosine/k-NN + centroid anomaly score, exactly as used for the
    locked-test evaluation. Returns one score per row of X_query."""
    normal_mask = y_train < 0.5
    c_normal = X_train[normal_mask].mean(axis=0)
    c_normal = c_normal / (np.linalg.norm(c_normal) + 1e-8)

    q_norm = np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8
    q_unit = X_query / q_norm
    sims_to_normal_centroid = q_unit @ c_normal
    base_anomaly = 1.0 - np.clip(sims_to_normal_centroid, -1.0, 1.0)
    base_anomaly = (base_anomaly + 1.0) / 2.0  # map [0,2] cosine-distance-like range to [0,1]

    train_norm = np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8
    train_unit = X_train / train_norm
    sims = q_unit @ train_unit.T
    k_eff = min(k, sims.shape[1])
    top_k = np.argsort(sims, axis=1)[:, -k_eff:]
    attack_votes = np.array([(y_train[idx] > 0.5).sum() for idx in top_k])
    normal_votes = k_eff - attack_votes
    bump = np.where(attack_votes > normal_votes, anomaly_bump, 0.0)
    score = np.clip(base_anomaly + bump, 0.0, 1.0)
    return score


def cosine_knn_score_single(X_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray,
                             k: int = 5, anomaly_bump: float = 0.2) -> float:
    """Single-event convenience wrapper (used by the latency benchmark and the
    SSH case study) around the identical batch rule above -- never a
    simplified stand-in for it."""
    return float(cosine_knn_scores(X_train, y_train, x_query.reshape(1, -1), k=k,
                                    anomaly_bump=anomaly_bump)[0])
