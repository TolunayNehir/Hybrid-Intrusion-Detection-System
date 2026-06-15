"""
Cosine Similarity Analysis Agent
==================================
Part of the Embedding path in the General System Diagram:
  Embedding Vector Warehouse → Cosine Similarity Analysis → Similarity Score / Anomaly Score → Decision Fusion Layer

Compares event embeddings against stored vectors to detect anomalies.
"""

import numpy as np
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_models import EmbeddingVector, PreprocessedData, ClassificationResult, SimilarityResult
from utils.logger import get_logger
from config.settings import SIMILARITY

logger = get_logger("CosineSimilarity")


class CosineSimilarityAnalyzer:
    """
    Cosine Similarity Analysis module.
    Compares new event embeddings against known normal patterns 
    in the Embedding Vector Warehouse to compute anomaly scores.
    """

    def __init__(self):
        self.normal_centroids: np.ndarray = None
        self.threshold = SIMILARITY["anomaly_threshold"]
        self.top_k = SIMILARITY["top_k_matches"]
        self.reference_vectors: List[np.ndarray] = []
        self.reference_labels: List[str] = []
        logger.info(f"CosineSimilarityAnalyzer initialized: threshold={self.threshold}, top_k={self.top_k}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def build_reference(self, normal_embeddings: List[EmbeddingVector],
                        attack_embeddings: List[EmbeddingVector] = None):
        """Build reference from known normal patterns."""
        if normal_embeddings:
            normal_vecs = np.array([e.vector for e in normal_embeddings])
            self.normal_centroids = np.mean(normal_vecs, axis=0)
            for e in normal_embeddings:
                self.reference_vectors.append(e.vector)
                self.reference_labels.append("Normal")

        if attack_embeddings:
            for e in attack_embeddings:
                self.reference_vectors.append(e.vector)
                self.reference_labels.append("Attack")

        logger.info(f"Reference built: {len(self.reference_vectors)} vectors")

    def analyze(self, embedding: EmbeddingVector, preprocessed: PreprocessedData) -> ClassificationResult:
        """
        Perform cosine similarity analysis.
        High similarity to normal = low anomaly score.
        Low similarity to normal = high anomaly score (potential intrusion).
        """
        if self.normal_centroids is None or len(self.reference_vectors) == 0:
            return ClassificationResult(
                agent_name="CosineSimilarity",
                event_id=preprocessed.event_id,
                prediction=0,
                confidence=0.5,
                label="Normal",
                details={"status": "no_reference"}
            )

        # Compute similarity to normal centroid
        sim_to_normal = self._cosine_similarity(embedding.vector, self.normal_centroids)

        # Compute similarities to all reference vectors
        similarities = []
        for ref_vec, ref_label in zip(self.reference_vectors, self.reference_labels):
            sim = self._cosine_similarity(embedding.vector, ref_vec)
            similarities.append({"similarity": sim, "label": ref_label})

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = similarities[:self.top_k]

        # Anomaly score: inverse of similarity to normal
        anomaly_score = 1.0 - max(0.0, min(1.0, sim_to_normal))

        # Check top-k voting
        attack_votes = sum(1 for m in top_matches if m["label"] != "Normal")
        normal_votes = len(top_matches) - attack_votes

        # Combine anomaly score with voting
        if attack_votes > normal_votes:
            anomaly_score = min(1.0, anomaly_score + 0.2)

        is_intrusion = anomaly_score > self.threshold
        prediction = 1 if is_intrusion else 0
        confidence = anomaly_score if is_intrusion else (1 - anomaly_score)
        label = preprocessed.metadata.get("attack_type", "Anomaly") if is_intrusion else "Normal"

        return ClassificationResult(
            agent_name="CosineSimilarity",
            event_id=preprocessed.event_id,
            prediction=prediction,
            confidence=float(confidence),
            label=label,
            details={
                "similarity_to_normal": float(sim_to_normal),
                "anomaly_score": float(anomaly_score),
                "top_k_matches": top_matches,
                "attack_votes": attack_votes,
                "normal_votes": normal_votes,
            }
        )

    def save_model(self) -> bool:
        """Save Cosine Similarity model checkpoint."""
        from utils.model_persistence import save_cosine_similarity, ensure_model_dirs
        ensure_model_dirs()
        return save_cosine_similarity(self)

    def load_model(self) -> bool:
        """Load Cosine Similarity model from checkpoint."""
        from utils.model_persistence import load_cosine_similarity
        return load_cosine_similarity(self)

    def analyze_batch(self, embeddings: List[EmbeddingVector],
                      preprocessed_batch: List[PreprocessedData]) -> List[ClassificationResult]:
        results = [self.analyze(e, p) for e, p in zip(embeddings, preprocessed_batch)]
        logger.info(f"CosineSimilarity analyzed {len(results)} events: "
                     f"{sum(r.prediction for r in results)} anomalies")
        return results
