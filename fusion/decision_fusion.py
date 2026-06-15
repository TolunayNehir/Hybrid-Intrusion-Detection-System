"""
Decision Fusion Layer
======================
Central convergence point for all 5 analysis agents.

Inputs from:
  1. QuantumAI → Classification Outputs
  2. CosineSimilarity → Similarity Score / Anomaly Score
  3. RNN_Classical → RNN Classification Output
  4. CNN_Classical → CNN Classification Output
  5. LLM_Analysis → LLM Analysis Output

Output to:
  → Intrusion / Violation Detection
"""

import numpy as np
from typing import List, Dict
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import ClassificationResult, FusionResult
from utils.logger import get_logger
from config.settings import FUSION

logger = get_logger("DecisionFusion")


class DecisionFusionLayer:
    """
    Decision Fusion Layer - combines outputs from all 5 agent modules
    into a unified intrusion detection decision.
    
    Supports:
    - Weighted Average fusion
    - Majority Voting fusion
    - Stacking (meta-learner) fusion
    """

    def __init__(self):
        self.method = FUSION["method"]
        self.weights = FUSION["weights"]
        self.threshold = FUSION["intrusion_threshold"]
        self.fusion_history: List[FusionResult] = []

        # Agent name mapping
        self.agent_weight_map = {
            "QuantumAI": "quantum_ai",
            "CosineSimilarity": "cosine_similarity",
            "RNN_Classical": "rnn_classical",
            "CNN_Classical": "cnn_classical",
            "LLM_Analysis": "llm_analysis",
        }
        logger.info(f"DecisionFusionLayer initialized: method={self.method}, threshold={self.threshold}")

    def _weighted_average(self, agent_results: Dict[str, ClassificationResult]) -> float:
        """Weighted average fusion of agent confidence scores."""
        total_score = 0.0
        total_weight = 0.0

        for agent_name, result in agent_results.items():
            weight_key = self.agent_weight_map.get(agent_name, agent_name)
            weight = self.weights.get(weight_key, 0.2)

            # Score: confidence * prediction direction
            score = result.confidence if result.prediction == 1 else (1 - result.confidence)
            total_score += weight * score
            total_weight += weight

        return total_score / (total_weight + 1e-8)

    def _majority_voting(self, agent_results: Dict[str, ClassificationResult]) -> float:
        """Majority voting fusion."""
        votes_attack = sum(1 for r in agent_results.values() if r.prediction == 1)
        votes_normal = len(agent_results) - votes_attack

        if votes_attack > votes_normal:
            avg_confidence = np.mean([r.confidence for r in agent_results.values() if r.prediction == 1])
            return float(avg_confidence)
        else:
            avg_confidence = np.mean([r.confidence for r in agent_results.values() if r.prediction == 0])
            return 1.0 - float(avg_confidence)

    def _stacking(self, agent_results: Dict[str, ClassificationResult]) -> float:
        """Simple stacking meta-learner fusion."""
        features = []
        for agent_name in ["QuantumAI", "CosineSimilarity", "RNN_Classical", "CNN_Classical", "LLM_Analysis"]:
            if agent_name in agent_results:
                r = agent_results[agent_name]
                features.extend([
                    float(r.prediction),
                    float(r.confidence),
                    float(r.confidence if r.prediction == 1 else 1 - r.confidence)
                ])
            else:
                features.extend([0.0, 0.5, 0.5])

        # Simple logistic combination
        feature_vec = np.array(features)
        # Meta-weights (pre-defined for simplicity)
        np.random.seed(99)
        meta_weights = np.random.randn(len(features)) * 0.1
        meta_bias = -0.5
        logit = np.dot(meta_weights, feature_vec) + meta_bias
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10))))

    def fuse(self, event_id: str, agent_results: Dict[str, ClassificationResult]) -> FusionResult:
        """
        Fuse all agent results into a single decision.
        
        Args:
            event_id: The event being classified
            agent_results: Dict mapping agent_name → ClassificationResult
        
        Returns:
            FusionResult with final intrusion decision
        """
        # Select fusion method
        if self.method == "weighted_average":
            final_score = self._weighted_average(agent_results)
        elif self.method == "voting":
            final_score = self._majority_voting(agent_results)
        elif self.method == "stacking":
            final_score = self._stacking(agent_results)
        else:
            final_score = self._weighted_average(agent_results)

        is_intrusion = final_score > self.threshold

        # Collect individual agent scores
        agent_scores = {}
        for name, result in agent_results.items():
            score = result.confidence if result.prediction == 1 else (1 - result.confidence)
            agent_scores[name] = float(score)

        fusion_result = FusionResult(
            event_id=event_id,
            final_score=float(final_score),
            is_intrusion=is_intrusion,
            agent_scores=agent_scores,
            fusion_method=self.method,
            timestamp=datetime.now(),
        )

        self.fusion_history.append(fusion_result)
        return fusion_result

    def fuse_batch(self, batch_results: List[Dict[str, ClassificationResult]],
                   event_ids: List[str]) -> List[FusionResult]:
        """Fuse results for a batch of events."""
        fusion_results = [self.fuse(eid, agents) for eid, agents in zip(event_ids, batch_results)]

        intrusion_count = sum(1 for fr in fusion_results if fr.is_intrusion)
        logger.info(f"Fusion completed for {len(fusion_results)} events: "
                     f"{intrusion_count} intrusions, {len(fusion_results) - intrusion_count} normal "
                     f"(method={self.method})")
        return fusion_results

    def update_weights(self, new_weights: Dict[str, float]):
        """Update fusion weights based on feedback."""
        self.weights.update(new_weights)
        logger.info(f"Fusion weights updated: {self.weights}")
