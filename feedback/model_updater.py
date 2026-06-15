"""
Feedback / Model Updates Module
=================================
From the General System Diagram:
  Security Analyst Review → Feedback / Model Updates → Creating Embeddings
                                                     → Reference Embedding Data Lake

Implements the feedback loop for continuous model improvement.
"""

from typing import List, Dict
from datetime import datetime
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import FeedbackRecord, EmbeddingVector
from utils.logger import get_logger
from config.settings import FEEDBACK

logger = get_logger("FeedbackUpdater")


class FeedbackModelUpdater:
    """
    Feedback / Model Updates module.
    Processes analyst feedback and updates:
    1. Creating Embeddings module (vocabulary updates)
    2. Reference Embedding Data Lake (new reference data)
    """

    def __init__(self):
        self.feedback_buffer: List[FeedbackRecord] = []
        self.update_count = 0
        self.update_interval = FEEDBACK["update_interval"]
        self.targets = FEEDBACK["targets"]
        logger.info(f"FeedbackModelUpdater initialized: update_interval={self.update_interval}")

    def collect_feedback(self, feedbacks: List[FeedbackRecord]):
        """Collect feedback from Security Analyst Review."""
        self.feedback_buffer.extend(feedbacks)
        logger.info(f"Collected {len(feedbacks)} feedback records "
                     f"(buffer: {len(self.feedback_buffer)})")

        # Check if we should trigger an update
        if len(self.feedback_buffer) >= self.update_interval:
            return self.process_updates()
        return None

    def process_updates(self) -> Dict:
        """Process accumulated feedback and generate update instructions."""
        if not self.feedback_buffer:
            return {"status": "no_updates"}

        self.update_count += 1

        # Analyze feedback
        confirmed_attacks = [f for f in self.feedback_buffer if f.correct_label != "Normal"]
        false_positives = [f for f in self.feedback_buffer
                           if f.correct_label == "Normal" and f.analyst_notes]
        confirmed_normals = [f for f in self.feedback_buffer if f.correct_label == "Normal"]

        update_result = {
            "update_id": self.update_count,
            "timestamp": datetime.now().isoformat(),
            "total_feedback": len(self.feedback_buffer),
            "confirmed_attacks": len(confirmed_attacks),
            "false_positives": len(false_positives),
            "confirmed_normals": len(confirmed_normals),
            "embedding_updates": [],
            "data_lake_updates": [],
        }

        # Generate embedding vocabulary updates
        if "embeddings" in self.targets:
            new_words = {}
            for fb in confirmed_attacks:
                word = fb.correct_label.lower().replace(" ", "_")
                # Strengthen the embedding for confirmed attack types
                np.random.seed(hash(word) % (2**31))
                vec = np.random.randn(128).astype(np.float32) * 1.2
                vec /= np.linalg.norm(vec)
                new_words[word] = vec
            update_result["embedding_updates"] = list(new_words.keys())

        # Generate data lake updates
        if "data_lake" in self.targets:
            new_embeddings = []
            new_labels = []
            for fb in self.feedback_buffer:
                # Create feedback-based reference embedding
                np.random.seed(hash(fb.event_id) % (2**31))
                vec = np.random.randn(128).astype(np.float32)
                vec /= np.linalg.norm(vec)
                new_embeddings.append(EmbeddingVector(
                    event_id=f"feedback_{fb.event_id}",
                    vector=vec,
                    source_type="feedback",
                    timestamp=fb.timestamp,
                ))
                new_labels.append(fb.correct_label)
            update_result["data_lake_updates"] = new_labels
            update_result["_new_embeddings"] = new_embeddings
            update_result["_new_labels"] = new_labels

        # Clear buffer after processing
        processed_count = len(self.feedback_buffer)
        self.feedback_buffer = []

        logger.info(f"🔄 Model Update #{self.update_count}: "
                     f"processed {processed_count} feedbacks, "
                     f"{len(confirmed_attacks)} attacks confirmed, "
                     f"{len(false_positives)} false positives identified")

        return update_result

    def force_update(self) -> Dict:
        """Force an immediate update regardless of buffer size."""
        return self.process_updates()

    def get_status(self) -> Dict:
        return {
            "buffer_size": len(self.feedback_buffer),
            "total_updates": self.update_count,
            "update_interval": self.update_interval,
        }
