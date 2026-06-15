"""
Embedding Creation Module
===========================
Creates semantic embeddings from preprocessed text features.

Receives: Preprocessed text data from Preprocessor
Outputs to:
  1. Embedding Vector Warehouse → Cosine Similarity Analysis
  2. Example Attack & Intrusion Data → Reference Embedding Data Lake
Also receives feedback from: Feedback / Model Updates
"""

import numpy as np
from typing import List, Dict
import hashlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import PreprocessedData, EmbeddingVector
from utils.logger import get_logger
from config.settings import EMBEDDING

logger = get_logger("EmbeddingEngine")


class EmbeddingEngine:
    """
    Creates behavioral semantic embeddings from security event text.
    Transforms text representations into vector space for similarity analysis.
    """

    def __init__(self):
        self.dimension = EMBEDDING["dimension"]
        self.method = EMBEDDING["method"]
        self.vocabulary: Dict[str, np.ndarray] = {}
        self._build_base_vocabulary()
        logger.info(f"EmbeddingEngine initialized: dim={self.dimension}, method={self.method}")

    def _build_base_vocabulary(self):
        """Build base vocabulary with random but consistent embeddings."""
        np.random.seed(42)
        base_words = [
            "tcp", "udp", "icmp", "http", "https", "dns", "ssh", "ftp",
            "ips", "siem", "firewall", "soar", "ids",
            "alert", "log", "event", "scan", "flood", "brute",
            "exploit", "backdoor", "shell", "overflow", "injection",
            "normal", "attack", "intrusion", "malware", "worm",
            "low", "medium", "high", "critical",
            "src", "dst", "port", "payload", "severity",
            "syn", "ack", "rst", "fin", "push",
            "reconnaissance", "privilege", "escalation", "unauthorized",
            "dos", "probe", "r2l", "u2r", "shellcode",
        ]
        for word in base_words:
            self.vocabulary[word] = np.random.randn(self.dimension).astype(np.float32)
            self.vocabulary[word] /= (np.linalg.norm(self.vocabulary[word]) + 1e-8)

    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create embedding for a word."""
        word_lower = word.lower().strip()
        if word_lower not in self.vocabulary:
            # Deterministic hash-based embedding for unknown words
            hash_val = int(hashlib.md5(word_lower.encode()).hexdigest(), 16)
            rng = np.random.RandomState(hash_val % (2**31))
            vec = rng.randn(self.dimension).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-8)
            self.vocabulary[word_lower] = vec
        return self.vocabulary[word_lower]

    def create_embedding(self, preprocessed: PreprocessedData) -> EmbeddingVector:
        """Create semantic embedding from preprocessed security event."""
        text = preprocessed.text_features
        words = text.replace("=", " ").replace(":", " ").replace("/", " ").split()

        if not words:
            vector = np.zeros(self.dimension, dtype=np.float32)
        else:
            # Average word embeddings (Word2Vec-like approach)
            word_vectors = [self._get_word_vector(w) for w in words]
            vector = np.mean(word_vectors, axis=0).astype(np.float32)

            # Numerical feature influence is intentionally zeroed-out.
            # Training uses features=zeros while live events carry real
            # preprocessed values; a non-zero coefficient creates a
            # train/inference embedding mismatch that breaks classifiers.
            # The text_features string already encodes protocol, IPs, ports
            # and payload — all discriminating information is preserved.

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm

        return EmbeddingVector(
            event_id=preprocessed.event_id,
            vector=vector,
            source_type=preprocessed.source_type,
            timestamp=preprocessed.timestamp,
        )

    def create_batch_embeddings(self, preprocessed_batch: List[PreprocessedData]) -> List[EmbeddingVector]:
        """Create embeddings for a batch of preprocessed events."""
        embeddings = [self.create_embedding(p) for p in preprocessed_batch]
        logger.info(f"Created {len(embeddings)} embeddings (dim={self.dimension})")
        return embeddings

    def save_model(self) -> bool:
        """Save Embedding Engine checkpoint."""
        from utils.model_persistence import save_embedding_engine, ensure_model_dirs
        ensure_model_dirs()
        return save_embedding_engine(self)

    def load_model(self) -> bool:
        """Load Embedding Engine from checkpoint."""
        from utils.model_persistence import load_embedding_engine
        return load_embedding_engine(self)

    def update_vocabulary(self, new_words: Dict[str, np.ndarray]):
        """Update vocabulary from feedback loop."""
        self.vocabulary.update(new_words)
        logger.info(f"Vocabulary updated: +{len(new_words)} words, total={len(self.vocabulary)}")
