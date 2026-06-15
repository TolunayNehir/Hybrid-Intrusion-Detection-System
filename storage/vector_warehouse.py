"""
Storage Modules
================
1. Embedding Vector Warehouse - stores real-time embedding vectors
2. Example Attack & Intrusion Data - known attack patterns
3. Reference Embedding Data Lake - reference embeddings for classical ML models

Data flow:
  Creating Embeddings → Embedding Vector Warehouse → Cosine Similarity Analysis
  Creating Embeddings → Example Attack & Intrusion Data → Reference Embedding Data Lake
  Reference Embedding Data Lake → RNN Model (Classical)
  Reference Embedding Data Lake → CNN Model (Classical)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import EmbeddingVector, RawSecurityEvent
from utils.logger import get_logger

logger = get_logger("Storage")


# ──────────────────────────────────────────────────────────────────────
# Synthetic event templates — gives RNN/Quantum learnable structure
# ──────────────────────────────────────────────────────────────────────
# Each attack family has distinctive structured features (ports, payload
# length, severity, keywords, source-type). The Preprocessor turns these
# into a 28-dim real feature vector; we then tile it into the first half
# of the 128-D embedding so RNN can see a coherent sequence and Quantum's
# PCA picks up genuine class-separating directions. The second half holds
# a per-class deterministic signature so cosine-similarity clustering
# (attack-type identity) is preserved.

_ATTACK_TEMPLATES = {
    "DoS": dict(dst_ports=[80, 443, 8080], protocol="TCP", severity="HIGH",
                payload_kw="flood ", payload_len=(800, 1500), src_type="Firewall"),
    "Probe": dict(dst_ports=list(range(20, 60)), protocol="TCP", severity="LOW",
                  payload_kw="scan ", payload_len=(20, 80), src_type="IDS"),
    "R2L": dict(dst_ports=[22, 23, 21], protocol="SSH", severity="HIGH",
                payload_kw="brute ", payload_len=(60, 200), src_type="IPS"),
    "U2R": dict(dst_ports=[22, 3389], protocol="SSH", severity="CRITICAL",
                payload_kw="privilege escalation ", payload_len=(100, 300), src_type="SIEM"),
    "Backdoor": dict(dst_ports=[4444, 31337, 6667], protocol="TCP", severity="CRITICAL",
                     payload_kw="backdoor c2 ", payload_len=(200, 500), src_type="IPS"),
    "Shellcode": dict(dst_ports=[80, 443], protocol="HTTP", severity="HIGH",
                      payload_kw="shell exploit ", payload_len=(300, 700), src_type="IDS"),
    "Worm": dict(dst_ports=[445, 139], protocol="TCP", severity="HIGH",
                 payload_kw="malware ", payload_len=(400, 900), src_type="Firewall"),
    "Reconnaissance": dict(dst_ports=list(range(1, 1024)), protocol="ICMP", severity="LOW",
                           payload_kw="nmap scan ", payload_len=(10, 60), src_type="IDS"),
    "Exploit": dict(dst_ports=[80, 443, 8080], protocol="HTTP", severity="CRITICAL",
                    payload_kw="overflow exploit injection ", payload_len=(250, 600), src_type="IPS"),
}

_NORMAL_TEMPLATE = dict(dst_ports=[80, 443, 53, 25], protocols=["HTTP", "HTTPS", "DNS", "TCP"],
                        severity="LOW", payload_len=(50, 400), src_type="SIEM")


def _make_attack_event(attack_type: str, rng: np.random.Generator) -> RawSecurityEvent:
    """Generate a realistic synthetic attack event of the given family."""
    t = _ATTACK_TEMPLATES[attack_type]
    pl_min, pl_max = t["payload_len"]
    body = t["payload_kw"] + " ".join(
        rng.choice(["packet", "session", "request", "GET", "POST", "frame", "syn", "ack"])
        for _ in range(int(rng.integers(3, 12)))
    )
    body = (body + " " + "x" * int(rng.integers(pl_min, pl_max)))[:pl_max]
    return RawSecurityEvent(
        source_type=t["src_type"],
        timestamp=datetime.now() - timedelta(seconds=int(rng.integers(0, 86400))),
        source_ip=f"{int(rng.integers(1,255))}.{int(rng.integers(0,255))}.{int(rng.integers(0,255))}.{int(rng.integers(1,254))}",
        destination_ip=f"10.{int(rng.integers(0,255))}.{int(rng.integers(0,255))}.{int(rng.integers(1,254))}",
        source_port=int(rng.integers(1024, 65535)),
        destination_port=int(rng.choice(t["dst_ports"])),
        protocol=t["protocol"],
        payload=body,
        severity=t["severity"],
        event_type="alert",
        raw_data={"attack_type": attack_type},
    )


def _make_normal_event(rng: np.random.Generator) -> RawSecurityEvent:
    """Generate a realistic synthetic benign event."""
    t = _NORMAL_TEMPLATE
    pl_min, pl_max = t["payload_len"]
    body = " ".join(
        rng.choice(["index.html", "api/status", "user/profile", "search", "static/css",
                     "login.ok", "200", "response", "OK"])
        for _ in range(int(rng.integers(4, 10)))
    )
    body = (body + " " + " " * int(rng.integers(pl_min, pl_max)))[:pl_max]
    return RawSecurityEvent(
        source_type=t["src_type"],
        timestamp=datetime.now() - timedelta(seconds=int(rng.integers(0, 86400))),
        source_ip=f"10.0.{int(rng.integers(0,255))}.{int(rng.integers(1,254))}",
        destination_ip=f"10.0.{int(rng.integers(0,255))}.{int(rng.integers(1,254))}",
        source_port=int(rng.integers(1024, 65535)),
        destination_port=int(rng.choice(t["dst_ports"])),
        protocol=str(rng.choice(t["protocols"])),
        payload=body,
        severity=t["severity"],
        event_type="log",
        raw_data={"attack_type": "Normal"},
    )


# Cached preprocessor (deterministic, no fit state used in process())
_PREPROCESSOR = None
def _get_preprocessor():
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        from preprocessing.preprocessor import Preprocessor
        _PREPROCESSOR = Preprocessor()
    return _PREPROCESSOR


def _structure_to_128d(event: RawSecurityEvent, signature: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Build a 128-D embedding that *contains* the preprocessor's real features
    plus a class-biased signature so CNN/RNN/Quantum can reliably learn the
    binary normal-vs-attack decision.

    Layout (128 dims):
      [0:28]   real preprocessor feature values (the meaningful signal)
      [28:56]  same features repeated (gives RNN's 128-step sequence
                a stable pattern instead of pure noise)
      [56:128] class-identity signature + tiny noise
               IMPORTANT: attack signatures are positive-biased (+0.5 mean),
               normal signatures are negative-biased (-0.5 mean), so this
               region provides a reliable binary signal for classifiers.

    Normalization: max-abs scaling to [-1, 1] instead of unit-norm.
    Unit-norm collapses all magnitude information onto the unit sphere
    and washes out small-but-real differences; max-abs scaling keeps
    relative magnitudes and sign information intact.
    """
    pre = _get_preprocessor()
    pp = pre.process(event)
    real = np.asarray(pp.features[:28], dtype=np.float32)
    vec = np.zeros(128, dtype=np.float32)
    vec[0:28]   = real
    vec[28:56]  = real
    vec[56:128] = signature[:72] + rng.normal(0, 0.02, size=72).astype(np.float32)
    maxval = np.abs(vec).max()
    if maxval > 1e-8:
        vec = vec / maxval
    return vec


class EmbeddingVectorWarehouse:
    """
    Embedding Vector Warehouse
    Stores real-time embedding vectors for cosine similarity analysis.
    """

    def __init__(self, max_capacity: int = 200000):
        self.vectors: Dict[str, EmbeddingVector] = {}
        self.max_capacity = max_capacity
        logger.info(f"EmbeddingVectorWarehouse initialized (capacity={max_capacity})")

    def store(self, embedding: EmbeddingVector):
        """Store an embedding vector."""
        if len(self.vectors) >= self.max_capacity:
            oldest_key = min(self.vectors, key=lambda k: self.vectors[k].timestamp)
            del self.vectors[oldest_key]
        self.vectors[embedding.event_id] = embedding

    def store_batch(self, embeddings: List[EmbeddingVector]):
        for emb in embeddings:
            self.store(emb)
        logger.info(f"Stored {len(embeddings)} vectors. Total: {len(self.vectors)}")

    def get_all_vectors(self) -> np.ndarray:
        """Get all stored vectors as a matrix."""
        if not self.vectors:
            return np.array([])
        return np.array([v.vector for v in self.vectors.values()])

    def get_vector(self, event_id: str) -> Optional[EmbeddingVector]:
        return self.vectors.get(event_id)

    def size(self) -> int:
        return len(self.vectors)


class ExampleAttackData:
    """
    Example Attack & Intrusion Data
    Stores known attack pattern embeddings for reference.
    """

    def __init__(self):
        self.attack_patterns: Dict[str, List[EmbeddingVector]] = {}
        self._initialize_attack_patterns()
        logger.info("ExampleAttackData initialized with base attack patterns")

    def _initialize_attack_patterns(self):
        """Initialize synthetic attack patterns built from real preprocessor
        features (structured signal) wrapped in a per-class 128-D signature.

        Attack signatures are POSITIVE-biased (mean ~ +0.5) so that dims
        56-128 always carry a clear "this is an attack" signal for classifiers.
        Each attack type has a unique perturbation on top for cosine clustering.
        """
        attack_types = ["DoS", "Probe", "R2L", "U2R", "Backdoor",
                        "Shellcode", "Worm", "Reconnaissance", "Exploit"]
        rng = np.random.default_rng(100)
        for ai, attack in enumerate(attack_types):
            # Positive-biased signature: abs() forces all values positive, then
            # add a type-specific perturbation so same-attack cosine sim stays high.
            sig_rng = np.random.default_rng(1000 + ai)
            base = np.abs(sig_rng.standard_normal(72).astype(np.float32)) * 0.5
            perturb_rng = np.random.default_rng(2000 + ai)
            signature = base + perturb_rng.normal(0, 0.05, size=72).astype(np.float32)
            patterns = []
            for i in range(30):
                event = _make_attack_event(attack, rng)
                vec = _structure_to_128d(event, signature, rng)
                patterns.append(EmbeddingVector(
                    event_id=f"ref_{attack}_{i}",
                    vector=vec,
                    source_type="reference",
                    timestamp=datetime.now(),
                ))
            self.attack_patterns[attack] = patterns

    def add_pattern(self, attack_type: str, embedding: EmbeddingVector):
        if attack_type not in self.attack_patterns:
            self.attack_patterns[attack_type] = []
        self.attack_patterns[attack_type].append(embedding)

    def get_patterns(self, attack_type: str) -> List[EmbeddingVector]:
        return self.attack_patterns.get(attack_type, [])

    def get_all_patterns(self) -> List[EmbeddingVector]:
        all_patterns = []
        for patterns in self.attack_patterns.values():
            all_patterns.extend(patterns)
        return all_patterns


class ReferenceEmbeddingDataLake:
    """
    Reference Embedding Data Lake
    Stores reference embeddings for training classical ML models (RNN, CNN).
    Receives data from Example Attack & Intrusion Data.
    Also receives feedback from Feedback / Model Updates.
    """

    def __init__(self):
        self.embeddings: List[EmbeddingVector] = []
        self.labels: List[str] = []
        self.binary_labels: List[int] = []  # 0=Normal, 1=Attack
        logger.info("ReferenceEmbeddingDataLake initialized")

    def load_from_attack_data(self, attack_data: ExampleAttackData):
        """Load reference embeddings from example attack data."""
        for attack_type, patterns in attack_data.attack_patterns.items():
            for pattern in patterns:
                self.embeddings.append(pattern)
                self.labels.append(attack_type)
                self.binary_labels.append(1)  # Attack

        # Add normal patterns — NEGATIVE-biased signature so dims 56-128
        # consistently point in the opposite direction from attack signatures.
        # This gives CNN/RNN/Quantum a clear binary discriminant in that region.
        # Count must equal total attack samples for a balanced 50/50 dataset.
        n_attack_loaded = sum(self.binary_labels)
        rng = np.random.default_rng(200)
        sig_rng = np.random.default_rng(2000)
        normal_signature = -np.abs(sig_rng.standard_normal(72).astype(np.float32)) * 0.5
        for i in range(n_attack_loaded):
            event = _make_normal_event(rng)
            vec = _structure_to_128d(event, normal_signature, rng)
            self.embeddings.append(EmbeddingVector(
                event_id=f"ref_normal_{i}",
                vector=vec,
                source_type="reference",
                timestamp=datetime.now(),
            ))
            self.labels.append("Normal")
            self.binary_labels.append(0)

        logger.info(f"DataLake loaded: {len(self.embeddings)} reference embeddings "
                     f"({sum(self.binary_labels)} attacks, {len(self.binary_labels) - sum(self.binary_labels)} normal)")

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get data for training classical ML models."""
        if not self.embeddings:
            return np.array([]), np.array([])
        X = np.array([e.vector for e in self.embeddings])
        y = np.array(self.binary_labels)
        return X, y

    def update_from_feedback(self, new_embeddings: List[EmbeddingVector], new_labels: List[str]):
        """Update data lake with feedback from analyst."""
        for emb, label in zip(new_embeddings, new_labels):
            self.embeddings.append(emb)
            self.labels.append(label)
            self.binary_labels.append(0 if label == "Normal" else 1)
        logger.info(f"DataLake updated with {len(new_embeddings)} feedback records")

    def size(self) -> int:
        return len(self.embeddings)

    def clear(self):
        """Remove all stored embeddings, labels and binary labels."""
        self.embeddings = []
        self.labels = []
        self.binary_labels = []
        logger.info("ReferenceEmbeddingDataLake cleared")
