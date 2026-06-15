"""Data models and structures used across the system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


@dataclass
class RawSecurityEvent:
    """Raw security event from any data source."""
    source_type: str          # IPS, SIEM, Firewall, SOAR, IDS
    timestamp: datetime
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    payload: str
    severity: str             # LOW, MEDIUM, HIGH, CRITICAL
    event_type: str           # alert, log, event
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedData:
    """Cleaned and feature-extracted data."""
    event_id: str
    features: np.ndarray      # Numerical feature vector
    text_features: str        # Text representation for embedding
    source_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingVector:
    """Semantic embedding representation of an event."""
    event_id: str
    vector: np.ndarray
    source_type: str
    timestamp: datetime


@dataclass
class ClassificationResult:
    """Output from any classification agent."""
    agent_name: str
    event_id: str
    prediction: int           # 0=Normal, 1=Intrusion
    confidence: float         # 0.0 - 1.0
    label: str                # "Normal" or attack type
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """Output from cosine similarity analysis."""
    event_id: str
    anomaly_score: float      # 0.0 - 1.0
    similarity_score: float
    top_matches: List[Dict[str, Any]] = field(default_factory=list)
    is_anomaly: bool = False


@dataclass
class FusionResult:
    """Combined decision from all agents via fusion layer."""
    event_id: str
    final_score: float
    is_intrusion: bool
    agent_scores: Dict[str, float] = field(default_factory=dict)
    fusion_method: str = "weighted_average"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DetectionAlert:
    """Final intrusion/violation detection alert."""
    alert_id: str
    event_id: str
    severity: str
    attack_type: str
    confidence: float
    source_ip: str
    destination_ip: str
    description: str
    timestamp: datetime
    fusion_result: Optional[FusionResult] = None
    analyst_reviewed: bool = False
    analyst_feedback: Optional[str] = None


@dataclass
class FeedbackRecord:
    """Feedback from security analyst for model updates."""
    event_id: str
    correct_label: str
    analyst_notes: str
    timestamp: datetime
    update_embeddings: bool = True
    update_data_lake: bool = True
