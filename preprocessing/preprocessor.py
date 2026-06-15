"""
Preprocessing Module
=====================
Cleaning, Parsing, Normalization, Feature Extraction

Receives raw data from all 5 data sources.
Outputs:
  1. Numerical features → Quantum AI path (Dimension Reduction) 
  2. Text features → Embedding Creation path
"""

import numpy as np
from typing import List, Tuple
from datetime import datetime
import uuid
import hashlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import RawSecurityEvent, PreprocessedData
from utils.logger import get_logger
from config.settings import PREPROCESSING

logger = get_logger("Preprocessor")

# Protocol encoding map
PROTOCOL_MAP = {
    "TCP": 0, "UDP": 1, "ICMP": 2, "HTTP": 3,
    "HTTPS": 4, "DNS": 5, "SSH": 6, "FTP": 7,
}

# Source type encoding
SOURCE_MAP = {
    "IPS": 0, "SIEM": 1, "Firewall": 2, "SOAR": 3, "IDS": 4,
}

# Severity encoding
SEVERITY_MAP = {
    "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3,
}


class Preprocessor:
    """
    Preprocessing: Cleaning, Parsing, Normalization, Feature Extraction
    
    This module standardizes raw security events from heterogeneous sources
    and extracts both numerical and text features for downstream analysis.
    """

    def __init__(self):
        self.max_features = PREPROCESSING["max_features"]
        self.normalization = PREPROCESSING["normalization"]
        self.feature_stats = {"min": None, "max": None, "mean": None, "std": None}
        self.fitted = False
        logger.info(f"Preprocessor initialized: max_features={self.max_features}, norm={self.normalization}")

    def clean(self, event: RawSecurityEvent) -> RawSecurityEvent:
        """Step 1: Clean the raw event - remove noise, handle missing values."""
        # Sanitize IP addresses
        event.source_ip = event.source_ip.strip() if event.source_ip else "0.0.0.0"
        event.destination_ip = event.destination_ip.strip() if event.destination_ip else "0.0.0.0"

        # Validate ports
        event.source_port = max(0, min(65535, event.source_port))
        event.destination_port = max(0, min(65535, event.destination_port))

        # Default protocol
        if event.protocol not in PROTOCOL_MAP:
            event.protocol = "TCP"

        return event

    def parse(self, event: RawSecurityEvent) -> dict:
        """Step 2: Parse event into structured fields."""
        ip_parts_src = [int(x) for x in event.source_ip.split(".")]
        ip_parts_dst = [int(x) for x in event.destination_ip.split(".")]

        return {
            "source_ip_octets": ip_parts_src,
            "dest_ip_octets": ip_parts_dst,
            "source_port": event.source_port,
            "dest_port": event.destination_port,
            "protocol_id": PROTOCOL_MAP.get(event.protocol, 0),
            "source_type_id": SOURCE_MAP.get(event.source_type, 0),
            "severity_id": SEVERITY_MAP.get(event.severity, 0),
            "payload_length": len(event.payload),
            "hour_of_day": event.timestamp.hour,
            "day_of_week": event.timestamp.weekday(),
            "payload_entropy": self._calculate_entropy(event.payload),
            "has_suspicious_keywords": self._check_suspicious(event.payload),
        }

    def extract_features(self, parsed: dict) -> np.ndarray:
        """Step 3: Feature Extraction - convert parsed data to numerical vector."""
        features = []

        # IP octets (8 features)
        features.extend(parsed["source_ip_octets"])
        features.extend(parsed["dest_ip_octets"])

        # Port features (4 features)
        features.append(parsed["source_port"] / 65535.0)
        features.append(parsed["dest_port"] / 65535.0)
        features.append(1.0 if parsed["dest_port"] in [22, 23, 3389] else 0.0)  # sensitive port
        features.append(1.0 if parsed["source_port"] < 1024 else 0.0)  # well-known port

        # Protocol and source (3 features)
        features.append(parsed["protocol_id"] / 7.0)
        features.append(parsed["source_type_id"] / 4.0)
        features.append(parsed["severity_id"] / 3.0)

        # Payload features (4 features)
        features.append(parsed["payload_length"] / 1500.0)
        features.append(parsed["payload_entropy"])
        features.append(float(parsed["has_suspicious_keywords"]))
        features.append(min(parsed["payload_length"] / 100.0, 1.0))

        # Temporal features (4 features)
        features.append(parsed["hour_of_day"] / 23.0)
        features.append(parsed["day_of_week"] / 6.0)
        features.append(1.0 if parsed["hour_of_day"] < 6 or parsed["hour_of_day"] > 22 else 0.0)
        features.append(1.0 if parsed["day_of_week"] >= 5 else 0.0)

        # Statistical features (5 features)
        ip_vals = parsed["source_ip_octets"] + parsed["dest_ip_octets"]
        features.append(np.mean(ip_vals) / 255.0)
        features.append(np.std(ip_vals) / 128.0)
        features.append(abs(parsed["source_port"] - parsed["dest_port"]) / 65535.0)
        features.append(1.0 if parsed["source_ip_octets"][0] in [10, 172, 192] else 0.0)
        features.append(1.0 if parsed["dest_ip_octets"][0] in [10, 172, 192] else 0.0)

        feature_vec = np.array(features, dtype=np.float32)

        # Pad or truncate to max_features
        if len(feature_vec) < self.max_features:
            feature_vec = np.pad(feature_vec, (0, self.max_features - len(feature_vec)))
        else:
            feature_vec = feature_vec[:self.max_features]

        return feature_vec

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Step 4: Normalization."""
        if self.normalization == "minmax":
            f_min = np.min(features) if np.min(features) != np.max(features) else 0
            f_max = np.max(features) if np.min(features) != np.max(features) else 1
            return (features - f_min) / (f_max - f_min + 1e-8)
        elif self.normalization == "standard":
            return (features - np.mean(features)) / (np.std(features) + 1e-8)
        return features

    def generate_text_representation(self, event: RawSecurityEvent) -> str:
        """Generate text representation for embedding creation."""
        return (
            f"{event.source_type} {event.event_type} {event.protocol} "
            f"src={event.source_ip}:{event.source_port} "
            f"dst={event.destination_ip}:{event.destination_port} "
            f"severity={event.severity} "
            f"payload={event.payload[:200]}"
        )

    def process(self, event: RawSecurityEvent) -> PreprocessedData:
        """Full preprocessing pipeline: Clean → Parse → Extract → Normalize."""
        # Step 1: Clean
        event = self.clean(event)

        # Step 2: Parse
        parsed = self.parse(event)

        # Step 3: Feature Extraction
        features = self.extract_features(parsed)

        # Step 4: Normalize
        features = self.normalize(features)

        # Generate text for embedding path
        text_repr = self.generate_text_representation(event)

        event_id = event.raw_data.get("event_id", str(uuid.uuid4())[:8])

        return PreprocessedData(
            event_id=event_id,
            features=features,
            text_features=text_repr,
            source_type=event.source_type,
            timestamp=event.timestamp,
            metadata={
                "severity": event.severity,
                "attack_type": event.raw_data.get("attack_type", "Unknown"),
                "protocol": event.protocol,
                "source_ip": event.source_ip,
                "destination_ip": event.destination_ip,
            }
        )

    def process_batch(self, events: List[RawSecurityEvent]) -> List[PreprocessedData]:
        """Process a batch of raw security events."""
        results = [self.process(e) for e in events]
        logger.info(f"Preprocessed {len(results)} events → features shape: ({len(results)}, {self.max_features})")
        return results

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        if not text:
            return 0.0
        probs = np.array([text.count(c) / len(text) for c in set(text)])
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    @staticmethod
    def _check_suspicious(payload: str) -> bool:
        keywords = ["exploit", "overflow", "injection", "brute", "scan",
                     "flood", "backdoor", "shell", "privilege", "escalation",
                     "c2", "malware", "nmap", "unauthorized"]
        return any(kw in payload.lower() for kw in keywords)
