"""
Intrusion / Violation Detection Module
========================================
From the General System Diagram:
  Decision Fusion Layer → Intrusion/Violation Detection → Dashboard/Reporting
                                                       → Alerts and Event Management

Converts fusion results into actionable detection alerts.
"""

import uuid
from typing import List, Dict
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import FusionResult, DetectionAlert, PreprocessedData
from utils.logger import get_logger

logger = get_logger("IntrusionDetector")


class IntrusionDetector:
    """
    Intrusion / Violation Detection module.
    Receives fused decisions and generates structured detection alerts
    for the Dashboard and Alert Management systems.
    """

    def __init__(self):
        self.alert_history: List[DetectionAlert] = []
        self.detection_count = 0
        self.false_positive_count = 0
        logger.info("IntrusionDetector initialized")

    def _determine_severity(self, fusion_result: FusionResult) -> str:
        """Determine alert severity based on fusion score."""
        score = fusion_result.final_score
        if score > 0.9:
            return "CRITICAL"
        elif score > 0.75:
            return "HIGH"
        elif score > 0.6:
            return "MEDIUM"
        return "LOW"

    def _determine_attack_type(self, fusion_result: FusionResult,
                                preprocessed: PreprocessedData) -> str:
        """Determine the most likely attack type from agent consensus."""
        return preprocessed.metadata.get("attack_type", "Unknown Threat")

    def _generate_description(self, fusion_result: FusionResult,
                               preprocessed: PreprocessedData) -> str:
        """Generate human-readable description of the detection."""
        agent_details = []
        for agent_name, score in fusion_result.agent_scores.items():
            status = "⚠ ALERT" if score > 0.5 else "✓ OK"
            agent_details.append(f"{agent_name}: {score:.2f} ({status})")

        return (
            f"Intrusion detected from {preprocessed.metadata.get('source_ip', 'N/A')} "
            f"→ {preprocessed.metadata.get('destination_ip', 'N/A')} "
            f"via {preprocessed.metadata.get('protocol', 'N/A')}. "
            f"Fusion score: {fusion_result.final_score:.3f} "
            f"({fusion_result.fusion_method}). "
            f"Agent scores: {', '.join(agent_details)}"
        )

    def detect(self, fusion_result: FusionResult,
               preprocessed: PreprocessedData) -> DetectionAlert:
        """
        Process a fusion result and generate a detection alert.
        
        Returns:
            DetectionAlert for Dashboard/Reporting and Alert Management
        """
        if not fusion_result.is_intrusion:
            return None

        self.detection_count += 1

        alert = DetectionAlert(
            alert_id=f"ALERT-{uuid.uuid4().hex[:8].upper()}",
            event_id=fusion_result.event_id,
            severity=self._determine_severity(fusion_result),
            attack_type=self._determine_attack_type(fusion_result, preprocessed),
            confidence=fusion_result.final_score,
            source_ip=preprocessed.metadata.get("source_ip", "Unknown"),
            destination_ip=preprocessed.metadata.get("destination_ip", "Unknown"),
            description=self._generate_description(fusion_result, preprocessed),
            timestamp=datetime.now(),
            fusion_result=fusion_result,
        )

        self.alert_history.append(alert)
        logger.info(f"🚨 Detection #{self.detection_count}: {alert.alert_id} "
                     f"[{alert.severity}] {alert.attack_type} "
                     f"(confidence={alert.confidence:.3f})")
        return alert

    def detect_batch(self, fusion_results: List[FusionResult],
                     preprocessed_batch: List[PreprocessedData]) -> List[DetectionAlert]:
        """Process batch of fusion results."""
        alerts = []
        for fr, pp in zip(fusion_results, preprocessed_batch):
            alert = self.detect(fr, pp)
            if alert:
                alerts.append(alert)
        logger.info(f"Detection batch complete: {len(alerts)} alerts generated from {len(fusion_results)} events")
        return alerts

    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for alert in self.alert_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        return {
            "total_detections": self.detection_count,
            "severity_distribution": severity_counts,
            "total_alerts": len(self.alert_history),
        }
