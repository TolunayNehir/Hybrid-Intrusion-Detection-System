"""
Dashboard / Reporting & Alerts and Event Management & Security Analyst Review
===============================================================================
From the General System Diagram (rightmost components):
  Intrusion/Violation Detection → Dashboard/Reporting → Security Analyst Review
  Intrusion/Violation Detection → Alerts and Event Management → Security Analyst Review

These three terminal components handle output presentation and human-in-the-loop review.
"""

from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import DetectionAlert, FeedbackRecord
from utils.logger import get_logger

logger = get_logger("Dashboard")


class DashboardReporting:
    """
    Dashboard / Reporting module.
    Provides consolidated view of detection results.
    Sends data to Security Analyst Review.
    """

    def __init__(self):
        self.reports: List[Dict] = []
        self.total_events_processed = 0
        self.total_alerts = 0
        logger.info("Dashboard/Reporting module initialized")

    def generate_report(self, alerts: List[DetectionAlert],
                        total_events: int) -> Dict:
        """Generate a detection report for the security analyst."""
        self.total_events_processed += total_events
        self.total_alerts += len(alerts)

        # Severity breakdown
        severity_counts = Counter(a.severity for a in alerts)

        # Attack type breakdown
        attack_counts = Counter(a.attack_type for a in alerts)

        # Source IP analysis
        source_ips = Counter(a.source_ip for a in alerts)
        top_attackers = source_ips.most_common(5)

        # Confidence statistics
        confidences = [a.confidence for a in alerts] if alerts else [0.0]

        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_events_analyzed": total_events,
                "total_alerts_generated": len(alerts),
                "detection_rate": len(alerts) / max(total_events, 1) * 100,
            },
            "severity_breakdown": dict(severity_counts),
            "attack_type_breakdown": dict(attack_counts),
            "top_attackers": [{"ip": ip, "count": cnt} for ip, cnt in top_attackers],
            "confidence_stats": {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
            },
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity,
                    "attack_type": a.attack_type,
                    "confidence": round(a.confidence, 3),
                    "source_ip": a.source_ip,
                    "destination_ip": a.destination_ip,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in alerts[:20]  # Top 20 alerts
            ],
        }

        self.reports.append(report)
        return report

    def display_report(self, report: Dict):
        """Display report to console."""
        print("\n" + "=" * 80)
        print(f"  📊 SECURITY DASHBOARD REPORT — {report['report_id']}")
        print("=" * 80)

        s = report["summary"]
        print(f"\n  📈 Summary:")
        print(f"     Events Analyzed:  {s['total_events_analyzed']}")
        print(f"     Alerts Generated: {s['total_alerts_generated']}")
        print(f"     Detection Rate:   {s['detection_rate']:.1f}%")

        print(f"\n  🔴 Severity Breakdown:")
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            cnt = report["severity_breakdown"].get(sev, 0)
            bar = "█" * cnt
            print(f"     {sev:>8s}: {cnt:3d} {bar}")

        print(f"\n  🎯 Attack Types:")
        for atype, cnt in report["attack_type_breakdown"].items():
            print(f"     {atype:>20s}: {cnt}")

        print(f"\n  🔍 Top Attackers:")
        for attacker in report["top_attackers"]:
            print(f"     {attacker['ip']:>18s}: {attacker['count']} alerts")

        cs = report["confidence_stats"]
        print(f"\n  📊 Confidence Stats:")
        print(f"     Mean: {cs['mean']:.3f}  Min: {cs['min']:.3f}  Max: {cs['max']:.3f}")

        if report["alerts"]:
            print(f"\n  🚨 Recent Alerts (top {min(len(report['alerts']), 10)}):")
            for a in report["alerts"][:10]:
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(a["severity"], "⚪")
                print(f"     {icon} [{a['severity']:>8s}] {a['alert_id']} | "
                      f"{a['attack_type']:>15s} | conf={a['confidence']:.3f} | "
                      f"{a['source_ip']} → {a['destination_ip']}")

        print("\n" + "=" * 80)


class AlertsEventManagement:
    """
    Alerts and Event Management module.
    Manages alert lifecycle, escalation, and notification.
    """

    def __init__(self):
        self.active_alerts: List[DetectionAlert] = []
        self.acknowledged_alerts: List[DetectionAlert] = []
        self.resolved_alerts: List[DetectionAlert] = []
        logger.info("Alerts and Event Management initialized")

    def receive_alerts(self, alerts: List[DetectionAlert]):
        """Receive new alerts from Intrusion Detection module."""
        for alert in alerts:
            self.active_alerts.append(alert)
            if alert.severity in ["CRITICAL", "HIGH"]:
                self._escalate(alert)

    def _escalate(self, alert: DetectionAlert):
        """Escalate high-severity alerts."""
        logger.warning(f"⚠️ ESCALATION: {alert.alert_id} [{alert.severity}] "
                       f"{alert.attack_type} from {alert.source_ip}")

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        for i, alert in enumerate(self.active_alerts):
            if alert.alert_id == alert_id:
                self.acknowledged_alerts.append(self.active_alerts.pop(i))
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False

    def resolve_alert(self, alert_id: str, resolution: str = ""):
        """Resolve an alert."""
        for alerts_list in [self.active_alerts, self.acknowledged_alerts]:
            for i, alert in enumerate(alerts_list):
                if alert.alert_id == alert_id:
                    alert.analyst_feedback = resolution
                    alert.analyst_reviewed = True
                    self.resolved_alerts.append(alerts_list.pop(i))
                    logger.info(f"Alert {alert_id} resolved: {resolution}")
                    return True
        return False

    def get_status(self) -> Dict:
        return {
            "active": len(self.active_alerts),
            "acknowledged": len(self.acknowledged_alerts),
            "resolved": len(self.resolved_alerts),
        }


class SecurityAnalystReview:
    """
    Security Analyst Review - Terminal component.
    Human-in-the-loop interface for reviewing system decisions
    and providing feedback for model updates.
    """

    def __init__(self):
        self.review_queue: List[DetectionAlert] = []
        self.reviewed: List[DetectionAlert] = []
        self.feedback_records: List[FeedbackRecord] = []
        logger.info("Security Analyst Review module initialized")

    def add_to_review(self, alerts: List[DetectionAlert]):
        """Add alerts for analyst review."""
        self.review_queue.extend(alerts)
        logger.info(f"Added {len(alerts)} alerts to analyst review queue "
                     f"(total queue: {len(self.review_queue)})")

    def review_alert(self, alert_id: str, correct_label: str,
                     notes: str = "") -> Optional[FeedbackRecord]:
        """
        Analyst reviews an alert and provides feedback.
        This feedback feeds back into Feedback / Model Updates.
        """
        for i, alert in enumerate(self.review_queue):
            if alert.alert_id == alert_id:
                alert.analyst_reviewed = True
                alert.analyst_feedback = correct_label
                self.reviewed.append(self.review_queue.pop(i))

                feedback = FeedbackRecord(
                    event_id=alert.event_id,
                    correct_label=correct_label,
                    analyst_notes=notes,
                    timestamp=datetime.now(),
                    update_embeddings=True,
                    update_data_lake=True,
                )
                self.feedback_records.append(feedback)
                logger.info(f"Alert {alert_id} reviewed: label={correct_label}")
                return feedback
        return None

    def auto_review(self, alerts: List[DetectionAlert]) -> List[FeedbackRecord]:
        """
        Simulate analyst review for demo purposes.
        In production, this would be a human interface.
        """
        feedbacks = []
        for alert in alerts:
            # Simulate: analyst confirms most detections
            import numpy as np
            if np.random.random() < 0.85:  # 85% confirmation rate
                correct_label = alert.attack_type
                notes = "Confirmed by automated review"
            else:
                correct_label = "Normal"
                notes = "False positive identified"

            alert.analyst_reviewed = True
            alert.analyst_feedback = correct_label
            self.reviewed.append(alert)

            feedback = FeedbackRecord(
                event_id=alert.event_id,
                correct_label=correct_label,
                analyst_notes=notes,
                timestamp=datetime.now(),
            )
            feedbacks.append(feedback)

        self.feedback_records.extend(feedbacks)
        logger.info(f"Auto-reviewed {len(alerts)} alerts → {len(feedbacks)} feedback records")
        return feedbacks

    def get_queue_size(self) -> int:
        return len(self.review_queue)
