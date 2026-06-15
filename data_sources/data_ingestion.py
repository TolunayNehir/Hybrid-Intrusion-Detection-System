"""
Data Sources Module
====================
Ingests real-time data from 5 cybersecurity data sources:
  1. IPS Alerts
  2. SIEM Events
  3. Firewall Logs
  4. SOAR Events
  5. IDS Alerts

All sources feed into the Preprocessing module.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Generator
import uuid
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_models import RawSecurityEvent
from utils.logger import get_logger

logger = get_logger("DataIngestion")

# ─── Attack type definitions ───
ATTACK_TYPES = [
    "Normal", "DoS", "Probe", "R2L", "U2R",
    "Backdoor", "Shellcode", "Worm", "Reconnaissance", "Exploit"
]

PROTOCOLS = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS", "SSH", "FTP"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class DataSourceBase:
    """Base class for all data sources."""

    def __init__(self, source_type: str, source_name: str):
        self.source_type = source_type
        self.source_name = source_name
        self.logger = get_logger(f"Source_{source_name}")

    def generate_event(self, is_attack: bool = False) -> RawSecurityEvent:
        """Generate a synthetic security event."""
        attack_type = np.random.choice(ATTACK_TYPES[1:]) if is_attack else "Normal"
        severity = np.random.choice(SEVERITIES, p=[0.1, 0.2, 0.4, 0.3]) if is_attack \
            else np.random.choice(SEVERITIES, p=[0.7, 0.2, 0.08, 0.02])

        return RawSecurityEvent(
            source_type=self.source_type,
            timestamp=datetime.now() - timedelta(seconds=np.random.randint(0, 3600)),
            source_ip=f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}",
            destination_ip=f"192.168.{np.random.randint(0,10)}.{np.random.randint(1,255)}",
            source_port=np.random.randint(1024, 65535),
            destination_port=np.random.choice([22, 80, 443, 3306, 8080, 21, 53, 25]),
            protocol=np.random.choice(PROTOCOLS),
            payload=self._generate_payload(attack_type),
            severity=severity,
            event_type="alert" if is_attack else "log",
            raw_data={
                "attack_type": attack_type,
                "source_name": self.source_name,
                "event_id": str(uuid.uuid4())[:8],
            }
        )

    def _generate_payload(self, attack_type: str) -> str:
        payloads = {
            "Normal": "GET /index.html HTTP/1.1 Host: example.com",
            "DoS": "SYN flood detected: 10000 packets/sec from single source",
            "Probe": "NMAP scan detected: sequential port scanning 1-1024",
            "R2L": "SSH brute force: 500 failed login attempts from remote host",
            "U2R": "Privilege escalation: unauthorized sudo command execution",
            "Backdoor": "Suspicious outbound connection to known C2 server",
            "Shellcode": "Buffer overflow attempt: NOP sled detected in payload",
            "Worm": "Self-replicating network traffic pattern: lateral movement",
            "Reconnaissance": "DNS enumeration: multiple zone transfer requests",
            "Exploit": "CVE-2024-1234 exploit attempt: malformed HTTP request",
        }
        return payloads.get(attack_type, "Unknown event payload")


class IPSAlertSource(DataSourceBase):
    """Intrusion Prevention System (IPS) Alert Source."""
    def __init__(self):
        super().__init__("IPS", "IPS_Alerts")


class SIEMEventSource(DataSourceBase):
    """Security Information and Event Management (SIEM) Event Source."""
    def __init__(self):
        super().__init__("SIEM", "SIEM_Events")


class FirewallLogSource(DataSourceBase):
    """Firewall Log Source."""
    def __init__(self):
        super().__init__("Firewall", "Firewall_Logs")


class SOAREventSource(DataSourceBase):
    """Security Orchestration, Automation and Response (SOAR) Event Source."""
    def __init__(self):
        super().__init__("SOAR", "SOAR_Events")


class IDSAlertSource(DataSourceBase):
    """Intrusion Detection System (IDS) Alert Source."""
    def __init__(self):
        super().__init__("IDS", "IDS_Alerts")


class DataIngestionEngine:
    """
    Main data ingestion engine that orchestrates all 5 data sources.
    Collects events and forwards them to the Preprocessing module.
    """

    def __init__(self):
        self.sources = {
            "IPS": IPSAlertSource(),
            "SIEM": SIEMEventSource(),
            "Firewall": FirewallLogSource(),
            "SOAR": SOAREventSource(),
            "IDS": IDSAlertSource(),
        }
        self.logger = get_logger("DataIngestionEngine")
        self.logger.info("Data Ingestion Engine initialized with 5 sources: IPS, SIEM, Firewall, SOAR, IDS")

    def collect_events(self, n_events: int = 100, attack_ratio: float = 0.3) -> List[RawSecurityEvent]:
        """Collect events from all 5 data sources."""
        events = []
        events_per_source = max(1, n_events // len(self.sources))

        for source_name, source in self.sources.items():
            for _ in range(events_per_source):
                is_attack = np.random.random() < attack_ratio
                event = source.generate_event(is_attack=is_attack)
                events.append(event)

        np.random.shuffle(events)
        self.logger.info(f"Collected {len(events)} events from {len(self.sources)} sources")
        return events

    def stream_events(self, attack_ratio: float = 0.3) -> Generator[RawSecurityEvent, None, None]:
        """Stream events continuously from all sources."""
        while True:
            source_name = np.random.choice(list(self.sources.keys()))
            source = self.sources[source_name]
            is_attack = np.random.random() < attack_ratio
            yield source.generate_event(is_attack=is_attack)
