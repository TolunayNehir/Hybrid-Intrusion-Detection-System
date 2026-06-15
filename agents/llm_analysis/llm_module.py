"""
LLM (Large Language Model) Analysis Agent — GPT-4o
=====================================================
From the General System Diagram:
  LLM Analysis → Decision Fusion Layer

Uses OpenAI GPT-4o (via Replit AI Integrations) to semantically analyse
security events and return a structured intrusion classification decision.
Falls back to the built-in rule-based engine when the API is unavailable.
"""

import os
import json
import re
from typing import List, Dict, Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_models import PreprocessedData, ClassificationResult
from utils.logger import get_logger
from config.settings import LLM_ANALYSIS

logger = get_logger("LLM_Analysis")

# ─────────────────────────────────────────────────────────────
# OpenAI client (Replit AI Integrations — no personal key needed)
# ─────────────────────────────────────────────────────────────

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from openai import OpenAI
        api_key  = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
        if not api_key or not base_url:
            return None
        _openai_client = OpenAI(api_key=api_key, base_url=base_url)
        return _openai_client
    except Exception as e:
        logger.warning(f"OpenAI client başlatılamadı: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Fallback: Rule-based knowledge base
# ─────────────────────────────────────────────────────────────

ATTACK_SIGNATURES = {
    "DoS":       {"keywords": ["flood","syn","dos","ddos","volume","packets/sec","overwhelm"],
                  "patterns": [r"(\d{4,})\s*packets", r"flood\s+detected", r"syn\s+flood"], "weight": 0.9},
    "Probe":     {"keywords": ["scan","nmap","port scan","enumeration","reconnaissance","fingerprint"],
                  "patterns": [r"port\s+scan", r"nmap", r"sequential\s+port"], "weight": 0.85},
    "R2L":       {"keywords": ["brute force","remote","login","authentication","password","ssh"],
                  "patterns": [r"brute\s+force", r"failed\s+login", r"(\d{2,})\s+attempts"], "weight": 0.88},
    "U2R":       {"keywords": ["privilege","escalation","sudo","root","unauthorized","admin"],
                  "patterns": [r"privilege\s+escalation", r"unauthorized\s+sudo", r"root\s+access"], "weight": 0.92},
    "Backdoor":  {"keywords": ["backdoor","c2","command and control","reverse shell","trojan"],
                  "patterns": [r"c2\s+server", r"backdoor", r"reverse\s+shell"], "weight": 0.95},
    "Shellcode": {"keywords": ["buffer overflow","shellcode","nop sled","stack smash","injection"],
                  "patterns": [r"buffer\s+overflow", r"nop\s+sled", r"shellcode"], "weight": 0.93},
    "Worm":      {"keywords": ["worm","self-replicat","lateral movement","propagat","spread"],
                  "patterns": [r"self-replicat", r"lateral\s+movement", r"worm"], "weight": 0.90},
    "Exploit":   {"keywords": ["exploit","cve","vulnerability","malformed","zero-day"],
                  "patterns": [r"CVE-\d{4}-\d+", r"exploit\s+attempt", r"malformed"], "weight": 0.91},
}

SUSPICIOUS_PORTS  = {22, 23, 3389, 445, 139, 1433, 3306, 5432, 27017}
SUSPICIOUS_HOURS  = set(range(0, 6)) | {23, 22}


def _rule_based_analyze(preprocessed: PreprocessedData) -> dict:
    """Built-in rule-based fallback analysis."""
    text_lower = preprocessed.text_features.lower()
    keyword_scores: Dict[str, float] = {}
    for atype, sig in ATTACK_SIGNATURES.items():
        kw_hits = sum(1 for kw in sig["keywords"] if kw in text_lower)
        pat_hits = sum(1 for p in sig["patterns"] if re.search(p, text_lower))
        if kw_hits or pat_hits:
            keyword_scores[atype] = min(1.0, (kw_hits*0.15 + pat_hits*0.3) * sig["weight"])

    meta = preprocessed.metadata
    risk = 0.0
    dst_port = meta.get("destination_port", 0)
    if isinstance(dst_port, (int, float)) and int(dst_port) in SUSPICIOUS_PORTS:
        risk += 0.15
    risk += {"LOW": 0.0, "MEDIUM": 0.1, "HIGH": 0.25, "CRITICAL": 0.4}.get(
        meta.get("severity", "LOW"), 0.0)
    if preprocessed.timestamp.hour in SUSPICIOUS_HOURS:
        risk += 0.1
    risk += {"IPS": 0.15, "IDS": 0.15, "SIEM": 0.1, "SOAR": 0.05, "Firewall": 0.05}.get(
        preprocessed.source_type, 0.0)
    risk = min(1.0, risk)

    if not keyword_scores:
        if risk > 0.5:
            return {"prediction": 1, "confidence": risk * 0.6,
                    "label": "Suspicious", "reasoning": "High contextual risk (no keyword match)"}
        return {"prediction": 0, "confidence": 1.0 - risk,
                "label": "Normal", "reasoning": "No signatures, low context risk"}

    best = max(keyword_scores, key=keyword_scores.get)
    score = keyword_scores[best] * 0.7 + risk * 0.3
    threshold = LLM_ANALYSIS.get("confidence_threshold", 0.5)
    if score > threshold:
        return {"prediction": 1, "confidence": score, "label": best,
                "reasoning": f"Rule match: {best} (kw_score={keyword_scores[best]:.2f}, ctx={risk:.2f})"}
    return {"prediction": 0, "confidence": 1.0 - score,
            "label": "Normal", "reasoning": f"Low confidence: {best} ({score:.2f})"}


# ─────────────────────────────────────────────────────────────
# GPT-4o analysis
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert Intrusion Detection System (IDS) analyst.
Analyze the network security event provided and respond ONLY with a valid JSON object:

{
  "prediction": 0 or 1,
  "confidence": float between 0.0 and 1.0,
  "label": "attack category or Normal",
  "reasoning": "brief one-sentence explanation"
}

prediction: 1 if intrusion/attack, 0 if normal traffic.
confidence: how certain you are (0.0–1.0).
label: one of [Normal, DoS, DDoS, Probe, PortScan, R2L, U2R, Backdoor, Shellcode, Worm, Exploit, SQLi, XSS, Brute Force, Suspicious].
Respond ONLY with the JSON — no markdown, no explanation."""


def _gpt4o_analyze(preprocessed: PreprocessedData) -> Optional[dict]:
    """Call GPT-4o and parse the structured JSON response."""
    client = _get_openai_client()
    if client is None:
        return None

    meta = preprocessed.metadata
    event_desc = (
        f"Source IP: {meta.get('source_ip', 'N/A')}\n"
        f"Destination IP: {meta.get('destination_ip', 'N/A')}\n"
        f"Source Port: {meta.get('source_port', 'N/A')}\n"
        f"Destination Port: {meta.get('destination_port', 'N/A')}\n"
        f"Protocol: {meta.get('protocol', 'N/A')}\n"
        f"Severity: {meta.get('severity', 'N/A')}\n"
        f"Source Type: {preprocessed.source_type}\n"
        f"Attack Type Hint: {meta.get('attack_type', 'Unknown')}\n"
        f"Payload / Text: {preprocessed.text_features[:500]}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Security Event:\n{event_desc}"},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        result = json.loads(raw)

        prediction = int(bool(result.get("prediction", 0)))
        confidence = float(max(0.0, min(1.0, result.get("confidence", 0.5))))
        label      = str(result.get("label", "Normal"))
        reasoning  = str(result.get("reasoning", ""))

        return {"prediction": prediction, "confidence": confidence,
                "label": label, "reasoning": reasoning}

    except json.JSONDecodeError as e:
        logger.warning(f"GPT-4o JSON parse hatası: {e} | raw={raw[:200]}")
        return None
    except Exception as e:
        err = str(e)
        if "FREE_CLOUD_BUDGET_EXCEEDED" in err:
            logger.error("AI Integrations cloud bütçesi aşıldı.")
        else:
            logger.warning(f"GPT-4o API hatası: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Public Agent Class
# ─────────────────────────────────────────────────────────────

class LLMAnalysisModule:
    """
    LLM Analysis module — uses GPT-4o (Replit AI Integrations).
    Falls back to rule-based engine when the API is unavailable.
    """

    def __init__(self):
        self.confidence_threshold = LLM_ANALYSIS.get("confidence_threshold", 0.5)
        self.analysis_count = 0
        self.gpt4o_calls   = 0
        self.fallback_calls = 0

        # Eagerly probe the client so startup logs are accurate
        client_ok = _get_openai_client() is not None
        logger.info(
            f"LLMAnalysisModule init | GPT-4o={'enabled' if client_ok else 'unavailable (fallback)'}"
        )

    def analyze(self, preprocessed: PreprocessedData) -> ClassificationResult:
        """Analyze a security event; GPT-4o first, rule-based fallback."""
        self.analysis_count += 1
        source = "gpt-4o"

        result = _gpt4o_analyze(preprocessed)
        if result is None:
            result = _rule_based_analyze(preprocessed)
            source = "rule_based_fallback"
            self.fallback_calls += 1
        else:
            self.gpt4o_calls += 1

        logger.debug(
            f"LLM [{source}] event={preprocessed.event_id} "
            f"pred={result['prediction']} conf={result['confidence']:.3f} "
            f"label={result['label']}"
        )

        return ClassificationResult(
            agent_name="LLM_Analysis",
            event_id=preprocessed.event_id,
            prediction=result["prediction"],
            confidence=float(result["confidence"]),
            label=result["label"],
            details={
                "reasoning":  result["reasoning"],
                "source":     source,
                "model":      "gpt-4o" if source == "gpt-4o" else "rule_based",
                "gpt4o_calls":    self.gpt4o_calls,
                "fallback_calls": self.fallback_calls,
            },
        )

    def analyze_batch(self, preprocessed_batch: List[PreprocessedData]) -> List[ClassificationResult]:
        results = [self.analyze(p) for p in preprocessed_batch]
        gpt_count = sum(1 for r in results if r.details.get("source") == "gpt-4o")
        logger.info(
            f"LLM_Analysis batch: {len(results)} events | "
            f"{gpt_count} via GPT-4o | {len(results)-gpt_count} via fallback | "
            f"{sum(r.prediction for r in results)} threats"
        )
        return results
