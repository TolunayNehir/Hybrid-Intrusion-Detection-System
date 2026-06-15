"""
Hybrid IDS — Flask Web Application
=====================================
Provides a REST API + HTML dashboard that wraps the HybridIDSOrchestrator.
"""

import os
import sys
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from utils.dataset_analyzer import DatasetAnalyzer, label_to_binary as _label_to_binary
from storage.training_db import init_db as _db_init, save_run as _db_save, get_history as _db_history, clear_history as _db_clear_history

# Persisted train-metrics files (survive restarts)
_TRAIN_METRICS_PATH       = os.path.join(PROJECT_ROOT, "data", "last_train_metrics.json")
_MODEL_TRAIN_STATUS_PATH  = os.path.join(PROJECT_ROOT, "data", "model_train_status.json")


def _save_train_metrics():
    """Persist _train_job to disk so metrics survive restarts."""
    try:
        os.makedirs(os.path.dirname(_TRAIN_METRICS_PATH), exist_ok=True)
        snapshot = {k: v for k, v in _train_job.items() if k != "running"}
        with open(_TRAIN_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"Train metrics save error: {e}")


def _load_train_metrics():
    """Load persisted train metrics into _train_job on startup."""
    if not os.path.exists(_TRAIN_METRICS_PATH):
        return
    try:
        with open(_TRAIN_METRICS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        _train_job.update(saved)
        _train_job["running"] = False  # Never running after restart
        logger.info(
            f"Previous training metrics loaded: "
            f"stage={_train_job.get('stage')} | "
            f"last_run={_train_job.get('last_run')}"
        )
    except Exception as e:
        logger.warning(f"Train metrics load error: {e}")


def _save_model_train_status():
    """Persist per-model train statuses to disk so they survive restarts."""
    try:
        os.makedirs(os.path.dirname(_MODEL_TRAIN_STATUS_PATH), exist_ok=True)
        snapshot = {
            k: {ek: ev for ek, ev in v.items() if ek != "running"}
            for k, v in _model_train_status.items()
        }
        with open(_MODEL_TRAIN_STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"Model train status save error: {e}")


def _load_model_train_status():
    """Restore per-model train statuses from disk on startup."""
    if not os.path.exists(_MODEL_TRAIN_STATUS_PATH):
        return
    try:
        with open(_MODEL_TRAIN_STATUS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key in _model_train_status:
            if key in saved:
                _model_train_status[key].update(saved[key])
                _model_train_status[key]["running"] = False
        logger.info("Per-model train statuses restored from disk")
    except Exception as e:
        logger.warning(f"Model train status load error: {e}")

logger = get_logger("WebApp")

app = Flask(__name__)
CORS(app)
_db_init()   # ensure training_history.db exists

# ─────────────────────────────────────────────
# Global orchestrator state (lazy init)
# ─────────────────────────────────────────────

_orchestrator = None
_init_lock    = threading.Lock()
_init_status  = {
    "ready":   False,
    "stage":   "idle",      # idle | initializing | training | ready | error
    "message": "System not started",
    "progress": 0,
    "error":   None,
}

# In-memory accumulation of processed results across calls
_session = {
    "total_events":    0,
    "total_alerts":    0,
    "all_alerts":      [],   # list of serialised alert dicts
    "latest_report":   None,
    "alert_status":    {"active": 0, "acknowledged": 0, "resolved": 0},
    "processing":      False,
}


# ─────────────────────────────────────────────
# Orchestrator initialisation (background)
# ─────────────────────────────────────────────

def _update_status(stage: str, message: str, progress: int = 0):
    _init_status.update({"stage": stage, "message": message, "progress": progress})
    logger.info(f"[{stage.upper()}] {message}")


def _init_orchestrator(force_retrain: bool = False):
    global _orchestrator
    with _init_lock:
        # Already ready and no retrain requested — restore status and exit
        if _orchestrator is not None and not force_retrain:
            _init_status.update({
                "ready":    True,
                "stage":    "ready",
                "message":  "System ready",
                "progress": 100,
                "error":    None,
            })
            return

        try:
            _update_status("initializing", "Loading modules…", 10)

            from main import HybridIDSOrchestrator
            orch = HybridIDSOrchestrator(force_retrain=force_retrain)

            _update_status("training", "Training models / loading checkpoints…", 40)
            orch.setup_and_train()

            _orchestrator = orch
            _init_status.update({
                "ready":    True,
                "stage":    "ready",
                "message":  "System ready",
                "progress": 100,
                "error":    None,
            })
            _load_train_metrics()         # restore dataset training metrics from disk
            _load_model_train_status()   # restore per-model (CNN/RNN/Quantum) metrics
            logger.info("Orchestrator ready.")

        except Exception as exc:
            _init_status.update({
                "ready":    False,
                "stage":    "error",
                "message":  str(exc),
                "progress": 0,
                "error":    str(exc),
            })
            logger.error(f"Orchestrator error: {exc}", exc_info=True)


def _start_init(force_retrain: bool = False):
    # Already running — don't restart
    if _init_status["stage"] in ("initializing", "training"):
        return
    # Already ready and no forced retrain — nothing to do
    if _init_status["ready"] and not force_retrain:
        return
    _init_status.update({
        "stage":    "initializing",
        "ready":    False,
        "error":    None,
        "progress": 0,
        "message":  "Starting…",
    })
    t = threading.Thread(target=_init_orchestrator, args=(force_retrain,), daemon=True)
    t.start()


# ─────────────────────────────────────────────
# Helper: serialise alert
# ─────────────────────────────────────────────

def _alert_to_dict(alert) -> Dict[str, Any]:
    return {
        "alert_id":       alert.alert_id,
        "event_id":       alert.event_id,
        "severity":       alert.severity,
        "attack_type":    alert.attack_type,
        "confidence":     round(alert.confidence, 4),
        "source_ip":      alert.source_ip,
        "destination_ip": alert.destination_ip,
        "description":    alert.description,
        "timestamp":      alert.timestamp.isoformat(),
        "reviewed":       alert.analyst_reviewed,
        "feedback":       alert.analyst_feedback,
    }


# ─────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────

@app.route("/")
def index():
    import time
    return render_template("index.html", cache_bust=int(time.time()))


# ─────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """System initialisation status + live counters."""
    from utils.model_persistence import get_all_checkpoint_status
    try:
        checkpoint_status = get_all_checkpoint_status()
    except Exception:
        checkpoint_status = {}

    alert_status = {"active": 0, "acknowledged": 0, "resolved": 0}
    if _orchestrator:
        alert_status = _orchestrator.alert_manager.get_status()

    return jsonify({
        "init":        _init_status.copy(),
        "checkpoints": checkpoint_status,
        "session": {
            "total_events": _session["total_events"],
            "total_alerts": _session["total_alerts"],
            "processing":   _session["processing"],
        },
        "alert_status": alert_status,
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start/restart the orchestrator (optionally force-retrain)."""
    data = request.get_json(silent=True) or {}
    force = bool(data.get("force_retrain", False))
    _start_init(force_retrain=force)
    return jsonify({"ok": True, "message": "Initialization started"})


@app.route("/api/process", methods=["POST"])
def api_process():
    """
    Process N security events through the full pipeline.
    Body: { "n_events": 20, "attack_ratio": 0.3 }
    """
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready yet"}), 503

    if _session["processing"]:
        return jsonify({"error": "Processing already in progress"}), 429

    data = request.get_json(silent=True) or {}
    n_events     = int(data.get("n_events", 20))
    attack_ratio = float(data.get("attack_ratio", 0.3))

    n_events     = max(1, min(n_events, 200))
    attack_ratio = max(0.0, min(attack_ratio, 1.0))

    _session["processing"] = True

    def _run():
        try:
            # Snapshot alert count before processing
            prev_alert_count = len(_orchestrator.detector.alert_history)

            result = _orchestrator.process_events(n_events, attack_ratio)

            # Grab newly added alerts from detector history
            new_alerts = _orchestrator.detector.alert_history[prev_alert_count:]

            # Grab latest dashboard report
            report = (
                _orchestrator.dashboard.reports[-1]
                if _orchestrator.dashboard.reports else {}
            )

            _session["total_events"] += result.get("total_events", n_events)
            _session["total_alerts"] += len(new_alerts)
            _session["latest_report"]  = report

            for a in reversed(new_alerts):
                _session["all_alerts"].insert(0, _alert_to_dict(a))

            # Keep only last 500
            _session["all_alerts"] = _session["all_alerts"][:500]

        except Exception as exc:
            logger.error(f"process_events error: {exc}", exc_info=True)
        finally:
            _session["processing"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": f"{n_events} events queued for processing"})


@app.route("/api/alerts")
def api_alerts():
    """Return recent alerts with optional filters."""
    limit    = int(request.args.get("limit", 50))
    severity = request.args.get("severity", "")    # CRITICAL|HIGH|MEDIUM|LOW
    reviewed = request.args.get("reviewed", "")    # true|false

    alerts = _session["all_alerts"]

    if severity:
        alerts = [a for a in alerts if a["severity"] == severity.upper()]
    if reviewed == "true":
        alerts = [a for a in alerts if a["reviewed"]]
    elif reviewed == "false":
        alerts = [a for a in alerts if not a["reviewed"]]

    return jsonify({
        "alerts": alerts[:limit],
        "total":  len(_session["all_alerts"]),
    })


@app.route("/api/report")
def api_report():
    """Return the latest dashboard report."""
    return jsonify(_session["latest_report"] or {})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Analyze a single custom security event.
    Body fields: source_ip, destination_ip, source_port, destination_port,
                 protocol, payload, severity, source_type
    """
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready yet"}), 503

    data = request.get_json(silent=True) or {}

    from utils.data_models import RawSecurityEvent
    try:
        event = RawSecurityEvent(
            source_type=data.get("source_type", "IPS"),
            timestamp=datetime.now(),
            source_ip=data.get("source_ip", "192.168.1.100"),
            destination_ip=data.get("destination_ip", "10.0.0.1"),
            source_port=int(data.get("source_port", 12345)),
            destination_port=int(data.get("destination_port", 80)),
            protocol=data.get("protocol", "TCP"),
            payload=data.get("payload", ""),
            severity=data.get("severity", "MEDIUM"),
            event_type="alert",
            raw_data={"event_id": f"manual-{int(time.time())}",
                      "attack_type": data.get("attack_type", "Unknown")},
        )

        preprocessed  = _orchestrator.preprocessor.process(event)
        embedding     = _orchestrator.embedding_engine.create_embedding(preprocessed)
        _orchestrator.vector_warehouse.store(embedding)

        results = {}

        # CNN — always return a result even if predict() raises
        try:
            _cnn_r = _orchestrator.cnn_classical.predict(embedding, preprocessed)
            results["cnn"] = vars(_cnn_r)
            results["cnn"]["details"] = dict(results["cnn"].get("details", {}))
        except Exception as _cnn_err:
            logger.warning(f"CNN predict error (returning fallback): {_cnn_err}")
            results["cnn"] = {
                "agent_name": "CNN_Classical", "prediction": 0,
                "confidence": 0.5, "label": "Hata",
                "details": {"error": str(_cnn_err), "model_type": "pytorch_cnn_v4"},
            }

        # RNN
        try:
            _rnn_r = _orchestrator.rnn_classical.predict(embedding, preprocessed)
            results["rnn"] = vars(_rnn_r)
            results["rnn"]["details"] = dict(results["rnn"].get("details", {}))
        except Exception as _rnn_err:
            logger.warning(f"RNN predict error (returning fallback): {_rnn_err}")
            results["rnn"] = {
                "agent_name": "RNN_Classical", "prediction": 0,
                "confidence": 0.5, "label": "Hata",
                "details": {"error": str(_rnn_err), "model_type": "pytorch_rnn"},
            }

        # Cosine similarity — returns ClassificationResult; scores live in .details
        sim = _orchestrator.cosine_analyzer.analyze(embedding, preprocessed)
        sim_details = sim.details if isinstance(sim.details, dict) else {}
        results["similarity"] = {
            "anomaly_score":    round(sim_details.get("anomaly_score",
                                     1.0 - sim_details.get("similarity_to_normal", 0.0)), 4),
            "similarity_score": round(sim_details.get("similarity_to_normal", 0.0), 4),
            "is_anomaly":       bool(sim.prediction == 1),
        }

        # LLM
        llm_r = _orchestrator.llm_analysis.analyze(preprocessed)
        results["llm"] = {
            "prediction":  llm_r.prediction,
            "confidence":  round(llm_r.confidence, 4),
            "label":       llm_r.label,
        }

        # Quantum AI
        qai_r = _orchestrator.quantum_ai.predict(preprocessed, embedding)
        results["quantum"] = {
            "prediction":  qai_r.prediction,
            "confidence":  round(qai_r.confidence, 4),
            "label":       qai_r.label,
            "details":     dict(qai_r.details) if qai_r.details else {},
        }

        # Fusion
        agent_results_dict = {
            "QuantumAI":        _orchestrator.quantum_ai.predict(preprocessed, embedding),
            "CosineSimilarity": _orchestrator.cosine_analyzer.analyze(embedding, preprocessed),
            "RNN_Classical":    _orchestrator.rnn_classical.predict(embedding, preprocessed),
            "CNN_Classical":    _orchestrator.cnn_classical.predict(embedding, preprocessed),
            "LLM_Analysis":     llm_r,
        }
        fusion = _orchestrator.fusion_layer.fuse(preprocessed.event_id, agent_results_dict)
        alert_list = [_orchestrator.detector.detect(fusion, preprocessed)] if fusion.is_intrusion else []

        fusion_out = {
            "final_score":   round(fusion.final_score, 4),
            "is_intrusion":  fusion.is_intrusion,
            "agent_scores":  {k: round(v, 4) for k, v in fusion.agent_scores.items()},
            "fusion_method": fusion.fusion_method,
        }

        alert_out = None
        if alert_list:
            a = alert_list[0]
            alert_out = _alert_to_dict(a)
            _orchestrator.alert_manager.receive_alerts(alert_list)
            _session["all_alerts"].insert(0, alert_out)
            _session["total_alerts"] += 1

        return jsonify({
            "ok":      True,
            "event_id": preprocessed.event_id,
            "agents":   results,
            "fusion":   fusion_out,
            "alert":    alert_out,
        })

    except Exception as exc:
        logger.error(f"analyze error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/alert/acknowledge", methods=["POST"])
def api_acknowledge():
    data = request.get_json(silent=True) or {}
    alert_id = data.get("alert_id", "")
    if _orchestrator:
        _orchestrator.alert_manager.acknowledge_alert(alert_id)
    for a in _session["all_alerts"]:
        if a["alert_id"] == alert_id:
            a["reviewed"] = True
            break
    return jsonify({"ok": True})


@app.route("/api/alert/resolve", methods=["POST"])
def api_resolve():
    data     = request.get_json(silent=True) or {}
    alert_id = data.get("alert_id", "")
    label    = data.get("label", "Confirmed")
    if _orchestrator:
        _orchestrator.alert_manager.resolve_alert(alert_id, label)
    for a in _session["all_alerts"]:
        if a["alert_id"] == alert_id:
            a["reviewed"]  = True
            a["feedback"]  = label
            break
    return jsonify({"ok": True})


@app.route("/api/metrics")
def api_metrics():
    """Aggregated metrics from session alerts for charts."""
    alerts = _session["all_alerts"]

    sev_counts  = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    type_counts = {}

    for a in alerts:
        s = a.get("severity", "LOW")
        sev_counts[s] = sev_counts.get(s, 0) + 1
        t = a.get("attack_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    # Timeline: last 20 alerts timestamps
    timeline = [
        {"ts": a["timestamp"][:19], "sev": a["severity"]}
        for a in alerts[:20]
    ]

    return jsonify({
        "severity_counts": sev_counts,
        "top_attack_types": [{"type": t, "count": c} for t, c in top_types],
        "timeline": timeline,
        "total_events": _session["total_events"],
        "total_alerts": _session["total_alerts"],
        "detection_rate": round(
            _session["total_alerts"] / max(_session["total_events"], 1) * 100, 1
        ),
    })


# ─────────────────────────────────────────────
# Batch prediction job store
# ─────────────────────────────────────────────

_batch_jobs: Dict[str, Any] = {}      # job_id → { status, rows, done, total, error }
_datalake_jobs: Dict[str, Any] = {}  # job_id → { status, done, total, added, skipped, error, type_counts }

# ─────────────────────────────────────────────
# Dataset-training state
# ─────────────────────────────────────────────

_train_job = {
    "running":  False,
    "stage":    "idle",      # idle | analyzing | training_cnn | training_rnn | saving | done | error
    "message":  "",
    "progress": 0,
    "rows":     0,
    "error":    None,
    "last_run": None,
    "metrics":  {},
    "schema":   {},          # dataset schema info from DatasetAnalyzer
}


# ─────────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────────

_META_COLS = {
    "source_ip", "src_ip",
    "destination_ip", "dst_ip",
    "source_port", "src_port",
    "destination_port", "dst_port",
    "protocol", "payload", "attack_type",
    "label", "class", "target",
}
_LABEL_NAMES = {
    "label", "class", "target", "attack_label", "is_attack",
    "attack", "attack_type", "category", "type", "y",
    "intrusion", "anomaly", "malicious",
}


def _parse_csv_bytes(raw: bytes):
    """
    Parse uploaded CSV bytes.

    Returns
    -------
    headers : list[str]
    rows    : list[dict]
    label_col : str | None   — detected label column name
    """
    import csv, io
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    rows = list(reader)

    label_col = None
    for h in headers:
        if h.strip().lower() in _LABEL_NAMES:
            label_col = h
            break

    return headers, rows, label_col


def _synthesize_payload(label: str, port: int, protocol: str) -> str:
    """
    Build a descriptive text payload from structured fields when the CSV
    payload column is empty.  The text is word-tokenised identically to real
    events so the word2vec embedding space stays consistent.
    """
    lbl = label.lower().strip()
    proto = protocol.lower() if protocol and protocol.lower() not in ("unknown", "") else ""

    # ── Port-based service hint ──────────────────────────────────────────
    _PORT_HINTS = {
        22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 80: "http",
        110: "pop3", 123: "ntp", 135: "rpc", 137: "netbios", 143: "imap",
        161: "snmp", 389: "ldap", 443: "https", 445: "smb", 3306: "mysql",
        3389: "rdp", 5432: "postgres", 6379: "redis", 8080: "http proxy",
        27017: "mongodb",
    }
    svc = _PORT_HINTS.get(port, f"port {port}")

    # ── Label → descriptive text map ────────────────────────────────────
    if "benign" in lbl or lbl in ("normal", "0", "legitimate", "safe", "clean"):
        severity = "LOW"
        desc = f"normal {svc} traffic ok benign low no anomaly"
    elif "ddos" in lbl or "dos" in lbl:
        severity = "CRITICAL"
        desc = f"attack dos ddos flood {svc} volumetric syn packets critical high"
    elif "portscan" in lbl or "port scan" in lbl or "reconnaissance" in lbl or "scan" in lbl:
        severity = "MEDIUM"
        desc = f"attack reconnaissance nmap port scan probe {svc} medium"
    elif "brute force" in lbl or "brute" in lbl:
        severity = "HIGH"
        desc = f"attack brute force {svc} login attempts repeated failed high intrusion"
    elif "sql injection" in lbl or "sqli" in lbl:
        severity = "CRITICAL"
        desc = f"attack sql injection exploit database {svc} critical web"
    elif "xss" in lbl or "cross site" in lbl:
        severity = "MEDIUM"
        desc = f"attack xss cross site scripting {svc} web exploit medium"
    elif "web attack" in lbl:
        severity = "HIGH"
        desc = f"attack web exploit {svc} http high intrusion"
    elif "botnet" in lbl:
        severity = "HIGH"
        desc = f"attack botnet c2 beacon {svc} malware high"
    elif "infiltration" in lbl:
        severity = "CRITICAL"
        desc = f"attack infiltration lateral movement {svc} critical"
    elif "heartbleed" in lbl:
        severity = "CRITICAL"
        desc = f"attack heartbleed ssl exploit {svc} critical overflow"
    else:
        severity = "MEDIUM"
        desc = f"alert {svc} {lbl} medium"

    parts = ["alert"]
    if proto:
        parts.append(proto)
    parts.append(f"dst port {port}")
    parts.append(f"severity {severity.lower()}")
    parts.append("payload")
    parts.append(desc)
    return " ".join(parts)


def _row_to_event(row: dict, idx: int, label_hint: str = ""):
    """Convert a CSV row dict → RawSecurityEvent.

    label_hint: if provided and payload is empty, synthesize a descriptive
    payload text so the word2vec embedding carries label-aware signal.
    """
    from utils.data_models import RawSecurityEvent

    def _get(*keys, default=""):
        for k in keys:
            for h, v in row.items():
                if h.strip().lower() == k:
                    return v
        return default

    def _int(*keys, default=0):
        v = _get(*keys, default=str(default))
        try:
            return int(float(v))
        except Exception:
            return default

    payload  = _get("payload", default="").strip()
    dst_port = _int("destination_port", "dst_port", "port", default=80)
    protocol = _get("protocol", default="TCP")

    if not payload and label_hint:
        payload = _synthesize_payload(label_hint, dst_port, protocol)

    return RawSecurityEvent(
        source_type=_get("source_type", default="IPS"),
        timestamp=datetime.now(),
        source_ip=_get("source_ip", "src_ip", default=f"10.0.0.{(idx % 254) + 1}"),
        destination_ip=_get("destination_ip", "dst_ip", default="192.168.1.1"),
        source_port=_int("source_port", "src_port", default=10000 + idx),
        destination_port=dst_port,
        protocol=protocol,
        payload=payload,
        severity=_get("severity", default="MEDIUM"),
        event_type="alert",
        raw_data={
            "event_id": f"batch-{idx}",
            "attack_type": _get("attack_type", default="Unknown"),
            "row_index": idx,
        },
    )


def _label_to_int(val: str) -> int:
    """Convert label string → binary int. 0=Normal, 1=Intrusion."""
    return _label_to_binary(val)


# ─────────────────────────────────────────────
# Route: Batch prediction
# ─────────────────────────────────────────────

@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """
    Upload a CSV file for bulk IDS prediction.
    Multipart form field: file=<csv>
    Returns: { job_id, total }
    """
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready yet"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No CSV file provided"}), 400

    raw = request.files["file"].read()
    try:
        headers, rows, _ = _parse_csv_bytes(raw)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    if not rows:
        return jsonify({"error": "CSV file is empty"}), 400

    rows = rows[:1000]   # max 1000 rows

    import uuid
    job_id = str(uuid.uuid4())[:8]
    _batch_jobs[job_id] = {
        "status":   "running",
        "done":     0,
        "total":    len(rows),
        "results":  [],
        "error":    None,
        "headers":  headers,
    }

    def _run(jid, row_list):
        import numpy as np
        import torch
        job = _batch_jobs[jid]

        # Decide if direct-mode inference is possible for this batch
        cnn_model = _orchestrator.cnn_classical
        rnn_model = _orchestrator.rnn_classical
        use_direct = (
            cnn_model.is_trained
            and getattr(cnn_model, "training_mode", "embedding") == "direct"
            and cnn_model.feature_columns
            and cnn_model.feature_mean is not None
        )
        if use_direct:
            avail_cols = list((row_list[0] if row_list else {}).keys())
            da = DatasetAnalyzer()
            da.feature_columns = cnn_model.feature_columns
            da.feature_mean    = cnn_model.feature_mean
            da.feature_std     = cnn_model.feature_std
            overlap = da.columns_overlap(avail_cols)
            if overlap < 0.5:
                use_direct = False
                logger.info(
                    f"Batch predict: direct mode skipped "
                    f"(column overlap={overlap:.0%} < 50%)"
                )
            else:
                logger.info(
                    f"Batch predict: direct mode "
                    f"(column overlap={overlap:.0%}, {len(cnn_model.feature_columns)} features)"
                )

        try:
            for idx, row in enumerate(row_list):
                event        = _row_to_event(row, idx)
                preprocessed = _orchestrator.preprocessor.process(event)
                embedding    = _orchestrator.embedding_engine.create_embedding(preprocessed)
                _orchestrator.vector_warehouse.store(embedding)

                # CNN + RNN — direct mode or embedding mode
                if use_direct:
                    X_row = da.transform([row])  # (1, F)
                    x_t   = torch.tensor(X_row, dtype=torch.float32)
                    cnn_model.net.eval()
                    rnn_model.net.eval()
                    with torch.no_grad():
                        cnn_prob = float(cnn_model.net(x_t).item())
                        rnn_prob = float(rnn_model.net(x_t).item())

                    cnn_pred = int(cnn_prob > 0.5)
                    rnn_pred = int(rnn_prob > 0.5)
                    from utils.data_models import ClassificationResult
                    cnn_r = ClassificationResult(
                        agent_name="CNN_Classical",
                        event_id=preprocessed.event_id,
                        prediction=cnn_pred,
                        confidence=cnn_prob if cnn_pred else 1.0 - cnn_prob,
                        label=preprocessed.metadata.get("attack_type", "Intrusion") if cnn_pred else "Normal",
                        details={"cnn_probability": cnn_prob, "mode": "direct"},
                    )
                    rnn_r = ClassificationResult(
                        agent_name="RNN_Classical",
                        event_id=preprocessed.event_id,
                        prediction=rnn_pred,
                        confidence=rnn_prob if rnn_pred else 1.0 - rnn_prob,
                        label=preprocessed.metadata.get("attack_type", "Intrusion") if rnn_pred else "Normal",
                        details={"rnn_probability": rnn_prob, "mode": "direct"},
                    )
                else:
                    cnn_r = _orchestrator.cnn_classical.predict(embedding, preprocessed)
                    rnn_r = _orchestrator.rnn_classical.predict(embedding, preprocessed)

                # Cosine
                sim_r      = _orchestrator.cosine_analyzer.analyze(embedding, preprocessed)
                sim_det    = sim_r.details if isinstance(sim_r.details, dict) else {}
                sim_score  = round(sim_det.get("anomaly_score",
                                   1.0 - sim_det.get("similarity_to_normal", 0.0)), 4)

                # LLM
                llm_r = _orchestrator.llm_analysis.analyze(preprocessed)

                # Quantum
                qai_r = _orchestrator.quantum_ai.predict(preprocessed, embedding)

                # Fusion
                fusion = _orchestrator.fusion_layer.fuse(
                    preprocessed.event_id,
                    {
                        "QuantumAI":        qai_r,
                        "CosineSimilarity": sim_r,
                        "RNN_Classical":    rnn_r,
                        "CNN_Classical":    cnn_r,
                        "LLM_Analysis":     llm_r,
                    },
                )

                alert_id = None
                if fusion.is_intrusion:
                    alerts = _orchestrator.detector.detect(fusion, preprocessed)
                    if alerts:
                        alert_out = _alert_to_dict(alerts)
                        _orchestrator.alert_manager.receive_alerts([alerts])
                        _session["all_alerts"].insert(0, alert_out)
                        _session["total_alerts"] += 1
                        alert_id = alert_out["alert_id"]

                _session["total_events"] += 1

                job["results"].append({
                    "row":         idx + 1,
                    "event_id":    preprocessed.event_id,
                    "is_intrusion": fusion.is_intrusion,
                    "score":       round(fusion.final_score, 4),
                    "label":       cnn_r.label if fusion.is_intrusion else "Normal",
                    "cnn":         round(cnn_r.confidence, 3),
                    "rnn":         round(rnn_r.confidence, 3),
                    "sim":         round(sim_score, 3),
                    "llm":         round(llm_r.confidence, 3),
                    "quantum":     round(qai_r.confidence, 3),
                    "alert_id":    alert_id,
                })
                job["done"] = idx + 1

            job["status"] = "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"]  = str(exc)
            logger.error(f"Batch predict error: {exc}", exc_info=True)

    threading.Thread(target=_run, args=(job_id, rows), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "total": len(rows)})


@app.route("/api/batch-status/<job_id>")
def api_batch_status(job_id):
    """Poll batch prediction job status."""
    job = _batch_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":  job["status"],
        "done":    job["done"],
        "total":   job["total"],
        "results": job["results"],
        "error":   job.get("error"),
    })


# ─────────────────────────────────────────────
# Route: Dataset training
# ─────────────────────────────────────────────

@app.route("/api/dataset-train", methods=["POST"])
def api_dataset_train():
    """
    Upload a labeled CSV to retrain CNN and RNN models.

    The endpoint auto-detects the dataset format (KDD Cup, CICIDS, generic numeric…),
    extracts all numeric and categorical features directly from the CSV, normalises them,
    and trains CNN + RNN in "direct" mode so the full information content of the dataset
    is used — not just the 24 handcrafted IPS/firewall features.

    Required: at least one column named label / class / target / attack / y (etc.)
    Returns: { ok, message }
    """
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready yet"}), 503

    if _train_job["running"]:
        return jsonify({"error": "Training already in progress"}), 429

    if "file" not in request.files:
        return jsonify({"error": "No CSV file provided"}), 400

    raw = request.files["file"].read()
    try:
        headers, rows, label_col = _parse_csv_bytes(raw)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    if not label_col:
        return jsonify({
            "error": (
                "Label column not found. "
                "CSV must contain one of: "
                "label, class, target, attack, attack_type, category, y, intrusion, anomaly."
            )
        }), 400

    if len(rows) < 10:
        return jsonify({"error": "At least 10 rows required"}), 400

    rows = rows[:200000]

    def _run():
        import numpy as np
        import torch
        _train_job.update({
            "running":  True,
            "stage":    "analyzing",
            "message":  "Analyzing dataset structure…",
            "progress": 5,
            "rows":     len(rows),
            "error":    None,
            "metrics":  {},
            "schema":   {},
        })
        try:
            # ── Step 1: Schema analysis ─────────────────────────────────
            analyzer = DatasetAnalyzer()
            schema   = analyzer.analyze(headers, rows, label_col=label_col)
            _train_job["schema"] = {
                "format":           schema["format"],
                "n_numeric_cols":   schema["n_numeric_features"],
                "n_categorical_cols": len(schema["categorical_cols"]),
                "label_distribution": schema["label_distribution"],
            }

            n_numeric = schema["n_numeric_features"]
            logger.info(
                f"Dataset format: {schema['format']} | "
                f"{n_numeric} numeric cols | {len(rows)} rows"
            )

            # ── Step 2: Feature matrix ──────────────────────────────────
            _train_job.update({
                "stage":   "extracting",
                "message": f"Building feature matrix ({n_numeric} numeric columns)…",
                "progress": 20,
            })

            X, y, feat_info = analyzer.build_feature_matrix(
                rows, label_col, use_categorical=True, max_features=128
            )

            n_attack = int(y.sum())
            n_normal = len(y) - n_attack
            n_feat   = X.shape[1]

            logger.info(
                f"Feature matrix: shape={X.shape} | "
                f"normal={n_normal} | attack={n_attack}"
            )

            _train_job.update({
                "stage":   "training_cnn",
                "message": f"Training CNN… ({n_feat} features, {len(X)} samples)",
                "progress": 40,
            })

            # ── Step 3: Train CNN (direct mode) ────────────────────────
            cnn = _orchestrator.cnn_classical
            cnn.train(X, y)
            cnn.training_mode   = "direct"
            cnn.feature_columns = feat_info["feature_columns"]
            cnn.feature_mean    = np.array(feat_info["feature_mean"], dtype=np.float32)
            cnn.feature_std     = np.array(feat_info["feature_std"],  dtype=np.float32)
            cnn.save_model()

            _train_job.update({
                "stage":   "training_rnn",
                "message": f"Training RNN… ({n_feat} features, {len(X)} samples)",
                "progress": 65,
            })

            # ── Step 4: Train RNN (direct mode) ────────────────────────
            rnn = _orchestrator.rnn_classical
            rnn.train(X, y)
            rnn.training_mode   = "direct"
            rnn.feature_columns = feat_info["feature_columns"]
            rnn.feature_mean    = np.array(feat_info["feature_mean"], dtype=np.float32)
            rnn.feature_std     = np.array(feat_info["feature_std"],  dtype=np.float32)
            rnn.save_model()

            # ── Step 5: Quick accuracy eval ─────────────────────────────
            _train_job.update({
                "stage":   "saving",
                "message": "Computing accuracy and saving checkpoints…",
                "progress": 90,
            })

            cnn.net.eval()
            cnn_preds = []
            with torch.no_grad():
                for i in range(0, len(X), 64):
                    bx   = torch.tensor(X[i:i+64], dtype=torch.float32)
                    out  = cnn.net(bx).cpu().numpy()
                    cnn_preds.extend((out > 0.5).astype(int).tolist())

            rnn.net.eval()
            rnn_preds = []
            with torch.no_grad():
                for i in range(0, len(X), 64):
                    bx   = torch.tensor(X[i:i+64], dtype=torch.float32)
                    out  = rnn.net(bx).cpu().numpy()
                    rnn_preds.extend((out > 0.5).astype(int).tolist())

            labels_int = y.astype(int).tolist()
            cnn_acc = sum(int(p) == l for p, l in zip(cnn_preds, labels_int)) / len(labels_int)
            rnn_acc = sum(int(p) == l for p, l in zip(rnn_preds, labels_int)) / len(labels_int)

            # ── Step 6: Retrain QNN on dataset (SPSA, ~30 iters, <1 s) ──
            qnn_acc = None
            try:
                _train_job.update({
                    "stage":   "training_qnn",
                    "message": f"Training QNN (Quantum AI)… ({len(X)} samples, SPSA)",
                    "progress": 82,
                })
                qai = _orchestrator.quantum_ai
                qai.train(X, y)           # refits reducer + SPSA + sets train_accuracy
                qai.dataset_accuracy = qai.train_accuracy  # same dataset
                qai.save_model()
                qnn_acc = qai.train_accuracy / 100.0
                logger.info(f"QNN retrained on dataset | accuracy={qai.train_accuracy:.1f}%")
            except Exception as qe:
                logger.warning(f"QNN training skipped: {qe}")

            _train_job.update({
                "stage":    "done",
                "message":  (
                    f"Training complete! "
                    f"CNN: {cnn_acc*100:.1f}%  RNN: {rnn_acc*100:.1f}%  "
                    + (f"QNN: {qnn_acc*100:.1f}%  " if qnn_acc is not None else "")
                    + f"({n_feat} features, format: {schema['format']})"
                ),
                "progress": 100,
                "metrics": {
                    "samples":        len(X),
                    "normal":         n_normal,
                    "attack":         n_attack,
                    "n_features":     n_feat,
                    "dataset_format": schema["format"],
                    "cnn_accuracy":   round(cnn_acc * 100, 1),
                    "rnn_accuracy":   round(rnn_acc * 100, 1),
                    "qnn_accuracy":   round(qnn_acc * 100, 1) if qnn_acc is not None else None,
                    "training_mode":  "direct",
                },
            })
            logger.info(
                f"Dataset training done | CNN acc={cnn_acc:.4f} | "
                f"RNN acc={rnn_acc:.4f} | "
                + (f"QNN acc={qnn_acc:.4f} | " if qnn_acc is not None else "")
                + f"features={n_feat} | format={schema['format']}"
            )

        except Exception as exc:
            _train_job.update({
                "stage": "error", "message": str(exc),
                "progress": 0, "error": str(exc),
            })
            logger.error(f"Dataset training error: {exc}", exc_info=True)
        finally:
            _train_job["running"]  = False
            _train_job["last_run"] = datetime.now().isoformat()
            if _train_job["stage"] == "done":
                _save_train_metrics()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": f"{len(rows)} rows queued for training"})


@app.route("/api/train-status")
def api_train_status():
    """Return dataset training job status."""
    return jsonify(_train_job.copy())


@app.route("/api/quantum-metrics")
def api_quantum_metrics():
    """Return Quantum AI model metadata from its checkpoint."""
    meta_path = os.path.join(PROJECT_ROOT, "models", "quantum", "quantum_ai_metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"available": False})
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        return jsonify(data)
    except Exception as e:
        logger.warning(f"Quantum metrics read error: {e}")
        return jsonify({"available": False, "error": str(e)})


def _parse_upload_to_rows(raw: bytes, filename: str):
    """
    Parse an uploaded file (CSV / TXT / JSON) into a list of dicts.
    Returns (rows, label_col, error_str).

    CSV : standard comma-separated with header row containing a label column.
    TXT : tab-separated with header  OR  one label per line (plain list).
    JSON: array of objects  OR  array of label strings.
    """
    import csv, io, json as _json

    text = raw.decode("utf-8", errors="replace").strip()
    ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    LABEL_COL_NAMES = {"label", "attack_type", "class", "category", "type"}

    # ── JSON ──────────────────────────────────────────────────────────────────
    if ext == "json":
        try:
            data = _json.loads(text)
        except Exception as je:
            return None, None, f"JSON parse error: {je}"

        if isinstance(data, list):
            if not data:
                return None, None, "JSON array is empty"
            # Array of strings → treat each as a label
            if isinstance(data[0], str):
                rows = [{"label": v} for v in data]
                return rows, "label", None
            # Array of objects
            if isinstance(data[0], dict):
                rows = [dict(item) for item in data]
                headers = list(rows[0].keys()) if rows else []
                label_col = next((h for h in headers if h.strip().lower() in LABEL_COL_NAMES), None)
                if label_col is None:
                    return None, None, (
                        "Label key not found in JSON objects. "
                        "Supported keys: label, attack_type, class, category, type"
                    )
                return rows, label_col, None
        return None, None, "JSON format not supported (array expected)"

    # ── TXT ───────────────────────────────────────────────────────────────────
    if ext == "txt":
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return None, None, "TXT file is empty"

        # Try tab-separated with header
        if "\t" in lines[0]:
            reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
            rows = list(reader)
            headers = list(reader.fieldnames or [])
            label_col = next((h for h in headers if h.strip().lower() in LABEL_COL_NAMES), None)
            if label_col and rows:
                return rows, label_col, None

        # Plain list: one label per line (skip first line if it looks like a header)
        start = 1 if lines[0].strip().lower() in LABEL_COL_NAMES else 0
        rows = [{"label": l.strip()} for l in lines[start:] if l.strip()]
        if not rows:
            return None, None, "No valid rows found in TXT file"
        return rows, "label", None

    # ── CSV (default) ─────────────────────────────────────────────────────────
    reader = csv.DictReader(io.StringIO(text))
    headers = list(reader.fieldnames or [])
    rows    = list(reader)
    if not rows:
        return None, None, "CSV is empty or could not be read"
    label_col = next((h for h in headers if h.strip().lower() in LABEL_COL_NAMES), None)
    if label_col is None:
        return None, None, (
            "Label column not found in CSV. "
            "Supported column names: label, attack_type, class, category, type"
        )
    return rows, label_col, None


@app.route("/api/datalake/upload", methods=["POST"])
def api_datalake_upload():
    """
    Async upload: parse CSV/TXT/JSON immediately, return job_id, process in background.
    Max rows: 5 000.  Poll /api/datalake/upload-status/<job_id> for progress.
    """
    if _orchestrator is None:
        return jsonify({"success": False, "error": "System not ready yet"}), 503

    f        = request.files.get("file")
    filename = f.filename if f and f.filename else "upload.csv"
    raw      = f.read() if f else request.get_data()
    if not raw:
        return jsonify({"success": False, "error": "No file provided"}), 400

    rows, label_col, parse_err = _parse_upload_to_rows(raw, filename)
    if parse_err:
        return jsonify({"success": False, "error": parse_err}), 400

    rows = rows[:200000]
    if not rows:
        return jsonify({"success": False, "error": "No valid rows found in file"}), 400

    import uuid
    job_id = str(uuid.uuid4())[:8]
    _datalake_jobs[job_id] = {
        "status":      "running",
        "done":        0,
        "total":       len(rows),
        "added":       0,
        "skipped":     0,
        "type_counts": {},
        "total_now":   0,
        "error":       None,
    }

    def _run(jid, row_list, lbl_col):
        import numpy as np
        from collections import Counter
        job = _datalake_jobs[jid]
        NORMAL_LABELS = {"normal", "0", "benign", "legitimate", "safe", "clean"}
        dl = _orchestrator.data_lake
        added   = 0
        skipped = 0
        labels_added = []
        CHUNK = 100

        try:
            for chunk_start in range(0, len(row_list), CHUNK):
                chunk = row_list[chunk_start:chunk_start + CHUNK]

                # Build events and preprocess as a batch
                events = []
                metas  = []
                for i, row in enumerate(chunk):
                    idx = chunk_start + i
                    label_raw = next(
                        (v.strip() for h, v in row.items()
                         if h.strip().lower() == lbl_col.strip().lower()),
                        ""
                    )
                    if not label_raw:
                        skipped += 1
                        metas.append(None)
                        events.append(None)
                        continue
                    ev = _row_to_event(row, idx, label_hint=label_raw)
                    ev.raw_data = ev.raw_data or {}
                    ev.raw_data["attack_type"] = label_raw
                    events.append(ev)
                    metas.append(label_raw)

                # Filter out skipped
                valid_pairs = [(ev, lbl) for ev, lbl in zip(events, metas) if ev is not None]
                if valid_pairs:
                    valid_evs  = [p[0] for p in valid_pairs]
                    valid_lbls = [p[1] for p in valid_pairs]

                    preprocessed_batch = _orchestrator.preprocessor.process_batch(valid_evs)
                    emb_batch          = _orchestrator.embedding_engine.create_batch_embeddings(preprocessed_batch)

                    for k, (emb, label) in enumerate(zip(emb_batch, valid_lbls)):
                        binary       = 0 if label.lower() in NORMAL_LABELS else 1
                        emb.event_id = f"upload_{chunk_start + k}_{label.replace(' ','_')[:20]}"
                        dl.embeddings.append(emb)
                        dl.labels.append(label)
                        dl.binary_labels.append(binary)
                        labels_added.append(label)
                        added += 1

                job["done"]    = chunk_start + len(chunk)
                job["added"]   = added
                job["skipped"] = skipped

            # Persist and refresh CNN centroids
            if added > 0:
                from utils.model_persistence import save_data_lake
                save_data_lake(dl)
                cnn = _orchestrator.cnn_classical
                if hasattr(cnn, "_compute_centroids") and cnn.is_trained:
                    X_new, y_new = dl.get_training_data()
                    cnn.train_X = X_new
                    cnn.train_y = y_new.astype(np.float32)
                    cnn._compute_centroids()
                    logger.info("CNN centroids refreshed after data lake upload")

            job["status"]      = "done"
            job["type_counts"] = dict(Counter(labels_added))
            job["total_now"]   = dl.size()
            logger.info(f"Data Lake async upload complete: {added} added, {skipped} skipped")

        except Exception as exc:
            job["status"] = "error"
            job["error"]  = str(exc)
            logger.error(f"Data Lake async upload error: {exc}", exc_info=True)

    threading.Thread(target=_run, args=(job_id, rows, label_col), daemon=True).start()
    return jsonify({"success": True, "job_id": job_id, "total": len(rows)})


@app.route("/api/datalake/upload-status/<job_id>")
def api_datalake_upload_status(job_id):
    """Poll async Data Lake upload job status."""
    job = _datalake_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":      job["status"],
        "done":        job["done"],
        "total":       job["total"],
        "added":       job["added"],
        "skipped":     job["skipped"],
        "type_counts": job["type_counts"],
        "total_now":   job["total_now"],
        "error":       job["error"],
    })


@app.route("/api/datalake")
def api_datalake():
    """Return the contents of the Reference Embedding Data Lake."""
    try:
        if _orchestrator is None:
            return jsonify({"available": False, "reason": "System not ready yet"})

        dl = _orchestrator.data_lake
        if dl.size() == 0:
            return jsonify({
                "available": True,
                "total": 0,
                "attack_count": 0,
                "normal_count": 0,
                "unique_attack_types": 0,
                "attack_type_dist": {},
                "entries": [],
            })

        # Attack-type distribution
        from collections import Counter
        type_dist = dict(Counter(dl.labels))

        # Entry list (capped at 500 rows for UI performance)
        import numpy as np
        entries = []
        for i, (emb, lbl, bin_lbl) in enumerate(
                zip(dl.embeddings, dl.labels, dl.binary_labels)):
            v_norm = float(np.linalg.norm(emb.vector))
            entries.append({
                "index":      i,
                "event_id":   emb.event_id,
                "label":      lbl,
                "binary":     int(bin_lbl),
                "vector_norm": round(v_norm, 4),
            })
            if i >= 499:
                break

        attack_count = int(sum(dl.binary_labels))
        return jsonify({
            "available":          True,
            "total":              dl.size(),
            "attack_count":       attack_count,
            "normal_count":       dl.size() - attack_count,
            "unique_attack_types": len([k for k in type_dist if k != "Normal"]),
            "attack_type_dist":   type_dist,
            "entries":            entries,
        })
    except Exception as e:
        logger.error(f"api_datalake error: {e}")
        return jsonify({"available": False, "error": str(e)}), 500


@app.route("/api/datalake/sample-csv")
def api_datalake_sample_csv():
    """Return a downloadable sample CSV with the correct column structure."""
    sample = (
        "label,payload,source_ip,destination_ip,protocol,port\n"
        "Normal,GET /index.html HTTP/1.1,192.168.1.10,10.0.0.1,TCP,80\n"
        "Normal,DNS query google.com,192.168.1.11,8.8.8.8,UDP,53\n"
        "DoS,SYN flood burst x1000,10.10.10.5,192.168.1.1,TCP,443\n"
        "DoS,UDP flood high-rate,10.10.10.6,192.168.1.2,UDP,0\n"
        "Exploit,Shellcode buffer overflow attempt,172.16.0.99,192.168.1.5,TCP,22\n"
        "Exploit,SQL injection ' OR 1=1 --,172.16.0.100,192.168.1.6,TCP,3306\n"
        "PortScan,SYN scan 0-65535,10.0.0.200,192.168.1.1,TCP,0\n"
        "Botnet,C2 beacon check-in,192.168.50.5,45.33.32.156,TCP,8080\n"
        "Normal,HTTPS POST /api/data,192.168.1.20,10.0.0.2,TCP,443\n"
        "Brute,SSH login attempt admin:password123,10.10.0.9,192.168.1.3,TCP,22\n"
    )
    from flask import Response
    return Response(
        sample,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=ornek_veri.csv"}
    )


@app.route("/api/datalake/export")
def api_datalake_export():
    """
    Export the full Reference Embedding Data Lake as a downloadable CSV file.
    Columns: index, event_id, label, binary_label, vector_norm, dim_0..dim_N-1
    """
    try:
        if _orchestrator is None:
            return jsonify({"error": "System not ready yet"}), 503

        import io, csv
        import numpy as np
        dl = _orchestrator.data_lake

        if dl.size() == 0:
            return jsonify({"error": "Data Lake is empty"}), 404

        output = io.StringIO()
        first_vec = dl.embeddings[0].vector
        dim = len(first_vec)
        dim_headers = [f"dim_{i}" for i in range(dim)]

        writer = csv.writer(output)
        writer.writerow(["index", "event_id", "label", "binary_label", "vector_norm"] + dim_headers)

        for i, (emb, lbl, bin_lbl) in enumerate(
                zip(dl.embeddings, dl.labels, dl.binary_labels)):
            v = emb.vector
            v_norm = round(float(np.linalg.norm(v)), 6)
            writer.writerow(
                [i, emb.event_id, lbl, int(bin_lbl), v_norm] + [round(float(x), 8) for x in v]
            )

        csv_bytes = output.getvalue().encode("utf-8")
        from flask import Response
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=data_lake_{dl.size()}records.csv",
                "Content-Length": str(len(csv_bytes)),
            },
        )
    except Exception as e:
        logger.error(f"api_datalake_export error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/datalake/reset", methods=["POST"])
def api_datalake_reset():
    """Clear all Data Lake entries and restore the built-in baseline patterns."""
    try:
        if _orchestrator is None:
            return jsonify({"success": False, "error": "System not ready yet"}), 503

        dl = _orchestrator.data_lake
        prev_total = dl.size()

        # Wipe in-memory data completely and persist empty state.
        dl.clear()
        from utils.model_persistence import save_data_lake
        save_data_lake(dl)

        logger.info(f"Data Lake reset: {prev_total} records deleted, Data Lake cleared")
        return jsonify({
            "success":    True,
            "prev_total": prev_total,
            "new_total":  0,
            "message":    f"{prev_total} records deleted · Data Lake cleared",
        })
    except Exception as e:
        logger.error(f"api_datalake_reset error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
# Architecture Configuration API
# ─────────────────────────────────────────────

_ARCH_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "arch_config.json")


def _load_arch_config() -> dict:
    """Load user-saved architecture config from disk, merged with settings defaults.

    Always merging with defaults ensures newly-added fields (e.g. n_conv_layers)
    appear even when an older arch_config.json predates them.
    """
    from config.settings import CLASSICAL_ML, QUANTUM_AI
    defaults = {
        "cnn": {**CLASSICAL_ML["cnn"]},
        "rnn": {**CLASSICAL_ML["rnn"]},
        "quantum": {
            "n_qubits":        QUANTUM_AI["n_qubits"],
            "n_hidden_layers": QUANTUM_AI["n_hidden_layers"],
            "max_iterations":  QUANTUM_AI["max_iterations"],
            "learning_rate":   QUANTUM_AI["learning_rate"],
            "perturbation":    QUANTUM_AI["perturbation"],
        },
    }
    if os.path.exists(_ARCH_CONFIG_PATH):
        try:
            with open(_ARCH_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge: saved values override defaults; new default keys fill gaps
            for section in ("cnn", "rnn", "quantum"):
                if section in saved:
                    defaults[section].update(saved[section])
        except Exception as e:
            logger.warning(f"arch_config.json read error: {e}")
    return defaults


def _apply_arch_config(cfg: dict):
    """Apply architecture config dict to live settings modules (in-memory)."""
    import config.settings as S
    if "cnn" in cfg:
        for k, v in cfg["cnn"].items():
            S.CLASSICAL_ML["cnn"][k] = v
    if "rnn" in cfg:
        for k, v in cfg["rnn"].items():
            S.CLASSICAL_ML["rnn"][k] = v
    if "quantum" in cfg:
        q = cfg["quantum"]
        for key in ("n_qubits", "n_hidden_layers", "max_iterations", "learning_rate", "perturbation"):
            if key in q:
                S.QUANTUM_AI[key] = q[key]
        if "n_hidden_layers" in q:
            S.QUANTUM_AI["n_layers"] = q["n_hidden_layers"]
    cc = cfg.get("cnn", {})
    logger.info(
        f"Architecture settings applied | "
        f"CNN layers={cc.get('n_conv_layers','?')} conv=[{cc.get('conv1_out','?')}→{cc.get('conv2_out','?')}] fc={cc.get('fc_hidden','?')} | "
        f"RNN hidden={cfg.get('rnn',{}).get('hidden_size','?')} layers={cfg.get('rnn',{}).get('num_layers','?')} | "
        f"Q qubits={cfg.get('quantum',{}).get('n_qubits','?')} layers={cfg.get('quantum',{}).get('n_hidden_layers','?')}"
    )


# ─────────────────────────────────────────────
# Auto Tuning — search grids
# ─────────────────────────────────────────────

_CNN_TUNE_GRID = [
    {"n_conv_layers": 1, "conv1_out":  8, "conv2_out": 16, "fc_hidden":  32, "dropout": 0.0, "kernel_size": 3, "_label": "1L · 8 filtre · FC32"},
    {"n_conv_layers": 1, "conv1_out": 16, "conv2_out": 32, "fc_hidden":  64, "dropout": 0.0, "kernel_size": 3, "_label": "1L · 16 filtre · FC64"},
    {"n_conv_layers": 2, "conv1_out":  8, "conv2_out": 16, "fc_hidden":  32, "dropout": 0.0, "kernel_size": 3, "_label": "2L · 8/16 · FC32"},
    {"n_conv_layers": 2, "conv1_out": 16, "conv2_out": 32, "fc_hidden":  64, "dropout": 0.0, "kernel_size": 3, "_label": "2L · 16/32 · FC64"},
    {"n_conv_layers": 2, "conv1_out": 16, "conv2_out": 32, "fc_hidden":  64, "dropout": 0.2, "kernel_size": 3, "_label": "2L · 16/32 · FC64 · dp0.2"},
    {"n_conv_layers": 2, "conv1_out": 32, "conv2_out": 64, "fc_hidden": 128, "dropout": 0.2, "kernel_size": 3, "_label": "2L · 32/64 · FC128"},
    {"n_conv_layers": 3, "conv1_out": 16, "conv2_out": 32, "fc_hidden":  64, "dropout": 0.0, "kernel_size": 3, "_label": "3L · 16/32 · FC64"},
    {"n_conv_layers": 3, "conv1_out": 16, "conv2_out": 32, "fc_hidden": 128, "dropout": 0.2, "kernel_size": 3, "_label": "3L · 16/32 · FC128"},
    {"n_conv_layers": 3, "conv1_out": 32, "conv2_out": 64, "fc_hidden": 128, "dropout": 0.3, "kernel_size": 3, "_label": "3L · 32/64 · FC128 · dp0.3"},
]

_RNN_TUNE_GRID = [
    {"hidden_size":  64, "num_layers": 1, "dropout": 0.0,  "_label": "h64 · 1 layer"},
    {"hidden_size": 128, "num_layers": 1, "dropout": 0.0,  "_label": "h128 · 1 layer"},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.2,  "_label": "h128 · 2 layers · dp0.2"},
    {"hidden_size": 256, "num_layers": 2, "dropout": 0.25, "_label": "h256 · 2 layers · dp0.25"},
    {"hidden_size": 256, "num_layers": 3, "dropout": 0.3,  "_label": "h256 · 3 layers · dp0.3"},
    {"hidden_size": 512, "num_layers": 2, "dropout": 0.3,  "_label": "h512 · 2 layers · dp0.3"},
]

_QUANTUM_TUNE_GRID = [
    {"n_qubits": 4, "n_hidden_layers": 2, "max_iterations":  80, "_label": "4 qubits · 2 layers"},
    {"n_qubits": 6, "n_hidden_layers": 3, "max_iterations":  80, "_label": "6 qubits · 3 layers"},
    {"n_qubits": 8, "n_hidden_layers": 4, "max_iterations": 100, "_label": "8 qubits · 4 layers"},
    {"n_qubits": 8, "n_hidden_layers": 6, "max_iterations": 100, "_label": "8 qubits · 6 layers"},
]

_TUNE_EPOCHS_FAST = 50    # trial epochs during grid search
_TUNE_EPOCHS_FULL = 150   # full epoch count for final retrain

# Overfitting guard: if (train_acc − val_acc) exceeds this threshold the trial
# is excluded from best-model selection.  Expressed as a fraction (0–1).
_OVERFIT_GAP_THRESHOLD = 0.40   # 40 percentage-point gap

_tune_custom_config: dict = {}   # populated by /api/tune/start from user input
_tune_cancel: bool = False        # set to True by /api/tune/cancel


def _build_grid_from_ranges(ranges: dict) -> list:
    """Build a trial grid (Cartesian product) from {param: [v1, v2, ...]} dict.
    Each entry gets a human-readable ``_label`` key.
    """
    import itertools
    keys   = [k for k in ranges if k and isinstance(ranges[k], list) and ranges[k]]
    values = [ranges[k] for k in keys]
    if not keys:
        return []
    grid = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg["_label"] = " · ".join(f"{k}={v}" for k, v in cfg.items())
        grid.append(cfg)
    return grid

_tune_status = {
    "running": False, "done": False, "error": None,
    "progress": 0, "total_trials": 0, "completed_trials": 0,
    "phase": "",
    "message": "",
    "trials": [],
    "best": {"cnn": None, "rnn": None, "quantum": None},
    "saved": False,
}


def _run_autotuning():
    """Grid-search over CNN/RNN/Quantum configs; pick best by 20% val accuracy."""
    global _tune_cancel
    import torch as _t
    import numpy as _np
    ts = _tune_status
    _tune_cancel = False
    try:
        ts.update({
            "running": True, "done": False, "error": None, "cancelled": False,
            "trials": [], "best": {"cnn": None, "rnn": None, "quantum": None},
            "progress": 0, "completed_trials": 0,
            "phase": "start", "message": "Scanning Data Lake…",
            "saved": False,
        })

        if _orchestrator is None:
            raise RuntimeError("Orchestrator not initialized — start the system first")

        X_all, y_all = _orchestrator.data_lake.get_training_data()
        n = len(X_all)
        if n < 10:
            raise ValueError(f"Data Lake has too few samples ({n}, min 10 required)")

        rng = _np.random.RandomState(42)
        idx = rng.permutation(n)
        n_train = int(n * 0.8)
        X_tr, X_val = X_all[idx[:n_train]], X_all[idx[n_train:]]
        y_tr, y_val = y_all[idx[:n_train]], y_all[idx[n_train:]]

        # Use user-defined grids and epochs if provided via /api/tune/start
        _cnn_grid       = _tune_custom_config.get("cnn_grid",          _CNN_TUNE_GRID)
        _rnn_grid       = _tune_custom_config.get("rnn_grid",          _RNN_TUNE_GRID)
        _q_grid         = _tune_custom_config.get("q_grid",            _QUANTUM_TUNE_GRID)
        _tune_fast      = _tune_custom_config.get("epochs_fast",       _TUNE_EPOCHS_FAST)
        _tune_full      = _tune_custom_config.get("epochs_full",       _TUNE_EPOCHS_FULL)
        _tune_skip_full = _tune_custom_config.get("skip_full_retrain", False)

        total = len(_cnn_grid) + len(_rnn_grid) + len(_q_grid)
        ts["total_trials"] = total
        done_n = [0]

        def _prog():
            return max(1, int(100 * done_n[0] / total))

        def _batch_val(net, Xv, yv):
            net.eval()
            with _t.no_grad():
                logits = net(_t.tensor(Xv, dtype=_t.float32))
                preds  = (_t.sigmoid(logits) >= 0.5).numpy().astype(int)
            return float((preds == yv).mean())

        # ── Phase 1: CNN ──────────────────────────────────────
        ts.update({"phase": "cnn", "message": "Trying CNN configurations…"})
        best_cnn_acc = -1.0
        best_cnn_params = None
        _cnn_fallback_gap, _cnn_fallback_params, _cnn_fallback_acc = 999.0, None, -1.0

        for i, cfg in enumerate(_cnn_grid):
            label = cfg["_label"]
            clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
            ts.update({"message": f"CNN [{i+1}/{len(_cnn_grid)}] → {label}", "progress": _prog()})

            from agents.classical_ml.cnn_model import ClassicalCNNModel
            m = ClassicalCNNModel()
            for k, v in clean.items():
                setattr(m, k, v)
            m.epochs = _tune_fast
            m.train(X_tr, y_tr)
            train_acc = _batch_val(m.net, X_tr, y_tr)
            val_acc   = _batch_val(m.net, X_val, y_val)
            gap       = train_acc - val_acc
            overfit   = gap > _OVERFIT_GAP_THRESHOLD
            n_par     = sum(p.numel() for p in m.net.parameters())

            ts["trials"].append({
                "model": "CNN", "label": label, "params": clean,
                "val_acc": round(val_acc * 100, 1), "n_params": n_par,
                "train_acc": round(train_acc * 100, 1),
                "gap": round(gap * 100, 1),
                "overfit": overfit,
            })
            try:
                _db_save(source="tuning_trial", model="cnn", val_acc=val_acc,
                         n_train=int(n_train), n_val=int(n - n_train),
                         epochs=_tune_fast, config=clean, label=label)
            except Exception as _dbe:
                logger.warning(f"DB save error (CNN trial): {_dbe}")
            if overfit:
                logger.warning(f"CNN trial skipped (overfit): {label} | gap={gap*100:.1f}pp")
            else:
                if val_acc > best_cnn_acc:
                    best_cnn_acc, best_cnn_params = val_acc, clean
            if gap < _cnn_fallback_gap:
                _cnn_fallback_gap, _cnn_fallback_params, _cnn_fallback_acc = gap, clean, val_acc
            done_n[0] += 1
            ts["completed_trials"] = done_n[0]
            ts["progress"] = _prog()
            if _tune_cancel:
                ts["message"] = "Stopping after CNN trials…"
                break

        if best_cnn_params is None:
            logger.warning("All CNN trials exceeded overfit threshold — least overfit selected")
            best_cnn_params, best_cnn_acc = _cnn_fallback_params, _cnn_fallback_acc
        ts["best"]["cnn"] = {**best_cnn_params, "val_acc": round(best_cnn_acc * 100, 1)}

        # ── Phase 2: RNN ──────────────────────────────────────
        ts.update({"phase": "rnn", "message": "Trying RNN configurations…"})
        best_rnn_acc = -1.0
        best_rnn_params = None
        _rnn_fallback_gap, _rnn_fallback_params, _rnn_fallback_acc = 999.0, None, -1.0

        for i, cfg in enumerate(_rnn_grid):
            label = cfg["_label"]
            clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
            ts.update({"message": f"RNN [{i+1}/{len(_rnn_grid)}] → {label}", "progress": _prog()})

            from agents.classical_ml.rnn_model import ClassicalRNNModel
            m = ClassicalRNNModel()
            for k, v in clean.items():
                setattr(m, k, v)
            m.epochs = _tune_fast
            m.train(X_tr, y_tr)
            train_acc = _batch_val(m.net, X_tr, y_tr)
            val_acc   = _batch_val(m.net, X_val, y_val)
            gap       = train_acc - val_acc
            overfit   = gap > _OVERFIT_GAP_THRESHOLD
            n_par     = sum(p.numel() for p in m.net.parameters())

            ts["trials"].append({
                "model": "RNN", "label": label, "params": clean,
                "val_acc": round(val_acc * 100, 1), "n_params": n_par,
                "train_acc": round(train_acc * 100, 1),
                "gap": round(gap * 100, 1),
                "overfit": overfit,
            })
            try:
                _db_save(source="tuning_trial", model="rnn", val_acc=val_acc,
                         n_train=int(n_train), n_val=int(n - n_train),
                         epochs=_tune_fast, config=clean, label=label)
            except Exception as _dbe:
                logger.warning(f"DB save error (RNN trial): {_dbe}")
            if overfit:
                logger.warning(f"RNN trial skipped (overfit): {label} | gap={gap*100:.1f}pp")
            else:
                if val_acc > best_rnn_acc:
                    best_rnn_acc, best_rnn_params = val_acc, clean
            if gap < _rnn_fallback_gap:
                _rnn_fallback_gap, _rnn_fallback_params, _rnn_fallback_acc = gap, clean, val_acc
            done_n[0] += 1
            ts["completed_trials"] = done_n[0]
            ts["progress"] = _prog()
            if _tune_cancel:
                ts["message"] = "Stopping after RNN trials…"
                break

        if best_rnn_params is None:
            logger.warning("All RNN trials exceeded overfit threshold — least overfit selected")
            best_rnn_params, best_rnn_acc = _rnn_fallback_params, _rnn_fallback_acc
        ts["best"]["rnn"] = {**best_rnn_params, "val_acc": round(best_rnn_acc * 100, 1)}

        # ── Phase 3: Quantum ──────────────────────────────────
        ts.update({"phase": "quantum", "message": "Trying Quantum configurations…"})
        best_q_acc = -1.0
        best_q_params = None
        _q_fallback_gap, _q_fallback_params, _q_fallback_acc = 999.0, None, -1.0

        import config.settings as _S
        _q_backup = {k: _S.QUANTUM_AI[k]
                     for k in ("n_qubits", "n_hidden_layers", "n_layers", "max_iterations")}

        for i, cfg in enumerate(_q_grid):
            label = cfg["_label"]
            clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
            ts.update({"message": f"Quantum [{i+1}/{len(_q_grid)}] → {label}", "progress": _prog()})

            _S.QUANTUM_AI["n_qubits"]        = clean["n_qubits"]
            _S.QUANTUM_AI["n_hidden_layers"] = clean["n_hidden_layers"]
            _S.QUANTUM_AI["n_layers"]        = clean["n_hidden_layers"]
            _S.QUANTUM_AI["max_iterations"]  = clean["max_iterations"]

            from agents.quantum_ai.quantum_module import QuantumAIModule
            m = QuantumAIModule()
            m.train(X_tr, y_tr, max_iter=clean["max_iterations"])

            # Apply reducer (PCA) before QNN — qnn.forward expects n_qubits-dim
            # input; calling qnn.predict on raw 128-D embeddings silently
            # truncates via zip() and defeats the architecture (every trial sees
            # only the first 4-8 features → near-identical accuracy).
            X_tr_red  = m.reducer.transform(X_tr)
            X_val_red = m.reducer.transform(X_val)
            tr_probs  = m.qnn.forward_batch(X_tr_red)
            val_probs = m.qnn.forward_batch(X_val_red)
            _thr      = _S.QUANTUM_AI["threshold"]
            tr_preds  = (tr_probs  > _thr).astype(int)
            preds     = (val_probs > _thr).astype(int)
            train_acc = float((tr_preds == y_tr).mean())
            val_acc   = float((preds == y_val).mean())
            gap       = train_acc - val_acc
            overfit   = gap > _OVERFIT_GAP_THRESHOLD
            n_par     = m.qnn.n_params

            # restore settings immediately
            for k, v in _q_backup.items():
                _S.QUANTUM_AI[k] = v

            ts["trials"].append({
                "model": "Quantum", "label": label, "params": clean,
                "val_acc": round(val_acc * 100, 1), "n_params": n_par,
                "train_acc": round(train_acc * 100, 1),
                "gap": round(gap * 100, 1),
                "overfit": overfit,
            })
            try:
                _db_save(source="tuning_trial", model="quantum", val_acc=val_acc,
                         n_train=int(n_train), n_val=int(n - n_train),
                         epochs=int(clean.get("max_iterations", 0)), config=clean, label=label)
            except Exception as _dbe:
                logger.warning(f"DB save error (Quantum trial): {_dbe}")
            if overfit:
                logger.warning(f"Quantum trial skipped (overfit): {label} | gap={gap*100:.1f}pp")
            else:
                if val_acc > best_q_acc:
                    best_q_acc, best_q_params = val_acc, clean
            if gap < _q_fallback_gap:
                _q_fallback_gap, _q_fallback_params, _q_fallback_acc = gap, clean, val_acc
            done_n[0] += 1
            ts["completed_trials"] = done_n[0]
            ts["progress"] = _prog()
            if _tune_cancel:
                ts["message"] = "Stopping after Quantum trials…"
                break

        if best_q_params is None:
            logger.warning("All Quantum trials exceeded overfit threshold — least overfit selected")
            best_q_params, best_q_acc = _q_fallback_params, _q_fallback_acc
        ts["best"]["quantum"] = {**best_q_params, "val_acc": round(best_q_acc * 100, 1)}

        # ── Phase 4: Full retrain + checkpoint save of winners ────
        # ── Early exit if user cancelled before any trials completed ────
        if _tune_cancel and done_n[0] == 0:
            ts.update({
                "running": False, "done": False, "cancelled": True,
                "phase": "cancelled", "progress": 0,
                "message": "⏹ Tuning cancelled — no trials completed.",
            })
            logger.info("Tuning cancelled before any trial completed.")
            return

        if _tune_cancel:
            ts.update({
                "phase": "retrain",
                "message": f"Cancelled — retraining best of {done_n[0]} completed trial(s)…",
                "progress": 75,
            })
        else:
            ts.update({
                "phase": "retrain",
                "message": "Retraining best models at full epochs and saving…",
                "progress": 82,
            })

        from utils.model_persistence import (
            save_classical_cnn, save_classical_rnn, save_quantum_ai,
        )
        from agents.classical_ml.cnn_model     import ClassicalCNNModel  as _CCNN
        from agents.classical_ml.rnn_model     import ClassicalRNNModel  as _CRNN
        from agents.quantum_ai.quantum_module  import QuantumAIModule    as _QAM

        if not _tune_skip_full:
            # ── Faz 3: Full retrain with best hyperparams on full training set ──
            ts["message"] = f"CNN full training ({_tune_full} epochs) started…"
            ts["progress"] = 84
            m_cnn_full = _CCNN()
            for k, v in best_cnn_params.items():
                setattr(m_cnn_full, k, v)
            m_cnn_full.epochs = _tune_full
            m_cnn_full.train(X_all, y_all)
            _orchestrator.cnn_classical = m_cnn_full
            save_classical_cnn(m_cnn_full)
            ts["progress"] = 90
            logger.info("Tuning: CNN best model saved.")
            try:
                _db_save(source="tuning_final", model="cnn",
                         val_acc=best_cnn_acc, n_train=int(n), n_val=0,
                         epochs=_tune_full, config=best_cnn_params,
                         label="Tuning Winner — CNN")
            except Exception as _dbe:
                logger.warning(f"DB save error (CNN final): {_dbe}")

            ts["message"] = f"RNN full training ({_tune_full} epochs) started…"
            m_rnn_full = _CRNN()
            for k, v in best_rnn_params.items():
                setattr(m_rnn_full, k, v)
            m_rnn_full.epochs = _tune_full
            m_rnn_full.train(X_all, y_all)
            _orchestrator.rnn_classical = m_rnn_full
            save_classical_rnn(m_rnn_full)
            ts["progress"] = 96
            logger.info("Tuning: RNN best model saved.")
            try:
                _db_save(source="tuning_final", model="rnn",
                         val_acc=best_rnn_acc, n_train=int(n), n_val=0,
                         epochs=_tune_full, config=best_rnn_params,
                         label="Tuning Winner — RNN")
            except Exception as _dbe:
                logger.warning(f"DB save error (RNN final): {_dbe}")

            ts["message"] = f"Quantum full training ({best_q_params['max_iterations']} iterations) started…"
            _S.QUANTUM_AI["n_qubits"]        = best_q_params["n_qubits"]
            _S.QUANTUM_AI["n_hidden_layers"] = best_q_params["n_hidden_layers"]
            _S.QUANTUM_AI["n_layers"]        = best_q_params["n_hidden_layers"]
            _S.QUANTUM_AI["max_iterations"]  = best_q_params["max_iterations"]
            m_q_full = _QAM()
            m_q_full.train(X_all, y_all, max_iter=best_q_params["max_iterations"])
            for k, v in _q_backup.items():
                _S.QUANTUM_AI[k] = v
            _orchestrator.quantum_ai = m_q_full
            save_quantum_ai(m_q_full)
            ts["progress"] = 99
            logger.info("Tuning: Quantum best model saved.")
            try:
                _db_save(source="tuning_final", model="quantum",
                         val_acc=best_q_acc, n_train=int(n), n_val=0,
                         epochs=int(best_q_params.get("max_iterations", 0)),
                         config=best_q_params, label="Tuning Winner — Quantum")
            except Exception as _dbe:
                logger.warning(f"DB save error (Quantum final): {_dbe}")

        else:
            # ── Faz 3 Atlandı: Apply best hyperparams to model stubs ──────────
            # Word2vec realignment (later) will do the actual training.
            ts.update({
                "message": "Faz 3 atlandı — kazanan parametreler uygulanıyor…",
                "progress": 90,
            })
            m_cnn_full = _CCNN()
            for k, v in best_cnn_params.items():
                setattr(m_cnn_full, k, v)
            _orchestrator.cnn_classical = m_cnn_full

            m_rnn_full = _CRNN()
            for k, v in best_rnn_params.items():
                setattr(m_rnn_full, k, v)
            _orchestrator.rnn_classical = m_rnn_full

            _S.QUANTUM_AI["n_qubits"]        = best_q_params["n_qubits"]
            _S.QUANTUM_AI["n_hidden_layers"] = best_q_params["n_hidden_layers"]
            _S.QUANTUM_AI["n_layers"]        = best_q_params["n_hidden_layers"]
            _S.QUANTUM_AI["max_iterations"]  = best_q_params["max_iterations"]
            m_q_full = _QAM()
            for k, v in _q_backup.items():
                _S.QUANTUM_AI[k] = v
            _orchestrator.quantum_ai = m_q_full

            ts["progress"] = 99
            logger.info("Tuning: skip_full_retrain=True — best params applied, full data-lake retrain skipped.")

        # ── Compute final val metrics on 20% hold-out and update Models tab ──
        ts.update({"phase": "eval", "message": "Computing final validation metrics…", "progress": 99})
        if not _tune_skip_full:
            try:
                _cnn_val_acc, _cnn_val_loss = _val_metrics_classical(
                    m_cnn_full.net, X_val, y_val, use_logits=True)
                _rnn_val_acc, _rnn_val_loss = _val_metrics_classical(
                    m_rnn_full.net, X_val, y_val, use_logits=True)
                _q_val_acc,   _q_val_loss   = _val_metrics_quantum(m_q_full, X_val, y_val)
            except Exception as _me:
                logger.warning(f"Tuning final eval error: {_me}")
                _cnn_val_acc = best_cnn_acc;  _cnn_val_loss = None
                _rnn_val_acc = best_rnn_acc;  _rnn_val_loss = None
                _q_val_acc   = best_q_acc;    _q_val_loss   = None
        else:
            # Use best trial accuracies as proxy (full retrain was skipped)
            _cnn_val_acc = best_cnn_acc;  _cnn_val_loss = None
            _rnn_val_acc = best_rnn_acc;  _rnn_val_loss = None
            _q_val_acc   = best_q_acc;    _q_val_loss   = None

        _model_train_status["cnn"].update({
            "running": False, "done": True, "progress": 100, "error": None,
            "val_acc": _cnn_val_acc, "val_loss": _cnn_val_loss,
            "train_acc": None, "n_train": int(n_train), "n_val": int(n - n_train),
            "message": f"✓ CNN tuned — val accuracy: {round(_cnn_val_acc * 100, 1)}%",
        })
        _model_train_status["rnn"].update({
            "running": False, "done": True, "progress": 100, "error": None,
            "val_acc": _rnn_val_acc, "val_loss": _rnn_val_loss,
            "train_acc": None, "n_train": int(n_train), "n_val": int(n - n_train),
            "message": f"✓ RNN tuned — val accuracy: {round(_rnn_val_acc * 100, 1)}%",
        })
        _model_train_status["quantum"].update({
            "running": False, "done": True, "progress": 100, "error": None,
            "val_acc": _q_val_acc, "val_loss": _q_val_loss,
            "train_acc": None, "n_train": int(n_train), "n_val": int(n - n_train),
            "message": f"✓ Quantum tuned — val accuracy: {round(_q_val_acc * 100, 1)}%",
        })
        # update tuning best dict with final retrained model accuracies
        ts["best"]["cnn"]["val_acc"]     = round(_cnn_val_acc * 100, 1)
        ts["best"]["rnn"]["val_acc"]     = round(_rnn_val_acc * 100, 1)
        ts["best"]["quantum"]["val_acc"] = round(_q_val_acc   * 100, 1)
        _save_model_train_status()
        logger.info(
            f"Tuning final val metrics — CNN: {round(_cnn_val_acc*100,1)}%"
            f" | RNN: {round(_rnn_val_acc*100,1)}%"
            f" | Quantum: {round(_q_val_acc*100,1)}%"
        )

        # Persist best params to arch_config.json automatically
        _current_arch = _load_arch_config()
        _current_arch["cnn"].update({k: v for k, v in best_cnn_params.items() if k != "val_acc"})
        _current_arch["rnn"].update({k: v for k, v in best_rnn_params.items() if k != "val_acc"})
        _current_arch["quantum"].update({k: v for k, v in best_q_params.items() if k != "val_acc"})
        os.makedirs(os.path.dirname(_ARCH_CONFIG_PATH), exist_ok=True)
        with open(_ARCH_CONFIG_PATH, "w", encoding="utf-8") as _f:
            json.dump(_current_arch, _f, indent=2, ensure_ascii=False)
        _apply_arch_config(_current_arch)

        # ── Realign to word2vec inference space ───────────────────────────
        # Tuning trained models on data-lake biased 128-dim vectors.
        # Quick-scenario inference uses word2vec embeddings (EmbeddingEngine).
        # Realign CNN, RNN and Quantum to that same space so they generalise
        # at inference time.  Uses the best hyperparams already installed in
        # the orchestrator.  Re-save updated checkpoints afterward.
        ts.update({
            "phase": "retrain",
            "message": "Realigning models to word2vec embedding space…",
            "progress": 99,
        })
        try:
            _orchestrator._retrain_classifiers_on_embedding_space()
            from utils.model_persistence import (
                save_classical_cnn as _sv_cnn,
                save_classical_rnn as _sv_rnn,
                save_quantum_ai    as _sv_q,
            )
            _sv_cnn(_orchestrator.cnn_classical)
            _sv_rnn(_orchestrator.rnn_classical)
            _sv_q(_orchestrator.quantum_ai)
            logger.info("Tuning: word2vec realignment complete — checkpoints updated.")
        except Exception as _rea_err:
            logger.warning(f"Tuning realignment warning (non-fatal): {_rea_err}")

        # ── Done ──────────────────────────────────────────────
        best = ts["best"]
        _was_cancelled = _tune_cancel
        ts.update({
            "running": False, "done": True, "progress": 100,
            "phase": "cancelled" if _was_cancelled else "done",
            "cancelled": _was_cancelled,
            "saved": True,
            "message": (
                (f"⏹ Tuning stopped early — best models saved"
                 f" · CNN: {best['cnn']['val_acc']}%"
                 f" · RNN: {best['rnn']['val_acc']}%"
                 f" · Quantum: {best['quantum']['val_acc']}%")
                if _was_cancelled else
                (f"Tuning complete and models saved"
                 f" · CNN: {best['cnn']['val_acc']}%"
                 f" · RNN: {best['rnn']['val_acc']}%"
                 f" · Quantum: {best['quantum']['val_acc']}%")
            ),
        })
        logger.info("Auto-tuning complete and all models saved to checkpoint.")

    except Exception as exc:
        ts.update({
            "running": False, "done": False,
            "error": str(exc), "progress": 0,
            "message": f"✗ Error: {exc}",
        })
        logger.error(f"Tuning error: {exc}", exc_info=True)


@app.route("/api/training-history")
def api_training_history():
    try:
        rows = _db_history(limit=500)
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training-history/export")
def api_training_history_export():
    """Download all training records as CSV or JSON."""
    import csv
    import io
    fmt = request.args.get("format", "csv").lower()
    try:
        rows = _db_history(limit=10000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        payload = json.dumps(rows, ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="egitim_kayitlari_{ts_now}.json"',
                "Content-Type": "application/json; charset=utf-8",
            },
        )

    # Default: CSV
    cols = ["id", "timestamp", "source", "model", "label",
            "val_acc", "val_loss", "n_train", "n_val", "epochs", "config", "notes"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore",
                            lineterminator="\r\n")
    writer.writeheader()
    for r in rows:
        row = dict(r)
        # Flatten config dict → JSON string for the CSV cell
        if isinstance(row.get("config"), dict):
            row["config"] = json.dumps(row["config"], ensure_ascii=False)
        if row.get("val_acc") is not None:
            row["val_acc"] = round(row["val_acc"] * 100, 2)   # store as %
        writer.writerow(row)

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="egitim_kayitlari_{ts_now}.csv"',
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


@app.route("/api/training-history/reset", methods=["POST"])
def api_training_history_reset():
    """Delete all rows from the training history database."""
    try:
        _db_clear_history()
        return jsonify({"success": True, "message": "All training records deleted."})
    except Exception as e:
        logger.error(f"Training history reset error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tune/start", methods=["POST"])
def api_tune_start():
    global _tune_custom_config
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready yet — click Start first"}), 503
    if _tune_status["running"]:
        return jsonify({"error": "Tuning already running"}), 429

    body = request.get_json(force=True, silent=True) or {}

    cnn_grid = _build_grid_from_ranges(body.get("cnn", {}))   or _CNN_TUNE_GRID
    rnn_grid = _build_grid_from_ranges(body.get("rnn", {}))   or _RNN_TUNE_GRID
    q_grid   = _build_grid_from_ranges(body.get("quantum", {})) or _QUANTUM_TUNE_GRID

    _tune_custom_config = {
        "cnn_grid":          cnn_grid,
        "rnn_grid":          rnn_grid,
        "q_grid":            q_grid,
        "epochs_fast":       max(1, int(body.get("epochs_fast", _TUNE_EPOCHS_FAST))),
        "epochs_full":       max(1, int(body.get("epochs_full", _TUNE_EPOCHS_FULL))),
        "skip_full_retrain": bool(body.get("skip_full_retrain", False)),
    }
    logger.info(
        f"Starting tuning — CNN: {len(cnn_grid)} · RNN: {len(rnn_grid)} · Q: {len(q_grid)} trials "
        f"| fast={_tune_custom_config['epochs_fast']} full={_tune_custom_config['epochs_full']} epochs"
        f"| skip_full_retrain={_tune_custom_config['skip_full_retrain']}"
    )
    threading.Thread(target=_run_autotuning, daemon=True).start()
    return jsonify({"ok": True, "message": "Auto-tuning started",
                    "total": len(cnn_grid) + len(rnn_grid) + len(q_grid)})


@app.route("/api/tune/status", methods=["GET"])
def api_tune_status_route():
    return jsonify(_tune_status)


@app.route("/api/tune/cancel", methods=["POST"])
def api_tune_cancel():
    global _tune_cancel
    if not _tune_status.get("running"):
        return jsonify({"ok": False, "message": "No tuning is running"}), 400
    _tune_cancel = True
    _tune_status["message"] = "⏹ Cancellation requested — stopping after current trial…"
    logger.info("Auto-tuning cancel requested by user.")
    return jsonify({"ok": True, "message": "Cancellation requested"})


@app.route("/api/tune/apply", methods=["POST"])
def api_tune_apply():
    """Persist the best-found params to arch_config.json and apply in-memory."""
    best = _tune_status.get("best", {})
    if not any(best.get(k) for k in ("cnn", "rnn", "quantum")):
        return jsonify({"error": "No completed tuning results yet"}), 400

    current = _load_arch_config()

    if best.get("cnn"):
        b = {k: v for k, v in best["cnn"].items() if k != "val_acc"}
        current["cnn"].update(b)
    if best.get("rnn"):
        b = {k: v for k, v in best["rnn"].items() if k != "val_acc"}
        current["rnn"].update(b)
    if best.get("quantum"):
        b = {k: v for k, v in best["quantum"].items() if k != "val_acc"}
        current["quantum"].update(b)
        if "n_hidden_layers" in b:
            current["quantum"]["n_hidden_layers"] = b["n_hidden_layers"]

    os.makedirs(os.path.dirname(_ARCH_CONFIG_PATH), exist_ok=True)
    with open(_ARCH_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    _apply_arch_config(current)
    return jsonify({"ok": True, "message": "Best parameters applied", "config": current})


# ─────────────────────────────────────────────
# Trial model selection & loading
# ─────────────────────────────────────────────

_tune_select_status: dict = {
    "running": False, "done": False, "error": None,
    "model": "", "progress": 0, "message": "", "val_acc": None,
}


def _load_selected_trial(model_key: str, params: dict, epochs: int):
    """Background task: retrain a specific trial config at full depth, then save as active checkpoint."""
    ts = _tune_select_status
    try:
        ts.update({
            "running": True, "done": False, "error": None,
            "model": model_key, "progress": 5,
            "message": f"{model_key.upper()} full training starting…",
            "val_acc": None,
        })

        X_all, y_all = _orchestrator.data_lake.get_training_data()
        n = len(X_all)
        if n < 10:
            raise ValueError(f"Not enough data ({n} samples)")

        from utils.model_persistence import (
            save_classical_cnn, save_classical_rnn, save_quantum_ai,
        )

        ts["progress"] = 15

        if model_key == "cnn":
            from agents.classical_ml.cnn_model import ClassicalCNNModel
            m = ClassicalCNNModel()
            for k, v in params.items():
                if hasattr(m, k):
                    setattr(m, k, type(getattr(m, k))(v))
            m.epochs = epochs
            ts["message"] = f"Training CNN ({n} samples, {epochs} epochs)…"
            ts["progress"] = 30
            m.train(X_all, y_all)
            save_classical_cnn(m)
            _orchestrator.cnn_classical = m

        elif model_key == "rnn":
            from agents.classical_ml.rnn_model import ClassicalRNNModel
            m = ClassicalRNNModel()
            for k, v in params.items():
                if hasattr(m, k):
                    setattr(m, k, type(getattr(m, k))(v))
            m.epochs = epochs
            ts["message"] = f"Training RNN ({n} samples, {epochs} epochs)…"
            ts["progress"] = 30
            m.train(X_all, y_all)
            save_classical_rnn(m)
            _orchestrator.rnn_classical = m

        elif model_key == "quantum":
            import config.settings as _S
            _q_backup = {k: _S.QUANTUM_AI.get(k) for k in
                         ("n_qubits", "n_hidden_layers", "n_layers", "max_iterations")}
            if "n_qubits" in params:
                _S.QUANTUM_AI["n_qubits"] = int(params["n_qubits"])
            if "n_hidden_layers" in params:
                _S.QUANTUM_AI["n_hidden_layers"] = int(params["n_hidden_layers"])
                _S.QUANTUM_AI["n_layers"]        = int(params["n_hidden_layers"])
            max_iter = int(params.get("max_iterations", epochs))
            _S.QUANTUM_AI["max_iterations"] = max_iter
            from agents.quantum_ai.quantum_module import QuantumAIModule
            m = QuantumAIModule()
            ts["message"] = f"Training Quantum ({n} samples, {max_iter} iterations)…"
            ts["progress"] = 30
            m.train(X_all, y_all, max_iter=max_iter)
            for k, v in _q_backup.items():
                if v is not None:
                    _S.QUANTUM_AI[k] = v
            save_quantum_ai(m)
            _orchestrator.quantum_ai = m

        ts["progress"] = 85
        ts["message"] = "Updating arch config…"

        # Persist chosen params to arch_config.json
        current = _load_arch_config()
        for k, v in params.items():
            if k in current.get(model_key, {}):
                current[model_key][k] = v
        os.makedirs(os.path.dirname(_ARCH_CONFIG_PATH), exist_ok=True)
        with open(_ARCH_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        _apply_arch_config(current)

        # Log to DB
        try:
            _db_save(source="manual", model=model_key,
                     val_acc=None, n_train=n, n_val=0,
                     epochs=epochs, config=params,
                     label=f"Trial Selection — {model_key.upper()}")
        except Exception:
            pass

        ts.update({
            "running": False, "done": True, "progress": 100,
            "message": f"✓ {model_key.upper()} trial loaded and checkpoint saved",
        })
        logger.info(f"Trial model loaded: {model_key} | params={params} | epochs={epochs}")

    except Exception as exc:
        ts.update({
            "running": False, "done": False, "error": str(exc),
            "progress": 0, "message": f"✗ Hata: {exc}",
        })
        logger.error(f"Trial load error ({model_key}): {exc}", exc_info=True)


@app.route("/api/tune/load-trial", methods=["POST"])
def api_tune_load_trial():
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready"}), 503
    if _tune_select_status["running"]:
        return jsonify({"error": "A load operation is already running"}), 429
    body      = request.get_json(force=True, silent=True) or {}
    model_key = (body.get("model") or "").lower()
    if model_key not in ("cnn", "rnn", "quantum"):
        return jsonify({"error": "Invalid model — must be cnn / rnn / quantum"}), 400
    params = body.get("config") or {}
    epochs = max(1, int(body.get("epochs") or _TUNE_EPOCHS_FULL))
    threading.Thread(
        target=_load_selected_trial,
        args=(model_key, params, epochs),
        daemon=True,
    ).start()
    return jsonify({"ok": True})


@app.route("/api/tune/load-trial-status", methods=["GET"])
def api_tune_load_trial_status():
    return jsonify(_tune_select_status)


# ─────────────────────────────────────────────
# Per-model training status
# ─────────────────────────────────────────────

_model_train_lock = threading.Lock()

def _blank_train_status():
    return {
        "running": False, "progress": 0, "message": "", "error": None, "done": False,
        "val_acc": None, "val_loss": None, "train_acc": None, "n_train": 0, "n_val": 0,
    }

_model_train_status = {
    "cnn":     _blank_train_status(),
    "rnn":     _blank_train_status(),
    "quantum": _blank_train_status(),
}


def _val_metrics_classical(net, X_val, y_val, use_logits: bool):
    """Compute val accuracy and val loss for CNN (logits) or RNN (probs)."""
    import torch, torch.nn as _nn
    import numpy as _np
    net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_val, dtype=torch.float32)
        y_t = torch.tensor(y_val.astype(_np.float32), dtype=torch.float32)
        out = net(X_t).squeeze(-1)
        if use_logits:
            val_loss = _nn.BCEWithLogitsLoss()(out, y_t).item()
            preds = (torch.sigmoid(out) >= 0.5).numpy().astype(int)
        else:
            val_loss = _nn.BCELoss()(out, y_t).item()
            preds = (out >= 0.5).numpy().astype(int)
    val_acc = float((preds == y_val).mean())
    return round(val_acc, 4), round(val_loss, 4)


def _val_metrics_quantum(model, X_val, y_val):
    """Compute val accuracy and BCE loss for QuantumAIModule."""
    import numpy as _np
    preds = _np.array([model.qnn.predict(x)[0] for x in X_val])
    val_acc = float((preds == y_val).mean())
    probs = _np.clip(_np.array([model.qnn.forward(x) for x in X_val]), 1e-7, 1 - 1e-7)
    val_loss = float(-_np.mean(
        y_val * _np.log(probs) + (1 - y_val) * _np.log(1 - probs)
    ))
    return round(val_acc, 4), round(val_loss, 4)


def _train_single(model_key: str):
    """Background worker: retrain one model with 80/20 train/val split."""
    import numpy as _np
    status = _model_train_status[model_key]
    status.update({**_blank_train_status(),
                   "running": True, "progress": 5,
                   "message": f"Preparing data for {model_key.upper()}…"})
    try:
        if _orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")
        X_all, y_all = _orchestrator.data_lake.get_training_data()
        n = len(X_all)
        if n == 0:
            raise ValueError("Data Lake is empty — start the system first")

        # 80 / 20 stratified split
        rng = _np.random.RandomState(42)
        idx = rng.permutation(n)
        n_train = int(n * 0.8)
        X_tr, X_val = X_all[idx[:n_train]], X_all[idx[n_train:]]
        y_tr, y_val = y_all[idx[:n_train]], y_all[idx[n_train:]]
        status.update({"n_train": int(n_train), "n_val": int(n - n_train),
                       "progress": 12, "message": f"Data split: {n_train} train / {n - n_train} validation"})

        _apply_arch_config(_load_arch_config())

        if model_key == "cnn":
            from agents.classical_ml.cnn_model import ClassicalCNNModel
            from utils.model_persistence import save_classical_cnn
            status.update({"progress": 20, "message": f"Building CNN ({_orchestrator.cnn_classical.n_conv_layers} layers)…"})
            _orchestrator.cnn_classical = ClassicalCNNModel()
            epochs = _orchestrator.cnn_classical.epochs
            status.update({"progress": 30, "message": f"Training CNN ({n_train} samples, {epochs} epochs)…"})
            _orchestrator.cnn_classical.train(X_tr, y_tr)
            status.update({"progress": 85, "message": "Evaluating validation set…"})
            val_acc, val_loss = _val_metrics_classical(_orchestrator.cnn_classical.net, X_val, y_val, use_logits=True)
            save_classical_cnn(_orchestrator.cnn_classical)

        elif model_key == "rnn":
            from agents.classical_ml.rnn_model import ClassicalRNNModel
            from utils.model_persistence import save_classical_rnn
            status.update({"progress": 20, "message": "Building RNN…"})
            _orchestrator.rnn_classical = ClassicalRNNModel()
            epochs = _orchestrator.rnn_classical.epochs
            status.update({"progress": 30, "message": f"Training RNN ({n_train} samples, {epochs} epochs)…"})
            _orchestrator.rnn_classical.train(X_tr, y_tr)
            status.update({"progress": 85, "message": "Evaluating validation set…"})
            val_acc, val_loss = _val_metrics_classical(_orchestrator.rnn_classical.net, X_val, y_val, use_logits=True)
            save_classical_rnn(_orchestrator.rnn_classical)

        elif model_key == "quantum":
            from agents.quantum_ai.quantum_module import QuantumAIModule
            from utils.model_persistence import save_quantum_ai
            status.update({"progress": 20, "message": "Building Quantum QMLP…"})
            _orchestrator.quantum_ai = QuantumAIModule()
            import config.settings as _S_manual
            epochs = _S_manual.QUANTUM_AI.get("max_iterations", 0)
            status.update({"progress": 30, "message": f"Training Quantum AI ({n_train} samples)…"})
            _orchestrator.quantum_ai.train(X_tr, y_tr)
            status.update({"progress": 85, "message": "Evaluating validation set…"})
            val_acc, val_loss = _val_metrics_quantum(_orchestrator.quantum_ai, X_val, y_val)
            save_quantum_ai(_orchestrator.quantum_ai)
        else:
            raise ValueError(f"Unknown model: {model_key}")

        # Realign all models to word2vec inference space (same 128-dim pipeline
        # used at inference in /api/analyze).  The model was trained above on
        # data-lake biased vectors; without this step it cannot classify
        # quick-scenario events which are embedded with EmbeddingEngine.
        status.update({"progress": 92, "message": "Realigning to word2vec embedding space…"})
        try:
            _orchestrator._retrain_classifiers_on_embedding_space()
            from utils.model_persistence import (
                save_classical_cnn as _sv_cnn2,
                save_classical_rnn as _sv_rnn2,
                save_quantum_ai    as _sv_q2,
            )
            _sv_cnn2(_orchestrator.cnn_classical)
            _sv_rnn2(_orchestrator.rnn_classical)
            _sv_q2(_orchestrator.quantum_ai)
        except Exception as _rea2_err:
            logger.warning(f"Single-model realignment warning (non-fatal): {_rea2_err}")

        acc_pct = round(val_acc * 100, 1)
        status.update({
            "running": False, "progress": 100, "done": True,
            "val_acc": val_acc, "val_loss": val_loss,
            "message": f"✓ {model_key.upper()} done — val accuracy: {acc_pct}%",
        })
        logger.info(f"Single model training successful: {model_key} | val_acc={acc_pct}% val_loss={val_loss}")
        _save_model_train_status()
        try:
            _db_save(
                source="manual", model=model_key,
                val_acc=val_acc, val_loss=val_loss,
                n_train=int(n_train), n_val=int(n - n_train),
                epochs=int(epochs),
                config=_load_arch_config().get(model_key, {}),
                label="Manual Training",
            )
        except Exception as _dbe:
            logger.warning(f"DB save error (manual): {_dbe}")

    except Exception as e:
        status.update({"running": False, "progress": 0, "done": False,
                       "message": f"✗ Error: {e}", "error": str(e)})
        logger.error(f"Single model training error ({model_key}): {e}", exc_info=True)
        _save_model_train_status()


@app.route("/api/train/<model_key>", methods=["POST"])
def api_train_model(model_key: str):
    """Train a single model (cnn | rnn | quantum)."""
    if model_key not in _model_train_status:
        return jsonify({"error": f"Invalid model: {model_key}"}), 400
    if not _init_status["ready"]:
        return jsonify({"error": "System not ready yet — click Start first"}), 503
    if _model_train_status[model_key]["running"]:
        return jsonify({"error": f"{model_key.upper()} already training"}), 429

    t = threading.Thread(target=_train_single, args=(model_key,), daemon=True)
    t.start()
    return jsonify({"ok": True, "message": f"{model_key.upper()} training started"})


@app.route("/api/train/status", methods=["GET"])
def api_model_train_status():
    """Return current per-model training status."""
    return jsonify(_model_train_status)


@app.route("/api/arch/config", methods=["GET"])
def api_arch_config_get():
    """Return current architecture config and all available presets."""
    from config.settings import ARCH_PRESETS
    return jsonify({
        "current": _load_arch_config(),
        "presets": ARCH_PRESETS,
    })


@app.route("/api/arch/config", methods=["POST"])
def api_arch_config_post():
    """Save new architecture config and apply to live settings."""
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "Empty payload"}), 400

    try:
        os.makedirs(os.path.dirname(_ARCH_CONFIG_PATH), exist_ok=True)
        with open(_ARCH_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _apply_arch_config(data)
        return jsonify({
            "ok":      True,
            "message": "Architecture settings saved. Click 'Retrain' to apply them.",
        })
    except Exception as e:
        logger.error(f"arch_config save error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Module-level startup ────────────────────────────────────────────────
# Runs both when executed directly (`python web_app.py`) AND when gunicorn
# imports this module.  Gunicorn never enters `if __name__ == "__main__":`,
# so without this block the orchestrator would never initialise in production
# and every API call (including CSV upload) would return 503.
_apply_arch_config(_load_arch_config())
_start_init(force_retrain=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
