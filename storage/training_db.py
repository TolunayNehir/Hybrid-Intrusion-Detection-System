"""
Training History Database
=========================
SQLite-backed store for every model training run.
Records both manual single-model training and Auto Tuning runs.
"""

import sqlite3
import json
import os
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "training_history.db")


def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                source    TEXT    NOT NULL,
                model     TEXT    NOT NULL,
                val_acc   REAL,
                val_loss  REAL,
                n_train   INTEGER DEFAULT 0,
                n_val     INTEGER DEFAULT 0,
                epochs    INTEGER DEFAULT 0,
                config    TEXT,
                label     TEXT,
                notes     TEXT
            )
        """)
        conn.commit()


def save_run(source: str, model: str, val_acc=None, val_loss=None,
             n_train: int = 0, n_val: int = 0, epochs: int = 0,
             config=None, label: str = "", notes: str = ""):
    """Persist a single training run to the database."""
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    cfg_str = json.dumps(config, ensure_ascii=False) if config is not None else None
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """INSERT INTO training_runs
               (timestamp, source, model, val_acc, val_loss,
                n_train, n_val, epochs, config, label, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, source, model,
             float(val_acc) if val_acc is not None else None,
             float(val_loss) if val_loss is not None else None,
             int(n_train), int(n_val), int(epochs),
             cfg_str, label, notes),
        )
        conn.commit()


def get_history(limit: int = 500):
    """Return all training runs newest-first."""
    if not os.path.exists(_DB_PATH):
        return []
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("config"):
            try:
                d["config"] = json.loads(d["config"])
            except Exception:
                pass
        result.append(d)
    return result


def clear_history():
    """Delete all rows from training_runs and reset the auto-increment counter."""
    if not os.path.exists(_DB_PATH):
        return
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("DELETE FROM training_runs")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='training_runs'")
        conn.commit()


def get_best():
    """Return the run with the highest val_acc for each model type."""
    if not os.path.exists(_DB_PATH):
        return {}
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT t.*
            FROM training_runs t
            INNER JOIN (
                SELECT model, MAX(val_acc) AS best_acc
                FROM training_runs
                WHERE val_acc IS NOT NULL
                GROUP BY model
            ) b ON t.model = b.model AND t.val_acc = b.best_acc
            GROUP BY t.model
        """).fetchall()
    result = {}
    for r in rows:
        d = dict(r)
        if d.get("config"):
            try:
                d["config"] = json.loads(d["config"])
            except Exception:
                pass
        result[d["model"]] = d
    return result
