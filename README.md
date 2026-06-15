# Hybrid Intrusion Detection System

A research-oriented Hybrid Intrusion Detection System (Hybrid IDS) that combines classical machine learning, quantum machine learning, semantic similarity analysis, rule-based or LLM based analysis and decision-level fusion for network intrusion detection experiments.

The project implements a multi-agent IDS pipeline inspired by an academic general system diagram. It ingests security events from multiple source types, preprocesses them, creates embeddings, evaluates them through several detection agents, fuses the outputs, generates alerts and supports analyst feedback and model updates.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Detection Agents](#detection-agents)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [Web Dashboard and REST API](#web-dashboard-and-rest-api)
- [Dataset Format](#dataset-format)
- [Training and Model Persistence](#training-and-model-persistence)
- [Configuration](#configuration)
- [Generated Files and GitHub Cleanup](#generated-files-and-github-cleanup)
- [Security Notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Overview

Hybrid Intrusion Detection System is designed as an experimental cybersecurity platform where multiple detection approaches work together:

1. **Security event ingestion** from IPS, SIEM, firewall, SOAR and IDS-like sources.
2. **Preprocessing** for cleaning, parsing, normalization and feature extraction.
3. **Embedding generation** for semantic representation of events.
4. **Reference embedding storage** through a vector warehouse and Data Lake.
5. **Multi-agent analysis** using Quantum AI, CNN, RNN, cosine similarity and LLM/rule-based reasoning.
6. **Decision fusion** through weighted averaging or voting-style aggregation.
7. **Intrusion detection and alerting** with dashboard/reporting support.
8. **Feedback loop** for analyst review and model/data updates.

This project is useful for:

- academic IDS/IPS experiments,
- hybrid AI cybersecurity demonstrations,
- comparing classical and quantum-inspired approaches,
- testing decision fusion strategies,
- building a local Flask-based IDS dashboard prototype.

---

## Key Features

- Multi-source security event ingestion.
- Classical ML agents: CNN and RNN classifiers.
- Quantum AI path with QMLP/QNN-style processing and SPSA-based optimization.
- Cosine similarity anomaly scoring against reference embeddings.
- LLM analysis module with OpenAI-compatible API support and rule-based fallback.
- Decision fusion layer combining all agent outputs.
- Flask web dashboard with REST API endpoints.
- CSV dataset upload and training workflow.
- Data Lake upload/export/reset support.
- Training history storage through SQLite.
- Model checkpointing and metadata files.
- Architecture preset configuration for CNN, RNN and Quantum models.

---

## System Architecture

The system follows this high-level pipeline:

```text
Security Sources
  ├── IPS Alerts
  ├── SIEM Events
  ├── Firewall Logs
  ├── SOAR Events
  └── IDS Alerts
        │
        ▼
Preprocessing
  ├── Cleaning
  ├── Parsing
  ├── Normalization
  └── Feature Extraction
        │
        ▼
Embedding Creation and Feature Representation
        │
        ├── Quantum AI Agent
        ├── Classical RNN Agent
        ├── Classical CNN Agent
        ├── Cosine Similarity Agent
        └── LLM / Rule-Based Analysis Agent
        │
        ▼
Decision Fusion Layer
        │
        ▼
Intrusion / Violation Detection
        │
        ├── Dashboard / Reporting
        ├── Alerts and Event Management
        └── Security Analyst Review
        │
        ▼
Feedback / Model Updates
```

---

## Detection Agents

### 1. Quantum AI Agent

Located in:

```text
agents/quantum_ai/quantum_module.py
```

The Quantum AI path includes:

- dimensionality reduction and normalization,
- QMLP/QNN-style quantum circuit processing,
- SPSA/Adam-SPSA style optimization,
- a recurrent bridge layer,
- final binary classification output.

The active configuration can be adjusted in:

```text
config/settings.py
config/arch_config.json
```

### 2. Classical RNN Agent

Located in:

```text
agents/classical_ml/rnn_model.py
```

The RNN agent processes event embeddings or numeric dataset features and produces a binary intrusion classification.

### 3. Classical CNN Agent

Located in:

```text
agents/classical_ml/cnn_model.py
```

The CNN agent provides another independent classical ML prediction path. The model can be trained from uploaded numeric datasets or from the default internal training flow.

### 4. Cosine Similarity Agent

Located in:

```text
agents/similarity/cosine_analyzer.py
```

This module compares event embeddings against reference vectors and calculates anomaly/similarity scores.

### 5. LLM / Rule-Based Analysis Agent

Located in:

```text
agents/llm_analysis/llm_module.py
```

The module attempts to use an OpenAI-compatible client when the following environment variables are available:

```env
AI_INTEGRATIONS_OPENAI_API_KEY=
AI_INTEGRATIONS_OPENAI_BASE_URL=
```

If the API is unavailable, the project falls back to a built-in rule-based analysis engine.

---

## Project Structure

```text
hybrid_ids_system/
├── main.py                         # CLI orchestrator and full demo runner
├── web_app.py                      # Flask dashboard and REST API
├── requirements.txt                # Python dependencies
│
├── agents/
│   ├── classical_ml/
│   │   ├── cnn_model.py            # Classical CNN model
│   │   └── rnn_model.py            # Classical RNN model
│   ├── llm_analysis/
│   │   └── llm_module.py           # LLM / rule-based IDS analysis
│   ├── quantum_ai/
│   │   └── quantum_module.py       # Quantum AI / QMLP module
│   └── similarity/
│       └── cosine_analyzer.py      # Cosine similarity anomaly analysis
├── config/
│   ├── settings.py                 # Main system settings and presets
│   └── arch_config.json            # Active architecture configuration
│
├── dashboard/
│   └── dashboard_reporting.py      # Dashboard, alert management, analyst review
│
├── data_sources/
│   └── data_ingestion.py           # IPS, SIEM, Firewall, SOAR, IDS event sources
│
├── detection/
│   └── intrusion_detector.py       # Final intrusion/alert creation logic
│
├── embeddings/
│   └── embedding_engine.py         # Event embedding creation
│
├── feedback/
│   └── model_updater.py            # Feedback and model update loop
│
├── fusion/
│   └── decision_fusion.py          # Weighted multi-agent decision fusion
│
├── preprocessing/
│   └── preprocessor.py             # Cleaning, parsing, normalization, features
│
├── storage/
│   ├── training_db.py              # SQLite training history utilities
│   └── vector_warehouse.py         # Vector warehouse and reference Data Lake
│
├── static/
│   ├── app.js                      # Dashboard frontend logic
│   ├── style.css                   # Dashboard styling
│   └── favicon.ico
│
├── templates/
│   └── index.html                  # Flask dashboard page
│
└── utils/
    ├── data_models.py              # Shared dataclasses / data structures
    ├── dataset_analyzer.py         # CSV schema detection and feature extraction
    ├── logger.py                   # Logging configuration
    └── model_persistence.py        # Model save/load utilities
```

---

## Requirements

Recommended environment:

- Python 3.10+
- pip
- virtualenv or venv
- CPU is sufficient for local experiments

Core dependencies used by the project:

```text
numpy>=1.21.0
torch>=2.0.0
qiskit>=1.0.0
flask>=3.0.0
flask-cors>=4.0.0
openai>=1.0.0
gunicorn>=20.1.0
```


---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If Qiskit is missing from `requirements.txt`, install it manually:

```bash
pip install qiskit
```

---

## Environment Variables

The system can run without an LLM API key because it includes a rule-based fallback. However, to enable OpenAI-compatible LLM analysis, create a local `.env` file or configure your environment variables:

```env
AI_INTEGRATIONS_OPENAI_API_KEY=your_api_key_here
AI_INTEGRATIONS_OPENAI_BASE_URL=your_openai_compatible_base_url_here
PORT=5000
```

For GitHub, do **not** commit `.env` files. Commit only an example file:

```text
.env.example
```

Suggested `.env.example`:

```env
AI_INTEGRATIONS_OPENAI_API_KEY=
AI_INTEGRATIONS_OPENAI_BASE_URL=
PORT=5000
```

Suggested `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example
```

---

## Running the Project

### Option 1: Run the CLI Demo

```bash
python main.py
```

This command initializes the full orchestrator, loads existing checkpoints if available, or trains missing models.

Force retraining:

```bash
python main.py --retrain
```

### Option 2: Run the Flask Dashboard

```bash
python web_app.py
```

Then open:

```text
http://localhost:5000
```

The Flask app initializes the orchestrator automatically at startup.

### Option 3: Run with Gunicorn

```bash
gunicorn --bind 0.0.0.0:5000 web_app:app
```

---

## Web Dashboard and REST API

The project exposes a Flask dashboard and multiple API endpoints.

### Main Dashboard

```http
GET /
```

Opens the web interface.

### System Status

```http
GET /api/status
```

Returns initialization status, checkpoint status and session counters.

### Start or Retrain the System

```http
POST /api/start
```

Starts initialization or retraining depending on request parameters.

### Process Demo Events

```http
POST /api/process
```

Processes generated security events through the IDS pipeline.

### Analyze a Single Event

```http
POST /api/analyze
```

Example JSON body:

```json
{
  "source_type": "IDS",
  "source_ip": "192.168.1.100",
  "destination_ip": "10.0.0.5",
  "source_port": 44444,
  "destination_port": 22,
  "protocol": "TCP",
  "severity": "HIGH",
  "payload": "Repeated SSH login attempts with invalid credentials",
  "attack_type": "Brute Force"
}
```

### Alerts

```http
GET /api/alerts
POST /api/alert/acknowledge
POST /api/alert/resolve
```

### Reports and Metrics

```http
GET /api/report
GET /api/metrics
GET /api/quantum-metrics
```

### Batch Prediction

```http
POST /api/batch-predict
GET /api/batch-status/<job_id>
```

### Dataset Training

```http
POST /api/dataset-train
GET /api/train-status
```

### Data Lake Management

```http
POST /api/datalake/upload
GET /api/datalake/upload-status/<job_id>
GET /api/datalake
GET /api/datalake/sample-csv
GET /api/datalake/export
POST /api/datalake/reset
```

### Training History

```http
GET /api/training-history
GET /api/training-history/export
POST /api/training-history/reset
```

### Hyperparameter Tuning and Architecture

```http
POST /api/tune/start
GET /api/tune/status
POST /api/tune/cancel
POST /api/tune/apply
POST /api/tune/load-trial
GET /api/tune/load-trial-status
POST /api/train/<model_key>
GET /api/train/status
GET /api/arch/config
POST /api/arch/config
```

Valid `model_key` values depend on the code configuration, but typically include:

```text
cnn
rnn
quantum
```

---

## Dataset Format

The project can work with manually created CSV data and known intrusion detection dataset structures such as KDD/NSL-KDD-like data, CICIDS-style data and generic numeric datasets.

### Simple Sample CSV

```csv
label,payload,source_ip,destination_ip,protocol,port
Normal,GET /index.html HTTP/1.1,192.168.1.10,10.0.0.1,TCP,80
Normal,DNS query google.com,192.168.1.11,8.8.8.8,UDP,53
DoS,SYN flood burst x1000,10.10.10.5,192.168.1.1,TCP,443
DoS,UDP flood high-rate,10.10.10.6,192.168.1.2,UDP,0
Exploit,Shellcode buffer overflow attempt,172.16.0.99,192.168.1.5,TCP,22
Exploit,SQL injection ' OR 1=1 --,172.16.0.100,192.168.1.6,TCP,3306
PortScan,SYN scan 0-65535,10.0.0.200,192.168.1.1,TCP,0
Botnet,C2 beacon check-in,192.168.50.5,45.33.32.156,TCP,8080
Normal,HTTPS POST /api/data,192.168.1.20,10.0.0.2,TCP,443
Brute,SSH login attempt admin:password123,10.10.0.9,192.168.1.3,TCP,22
```

### Supported Label Names

The dataset analyzer searches for label columns such as:

```text
label, class, target, attack_label, is_attack, attack, attack_type,
category, type, y, intrusion, anomaly, malicious
```

### Label Mapping

Common normal labels are mapped to `0`:

```text
normal, benign, legitimate, safe, clean, 0, false, no
```

Common attack labels are mapped to `1`:

```text
attack, anomaly, intrusion, malicious, suspicious, ddos, dos,
portscan, brute force, sql injection, xss, botnet, infiltration,
heartbleed, probe, r2l, u2r
```

---

## Training and Model Persistence

The project supports automatic checkpoint loading and saving.

Model checkpoint locations:

```text
models/
├── quantum/
│   ├── quantum_ai_state.pkl
│   └── quantum_ai_metadata.json
├── classical/
│   ├── cnn/
│   │   ├── classical_cnn_state.pkl
│   │   └── classical_cnn_metadata.json
│   └── rnn/
│       ├── classical_rnn_state.pkl
│       └── classical_rnn_metadata.json
├── similarity/
│   ├── cosine_similarity_state.pkl
│   └── cosine_similarity_metadata.json
├── embeddings/
│   ├── embedding_engine_state.pkl
│   └── embedding_engine_metadata.json
└── datalake/
    ├── data_lake_state.pkl
    └── data_lake_metadata.json
```

On startup:

1. The system checks whether all expected checkpoints exist.
2. If all checkpoints are available, it loads them.
3. If checkpoints are missing, it trains the required components.
4. If `--retrain` is used, existing checkpoints are ignored and models are retrained.

> **GitHub recommendation**  
> Do not commit generated `.pkl` checkpoint files directly into the main repository. Use GitHub Releases, Git LFS, or provide instructions for users to train the checkpoints locally.

---

## Configuration

Main configuration file:

```text
config/settings.py
```

Architecture override file:

```text
config/arch_config.json
```

Important settings include:

- data source priorities,
- preprocessing parameters,
- embedding dimension,
- Quantum AI qubit/layer/optimizer settings,
- CNN and RNN training settings,
- cosine similarity thresholds,
- LLM confidence threshold,
- decision fusion weights,
- intrusion threshold,
- dashboard and feedback settings,
- model persistence options.

Example fusion configuration:

```python
FUSION = {
    "method": "weighted_average",
    "weights": {
        "quantum_ai": 0.20,
        "cosine_similarity": 0.20,
        "rnn_classical": 0.20,
        "cnn_classical": 0.20,
        "llm_analysis": 0.20,
    },
    "intrusion_threshold": 0.55,
}
```

---

## Security Notes

### 1. Do Not Commit Secrets

Do not commit:

- `.env` files,
- API keys,
- access tokens,
- private logs,
- local platform state,
- training databases containing sensitive data.

Only commit `.env.example`.

### 2. Pickle Checkpoint Warning

The project uses pickle-based model checkpoint loading.

```text
Never load pickle files from untrusted sources.
```

Pickle files can execute arbitrary code when loaded. For a production-grade project, consider replacing pickle checkpointing with safer formats such as:

- PyTorch `state_dict`,
- NumPy `.npz`,
- `safetensors`,
- JSON for metadata and non-binary configuration.

### 3. API Protection

The current Flask API is suitable for local experimentation. If deployed publicly, protect sensitive endpoints such as:

```text
/api/datalake/reset
/api/training-history/reset
/api/tune/start
/api/tune/apply
/api/train/<model_key>
/api/arch/config
```

Recommended production improvements:

- add authentication,
- add CSRF protection for browser-based actions,
- restrict CORS origins,
- set upload size limits,
- validate uploaded files strictly,
- avoid exposing debug information.

Example upload size limit:

```python
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
```

Example restricted CORS configuration:

```python
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000"]}})
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'qiskit'`

Install Qiskit:

```bash
pip install qiskit
```

Also add it to `requirements.txt`:

```text
qiskit>=1.0.0
```

### Flask App Starts but API Says System Is Not Ready

The orchestrator may still be initializing or training models. Check:

```http
GET /api/status
```

If an error is reported, inspect the terminal logs.

### LLM Agent Does Not Use GPT-4o

The LLM module falls back to rule-based analysis when the required environment variables are missing:

```env
AI_INTEGRATIONS_OPENAI_API_KEY=
AI_INTEGRATIONS_OPENAI_BASE_URL=
```

This fallback behavior is expected.

### Checkpoints Are Not Loading

Possible reasons:

- checkpoint files are missing,
- checkpoint metadata is incompatible with the current architecture,
- `.pkl` files were removed during cleanup,
- model version changed.

Run:

```bash
python main.py --retrain
```

### Training Database Should Not Be in Git

The file below is generated at runtime:

```text
storage/training_history.db
```

Do not commit it. The application can recreate it.

---

## Roadmap

Potential future improvements:

- Replace pickle checkpointing with safer model serialization.
- Add automated tests for preprocessing, model loading and API routes.
- Add Docker support.
- Add authentication for admin-level API routes.
- Improve dataset validation and upload size limits.
- Add CI workflow for linting and basic test execution.
- Add benchmark scripts for CICIDS/KDD-style datasets.
- Add reproducibility settings for random seeds.
- Separate research demo mode from production API mode.
- Provide example notebooks for model training and evaluation.

---


## Disclaimer

This project is an academic and experimental intrusion detection prototype. It is not a complete production-grade security product. Do not rely on it as the only defense mechanism in a real network environment without extensive validation, hardening, monitoring and expert review.
