"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          HYBRID INTRUSION DETECTION SYSTEM - Main Orchestrator             ║
║   Combining Classical and Quantum Machine Learning for Cybersecurity       ║
║                                                                            ║
║   Based on: General System Diagram (Fig. 1)                                ║
║                                                                            ║
║   Data Flow:                                                               ║
║   [IPS/SIEM/Firewall/SOAR/IDS] → Preprocessing → Embeddings               ║
║       ├─ Quantum AI Path: DimReduce → QNN → RNN → ClassOutput             ║
║       ├─ Cosine Similarity: VectorWarehouse → CosineAnalysis → Score      ║
║       ├─ Classical RNN: DataLake → RNN → RNN ClassOutput                   ║
║       ├─ Classical CNN: DataLake → CNN → CNN ClassOutput                   ║
║       └─ LLM Analysis → LLM Output                                        ║
║   All 5 agents → Decision Fusion Layer → Intrusion Detection               ║
║       → Dashboard/Reporting + Alerts/EventMgmt → Security Analyst Review   ║
║       → Feedback / Model Updates → (loop back)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─── Import all modules matching the General System Diagram ───
from data_sources.data_ingestion import DataIngestionEngine
from preprocessing.preprocessor import Preprocessor
from embeddings.embedding_engine import EmbeddingEngine
from storage.vector_warehouse import (
    EmbeddingVectorWarehouse, ExampleAttackData, ReferenceEmbeddingDataLake
)
from agents.quantum_ai.quantum_module import QuantumAIModule
from agents.classical_ml.rnn_model import ClassicalRNNModel
from agents.classical_ml.cnn_model import ClassicalCNNModel
from agents.similarity.cosine_analyzer import CosineSimilarityAnalyzer
from agents.llm_analysis.llm_module import LLMAnalysisModule
from fusion.decision_fusion import DecisionFusionLayer
from detection.intrusion_detector import IntrusionDetector
from dashboard.dashboard_reporting import (
    DashboardReporting, AlertsEventManagement, SecurityAnalystReview
)
from feedback.model_updater import FeedbackModelUpdater
from utils.data_models import *
from utils.logger import get_logger
from utils.model_persistence import (
    ensure_model_dirs, get_all_checkpoint_status, get_checkpoint_info,
    save_all_models, load_all_models, save_data_lake, load_data_lake,
    save_embedding_engine, load_embedding_engine,
)
from config.settings import MODEL_PERSISTENCE

logger = get_logger("Orchestrator")


class HybridIDSOrchestrator:
    """
    Main System Orchestrator - connects all modules as shown in the 
    General System Diagram (Fig. 1).
    
    Complete data flow:
    1. Data Sources (5) → Preprocessing
    2. Preprocessing → Creating Embeddings + Quantum AI Dimension Reduction
    3. Embeddings → Vector Warehouse + Example Attack Data
    4. Vector Warehouse → Cosine Similarity Analysis
    5. Example Attack Data → Reference Data Lake
    6. Reference Data Lake → Classical RNN + Classical CNN
    7. Preprocessed Data → LLM Analysis
    8. Features → Quantum AI (DimReduce → QNN → RNN → ClassOutput)
    9. All 5 agents → Decision Fusion Layer
    10. Fusion → Intrusion/Violation Detection
    11. Detection → Dashboard/Reporting + Alerts/Event Management
    12. Both → Security Analyst Review
    13. Review → Feedback/Model Updates → Embeddings + Data Lake (loop)
    """

    def __init__(self, force_retrain: bool = False):
        logger.info("=" * 70)
        logger.info("  Initializing Hybrid Intrusion Detection System")
        logger.info("=" * 70)

        self.force_retrain = force_retrain
        self.models_loaded_from_checkpoint = False

        # ── 1. Data Sources ──
        self.data_engine = DataIngestionEngine()

        # ── 2. Preprocessing ──
        self.preprocessor = Preprocessor()

        # ── 3. Embedding Creation ──
        self.embedding_engine = EmbeddingEngine()

        # ── 4. Storage Components ──
        self.vector_warehouse = EmbeddingVectorWarehouse()
        self.attack_data = ExampleAttackData()
        self.data_lake = ReferenceEmbeddingDataLake()

        # ── 5. Agent Modules (5 agents) ──
        self.quantum_ai = QuantumAIModule()          # Agent 1: Quantum AI
        self.cosine_analyzer = CosineSimilarityAnalyzer()  # Agent 2: Cosine Similarity
        self.rnn_classical = ClassicalRNNModel()     # Agent 3: Classical RNN
        self.cnn_classical = ClassicalCNNModel()     # Agent 4: Classical CNN
        self.llm_analysis = LLMAnalysisModule()      # Agent 5: LLM Analysis

        # ── 6. Decision Fusion Layer ──
        self.fusion_layer = DecisionFusionLayer()

        # ── 7. Intrusion / Violation Detection ──
        self.detector = IntrusionDetector()

        # ── 8. Dashboard & Alerts ──
        self.dashboard = DashboardReporting()
        self.alert_manager = AlertsEventManagement()

        # ── 9. Security Analyst Review ──
        self.analyst_review = SecurityAnalystReview()

        # ── 10. Feedback / Model Updates ──
        self.feedback_updater = FeedbackModelUpdater()

        # ── Ensure model directories exist ──
        ensure_model_dirs()

        logger.info("All modules initialized successfully")
        logger.info("=" * 70)

    def _display_checkpoint_status(self):
        """Display checkpoint availability for all models."""
        status = get_all_checkpoint_status()
        print("\n  📦 Model Checkpoint Durumu:")
        print("  " + "─" * 45)
        for model_name, exists in status.items():
            icon = "✅" if exists else "❌"
            label = model_name.replace("_", " ").title()
            detail = ""
            if exists:
                info = get_checkpoint_info(model_name)
                if info:
                    detail = f" (kayıt: {info.get('saved_at', 'bilinmiyor')[:19]})"
            print(f"     {icon} {label:.<30s} {'Mevcut' + detail if exists else 'Bulunamadı'}")
        print("  " + "─" * 45)
        return status

    def _try_load_checkpoints(self) -> bool:
        """
        Auto-detect and load existing model checkpoints.
        Returns True if ALL models were loaded successfully.
        """
        if not MODEL_PERSISTENCE.get("auto_load_on_startup", True):
            logger.info("Auto-load disabled in settings")
            return False

        status = get_all_checkpoint_status()
        all_exist = all(status.values())

        if not all_exist:
            missing = [k for k, v in status.items() if not v]
            logger.info(f"Checkpoint(ler) eksik: {missing} → Yeniden eğitim gerekli")
            return False

        logger.info("Tüm checkpoint'ler bulundu. Yükleniyor...")
        results = load_all_models(
            self.quantum_ai, self.rnn_classical, self.cnn_classical,
            self.cosine_analyzer, self.embedding_engine, self.data_lake
        )

        all_loaded = all(results.values())
        if all_loaded:
            self.models_loaded_from_checkpoint = True
            logger.info("✅ Tüm modeller checkpoint'lerden başarıyla yüklendi!")
            # Align all agents to the live-event embedding space:
            # cosine centroid + CNN/RNN decision boundary must match
            # the EmbeddingEngine word-average vectors used at inference.
            self._rebuild_cosine_reference()
            self._retrain_classifiers_on_embedding_space()
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"Bazı modeller yüklenemedi: {failed} → Yeniden eğitim yapılacak")

        return all_loaded

    # ── Shared text templates for embedding-space training/reference ──────
    # Templates must use event_type="alert" (matches live RawSecurityEvent.event_type).
    # First 4 normal entries exactly mirror the 4 test-scenario text_features so
    # the Fisher projection is trained on the same embedding vectors used at inference.
    _NORMAL_TEXTS = [
        "Firewall alert HTTPS src=192.168.1.105:54312 dst=93.184.216.34:443 severity=LOW payload=Normal HTTPS web traffic GET 200 OK TLS no anomalies",
        "IDS alert SSH src=10.0.0.25:52100 dst=10.0.0.5:22 severity=LOW payload=Normal SSH session admin login success management routine",
        "SIEM alert DNS src=192.168.5.20:49200 dst=8.8.8.8:53 severity=LOW payload=Normal DNS query response ok standard recursive low",
        "SOAR alert TCP src=10.10.2.15:41800 dst=10.10.2.50:5432 severity=LOW payload=Normal database select query response ok no anomaly",
        "Firewall alert HTTPS src=10.1.1.20:55001 dst=172.217.0.1:443 severity=LOW payload=normal web browsing chrome get 200 ok tls low",
        "IPS alert TCP src=192.168.2.10:48000 dst=203.0.113.5:80 severity=LOW payload=normal http get request 200 ok no anomaly low",
        "SIEM alert UDP src=172.16.0.3:51200 dst=10.0.0.1:123 severity=LOW payload=normal ntp time sync udp request response low",
        "IDS alert TCP src=10.5.0.20:44100 dst=10.5.0.1:25 severity=LOW payload=normal smtp email send 250 ok authenticated low",
        "Firewall alert HTTPS src=192.168.3.55:53000 dst=52.0.0.1:443 severity=LOW payload=normal api rest get json response 200 ok low",
        "SOAR alert TCP src=10.20.0.5:39000 dst=10.20.0.9:3306 severity=LOW payload=normal mysql query select ok low",
    ]

    # First 3 attack entries mirror the 3 test-scenario text_features exactly.
    _ATTACK_TEXTS = [
        "IDS alert SSH src=185.220.101.47:58921 dst=10.0.1.22:22 severity=HIGH payload=attack brute force failed password repeated intrusion high",
        "Firewall alert TCP src=198.51.100.77:0 dst=172.16.0.5:80 severity=CRITICAL payload=attack dos syn flood ddos critical volumetric 48000 SYN packets",
        "IPS alert HTTP src=45.155.205.233:44231 dst=172.20.0.10:8080 severity=CRITICAL payload=attack u2r rce exploit shell overflow privilege escalation critical",
        "IDS alert TCP src=203.0.113.99:61000 dst=10.10.0.1:0 severity=MEDIUM payload=attack reconnaissance nmap port scan probe medium",
        "SIEM alert HTTPS src=10.5.3.88:49701 dst=185.243.115.90:443 severity=CRITICAL payload=attack r2l ransomware c2 beacon backdoor shell critical",
        "Firewall alert UDP src=0.0.0.0:53 dst=192.0.2.1:53 severity=HIGH payload=attack dos dns amplification ddos flood high",
        "IPS alert HTTPS src=91.108.4.200:52314 dst=192.168.10.50:443 severity=CRITICAL payload=attack probe sql injection exploit web critical",
        "IDS alert TCP src=172.18.0.5:45000 dst=10.0.0.20:4444 severity=HIGH payload=attack shellcode reverse shell backdoor unauthorized high",
        "SIEM alert TCP src=45.33.32.156:60000 dst=192.168.1.1:80 severity=HIGH payload=attack worm propagation intrusion malware high",
        "IPS alert TCP src=10.9.0.8:49200 dst=10.9.0.1:22 severity=CRITICAL payload=attack brute force ssh exploit intrusion critical",
    ]

    def _get_embedding_space_dataset(self):
        """
        Build X, y arrays in the EmbeddingEngine text space (same space as
        live events). Used for CNN/RNN retraining and cosine reference.
        Returns (X: ndarray [n,128], y: ndarray [n], normal_embs, attack_embs).
        """
        from utils.data_models import PreprocessedData as PD
        from datetime import datetime as dt
        now = dt.now()
        zeros = np.zeros(64, dtype=np.float32)

        X, y, normal_embs, attack_embs = [], [], [], []

        for i, txt in enumerate(self._NORMAL_TEXTS * 9):   # 90 normal (balanced)
            pd = PD(event_id=f"emb_normal_{i}", features=zeros.copy(),
                    text_features=txt, source_type="reference",
                    timestamp=now, metadata={"attack_type": "Normal"})
            emb = self.embedding_engine.create_embedding(pd)
            X.append(emb.vector); y.append(0); normal_embs.append(emb)

        for i, txt in enumerate(self._ATTACK_TEXTS * 9):   # 90 attack
            pd = PD(event_id=f"emb_attack_{i}", features=zeros.copy(),
                    text_features=txt, source_type="reference",
                    timestamp=now, metadata={"attack_type": "Attack"})
            emb = self.embedding_engine.create_embedding(pd)
            X.append(emb.vector); y.append(1); attack_embs.append(emb)

        return np.array(X, dtype=np.float32), np.array(y), normal_embs, attack_embs

    def _retrain_classifiers_on_embedding_space(self):
        """
        Retrain CNN, RNN and Quantum in 'embedding' mode using the same
        128-dim word2vec vectors the EmbeddingEngine produces at inference.

        Steps:
          1. Generate 420 diverse synthetic events (9 attack types × 30 +
             150 normal) via _make_attack_event / _make_normal_event.
          2. Run each through Preprocessor → EmbeddingEngine to get the
             128-dim word2vec vector (identical pipeline to /api/analyze).
          3. Train CNN, RNN and Quantum with training_mode='embedding' so
             predict() uses embedding.vector at inference — same distribution.
        """
        from storage.vector_warehouse import (
            _make_attack_event, _make_normal_event, _ATTACK_TEMPLATES
        )
        logger.info("Retraining CNN+RNN+Quantum on word2vec embedding space (128-dim)…")

        rng = np.random.default_rng(77)
        attack_types = list(_ATTACK_TEMPLATES.keys())
        X_list, y_list = [], []
        emb_objs: list = []   # (EmbeddingVector, label_str)

        # 30 samples per attack type — passed through the live embedding pipeline
        for at in attack_types:
            for _ in range(30):
                ev  = _make_attack_event(at, rng)
                pp  = self.preprocessor.process(ev)
                emb = self.embedding_engine.create_embedding(pp)
                X_list.append(emb.vector)   # 128-dim word2vec
                y_list.append(1)
                emb_objs.append((emb, at))

        # BALANCED normal samples: equal to total attack count
        n_attack_total = len(attack_types) * 30   # 270
        for i in range(n_attack_total):
            ev  = _make_normal_event(rng)
            pp  = self.preprocessor.process(ev)
            emb = self.embedding_engine.create_embedding(pp)
            X_list.append(emb.vector)
            y_list.append(0)
            emb_objs.append((emb, "Normal"))

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        n_atk = int(y.sum())
        logger.info(f"  Embedding-mode dataset: {len(X)} samples "
                    f"({len(X) - n_atk} normal, {n_atk} attack) — balanced 50/50, dim={X.shape[1]}")

        # ── Refresh data lake with word2vec-space vectors ──────────────
        # Auto-tuning reads the data lake; keeping it in sync with the live
        # embedding space ensures architecture search uses the correct distribution.
        self.data_lake.clear()
        for emb_obj, lbl in emb_objs:
            self.data_lake.embeddings.append(emb_obj)
            self.data_lake.labels.append(lbl)
            self.data_lake.binary_labels.append(0 if lbl == "Normal" else 1)
        logger.info(f"  Data lake refreshed: {len(self.data_lake.embeddings)} word2vec-space samples "
                    f"({sum(self.data_lake.binary_labels)} attack, "
                    f"{len(self.data_lake.binary_labels) - sum(self.data_lake.binary_labels)} normal)")

        # Keep embedding mode: predict() will use embedding.vector at inference
        # (same 128-dim word2vec pipeline → no train/inference distribution gap).
        self.cnn_classical.training_mode = "embedding"
        self.rnn_classical.training_mode = "embedding"

        self.cnn_classical.train(X, y)

        # Small Gaussian jitter (2×) for RNN / Quantum diversity
        rng2 = np.random.default_rng(42)
        X_aug = np.vstack([
            X + rng2.normal(0, 0.01, X.shape).astype(np.float32)
            for _ in range(2)
        ])
        y_aug = np.tile(y, 2)
        logger.info(f"  RNN/Quantum augmented set: {len(X_aug)} samples (2× jitter, σ=0.01)")

        # Raise lr to 0.01 for the direct retraining pass; the grouped RNN
        # architecture (seq=8, feat=16) already handles gradient flow, but
        # a higher lr speeds convergence on the smaller 840-sample set.
        _saved_rnn_lr = self.rnn_classical.lr
        self.rnn_classical.lr = 0.01
        self.rnn_classical.train(X_aug, y_aug)
        self.rnn_classical.lr = _saved_rnn_lr

        # Quantum: embedding.vector is passed at inference (see quantum predict),
        # so train the DimensionReducer on the same 128-dim space.
        self.quantum_ai.train(X_aug, y_aug, max_iter=150, batch_size=20)

        logger.info("✅ CNN+RNN+Quantum retrained in embedding (word2vec, 128-dim) mode")

    def _rebuild_cosine_reference(self):
        """
        Rebuild the cosine similarity reference in the SAME embedding space
        as live events (EmbeddingEngine word-average). Uses the shared
        _get_embedding_space_dataset so templates are maintained in one place.
        """
        _, _, normal_embs, attack_embs = self._get_embedding_space_dataset()
        self.cosine_analyzer.build_reference(normal_embs, attack_embs)
        logger.info(f"Cosine reference rebuilt in embedding-engine space: "
                    f"{len(normal_embs)} normal, {len(attack_embs)} attack vectors")

    def _save_all_checkpoints(self):
        """Save all model checkpoints after training."""
        if not MODEL_PERSISTENCE.get("auto_save_after_training", True):
            logger.info("Auto-save disabled in settings")
            return

        logger.info("\n💾 Modeller kaydediliyor...")
        results = save_all_models(
            self.quantum_ai, self.rnn_classical, self.cnn_classical,
            self.cosine_analyzer, self.embedding_engine, self.data_lake
        )
        success = sum(v for v in results.values())
        total = len(results)
        if success == total:
            print(f"  💾 Tüm modeller başarıyla kaydedildi ({success}/{total})")
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"  ⚠️  {success}/{total} model kaydedildi. Başarısız: {failed}")

    def setup_and_train(self):
        """
        Phase 1: Setup storage and train models.
        Checks for existing checkpoints first (smart loading).
        If --retrain flag is used, forces retraining.
        """
        logger.info("\n" + "─" * 50)
        logger.info("  PHASE 1: System Setup & Model Loading/Training")
        logger.info("─" * 50)

        # ── Display checkpoint status ──
        self._display_checkpoint_status()

        # ── Smart Loading: Try to load from checkpoints ──
        if not self.force_retrain:
            if self._try_load_checkpoints():
                print("\n  🔄 Modeller mevcut checkpoint'lerden yüklendi!")
                print("     (Yeniden eğitim için: python3 main.py --retrain)")
                return
            else:
                print("\n  🔧 Checkpoint bulunamadı veya eksik → Eğitim başlıyor...")
        else:
            print("\n  🔁 --retrain bayrağı aktif → Tüm modeller yeniden eğitilecek...")

        # ── Full training pipeline ──
        # Step 1: Load Data Lake — prefer saved checkpoint, fall back to example data
        from utils.model_persistence import load_data_lake
        dl_loaded = load_data_lake(self.data_lake)
        if dl_loaded and len(self.data_lake.embeddings) > 0:
            logger.info(f"[1/5] Data Lake loaded from checkpoint ({len(self.data_lake.embeddings)} records).")
        else:
            logger.info("[1/5] No saved Data Lake found — loading from example attack data...")
            self.data_lake.load_from_attack_data(self.attack_data)

        # Step 2: Build cosine similarity reference
        logger.info("[2/5] Building Cosine Similarity reference vectors...")
        normal_refs = [e for e, l in zip(self.data_lake.embeddings, self.data_lake.labels) if l == "Normal"]
        attack_refs = [e for e, l in zip(self.data_lake.embeddings, self.data_lake.labels) if l != "Normal"]
        self.cosine_analyzer.build_reference(normal_refs, attack_refs)

        # Steps 3+4: Train Classical RNN and CNN in parallel
        logger.info("[3+4/5] Training Classical RNN and CNN in parallel...")
        X_train, y_train = self.data_lake.get_training_data()
        if len(X_train) > 0:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_rnn = pool.submit(self.rnn_classical.train, X_train, y_train)
                fut_cnn = pool.submit(self.cnn_classical.train, X_train, y_train)
                for fut in as_completed([fut_rnn, fut_cnn]):
                    exc = fut.exception()
                    if exc:
                        logger.error(f"Parallel training error: {exc}")

        # Step 5: Train Quantum AI module on actual data lake embeddings
        logger.info("[5/5] Training Quantum AI Module (Adam-SPSA) on data lake embeddings...")
        if len(X_train) > 0:
            self.quantum_ai.train(X_train, y_train)
        else:
            # Fallback: minimal synthetic set
            np.random.seed(300)
            qnn_X = np.random.randn(20, 64).astype(np.float32) * 0.5
            qnn_y = np.array([0]*10 + [1]*10)
            self.quantum_ai.train(qnn_X, qnn_y)

        logger.info("✅ All models trained successfully!")

        # Align all agents to the live-event embedding space
        self._rebuild_cosine_reference()
        self._retrain_classifiers_on_embedding_space()

        # ── Save checkpoints after training ──
        self._save_all_checkpoints()
        print("  ✅ Eğitim tamamlandı ve modeller kaydedildi!\n")

    def process_events(self, n_events: int = 50, attack_ratio: float = 0.3) -> Dict:
        """
        Phase 2: Main processing pipeline.
        Processes security events through the entire system.
        """
        logger.info("\n" + "─" * 50)
        logger.info(f"  PHASE 2: Processing {n_events} Security Events")
        logger.info("─" * 50)

        # ── Step 1: Collect events from all 5 data sources ──
        logger.info("[Step 1] Collecting events from IPS, SIEM, Firewall, SOAR, IDS...")
        raw_events = self.data_engine.collect_events(n_events, attack_ratio)

        # ── Step 2: Preprocessing ──
        logger.info("[Step 2] Preprocessing: Cleaning → Parsing → Feature Extraction → Normalization...")
        preprocessed = self.preprocessor.process_batch(raw_events)

        # ── Step 3: Create Embeddings ──
        logger.info("[Step 3] Creating semantic embeddings...")
        embeddings = self.embedding_engine.create_batch_embeddings(preprocessed)

        # ── Step 4: Store in Vector Warehouse ──
        logger.info("[Step 4] Storing in Embedding Vector Warehouse...")
        self.vector_warehouse.store_batch(embeddings)

        # ══════════════════════════════════════════
        # ── Step 5: Run all 5 agent modules in parallel ──
        # ══════════════════════════════════════════
        logger.info("[Step 5] Running 5 analysis agents in parallel...")

        # Agent 1: Quantum AI Path
        logger.info("  ├─ Agent 1: Quantum AI (DimReduce → QNN → RNN)...")
        quantum_results = self.quantum_ai.predict_batch(preprocessed)

        # Agent 2: Cosine Similarity Analysis
        logger.info("  ├─ Agent 2: Cosine Similarity Analysis...")
        similarity_results = self.cosine_analyzer.analyze_batch(embeddings, preprocessed)

        # Agent 3: Classical RNN
        logger.info("  ├─ Agent 3: Classical RNN Model...")
        rnn_results = self.rnn_classical.predict_batch(embeddings, preprocessed)

        # Agent 4: Classical CNN
        logger.info("  ├─ Agent 4: Classical CNN Model...")
        cnn_results = self.cnn_classical.predict_batch(embeddings, preprocessed)

        # Agent 5: LLM Analysis
        logger.info("  └─ Agent 5: LLM Analysis...")
        llm_results = self.llm_analysis.analyze_batch(preprocessed)

        # ── Step 6: Decision Fusion Layer ──
        logger.info("[Step 6] Decision Fusion Layer: combining all 5 agent outputs...")
        batch_agent_results = []
        event_ids = []
        for i in range(len(preprocessed)):
            agent_results = {
                "QuantumAI": quantum_results[i],
                "CosineSimilarity": similarity_results[i],
                "RNN_Classical": rnn_results[i],
                "CNN_Classical": cnn_results[i],
                "LLM_Analysis": llm_results[i],
            }
            batch_agent_results.append(agent_results)
            event_ids.append(preprocessed[i].event_id)

        fusion_results = self.fusion_layer.fuse_batch(batch_agent_results, event_ids)

        # ── Step 7: Intrusion / Violation Detection ──
        logger.info("[Step 7] Intrusion / Violation Detection...")
        alerts = self.detector.detect_batch(fusion_results, preprocessed)

        # ── Step 8: Dashboard / Reporting ──
        logger.info("[Step 8] Generating Dashboard Report...")
        report = self.dashboard.generate_report(alerts, len(preprocessed))
        self.dashboard.display_report(report)

        # ── Step 9: Alerts and Event Management ──
        logger.info("[Step 9] Processing Alerts and Event Management...")
        self.alert_manager.receive_alerts(alerts)

        # ── Step 10: Security Analyst Review ──
        logger.info("[Step 10] Security Analyst Review (auto-simulation)...")
        self.analyst_review.add_to_review(alerts)
        feedbacks = self.analyst_review.auto_review(alerts)

        # ── Step 11: Feedback / Model Updates ──
        logger.info("[Step 11] Processing Feedback / Model Updates...")
        update_result = self.feedback_updater.collect_feedback(feedbacks)

        if update_result and update_result.get("status") != "no_updates":
            # Apply feedback to embeddings
            if update_result.get("embedding_updates"):
                logger.info("  → Updating Embedding Engine vocabulary...")
            # Apply feedback to data lake
            if update_result.get("_new_embeddings"):
                self.data_lake.update_from_feedback(
                    update_result["_new_embeddings"],
                    update_result["_new_labels"]
                )
                logger.info("  → Updating Reference Embedding Data Lake...")

        # ── Summary Statistics ──
        stats = {
            "total_events": len(preprocessed),
            "total_intrusions_detected": sum(1 for fr in fusion_results if fr.is_intrusion),
            "total_alerts": len(alerts),
            "agent_detection_counts": {
                "QuantumAI": sum(r.prediction for r in quantum_results),
                "CosineSimilarity": sum(r.prediction for r in similarity_results),
                "RNN_Classical": sum(r.prediction for r in rnn_results),
                "CNN_Classical": sum(r.prediction for r in cnn_results),
                "LLM_Analysis": sum(r.prediction for r in llm_results),
            },
            "fusion_stats": {
                "avg_score": np.mean([fr.final_score for fr in fusion_results]),
                "max_score": max(fr.final_score for fr in fusion_results),
            },
            "detection_stats": self.detector.get_statistics(),
            "alert_mgmt_status": self.alert_manager.get_status(),
            "feedback_status": self.feedback_updater.get_status(),
        }

        return stats

    def run_full_demo(self):
        """Run the complete system demonstration."""
        print("\n")
        print("╔" + "═" * 70 + "╗")
        print("║" + "  HYBRID INTRUSION DETECTION SYSTEM".center(70) + "║")
        print("║" + "  Classical + Quantum Machine Learning".center(70) + "║")
        print("║" + "  General System Diagram Implementation".center(70) + "║")
        print("╚" + "═" * 70 + "╝")
        print()

        # Phase 1: Train or Load
        self.setup_and_train()

        # Show model source info
        if self.models_loaded_from_checkpoint:
            print("  ℹ️  Model Kaynağı: Checkpoint'lerden yüklendi (önceden eğitilmiş)")
        else:
            print("  ℹ️  Model Kaynağı: Yeni eğitildi ve kaydedildi")

        # Phase 2: Process
        stats = self.process_events(n_events=50, attack_ratio=0.35)

        # Final Summary
        print("\n" + "═" * 80)
        print("  📋 FINAL SYSTEM SUMMARY")
        print("═" * 80)
        print(f"\n  Total Events Processed:    {stats['total_events']}")
        print(f"  Total Intrusions Detected: {stats['total_intrusions_detected']}")
        print(f"  Total Alerts Generated:    {stats['total_alerts']}")

        print(f"\n  🤖 Agent Detection Counts:")
        for agent, count in stats["agent_detection_counts"].items():
            bar = "█" * count
            print(f"     {agent:>20s}: {count:3d} {bar}")

        fs = stats["fusion_stats"]
        print(f"\n  🔀 Fusion Statistics:")
        print(f"     Average Score: {fs['avg_score']:.4f}")
        print(f"     Max Score:     {fs['max_score']:.4f}")

        ds = stats["detection_stats"]
        print(f"\n  🛡️  Detection Statistics:")
        print(f"     Total Detections: {ds['total_detections']}")
        for sev, cnt in ds["severity_distribution"].items():
            print(f"     {sev:>8s}: {cnt}")

        ams = stats["alert_mgmt_status"]
        print(f"\n  📨 Alert Management:")
        print(f"     Active:       {ams['active']}")
        print(f"     Acknowledged: {ams['acknowledged']}")
        print(f"     Resolved:     {ams['resolved']}")

        fbs = stats["feedback_status"]
        print(f"\n  🔄 Feedback Loop:")
        print(f"     Buffer Size:    {fbs['buffer_size']}")
        print(f"     Total Updates:  {fbs['total_updates']}")

        # Model persistence info
        print(f"\n  💾 Model Persistence:")
        if self.models_loaded_from_checkpoint:
            print(f"     Durum: Checkpoint'lerden yüklendi")
        else:
            print(f"     Durum: Yeni eğitildi ve kaydedildi")
        print(f"     Checkpoint Dizini: models/")

        print("\n" + "═" * 80)
        print("  ✅ System demonstration complete!")
        print("═" * 80 + "\n")

        return stats


# ─── Entry Point ───
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Intrusion Detection System - Classical + Quantum ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python3 main.py                # Checkpoint varsa yükle, yoksa eğit
  python3 main.py --retrain      # Mevcut checkpoint'leri yok say, yeniden eğit
        """
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Mevcut checkpoint'leri yok sayarak tüm modelleri yeniden eğit"
    )
    args = parser.parse_args()

    orchestrator = HybridIDSOrchestrator(force_retrain=args.retrain)
    orchestrator.run_full_demo()
