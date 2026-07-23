# Supplementary Evidence Bundle

This bundle accompanies the manuscript **“Auditing Quantum Contributions in a Hybrid Intrusion Detection System: A Leakage-Controlled Evaluation Across Three Benchmarks.”** It contains the scripts, immutable split files, predictions, model states, statistical outputs, circuit specifications, and checksums needed to inspect and reproduce the quantitative claims reported in the article.

## 1. Scope and provenance

The raw CICIDS2017, NSL-KDD, and UNSW-NB15 source files are not redistributed. Their source locations, exact filenames, selection rules, and available checksums are documented in `source_data_manifest.md`. The bundle contains the derived locked datasets and all analysis outputs used in the manuscript.

`vendored_repo/` is an anonymized snapshot of the exact reference-implementation modules imported by the primary scripts. Set `HYBRID_IDS_REPO` to this directory to avoid dependence on an external checkout.

## 2. Environment

The reported environment is Python 3.12.2 on CPU. Create an isolated environment with:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export HYBRID_IDS_REPO="$(pwd)/vendored_repo"
```

The pinned packages include the tabular-baseline dependencies (`xgboost`, `lightgbm`, and `catboost`) used by `16_final_additions.py`.

## 3. Primary locked-test workflow

Run from the root of this bundle:

```bash
python 01_build_locked_split.py      # locked split and zero-overlap assertion
python 02_train_models.py            # all primary agents, five seeds
python 03_evaluate.py                # metrics, fusion, ablations
python 04_latency_benchmark.py       # repeated-session single-event latency
python 05_ssh_case_study.py          # numerical fusion false-positive analysis
python 06_make_figures.py            # principal manuscript figures
python 07_paired_analysis.py         # paired tests and per-seed transparency
```

Outputs are written below `results/`, including `split_manifest.csv`, `split_summary.json`, `locked_dataset.csv`, model checkpoints, per-seed predictions, metrics, fusion ablations, latency samples, and paired-test outputs.

## 4. Additional analyses reported in the manuscript

```bash
python 08_train_curves.py                # CNN/RNN train-validation curves
python 09_phaseVI_analysis.py            # calibration, CIs, McNemar analyses
python 10_circuit_spec.py                 # gate counts, logical/transpiled depth
python 11_replication.py                  # NSL-KDD and UNSW-NB15 replication
python 12_quantum_improvements.py         # readout/encoding/architecture study
python 13_quantum_improved_figure.py      # improvement-study figure
python 14_option4_optimization.py         # enhanced optimization protocol
python 15_circuit_figures.py              # circuit diagram exports
python 16_final_additions.py              # tabular, multiclass, shot-noise analyses
python 17_devconfig_locked_check.py       # development-record winner config under the locked protocol
```

Associated outputs are stored in:

- `results_phaseVI/`
- `results_replication/`
- `results_quantum_improved/`
- `results_final_additions/`
- `results_devconfig_check/`
- `figures/`

## 5. Key files and what they verify

| File or directory | Purpose |
|---|---|
| `01_build_locked_split.py`, `results/split_manifest.csv`, `results/split_summary.json` | Group-aware, feature-space-consistent split; 43 duplicate groups and 49 row reassignments; zero cross-partition fingerprint overlap. |
| `02_train_models.py`, `vqc_model.py`, `results/model_states/` | Frozen model definitions and trained states for the reported seeds. |
| `results/predictions/` | Train/validation/test predicted probabilities used by all primary metrics. |
| `03_evaluate.py`, `results/metrics_*.json` | Primary locked-test metrics and fusion results. |
| `cosine_scoring.py` | Single shared implementation of the cosine/k-NN anomaly score. |
| `04_latency_benchmark.py`, `results/latency_benchmark.json`, `results/raw_latency_*.npy` | Repeated-session latency values and raw timing samples. |
| `05_ssh_case_study.py`, `results/ssh_case_study.json` | Numerical analysis of the dashboard fusion-consistency example. |
| `07_paired_analysis.py`, `results/paired_analysis.json` | Paired tests, tie handling, and per-seed transparency. |
| `08_train_curves.py`, `results_phaseVI/train_curves.json` | CNN/RNN training dynamics. |
| `09_phaseVI_analysis.py`, `results_phaseVI/phaseVI_stats.json` | Calibration, confidence intervals, and McNemar tests. |
| `10_circuit_spec.py`, `results_phaseVI/circuit_spec.json` | Circuit resource specification. |
| `11_replication.py`, `results_replication/` | Frozen-protocol replication on NSL-KDD and UNSW-NB15. |
| `12_quantum_improvements.py`, `14_option4_optimization.py`, `results_quantum_improved/` | Readout, encoding, architecture, and optimization improvements. |
| `16_final_additions.py`, `results_final_additions/` | Strong tabular baselines, multiclass delimitation, and shot-noise robustness. |
| `17_devconfig_locked_check.py`, `results_devconfig_check/` | The development record's best auto-tuned quantum hyperparameter setting (4 qubits, 3 hidden layers) transferred to the locked protocol, with a matched Classical-4D baseline. |
| `source_data_manifest.md` | Raw-source provenance and transformation notes. |
| `SHA256SUMS.txt` | Integrity manifest for every file in the final evidence bundle. |

## 6. Verification

After extraction, verify integrity from the bundle root:

```bash
sha256sum -c SHA256SUMS.txt
```

All entries should report `OK`. The manifest is regenerated only after the bundle is frozen.

## 7. Known limitations

- The stochastic models use exactly the five reported seeds (0–4); no additional seeds were run and discarded.
- Classical-8D and cosine/k-NN are deterministic and are therefore evaluated once.
- All quantum circuits are evaluated with exact statevector simulation. The shot-noise analysis samples finite measurement counts, but no physical-QPU or hardware-noise experiment is included.
- The three benchmark corpora are scale-matched subsets rather than full temporal or cross-environment deployments.
