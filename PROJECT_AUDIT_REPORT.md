# Hybrid QCNN Project Audit Report

## Scope

This report summarizes the current repository state after scanning the core code, configuration, tests, docs, and saved evaluation artifacts. It focuses on what the project is, how training and evaluation work, what scores have been achieved, and where the results differ across experiment stages.

## Executive Summary

The workspace is a multilingual sentiment analysis system built around transformer embeddings plus a PennyLane-based QCNN / hybrid quantum-classical model. The repo has a mature experiment trail, but the metrics are not single-number results: there are strong production-style runs, weaker cross-language holdout runs, and older baseline comparisons that report different scales and different validation setups.

Best validated results in the repository are in the 0.879 to 0.884 accuracy range with F1 around 0.879, ECE around 0.042, and Brier around 0.089, while the README also documents a 79.4% accuracy / 0.794 F1 / 0.930 ROC-AUC production-style run. Generalization and zero-shot artifacts are materially lower, with some holdout runs around 0.318 to 0.410 accuracy depending on the split and language set.

## What The Project Is

The project is a Hybrid QCNN sentiment analysis platform for English, Hindi, Bhojpuri, and Maithili.

The current model path is:

1. Sentence-transformer embeddings produce 384-dimensional text vectors.
2. A classical projection / bridge layer compresses the representation.
3. A PennyLane QCNN layer processes the reduced quantum input.
4. A classical head predicts negative, neutral, or positive sentiment.

The codebase also contains a FastAPI inference service, dataset normalization and balancing logic, a multi-seed training pipeline, calibration utilities, and a fairly large set of evaluation artifacts and plots.

## Repository Structure

Main areas reviewed:

- `main.py` as the experiment entry point.
- `backend/` for data, features, models, quantum layers, training, and evaluation helpers.
- `configs/` for the master training configuration and experiment registry.
- `datasets/` for raw and processed multilingual corpora.
- `evaluation/` for metrics, plots, telemetry, and archived benchmark runs.
- `docs/` for the scientific narrative, reports, and stabilization notes.
- `scripts/` for training, benchmarking, dataset processing, and diagnostics.
- `tests/` for import and behavior checks.

## Training Pipeline

The current training path is centered in `backend/training/train.py` and `backend/models/standardized.py`.

Key behaviors:

- Dataset loading and normalization happen in `backend/data/loader.py`.
- The data pipeline cleans text, normalizes labels, balances classes, and deduplicates text.
- `train_and_predict()` computes embeddings, splits the data, builds either the QCNN model or a classical baseline, trains it, calibrates probabilities, and computes metrics.
- The standardized PyTorch wrapper uses CUDA if available, otherwise CPU fallback.
- Evaluation includes accuracy, weighted precision/recall/F1, macro metrics, ROC-AUC, ECE, Brier score, per-language breakdown, failure analysis, and calibration plots.

The repo’s config file currently sets a research-scale run profile with 12 qubits, 10 QCNN layers, mixed precision, gradient accumulation, and a CUDA-first device target.

## Model Architecture

The hybrid model in `backend/models/hybrid_qcnn.py` uses:

- A language embedding for multilingual context.
- A classical bridge from 384 dimensions to a lower latent representation.
- A quantum path that feeds PennyLane circuit output into the final feature fusion.
- A residual-style fusion head that combines classical, quantum, and language features.

The QCNN layer in `backend/quantum/layers.py` uses PennyLane `default.qubit` and `qml.qnode(..., interface="torch", diff_method="backprop")`. That means the circuit logic is still CPU-oriented for simulation, even though the surrounding model can run on GPU for some tensor work if the environment supports it.

## Hardware and Runtime Implications

The workstation inspected earlier is strong on CPU and RAM but modest on GPU:

- CPU: Intel Xeon Gold 5218, 16 cores, 32 logical processors.
- RAM: about 63.63 GB.
- GPU: NVIDIA Quadro P2200 with 5 GB VRAM.
- NVIDIA driver and CUDA tooling are present, but the active Python environment in the probe did not have `torch` installed at that moment.

For this repository, that means:

- CPU training is the safe fallback and likely the practical baseline for QCNN simulation.
- GPU helps mainly if the PyTorch and transformer stack is installed with CUDA support and the workload is large enough to benefit.
- The QCNN itself is not a high-VRAM training target; the circuit simulation and quantum gradient work remain the expensive part.

## Best Scores Found

The repository contains multiple score regimes. The main ones are below.

### Strong validated runs

- `docs/final_report.md`: 0.8842 accuracy, 0.8791 F1, 0.0421 ECE, 0.0892 Brier score.
- `docs/runner_fix_report.md`: 0.879 accuracy, 0.880 precision, 0.879 recall, 0.879 F1, 0.968 ROC-AUC after the runner fix.
- `README.md`: 79.4% accuracy, 0.794 F1, 0.930 ROC-AUC for the cited Hybrid QCNN production result.

### Benchmark and ablation results

- `docs/analysis_report.md` says transformer baselines reached a performance score of 0.906 with 87.9% accuracy, while the hybrid QCNN group was lower at 77.5% accuracy on that experimental family.
- The same report says the QCNN still had stable class separation with ROC-AUC around 0.90 despite lower absolute accuracy.

### Generalization / transfer results

- `evaluation/latest/metrics/cross_language_results.json`: 0.317919 accuracy, 0.153382 F1, 0.457781 ROC-AUC, 0.139153 ECE, 0.708235 Brier score for a zero-shot transfer stage.
- `evaluation/latest/metrics/language_metrics.json`: English stage at 0.7764 accuracy and 0.77535 F1 with ROC-AUC 0.91577; Hindi stage at 0.4246 accuracy; Bhojpuri stage at 0.55646 accuracy; these are language-specific and not directly comparable to the global production result.
- `evaluation/latest/metrics/quantum_gain.json`: hybrid_accuracy 0.592933, baseline_accuracy 0.598600, so the measured quantum gain there is slightly negative.

### Baseline comparison

- `evaluation/latest/global_benchmark/results_N50000_latest.csv`: classical baseline accuracy 0.5822 and F1_macro 0.58145 with ECE 0.02062.

## Why The Scores Differ

The numbers differ because the repo is not reporting one single training regime. It contains several experiment families:

1. Full production-style hybrid runs.
2. Classical baseline comparisons.
3. Language-specific training.
4. Zero-shot / holdout transfer.
5. Micro-overfit and sanity checks.

The docs explicitly note that the hybrid QCNN can trail the transformer ceiling on large-scale benchmarks while still showing stable class separation. That is why some reports show good calibration and ROC-AUC, while some holdout runs are much weaker on raw accuracy.

## Data Pipeline Notes

The data loader in `backend/data/loader.py` does the following:

- Normalizes text with Unicode NFKC handling.
- Preserves Indic scripts such as Devanagari.
- Maps labels to 0/1/2.
- Detects language when missing.
- Filters short or repetitive samples.
- Removes duplicate text.
- Balances classes and language groups.
- Produces a processed `final_merged.csv` and a smaller `qcnn_subset_100k.csv`.

That is a major reason the repo has more stable runs than a naive text pipeline: it is enforcing deduplication and balancing before training.

## Evaluation And Diagnostics

The evaluation path includes:

- Weighted and macro classification metrics.
- ROC-AUC from binarized probabilities.
- Calibration curves and ECE.
- Failure case export.
- Per-language confusion matrices.
- Divergence checks to make sure ablations are not collapsing into identical outputs.

This is a stronger-than-average research workflow because it checks both final accuracy and calibration / generalization behavior.

## Tests And Quality Gates

The test folder currently includes:

- An import integrity check for core modules.
- A behavior test that verifies hybrid QCNN forward shape on CPU and checks class collapse detection.

The tests are useful, but the suite is still small relative to the size of the repo. It validates structure and some core model behavior, not the full end-to-end data and training pipeline.

## Notable Risks And Inconsistencies

1. The repo contains several generations of architecture naming and reporting language, so older docs and newer code do not always describe exactly the same model.
2. Some reports cite excellent production-style results, while holdout and transfer metrics are much lower. Those should not be mixed into one headline number.
3. The active environment probe earlier showed `torch` missing, so the runtime setup still needs a correct Python environment before any real training run.
4. The GPU is a Quadro P2200 with 5 GB VRAM, which is useful but not large enough to treat this as a heavy GPU training workstation.
5. The QCNN circuit is still simulation-heavy, so GPU availability does not automatically translate into big end-to-end speedups.

## Practical Conclusion

The best way to describe the project today is:

This is a mature multilingual hybrid QCNN research repo with a strong evaluation framework, a credible production-style score around 0.88 accuracy / 0.88 F1, and several weaker transfer experiments that show the main remaining challenge is generalization rather than raw training stability.

If you want a single operational takeaway, it is this:

- Use the calibrated production-style hybrid run as the best headline result.
- Treat the zero-shot and holdout numbers as stress tests.
- Do not compare every score in the repo directly, because the datasets and experiment goals are different.

## Files Used In This Audit

- `README.md`
- `main.py`
- `configs/master_config.yaml`
- `backend/data/loader.py`
- `backend/models/hybrid_qcnn.py`
- `backend/models/standardized.py`
- `backend/quantum/layers.py`
- `backend/training/train.py`
- `backend/evaluation/core.py`
- `backend/api/main.py`
- `tests/test_imports.py`
- `tests/test_behavior.py`
- `docs/final_report.md`
- `docs/analysis_report.md`
- `docs/runner_fix_report.md`
- `docs/scientific_stabilization_report_v2.md`
- `evaluation/latest/metrics/cross_language_results.json`
- `evaluation/latest/metrics/language_metrics.json`
- `evaluation/latest/metrics/quantum_gain.json`
- `evaluation/latest/global_benchmark/results_N50000_latest.csv`
