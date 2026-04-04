# Project Structure Guide

## Directory Organization

This project follows a clean, modular architecture with strict separation of concerns.

### Core Pipeline Directories

```
core/                    # ⭐ Canonical production code (Single Source of Truth)
├── data/                # Data ingestion & processing
│   └── loader.py        # HF datasets, normalization, balancing, export
├── features/            # Feature extraction pipelines
│   ├── embedding.py     # Unified embedding (MiniLM 384-dim + TF-IDF fallback)
│   └── extractors.py    # Feature extraction utilities
├── models/              # Unified model interface
│   ├── standardized.py  # StandardModel wrapper (sklearn API contract)
│   └── builders.py      # Model factory (LogisticRegression, MLP variants)
├── training/            # Training orchestration
│   ├── train.py         # Main training with embedding cache & reproducibility
│   └── utils.py         # Training utilities
├── inference/           # Model deployment & prediction
│   ├── predictor.py     # Batch inference engine
│   └── registry.py      # Model registry & loading
└── utils/               # Shared utilities
    ├── config.py        # Config loading & validation
    └── hf_datasets_import.py  # Safe HuggingFace import shim
```

### Project Structure

```
scripts/                 # 🎯 Public CLI entry points (Thin wrappers)
├── run_data.py         # python scripts/run_data.py → core.data.loader
├── run_train.py        # python scripts/run_train.py → core.training.train
├── run_eval.py         # python scripts/run_eval.py → evaluation.metrics.evaluator
└── setup_environment.py # Environment bootstrap (first-time setup)

experiments/            # 🔬 Ablation & research orchestration
├── runner.py           # python experiments/runner.py → experiments.runner_impl
├── runner_impl.py      # M1–M12 experiment orchestration (clean logic)
├── configs/            # Experiment configuration matrix
└── results/            # Experiment output logs (git-ignored)

evaluation/             # 📊 Metrics & analysis
├── metrics/
│   ├── evaluator.py    # Metric computation & JSON export
│   └── reporters.py    # Result reporting utilities
└── ablation_results.json  # M1–M12 experiment results

configs/                # ⚙️  YAML configuration files
├── E1_full_qcnn.yaml   # Config experiments E1–E5
├── E2_no_qcnn.yaml
├── E3_reduced_qcnn.yaml
├── E4_no_projection.yaml
└── E5_transformer.yaml

datasets/               # 📁 Data storage
├── processed/          # Final merged & processed data
│   └── final_merged.csv  # Single source for training/evaluation
└── raw/                # Raw dataset cache (from HuggingFace)

models/                 # 🤖  Trained model artifacts
├── E1_best.pt         # Best model per config
├── M1_best.pt–M12_best.pt  # Ablation experiment models
├── model_bundle.joblib  # Sklearn model + preprocessor bundles
└── label_encoder.joblib # Class label encoder

docs/                   # 📖 Documentation
├── README.md           # Project overview (auto-generated)
├── architecture.md     # System design & pipeline flow
├── methodology.md      # Research methodology
├── dataset-pipeline.md # Data processing details
├── qcnn-model.md       # Quantum CNN architecture
├── training-workflow.md # Training pipeline details
└── deployment.md       # Production deployment guide

logs/                   # 📝 Runtime logs
├── training.log        # Training run logs
├── evaluation.log      # Evaluation logs
└── experiments.log     # Experiment runner logs

figures/                # 📈 Generated plots & visualizations
└── Results plots, confusion matrices, ROC curves, etc.

archive/                # 🗂️  Legacy & old code (NOT in active pipeline)
├── backend_old/        # Old Flask blueprint code
├── core_legacy/        # Superseded data/training modules
├── legacy_scripts/     # Old run_data_phase.py and variants
└── old_scripts/        # Unused utility scripts

metrics/                # 📊 Output metrics (generated, git-ignored)
└── classification_report.txt

report/                 # 📄 Project submission/reports
├── README.md
└── SUBMISSION_CHECKLIST.md

REPORT GUIDE/           # 📋 Submission template files (Word docs)
```

### Configuration Files (Root Level)

```
.venv/                  # Python virtual environment (git-ignored)
.vscode/                # VS Code workspace settings
├── settings.json       # Editor & terminal config
└── launch.json         # Debug configurations

pyrightconfig.json      # Type checking configuration
requirements.txt        # Python dependencies
docker-compose.yml      # Local Docker compose (dev/test)
Dockerfile.backend      # Backend application Dockerfile
.dockerignore           # Docker build exclusions
.gitignore              # Git exclusions

README.md               # Project overview
Research_Paper.md       # Research paper draft
```

## Key Invariants

### ✅ What You'll Find

1. **Single Source of Truth**: `core/` is the ONLY active code location
   - All imports resolve from `core.*` namespace
   - No sys.path hacks or manual path manipulation

2. **Clean Entry Points**: Three public commands
   - `python scripts/run_data.py` → Full data pipeline
   - `python experiments/runner.py` → M1–M12 ablation matrix
   - `python scripts/run_eval.py` → Final evaluation & metrics

3. **Deterministic Reproducibility**
   - Fixed seed=42 for all random operations
   - Embedding cache singleton for cross-config reuse
   - Stratified train/test split
   - Balanced class weights

4. **Modular Architecture**
   - Each `core/*` module has clear responsibility
   - Clean interfaces and dependency injection
   - No circular imports or implicit globals

### ❌ What You Won't Find

- ~~sys.path hacks~~ (removed)
- ~~Duplicate code~~ (consolidated)
- ~~Loose .py scripts~~ (archived)
- ~~backend/ API code~~ (archived to archive/backend_old/)
- ~~data_pipeline/ duplicates~~ (removed)
- ~~training/ duplicates~~ (removed)

## Command References

### Data Pipeline
```bash
python scripts/run_data.py
# Output: datasets/processed/final_merged.csv
```

### Training (Single Config)
```bash
python scripts/run_train.py
# Output: evaluation/train_result.json
```

### Evaluation (Final Metrics)
```bash
python scripts/run_eval.py
# Output: evaluation/results.json
```

### Experiments (Full Ablation: M1–M12)
```bash
python experiments/runner.py
# Output: evaluation/ablation_results.json
```

## Dependency Flow

```
Input Data (HuggingFace)
    ↓
core/data/loader.py (normalize, balance, merge)
    ↓
datasets/processed/final_merged.csv
    ↓
core/features/embedding.py (MiniLM 384-dim)
    ↓
core/models/standardized.py (LogisticRegression / MLP)
    ↓
core/training/train.py (train_and_predict)
    ↓
evaluation/metrics/evaluator.py (compute metrics)
    ↓
evaluation/results.json
```

## Best Practices

### When Adding New Code
1. Place it in `core/<module>/` following existing structure
2. Use clean imports: `from core.data import loader`
3. Add `__main__` entry point if standalone executable
4. Never use `sys.path.insert()` or relative imports

### When Modifying Data Pipeline
1. Update `core/data/loader.py` only
2. Never modify `datasets/` manually; regenerate via `python scripts/run_data.py`
3. Document schema changes in `docs/dataset-pipeline.md`

### When Training Models
1. Always use `core/training/train.py`
2. Modify configs in `configs/` if needed
3. Model artifacts auto-saved to `models/` with timestamp

### Debugging & Logs
1. Check `logs/` directory for execution traces
2. Use `python -m pdb` for interactive debugging
3. Run with `export PYTHONPATH=$PWD` if needed (shouldn't be necessary)

## Version History

- **2026-03-26**: Complete refactor - removed duplicates, cleaned structure
- **Previous**: Incremental experiments (see archive/ for history)

---

**Last Updated**: March 26, 2026  
**Maintainer**: Research Team  
**Status**: ✅ Production-Ready
