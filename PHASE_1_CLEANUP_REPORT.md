# Phase 1: Surgical Cleanup Report

**Generated:** 2026-04-03  
**Status:** READY FOR EXECUTION

---

## Summary

This report documents files identified for deletion (Phase 1) based on Phases 0 reconnaissance analysis. These files are confirmed orphaned, duplicated, or no longer used in the active research pipeline.

**Key Principle:** Delete only files that are:
- Never imported anywhere
- Not referenced in scripts, configs, or CLI
- Not part of active research pipeline
- Can be recovered from git history if needed

---

## Orphaned Modules - DELETE

### ✅ CONFIRMED ORPHANED (0 imports, safe to delete):

1. **`backend/models/classical_models.py`**
   - Status: Never imported anywhere
   - Content: Legacy sklearn models (TF-IDF, Random Forest, SVM)
   - Reason: Replaced by HybridQCNN and market BaseLine baselines
   - Action: **DELETE**

2. **`backend/models/transformer_model.py`**
   - Status: Never imported anywhere
   - Content: Unused transformer baseline
   - Reason: Not used in main pipeline or experiments
   - Action: **DELETE**

3. **`backend/models/model_registry.py` (backend/models/ only)**
   - Status: Duplicate of backend/inference/model_registry.py
   - Reason: Consolidate to ONE registry in inference/ for deployment
   - Action: **DELETE**

### 🟡 Actions to Consolidate:

4. **Dual Model Registries → Keep ONLY `backend/inference/model_registry.py`**
   - Delete: `backend/models/model_registry.py` (duplicate)
   - Keep: `backend/inference/model_registry.py` (for deployment)
   - Note: Update imports if any

5. **Dual Training Paths → Keep ONLY `train.py`**
   - Keep: `backend/training/train.py` (ACTIVE)
   - Freeze: `backend/training/train_v2.py` (as reference/archive)
   - New: `backend/training/train_simple.py` (Phase 5 baseline)
   - Reason: Avoid confusion; simplified training is the baseline going forward

---

## Diagnostic Scripts - ARCHIVE or DELETE

These are one-off diagnostic/utility scripts rarely used. Archive to `scripts/archive/` or delete:

### Safe to Archive:
- `scripts/build_massive_datasets.py` - Data generation (one-time use)
- `scripts/download_datasets.py` - Legacy data acquisition (one-time use)
- `scripts/data_harvester.py` - Harvester (one-time use)
- `scripts/frozen_embedding_test.py` - Debug utility
- `scripts/expand_datasets_to_2m.py` - Dataset expansion (one-time)
- `scripts/sanity_shuffle_test.py` - Diagnostic test
- `scripts/single_batch_overfit.py` - Diagnostic test
- `scripts/micro_overfit_test.py` - Diagnostic test
- `scripts/dedupe_dataset_files.py` - Data cleaning utility
- `scripts/maithili_extractor.py` - Language-specific extractor

### Keep:
- `scripts/prepare_split_data.py` - **Phase 3** (data splits)
- `scripts/troubleshoot_torch.py` - Debug utility
- `scripts/research_benchmark.py` - Benchmarking
- `scripts/setup_environment.py` - Environment setup
- `scripts/scan_hardware.py` - Hardware detection
- `scripts/verify_environment.py` - **Phase 2** (new)
- `scripts/train_curriculum.py` - **Phase 6** (new)
- `scripts/evaluate_model.py` - **Phase 10** (new)
- `scripts/track_reproducibility.py` - **Phase 11** (new)
- `scripts/orchestrate_phases.py` - Master orchestrator (new)

---

## Visualization & Reporting Scripts - CONSOLIDATE

These generate plots and reports. Consolidate into `evaluation/latest/`:

### Can be Archived:
- `scripts/generate_v31_global_dashboard.py` - Versioned report
- `scripts/generate_v30_elite_render.py` - Versioned report
- `scripts/generate_readme_plots.py` - README plots
- `scripts/generate_advanced_metrics.py` - Advanced metrics
- `evaluation/generate_new_plots.py` - Plotting utility
- `evaluation/plot_benchmarks.py` - Benchmark plots

### Keep:
- `evaluation/visualizer.py` - Core visualization module (imported elsewhere)

---

## Files to Keep (CRITICAL - DO NOT DELETE)

### Backend Core:
- `backend/training/train.py` - Main training pipeline
- `backend/training/train_simple.py` - Phase 5 baseline (new)
- `backend/models/hybrid_qcnn.py` - Quantum model (Phase 4 updated)
- `backend/models/standardized.py` - Model wrapper
- `backend/models/decision_fusion.py` - Fusion model
- `backend/quantum/layers.py` - Quantum circuits
- `backend/data/loader.py` - Data loading
- `backend/data/dataset_guard.py` - Data integrity
- `backend/features/embedding.py` - BERT embeddings
- `backend/evaluation/*.py` - All evaluation modules
- `backend/utils/*.py` - Logging and config

### Configs:
- `configs/*.yaml` - Experiment configs
- `configs/validator.py` - Config validation

### Datasets:
- `datasets/{lang}/{mode}.csv` - Raw data
- `datasets/splits/{lang}/{train,val,test}.csv` - **Phase 3** (new splits)

### Evaluation:
- `evaluation/latest/` - Current run results
- `evaluation/archive/` - Historical results
- `evaluation/metrics/` - Metric tracking

---

## Deletion Checklist

```bash
# DELETE: Confirmed orphaned modules
rm -f backend/models/classical_models.py
rm -f backend/models/transformer_model.py
rm -f backend/models/model_registry.py  # Keep inference version only

# ARCHIVE: One-off diagnostic scripts
mkdir -p scripts/archive
mv scripts/build_massive_datasets.py scripts/archive/
mv scripts/download_datasets.py scripts/archive/
mv scripts/frozen_embedding_test.py scripts/archive/
# ... (see list above)

# ARCHIVE: Versioned dashboards
mv scripts/generate_v31_global_dashboard.py scripts/archive_dashboards/
mv scripts/generate_v30_elite_render.py scripts/archive_dashboards/
mv scripts/generate_readme_plots.py scripts/archive_dashboards/
```

---

## Statistics

| Category | Count |
|----------|-------|
| **Confirmed Orphaned** | 3 |
| **Duplicates** | 1 |
| **One-off Diagnostics** | 10 |
| **Visualization Scripts** | 6 |
| **Safe to Archive** | 20 |
| **Core Research Modules** | ~40 |
| **Total .py files** | 79 |
| **After Cleanup** | ~50 |

---

## Impact Analysis

### ✅ Benefits of Cleanup:
- Reduces confusion (single training path)
- Eases maintenance (fewer files to scan)
- Improves clarity (orphaned code removed)
- Reduces import errors

### ⚠️ Risks (Mitigated):
- **Risk:** Loss of data generation scripts  
  **Mitigation:** Archive to git branch; recover if needed

- **Risk:** Loss of diagnostic utilities  
  **Mitigation:** Keep in archive/ subdirectory; still accessible

---

## Execution Steps

1. **Create archive directory:**
   ```bash
   mkdir -p scripts/archive
   mkdir -p scripts/archive_dashboards
   ```

2. **Move diagnostic scripts to archive** (keep git history):
   ```bash
   git mv scripts/build_massive_datasets.py scripts/archive/
   # ... repeat for all diagnostic scripts
   ```

3. **Delete confirmed orphaned modules:**
   ```bash
   git rm backend/models/classical_models.py
   git rm backend/models/transformer_model.py
   git rm backend/models/model_registry.py
   ```

4. **Commit:**
   ```bash
   git commit -m "Phase 1: Surgical cleanup - remove orphaned modules, archive diagnostics"
   ```

5. **Verify:**
   ```bash
   python scripts/verify_environment.py  # Should pass
   ```

---

## Sign-Off

**Status:** Ready for execution  
**Reviewed:** Phase 0 reconnaissance complete  
**Next:** Phase 2 - Environment Stabilization

---
