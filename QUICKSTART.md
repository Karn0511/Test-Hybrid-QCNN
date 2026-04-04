# QUICK START: EXECUTION GUIDE

**Status:** All infrastructure complete. Ready to execute.

---

## PHASE 1: CLEANUP (5 minutes)

```bash
cd d:\Hybrid-QCNN Sentiment Analysis

# Delete orphaned modules
git rm backend/models/classical_models.py
git rm backend/models/transformer_model.py
git rm backend/models/model_registry.py

# Create archive for diagnostic scripts
mkdir -p scripts/archive
git mv scripts/build_massive_datasets.py scripts/archive/
git mv scripts/download_datasets.py scripts/archive/
git mv scripts/frozen_embedding_test.py scripts/archive/
git mv scripts/expand_datasets_to_2m.py scripts/archive/
git mv scripts/sanity_shuffle_test.py scripts/archive/
git mv scripts/single_batch_overfit.py scripts/archive/
git mv scripts/micro_overfit_test.py scripts/archive/
git mv scripts/dedupe_dataset_files.py scripts/archive/

# Commit
git commit -m "Phase 1: Surgical cleanup - remove orphaned modules"
```

---

## PHASE 2: VERIFY ENVIRONMENT (2 minutes)

```bash
python scripts/verify_environment.py
```

**Expected Output:** `environment_verification_report.json`
```
{
  "torch_available": true,
  "cuda_available": false,
  "pennylane_available": true,
  "backend_imports": "OK",
  "nlp_stack": "OK",
  "data_guard": "OK",
  "overall_status": "PASS"
}
```

---

## PHASE 3: CREATE CLEAN SPLITS (10 minutes)

```bash
python scripts/prepare_split_data.py
```

**Expected Output:**
- `datasets/splits/english/{train,val,test}.csv`
- `datasets/splits/hindi/{train,val,test}.csv`
- `datasets/splits/bhojpuri/{train,val,test}.csv`
- `datasets/splits/maithili/{train,val,test}.csv`
- `datasets/splits/multilingual/{train,val,test}.csv`
- `data_integrity_report.json` (confirms zero leakage)

**Verification:**
```bash
ls -la datasets/splits/english/
# Should see: train.csv val.csv test.csv
head -5 datasets/splits/english/train.csv
# Should see: text,label format
```

---

## PHASE 4: TEST ARCHITECTURE (Optional - verify hybrid_qcnn.py v6 loads)

```python
import torch
from backend.models.hybrid_qcnn import HybridQCNN

model = HybridQCNN(
    input_dim=384,
    bridge_dim=128,
    quantum_dim=8,
    num_quantum_layers=2,
    num_qubits=8,
    num_classes=3,
    fusion_dim=152
)

# Test forward pass
batch_size = 8
x = torch.randn(batch_size, 384)
output = model(x)
print(output.shape)  # Should be [8, 3]
```

---

## PHASE 5: QUICK ENGLISH VALIDATION (30 minutes)

```bash
python backend/training/train_simple.py \
  --lang english \
  --max_rows 1000 \
  --batch_size 16 \
  --epochs 5 \
  --lr 1e-3 \
  --seed 42
```

**Expected Output:**
- `checkpoints/english_seed42_best.pt` (model checkpoint)
- Console logs showing per-epoch metrics
- **No class collapse detected** (all per-class recalls > 0.2)
- Training should complete in < 5 minutes on CPU

**Verification:**
```bash
ls -la checkpoints/english_seed42_best.pt
# File should exist and be >10MB
```

---

## PHASE 6: FULL CURRICULUM TRAINING (4-6 hours)

```bash
python scripts/train_curriculum.py \
  --seeds 42,43,44 \
  --max_rows 50000 \
  --batch_size 16 \
  --epochs 100 \
  --early_stopping_patience 5
```

**Expected Output:**
- `curriculum_training_report.json` (per-stage results)
- Checkpoints for each language/seed:
  - `checkpoints/english_seed{42,43,44}_best.pt`
  - `checkpoints/hindi_seed{42,43,44}_best.pt`
  - `checkpoints/bhojpuri_seed{42,43,44}_best.pt`
  - `checkpoints/maithili_seed{42,43,44}_best.pt`
  - `checkpoints/multilingual_seed{42,43,44}_best.pt`

**Monitoring:**
```bash
# In separate terminal, watch checkpoints directory
watch ls -lah checkpoints/*.pt
```

---

## PHASE 10: EVALUATE STABILITY (30 minutes)

```bash
python scripts/evaluate_model.py \
  --checkpoint_dir checkpoints/ \
  --seeds 42,43,44
```

**Expected Output:** `phase10_evaluation_results.json`
```json
{
  "english": {
    "accuracy": "0.94 ± 0.02",
    "macro_f1": "0.93 ± 0.02",
    "per_class_recall": [0.95, 0.92, 0.91],
    "publication_status": "PUBLICATION-READY"
  },
  "hindi": {
    "accuracy": "0.82 ± 0.03",
    "macro_f1": "0.81 ± 0.04",
    "per_class_recall": [0.82, 0.80, 0.78],
    "publication_status": "PUBLICATION-READY"
  },
  ...
}
```

**Publication Criterion:** std(macro_f1) < 0.05 across 3 seeds

---

## PHASE 11: REPRODUCIBILITY VERIFICATION (5 minutes)

```bash
python scripts/track_reproducibility.py \
  --checkpoint_dir checkpoints/ \
  --config_dir configs/
```

**Expected Output:** `reproducibility_manifest.json`
```json
{
  "runs": [
    {
      "run_id": "uuid-1234",
      "language": "english",
      "seed": 42,
      "macro_f1": 0.93,
      "dataset_hash": "abc123def456...",
      "checkpoint_hash": "def456abc123...",
      "reproducibility": "VERIFIED"
    },
    ...
  ],
  "stability_assessment": "PUBLICATION-READY",
  "date_generated": "2026-04-03T12:00:00Z"
}
```

---

## TROUBLESHOOTING

### Issue: Phase 2 verification fails
```bash
# Check torch version
python -c "import torch; print(torch.__version__)"  # Should be 2.2.2

# Check pennylane
python -c "import pennylane as qml; print(qml.__version__)"  # Should be 0.35.1

# Try reinstalling dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Phase 3 fails with data leakage
```bash
# Check for duplicates in raw data
python -c "import pandas as pd; df = pd.read_csv('datasets/english/raw.csv'); print(f'Duplicates: {df.duplicated().sum()}')"

# Run data guard check manually
python -c "from backend.data.dataset_guard import check_leakage; check_leakage(...)"
```

### Issue: Phase 5 shows class collapse
- Class collapse defined as: any class recall < 0.2
- Solutions:
  1. Increase patience in early stopping (patience=10)
  2. Lower learning rate (lr=5e-4)
  3. Check data balance with `data_integrity_report.json`
  4. Verify training data with Phase 3 splits

### Issue: Phase 6 training is too slow
- Reduce --max_rows to 10000 for quick validation
- Reduce --epochs to 20 for CPU testing
- Verify GPU usage: `nvidia-smi -l 1` (Windows requires nvidia-smi in PATH)

---

## SUCCESS CRITERIA

### ✅ Phase 1-11 Complete When:
- [ ] Phase 1: No .py files remain in backend/models/classical_models.py, etc.
- [ ] Phase 2: environment_verification_report.json has "overall_status": "PASS"
- [ ] Phase 3: data_integrity_report.json shows "leakage_detected": false for all languages
- [ ] Phase 4: model loads without assertion errors
- [ ] Phase 5: English training completes with std(recall) > 0.2 (no collapse)
- [ ] Phase 6: All 4 curriculum stages complete; multilingual std(F1) < 0.05
- [ ] Phase 10: phase10_evaluation_results.json all languages marked "PUBLICATION-READY"
- [ ] Phase 11: reproducibility_manifest.json confirms identical seed reproducibility

### 🏆 Research Grade When:
```
All phases complete AND
std(macro_f1) < 0.05 across 3 seeds (all languages) AND
Zero data leakage detected AND
No class collapse across all languages
```

---

## COMMAND TEMPLATES

### Run single language, single seed (quick test):
```bash
python backend/training/train_simple.py --lang english --max_rows 5000 --seed 42
```

### Run all languages, all seeds (full experiment):
```bash
python scripts/train_curriculum.py --seeds 42,43,44,45,46 --max_rows 100000
```

### Evaluate specific checkpoint:
```bash
python scripts/evaluate_model.py --checkpoint checkpoints/english_seed42_best.pt
```

### Generate reproducibility report:
```bash
python scripts/track_reproducibility.py --seeds 42,43,44
```

---

## MONITORING

### Watch training progress (real-time):
```bash
# Terminal 1: Start training
python scripts/train_curriculum.py --seeds 42,43,44

# Terminal 2: Monitor checkpoints
watch -n 5 "ls -lah checkpoints/*.pt | tail -10"

# Terminal 3: Monitor logs
tail -f logs/training.log
```

### Track metrics history:
```bash
# After each phase completes
python -c "
import json
with open('curriculum_training_report.json') as f:
    report = json.load(f)
    for stage in report['stages']:
        print(f\"Stage: {stage['stage_name']}, F1: {stage.get('macro_f1', 'N/A')}\")"
```

---

## NEXT STEPS AFTER PHASES COMPLETE

1. **Create publish-ready results table:**
   ```bash
   python -c "
   import json
   with open('phase10_evaluation_results.json') as f:
       results = json.load(f)
   # Generate LaTeX table for paper
   "
   ```

2. **Generate accuracy plots:**
   ```bash
   python evaluation/generate_new_plots.py --input phase10_evaluation_results.json
   ```

3. **Create experimental summary:**
   ```bash
   python scripts/track_reproducibility.py --generate_report
   ```

---

**Status:** READY FOR LAUNCH ✅
**Next Command:** `python scripts/verify_environment.py`
