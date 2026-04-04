# Workspace audit summary

## Archived legacy assets

The following workspace items were preserved under the root `archive/` folder:

- `legacy-react-dashboard/` — prior Vite/React UI, not aligned with the Angular delivery target.
- `legacy-sentiment-analysis-project/` — previous notebook-style training project, including raw data, trained models, and duplicate scripts.
- `legacy-root-results/` — legacy top-level results folder.
- `legacy-python-env/` — old virtual environment.
- `test.py` — obsolete local path smoke test.

## Duplicate or conflicting scripts detected

Within the legacy sentiment project:

- `src/preprocess.py` and `src/preprocess_optimized.py`
- `src/predict.py` and `src/predict_optimized.py`
- `src/train_model.py` and `src/train_optimized.py`

## Preserved useful artifacts

Copied into the new platform:

- `backend/datasets/raw/amazon_reviews_legacy.csv`
- `backend/experiments/models/legacy/`

## Obsolete characteristics in the legacy codebase

- Hard-coded absolute Windows paths
- Mixed experimental dependencies (`texthero`, `swifter`, `numba`) without environment management
- Empty model/result version folders
- Frontend stack mismatch: React app instead of Angular dashboard
