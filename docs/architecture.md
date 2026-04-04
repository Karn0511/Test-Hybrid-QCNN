# Quantum-AI Sentiment Intelligence Platform

## Table of Contents
1. Architecture
2. QCNN Model Design 
3. Datasets
4. Training Workflow
5. Deployment Process

---

## 1. Architecture
The repository has been structured exactly into:
- `frontend/`: Angular UI client.
- `backend/`: FastAPI controller including inference logic (`backend/server.py`).
- `datasets/`: Dataset configurations and runtime merging tools.
- `models/`: Classical machine learning and quantum ML classes (`QCNNConfig`, `HybridQCNNClassifier`).
- `training/`: Pipelines configuring ML models.
- `experiments/`: Location of generated checks and reporting graphics (`results/` vs `models/`).

## 2. QCNN Model Design
The primary Quantum Convolutional Neural Network consists of a hybrid stack: 
1. `feature_projection`: PyTorch Linear Layers embedding standard textual vectors into $n$-qubit quantum feature bounds.
2. `quantum_layer`: Specialized PennyLane (`default.qubit` simulator) variational circuit layering containing rotation parameters.
3. `classifier`: A trailing classical density reduction mapping qubits to deterministic labels (`positive`, `neutral`, `negative`).

## 3. Training Workflow
- Prepare and unify existing models: `python training/train_qcnn.py --prepare-datasets`
- `dataset_size` logging has been added immediately following compilation.
- Model tracking automatically captures running `train_loss`, `validation_accuracy`, and `throughput_samples_per_second`. Learning profiles are automatically saved to `experiments/results/training_curves.png`.
- Baseline comparisons run by accessing `python training/train_baselines.py`. This script executes logistic regression, SGD, explicit tree models, and automatically triggers DistilBERT transformers on huggingface layers securely. All metadata is serialized into `baseline_benchmarks.json`.

## 4. Deployment Process
Deploy the ML framework by resolving inference requests securely.
1. Make sure to first organize directories exactly following Phase 1. An automated tool `python reorganize_project.py` is written to map and transfer directories properly removing legacy remnants.
2. Models are pre-bound and automatically instantiate sequentially matching model endpoints. FastAPI routes automatically cache instantiated ML wrappers upon context initiation.
3. Start the ML service manually via `uvicorn backend.server:app --reload`.
4. Run UI metrics dashboard via Angular: `ng serve`. 
