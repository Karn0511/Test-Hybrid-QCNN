from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from backend.data.loader import load_processed, prepare_dataset, LANG_MAP
from backend.data.dataset_guard import get_dataset_hash, validate_integrity
from backend.features.embedding import EmbeddingPipeline
from backend.models.standardized import build_model
from backend.utils.logger import get_logger, configure_logging
from evaluation.metrics.evaluator import evaluate_predictions
from backend.evaluation.calibration import compute_ece, compute_brier_score
from backend.evaluation.failure_analyzer import FailureAnalyzer

configure_logging(level='INFO')
logger = get_logger(__name__)

# Standardized Output Paths
EVAL_BASE = Path("evaluation/latest")
METRICS_DIR = EVAL_BASE / "metrics"
ANALYSIS_DIR = EVAL_BASE / "analysis"
PLOTS_DIR = EVAL_BASE / "plots"

_EMBEDDER: EmbeddingPipeline | None = None
_EMBED_CACHE: dict[tuple, np.ndarray] = {}

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_predict(config: dict, dataset_path: Path | None = None, seed: int = 42, holdout_langs: list[str] | None = None) -> dict:
    set_seed(seed)

    if dataset_path is None:
        dataset_path = prepare_dataset(force_rebuild=False)

    df = load_processed(dataset_path)
    
    if not validate_integrity(df):
        raise ValueError("Dataset integrity check failed!")
    
    d_hash = get_dataset_hash(df)
    logger.info(f"Dataset Hash: {d_hash}")

    max_rows = int(config.get("max_rows", 3000)) # Elite v35.0: Balanced Research Scale Default
    logger.info(f"📊 [Active Dataset]: Running benchmark with N = {max_rows} rows.")
    
    # [v18.0 Universal Strike]: Hardware-Aware Adaptive Batching
    import psutil
    mem_gb = psutil.virtual_memory().total / (1024**3)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    if max_rows >= 10000:
        # Cloud/Server Optimization
        if gpu_mem >= 12: # T4/A100/P100
            config["batch_size"] = 128
            logger.info("🚀 [Overdrive View]: GPU VRAM >= 12GB. Setting Batch Size to 128.")
        else:
            # v25.0: Hyper-Drive - Pushing 'Beast PC' to 128
            config["batch_size"] = 128
            logger.info("🚀 [HYPER-DRIVE]: Beast GPU detected. Setting Batch Size to 128.")
    else:
        # Local/Small Dataset Optimization: More steps per epoch for better convergence
        config["batch_size"] = min(32, max_rows // 10 if max_rows > 100 else 8)
        logger.info(f"🔬 [High-Fidelity]: Local dataset detected. Adaptive Batch Size set to {config['batch_size']}.")

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    # Standardized labels (0, 1, 2)
    y = df["label"].astype(int).to_numpy()
    texts = df["text"].astype(str).tolist()
    languages = df["language"].astype(str).tolist()

    # Phase 6: Class Imbalance Fix (Class Weights)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    # --- Neutral Surge (v14.4 Acceleration) ---
    # We calibrate the weight of the Neutral class (index 1) for multi-dialect stability
    if len(class_weights) > 1:
        class_weights[1] *= 3.0 # v35.1 Neutral Anchor Upgrade
        
    config["class_weights"] = class_weights.tolist()
    logger.info(f"Computed Triple-Weighted Class Weights: {config['class_weights']}")

    global _EMBEDDER
    if _EMBEDDER is None: _EMBEDDER = EmbeddingPipeline()

    cache_key = (len(texts), d_hash, max_rows, seed)
    if cache_key in _EMBED_CACHE:
        x = _EMBED_CACHE[cache_key]
    else:
        # [v16.0 Quicksilver]: Direct NPY Bypass using Dataset Hash
        x = _EMBEDDER.fit_transform(texts, dataset_hash=d_hash)
        _EMBED_CACHE[cache_key] = x

    if holdout_langs:
        # Custom Holdout Split
        is_holdout = np.array([l in holdout_langs for l in languages])
        idx_test = np.where(is_holdout)[0]
        idx_train_val = np.where(~is_holdout)[0]
        
        # 80/20 train/val from the non-holdout set
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.2, random_state=seed, stratify=y[idx_train_val]
        )
    else:
        # Standard Random Split (Adjustable)
        test_size = float(config.get("test_size", 0.2))
        indices = np.arange(len(y))
        
        if test_size > 0:
            idx_train, idx_temp = train_test_split(
                indices, test_size=test_size, random_state=seed, stratify=y
            )
            val_size = 0.5 # Default 50/50 of the test_size for val/test
            idx_val, idx_test = train_test_split(
                idx_temp, test_size=val_size, random_state=seed, stratify=y[idx_temp]
            )
        else:
            # 100% Train mode for Sanity/Overfit
            idx_train = indices
            idx_val = indices
            idx_test = indices
        
    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    # --- Multi-Dialect Context Mapping (v34.5 None-Guard) ---
    langs_train = [languages[i] for i in idx_train]
    langs_val = [languages[i] for i in idx_val]
    langs_test = [languages[i] for i in idx_test]
    
    texts_train = [str(texts[i]) if texts[i] is not None else "" for i in idx_train]
    texts_val = [str(texts[i]) if texts[i] is not None else "" for i in idx_val]
    texts_test = [str(texts[i]) if texts[i] is not None else "" for i in idx_test]

    config["n_qubits"] = config.get("n_qubits", 12) # Scaled to 12 for local hardware stability
    model = build_model(config=config, seed=seed)
    param_count = getattr(model, "param_count", 0)
    logger.info(f"🦾 Model Built: {config.get('id', 'unknown')} | Parameters: {param_count:,}")
    
    if hasattr(model, "fit"):
        model.fit(x_train, y_train, x_val=x_val, y_val=y_val, languages=langs_train, languages_val=langs_val, texts=texts_train, texts_val=texts_val)
    
    if hasattr(model, "calibrate") and x_val is not None:
        l_val_ids = np.array([LANG_MAP.get(l.lower(), 0) for l in langs_val])
        model.calibrate(x_val, y_val, lang_ids=l_val_ids)

    l_test_ids = np.array([LANG_MAP.get(l.lower(), 0) for l in langs_test])
    y_pred = model.predict(x_test, lang_ids=l_test_ids)
    y_prob = model.predict_proba(x_test, lang_ids=l_test_ids)
    
    metrics = evaluate_predictions(
        y_true=y_test.tolist(),
        y_pred=y_pred.tolist(),
        y_prob=y_prob.tolist(),
    )
    metrics["ece"] = compute_ece(y_test, y_prob)
    metrics["brier_score"] = compute_brier_score(y_test, y_prob)

    # Phase 7: Language Metrics
    lang_test = [languages[i] for i in idx_test]
    text_test = [texts[i] for i in idx_test]
    metrics["per_language"] = {}
    
    unique_langs = set(lang_test)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    for lang in unique_langs:
        lang_mask = [l == lang for l in lang_test]
        if any(lang_mask):
            y_t_l = [y_test[i] for i, m in enumerate(lang_mask) if m]
            y_p_l = [y_pred[i] for i, m in enumerate(lang_mask) if m]
            
            metrics["per_language"][lang] = {
                "accuracy": float(accuracy_score(y_t_l, y_p_l)),
                "f1": float(f1_score(y_t_l, y_p_l, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_t_l, y_p_l).tolist()
            }
            
    # Phase 5: Failure Analysis
    failures = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            failures.append({
                "text": text_test[i],
                "predicted": int(y_pred[i]),
                "actual": int(y_test[i]),
                "confidence": float(max(y_prob[i])) if y_prob is not None else 0.0,
                "language": lang_test[i]
            })
            
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "failure_cases.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
        
    # [NEW]: Advanced Failure Diagnostics
    analyzer = FailureAnalyzer(output_dir=str(ANALYSIS_DIR))
    diag_results = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "texts": text_test,
        "embeddings": x_test # Passing embeddings for clustering hotspots
    }
    analyzer.run_full_diagnostics(diag_results)
        
    # Phase 6: Calibration Visuals
    try:
        from evaluation.visualizer import plot_reliability_diagram, plot_ece
        calib_dir = METRICS_DIR / "calibration" / f"{config.get('id', 'default')}"
        calib_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagram(y_test, y_prob, save_path=calib_dir / f"reliability_diagram_{seed}.png")
        plot_ece(metrics["ece"], save_path=calib_dir / f"ece_plot_{seed}.png")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not generate calibration plots: {e}")

    # Finalize Telemetry
    telemetry = getattr(model, "telemetry", {})
    if telemetry:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        tel_file = METRICS_DIR / f"telemetry_{config.get('id', 'default')}_{seed}.json"
        with open(tel_file, "w") as f:
            json.dump(telemetry, f, indent=4)

    return {
        "metrics": metrics,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if y_prob is not None else [],
        "texts": text_test,
        "languages": lang_test,
        "dataset_hash": d_hash,
        "history": getattr(model, "history", []),
        "telemetry": telemetry,
        "param_count": param_count,
        "model_id": config.get("id", "default"),
        "model": model,
        "embedder": _EMBEDDER,
        "x_test": x_test
    }

def run_multi_seed(config: dict, n_seeds: int = 3, dataset_path: Path | None = None, holdout_langs: list[str] | None = None) -> dict:
    # [STRICT-MODE]: Research Integrity Enforcement
    if n_seeds < 3 and config.get("enforce_publication_integrity", True):
        logger.warning(f"[!] [SCIENTIFIC INTEGRITY]: Requested n_seeds={n_seeds} is insufficient for Publication. Overriding to n=3.")
        n_seeds = 3

    all_metrics = []
    runs = []
    for i in range(n_seeds):
        res = train_and_predict(config, dataset_path=dataset_path, seed=42+i, holdout_langs=holdout_langs)
        all_metrics.append(res["metrics"])
        runs.append(res)
    
    summary = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            vals = [m[key] for m in all_metrics]
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    
    return {"summary": summary, "runs": runs}

if __name__ == "__main__":
    cfg = {"id": "M_FINAL", "use_qcnn": True, "qcnn_depth": 4, "projection": True}
    micro_res = train_and_predict(micro_cfg)
    if micro_res["metrics"]["accuracy"] < 0.85:
        logger.error("[!] [VALIDATION FAIL]: Micro-test failed to reach 85%+ Accuracy. Model logic glitch.")
