#!/usr/bin/env python
"""
Phase 10: Standardized Evaluation & Benchmarking
Computes all metrics: accuracy, macro-F1, ROC-AUC, ECE, Brier.
Multi-seed evaluation and baseline comparison.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

from backend.models.standardized import build_model
from backend.data.loader import EliteMultilingualDataset
from backend.evaluation.calibration import compute_ece, compute_brier_score
from backend.utils.logger import get_logger
from torch.utils.data import DataLoader

logger = get_logger("Evaluator")

def evaluate_checkpoint(model_path: str, lang: str, split: str = "test", device=None) -> Dict:
    """
    Evaluate a single checkpoint on a language/split.
    
    Args:
        model_path: Path to saved model weights
        lang: Language to evaluate on
        split: 'train', 'val', or 'test'
        device: torch device
        
    Returns:
        Metrics dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load model
        config = {
            "use_qcnn": True,
            "n_qubits": 8,
            "n_layers": 2,
        }
        model = build_model(config).model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Load dataset
        dataset = EliteMultilingualDataset(lang, split, max_samples=None)
        loader = DataLoader(dataset, batch_size=64, num_workers=0)
        
        # Inference
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y, lang_id in loader:
                x = x.to(device)
                logits = model(x, lang_ids=y.to(device) if hasattr(model, 'lang_embed') else None)
                probs = torch.softmax(logits, dim=1)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs_np = probs.cpu().numpy()
                
                all_preds.extend(preds)
                all_probs.extend(probs_np)
                all_labels.extend(y.numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        
        # ROC-AUC
        try:
            classes = sorted(set(all_labels))
            if len(classes) > 1:
                y_bin = label_binarize(all_labels, classes=classes)
                roc_auc = roc_auc_score(y_bin, all_probs, average="weighted", multi_class="ovr")
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        # Calibration
        ece = compute_ece(all_labels, all_probs)
        brier = compute_brier_score(all_labels, all_probs)
        
        result = {
            "language": lang,
            "split": split,
            "n_samples": len(all_labels),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "roc_auc": float(roc_auc),
            "ece": float(ece),
            "brier_score": float(brier),
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Evaluation failed for {lang}/{split}: {e}")
        return None

def run_multi_seed_evaluation(
    model_dir: Path,
    languages: List[str] = None,
    seeds: List[int] = [42, 43, 44],
) -> Dict:
    """
    Run evaluation across multiple seeds and languages.
    Compute mean ± std for stability assessment.
    """
    if languages is None:
        languages = ["english", "hindi", "bhojpuri", "maithili", "multilingual"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("="*70)
    logger.info(f"PHASE 10: MULTI-SEED EVALUATION")
    logger.info(f"Seeds: {seeds} | Languages: {languages}")
    logger.info("="*70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "phase": "10_evaluation",
        "device": str(device),
        "seeds": seeds,
        "languages": languages,
        "evaluations": {},
        "summary": {},
    }
    
    for lang in languages:
        logger.info(f"\n📊 {lang.upper()}")
        lang_results = {"seeds": {}}
        
        for seed in seeds:
            checkpoint_path = model_dir / f"{lang}_s{seed}_best.pt"
            if not checkpoint_path.exists():
                logger.warning(f"   Checkpoint not found: {checkpoint_path}")
                continue
            
            logger.info(f"   Seed {seed}...", end=" ")
            
            # Evaluate on test split
            test_metrics = evaluate_checkpoint(
                str(checkpoint_path), lang, split="test", device=device
            )
            
            if test_metrics:
                lang_results["seeds"][f"seed_{seed}"] = test_metrics
                logger.info(f"✓ F1={test_metrics['macro_f1']:.3f}")
            else:
                logger.info("✗ Failed")
        
        # Compute mean ± std
        if lang_results["seeds"]:
            metrics_list = list(lang_results["seeds"].values())
            
            f1_values = [m["macro_f1"] for m in metrics_list]
            acc_values = [m["accuracy"] for m in metrics_list]
            roc_values = [m["roc_auc"] for m in metrics_list]
            ece_values = [m["ece"] for m in metrics_list]
            
            summary = {
                "n_seeds": len(metrics_list),
                "accuracy_mean": float(np.mean(acc_values)),
                "accuracy_std": float(np.std(acc_values)),
                "macro_f1_mean": float(np.mean(f1_values)),
                "macro_f1_std": float(np.std(f1_values)),
                "roc_auc_mean": float(np.mean(roc_values)),
                "roc_auc_std": float(np.std(roc_values)),
                "ece_mean": float(np.mean(ece_values)),
                "ece_std": float(np.std(ece_values)),
                "stable": float(np.std(f1_values)) < 0.05,  # Threshold for stability
            }
            
            lang_results["summary"] = summary
            logger.info(f"   → ACC: {summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f}")
            logger.info(f"   → F1:  {summary['macro_f1_mean']:.3f}±{summary['macro_f1_std']:.3f}")
            logger.info(f"   → AUC: {summary['roc_auc_mean']:.3f}±{summary['roc_auc_std']:.3f}")
            logger.info(f"   → Stable: {summary['stable']}")
        
        results["evaluations"][lang] = lang_results
    
    # Save results
    results_path = Path("evaluation/latest/phase10_evaluation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ Evaluation complete. Results saved to {results_path}")
    logger.info("="*70)
    
    return results

if __name__ == "__main__":
    model_dir = Path("checkpoints")
    
    results = run_multi_seed_evaluation(
        model_dir=model_dir,
        languages=["english", "hindi", "bhojpuri", "maithili"],
        seeds=[42, 43, 44]
    )
    
    print(json.dumps(results, indent=2))
