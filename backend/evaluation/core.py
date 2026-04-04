import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.preprocessing import label_binarize
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def compute_metrics(y_true: list[int], y_pred: list[int], y_prob: list[list[float]]) -> dict:
    """Addition: Centralized metric computation (Single Source of Truth)."""
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="macro",
        zero_division=0,
    )

    roc_auc = 0.0
    try:
        classes = sorted(np.unique(y_true_arr).tolist())
        if len(classes) > 1:
            y_bin = label_binarize(y_true_arr, classes=classes)
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == len(classes):
                roc_auc = float(roc_auc_score(y_bin, y_prob_arr, average="weighted", multi_class="ovr"))
    except Exception as e:
        logger.warning(f"ROC-AUC calculation failed: {str(e)}")
        roc_auc = 0.0

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1-score": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
    }

def generate_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path, labels: list[str]):
    """Addition: Centralized Confusion Matrix generation with normalization."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix', fontweight='bold')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion Matrix saved to {output_path}")

def generate_classification_report_extended(y_true: list[int], y_pred: list[int], labels: list[str]) -> str:
    """Addition: Centralized Classification Report."""
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)

def check_model_divergence(preds_A: list[int], preds_B: list[int], threshold: float = 0.01) -> bool:
    """
    Addition C: Model Output Divergence Test.
    Ensures different configurations actually produce different predictions.
    """
    if len(preds_A) != len(preds_B):
        logger.error("Divergence Test Error: Prediction lengths mismatch.")
        return False
        
    diff = np.mean(np.array(preds_A) != np.array(preds_B))
    logger.info(f"Model Divergence: {diff:.4f} (threshold={threshold})")
    
    if diff <= threshold:
        logger.warning(f"Ablation Collapse Warning: Predictions are { (1-diff)*100 } % identical.")
        return False
    return True

def save_final_metrics(payload: dict, output_path: Path = Path("evaluation/final_metrics.json")):
    """Saves the single source of truth for metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Final metrics saved to {output_path}")
