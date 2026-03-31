from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize


def evaluate_predictions(y_true: list[int], y_pred: list[int], y_prob: list[list[float]]) -> dict:
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="weighted",
        zero_division=0,
    )

    roc_auc = 0.0
    try:
        classes = sorted(np.unique(y_true_arr).tolist())
        y_bin = label_binarize(y_true_arr, classes=classes)
        if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == len(classes):
            roc_auc = float(roc_auc_score(y_bin, y_prob_arr, average="weighted", multi_class="ovr"))
    except Exception:
        roc_auc = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1), # Weighted
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc),
    }
    return metrics


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    source = Path("evaluation/train_result.json")
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        metrics = evaluate_predictions(payload["y_true"], payload["y_pred"], payload["y_prob"])
    else:
        ablation = Path("evaluation/ablation_results.json")
        if not ablation.exists():
            raise FileNotFoundError("Run training or experiments first to create evaluation inputs.")
        blob = json.loads(ablation.read_text(encoding="utf-8"))
        best = max(blob.get("experiments", []), key=lambda x: x["metrics"]["f1"])
        metrics = evaluate_predictions(best["y_true"], best["y_pred"], best["y_prob"])

    save_results(Path("evaluation/results.json"), metrics)
    print(json.dumps({"output": "evaluation/results.json", "metrics": metrics}, indent=2))
