import json
import numpy as np
from scipy.stats import entropy
from pathlib import Path
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def profile_sample_difficulty(y_prob: list[list[float]], texts: list[str]) -> list[dict]:
    """
    Addition F: Data Difficulty Profiling.
    Uses entropy of predicted probabilities to label difficult samples.
    """
    logger.info("Profiling sample difficulty via entropy calculation...")
    
    y_prob_arr = np.array(y_prob)
    # Calculate entropy along the class axis
    # High entropy means the model is uncertain
    entropies = entropy(y_prob_arr, axis=1)
    
    profiling = []
    for idx, e in enumerate(entropies):
        pred_class = int(np.argmax(y_prob_arr[idx]))
        profiling.append({
            "text": texts[idx],
            "entropy": float(e),
            "predicted_class": pred_class,
            "confidence": float(y_prob_arr[idx][pred_class]),
            "difficulty_label": "hard" if e > 0.8 else "medium" if e > 0.4 else "easy"
        })
        
    return profiling

def save_profiling_results(results: list[dict], output_path: Path = Path("analysis/sample_difficulty.json")):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sample difficulty profiling saved to {output_path}")
