import json
import numpy as np
from pathlib import Path
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def approximate_token_importance(model, texts: list[str], tokenizer_func, top_n: int = 10) -> dict:
    """
    Addition D: Explain why predictions happen.
    Uses a simple leave-one-out perturbation to approximate token importance.
    """
    logger.info("Calculating approximate token importance (SHAP-lite)...")
    
    importance_summary = {}
    
    # Process a subset of samples for speed in this research phase
    for text in texts[:50]:
        tokens = tokenizer_func(text)
        if not tokens:
            continue
            
        full_prob = model.predict_proba([text])[0]
        base_class = np.argmax(full_prob)
        base_score = full_prob[base_class]
        
        token_scores = {}
        for i, token in enumerate(tokens):
            # Create a version of the text without the current token
            perturbed_tokens = tokens[:i] + tokens[i+1:]
            perturbed_text = " ".join(perturbed_tokens)
            
            p_prob = model.predict_proba([perturbed_text])[0]
            p_score = p_prob[base_class]
            
            # Change in score (importance)
            token_scores[token] = float(base_score - p_score)
            
        importance_summary[text] = token_scores
        
    return importance_summary

def save_importance_results(results: dict, output_path: Path = Path("analysis/feature_importance.json")):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Feature importance saved to {output_path}")
