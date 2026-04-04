import pandas as pd
import numpy as np
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def extract_hard_negatives(y_true: list[int], y_prob: list[list[float]], texts: list[str], languages: list[str] = None, top_k: int = 100) -> pd.DataFrame:
    """
    v15.0 Elite: Identifies samples where the model was most confidently wrong.
    Enforces a Language Diversity Guard to prevent monolingual bias.
    """
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    
    # Probability of the correct class
    true_probs = y_prob_arr[np.arange(len(y_true_arr)), y_true_arr]
    
    # Difficulty index (1 - prob_of_true_class)
    difficulty = 1.0 - true_probs
    
    # Find indices of misclassified samples
    y_pred = np.argmax(y_prob_arr, axis=1)
    misclassified = (y_pred != y_true_arr)
    
    hard_indices = np.where(misclassified)[0]
    hard_difficulties = difficulty[hard_indices]
    
    # Sort by difficulty
    sorted_idx = np.argsort(hard_difficulties)[::-1]
    all_hard_idx = hard_indices[sorted_idx]
    
    # --- v35.0 Elite Precision Guard (4-Way Balance) ---
    if texts is not None and languages is not None and len(languages) == len(texts):
        langs_arr = np.array([l.lower() for l in languages])
        
        # Target 25% for each dialect to ensure multi-dialect superiority
        target_per_lang = max(1, top_k // 4)
        selected_idx_list = []
        
        for lang in ["english", "hindi", "bhojpuri", "maithili"]:
            lang_mask = (langs_arr[all_hard_idx] == lang)
            lang_idx = all_hard_idx[lang_mask]
            selected_idx_list.append(lang_idx[:target_per_lang])
            
        selected_idx = np.concatenate(selected_idx_list)
        # If we didn't reach top_k, fill with remaining most difficult samples
        if len(selected_idx) < top_k:
            remaining_mask = ~np.isin(all_hard_idx, selected_idx)
            selected_idx = np.concatenate([selected_idx, all_hard_idx[remaining_mask][:top_k - len(selected_idx)]])
    else:
        selected_idx = all_hard_idx[:top_k]
    
    label_map_rev = {0: "negative", 1: "neutral", 2: "positive"}
    
    results = []
    for idx in selected_idx:
        results.append({
            "idx": int(idx), # v27.0: Quick-Slice Index for Zero-Inference Refinement
            "text": texts[idx] if texts is not None else "MISSING_TEXT",
            "label": label_map_rev[y_true[idx]],
            "predicted": label_map_rev[y_pred[idx]],
            "confidence": float(y_prob_arr[idx][y_pred[idx]]),
            "true_prob": float(true_probs[idx]),
            "language": languages[idx] if languages is not None else "unknown"
        })
        
    logger.info("Extracted %d hard negatives with Language Diversity Guard.", len(results))
    return pd.DataFrame(results)

def augment_with_hard_negatives(original_df: pd.DataFrame, hard_df: pd.DataFrame, multiplier: int = 2) -> pd.DataFrame:
    """Oversamples hard negatives to improve next iteration."""
    if hard_df.empty:
        return original_df
        
    logger.info(f"Augmenting dataset with {len(hard_df)} hard negatives (x{multiplier}).")
    augmented = pd.concat([original_df] + [hard_df[["text", "label"]]] * multiplier, ignore_index=True)
    return augmented.sample(frac=1.0).reset_index(drop=True)
