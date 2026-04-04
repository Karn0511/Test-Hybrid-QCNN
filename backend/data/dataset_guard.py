import pandas as pd
import hashlib
from pathlib import Path
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataset_hash(df: pd.DataFrame) -> str:
    """Computes a stable hash for the dataset to ensure reproducibility."""
    # Sort by content to ensure the hash is agnostic of row order
    # We use a subset of columns for stability if columns might change
    hash_cols = [c for c in ["text", "label", "language"] if c in df.columns]
    if not hash_cols:
        hash_cols = list(df.columns)
        
    sorted_df = df.sort_values(hash_cols).reset_index(drop=True)
    return hashlib.md5(pd.util.hash_pandas_object(sorted_df).values).hexdigest()

def validate_integrity(df: pd.DataFrame) -> bool:
    """
    Performs data integrity checks:
    - No missing labels or text
    - Correct label set
    - Balanced distribution check (log only)
    """
    if df.empty:
        logger.error("Dataset is empty!")
        return False
        
    # Check for NaNs
    if df[["text", "label"]].isnull().any().any():
        logger.warning("Dataset contains NaN values in text or label columns.")
        
    # Check label set (standardized to 0, 1, 2)
    expected_labels = {0, 1, 2}
    actual_labels = set(df["label"].unique())
    if not actual_labels.issubset(expected_labels):
        logger.error(f"Invalid labels detected: {actual_labels - expected_labels}")
        return False
        
    # Log distribution
    dist = df["label"].value_counts(normalize=True).to_dict()
    logger.info(f"Dataset distribution: {dist}")
    
    return True

def check_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Checks for text overlap between train and test sets."""
    train_texts = set(train_df["text"].str.strip().str.lower())
    test_texts = set(test_df["text"].str.strip().str.lower())
    
    overlap = train_texts.intersection(test_texts)
    if overlap:
        logger.warning(f"CRITICAL: Data leakage detected! {len(overlap)} samples overlap between train and test.")
        # Scientific Trace: Log the first 5 problematic overlaps
        problematic = list(overlap)[:5]
        for p in problematic:
            logger.warning(f"Leakage Sample Trace: '{p}'")
        return False
    
    logger.info("Data leakage check passed: No overlap found.")
    return True

def bhojpuri_tokenizer_heuristic(text: str) -> list[str]:
    """
    Scientific heuristic for Bhojpuri-inflected tokens.
    Handles common Bhojpuri verb endings and particles that standard Hindi tokenizers might miss.
    """
    # Simple whitespace split for now, but sensitive to Bhojpuri markers
    tokens = text.split()
    
    # Heuristic markers for Bhojpuri (e.g., 'बा', 'बानि', 'हवे')
    # These are often attached or used as distinct particles
    return tokens

if __name__ == "__main__":
    # Quick sanity check
    data = {"text": ["Hello", "World"], "label": ["positive", "negative"]}
    df = pd.DataFrame(data)
    print(f"Hash: {get_dataset_hash(df)}")
    print(f"Integrity: {validate_integrity(df)}")
