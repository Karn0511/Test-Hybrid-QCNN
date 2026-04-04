from __future__ import annotations

import re
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from backend.utils.hf_datasets_import import import_hf_datasets
from backend.utils.logger import get_logger

import torch
from torch.utils.data import Sampler, Dataset
from backend.utils.logger import get_logger

import random
from langdetect import detect, DetectorFactory

logger = get_logger(__name__)

LANG_MAP = {
    "english": 0,
    "hindi": 1,
    "bhojpuri": 2,
    "maithili": 3,
    "en": 0,
    "hi": 1,
    "bh": 2,
    "mai": 3
}

RAW_DIR = Path("datasets/raw")
PROCESSED_DIR = Path("datasets/processed")
MERGED_DIR = PROCESSED_DIR / "merged"
METADATA_DIR = PROCESSED_DIR / "metadata"

FINAL_PATH = MERGED_DIR / "final_merged.csv"
SUBSET_PATH = MERGED_DIR / "qcnn_subset_100k.csv"
SPLITS_DIR = Path("datasets/splits")

# --- Kaggle Awareness Layer (SOTA Shift) ---
def _discover_kaggle_data() -> Path | None:
    # Recursively scan /kaggle/input for the dataset to survive name changes
    base_input = Path("/kaggle/input")
    if not base_input.exists():
        return None
    
    # Priority 1: Exact matches in our expected slug
    expected_slug = base_input / "sentiment-platform-data"
    search_patterns = ["final_merged.csv", "final_merge.csv", "final_merged.CSV"]
    
    if expected_slug.exists():
        for pattern in search_patterns:
            target = expected_slug / pattern
            if target.exists():
                return target

    # Priority 2: Recursive scan for any final_*.csv file
    for csv_path in base_input.rglob("final_*.csv"):
        return csv_path
        
    return None

KAGGLE_PATH = _discover_kaggle_data()
if KAGGLE_PATH:
    logger.info(f"[!] Kaggle Logic: Auto-Discovered dataset at {KAGGLE_PATH}")
    FINAL_PATH = KAGGLE_PATH
# -------------------------------------------
    # Redirect all processed/merged outputs to /kaggle/working (writable)
    PROCESSED_DIR = Path("/kaggle/working/datasets/processed")
    MERGED_DIR = PROCESSED_DIR / "merged"
    METADATA_DIR = PROCESSED_DIR / "metadata"
# -------------------------------------------

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "0": 0,
    "1": 1,
    "2": 2,
    0: 0,
    1: 1,
    2: 2
}

def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # NFKC Normalization is critical for Indic scripts
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Keep alphanumeric and common Indic ranges (Devanagari, Bengali, etc.)
    # \u0900-\u097F: Devanagari (Hindi, Bhojpuri, Maithili)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _detect_lang_safe(text: str) -> str:
    try:
        if len(text) < 10: return "unknown"
        return detect(text)
    except:
        return "unknown"

def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["text", "label", "language", "source"]
    
    # Ensure columns exist
    for col in required:
        if col not in out.columns:
            out[col] = "unknown" if col in {"language", "source"} else ""

    out["text"] = out["text"].astype(str).map(_normalize_text)
    
    # Map labels to 0, 1, 2
    out["label"] = out["label"].astype(str).str.lower().str.strip().map(LABEL_MAP)
    
    # Filter out invalid labels
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    
    # Language handling
    mask = (out["language"] == "unknown") | (out["language"] == "")
    if mask.any():
        logger.info(f"Detecting language for {mask.sum()} samples...")
        out.loc[mask, "language"] = out.loc[mask, "text"].map(_detect_lang_safe)

    out["source"] = out["source"].astype(str).str.strip()
    
    # Quality filters (v3.0 Extreme Cleaning)
    # Remove very short samples (< 15 chars) which are often noise
    out = out[out["text"].str.len() > 15]
    
    # Remove common repetitive patterns (e.g., "ha ha ha", "no no no")
    def _is_repetitive(text):
        words = text.split()
        if len(words) < 3: return False
        # If one word makes up more than 70% of the text, it's repetitive
        from collections import Counter
        counts = Counter(words)
        most_common_pct = counts.most_common(1)[0][1] / len(words)
        return most_common_pct > 0.7

    out = out[~out["text"].map(_is_repetitive)]
    
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)

    
    return out[required]

def _stratified_cap(df: pd.DataFrame, max_samples: int, label_col: str = "label", min_ratio: float = 0.15) -> pd.DataFrame:
    """
    Elite v3.0 Stratedied Capper: Protects minority classes (Neutral) from starvation.
    Ensures every class has at least 'min_ratio' of the target budget.
    """
    if len(df) <= max_samples:
        return df.reset_index(drop=True)

    labels = sorted(df[label_col].astype(int).unique().tolist())
    class_parts = {c: df[df[label_col].astype(int) == c] for c in labels}
    
    min_per_class = int(max(1, round(max_samples * float(min_ratio))))
    selected_parts = []
    selected_indices = set()
    
    for c in labels:
        take = min(len(class_parts[c]), min_per_class)
        selected = class_parts[c].sample(n=take, random_state=42)
        selected_parts.append(selected)
        selected_indices.update(selected.index.tolist())

    remaining = max_samples - sum(len(p) for p in selected_parts)
    if remaining > 0:
        remainder = df[~df.index.isin(selected_indices)]
        selected_parts.append(remainder.sample(n=remaining, random_state=42))

    return pd.concat(selected_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

def build_tempered_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    PhD-Grade Tempered Weighting: Prevents over-correction in multi-dialect loops.
    Blends effective-number weights with uniform priors.
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    
    beta = 0.999
    effective_num = (1.0 - beta) / (1.0 - np.power(beta, counts))
    raw = effective_num / np.mean(effective_num)
    
    # Blending (v31.4 Elite): Prevents class oscillation
    alpha = 0.15
    weights = (1.0 - alpha) * np.ones_like(raw) + alpha * raw
    weights = np.clip(weights, 0.85, 1.25)
    return torch.tensor(weights, dtype=torch.float32)

def _balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    counts = df["label"].value_counts()
    min_count = counts.min()
    
    chunks = []
    for _, part in df.groupby("label"):
        chunks.append(part.sample(n=min_count, random_state=42))
    
    return pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

def balance_languages(df: pd.DataFrame, mode: str = "balanced") -> pd.DataFrame:
    """Implement the user-defined master balancing strategy with internal class balancing."""
    if mode == "real_world":
        logger.info("REAL-WORLD MODE: Preserving original distributions...")
        # Optionally shuffle, but keep original distributions
        return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    logger.info("BALANCED MODE: Balancing languages and classes for research-grade capability...")
    
    def _get_balanced_lang(lang_df, n_target, lang_name):
        if lang_df.empty: return lang_df
        # First balance classes
        balanced_classes = _balance_classes(lang_df)
        if len(balanced_classes) > n_target:
            return balanced_classes.sample(n=n_target, random_state=42)
        return balanced_classes

    # 1. English (Capped at 300k balanced)
    df_eng = _get_balanced_lang(df[df["language"] == "english"], 300_000, "English")
    
    # 2. Hindi (Target 100k balanced)
    df_hi = _get_balanced_lang(df[df["language"] == "hindi"], 100_000, "Hindi")
    
    # 3. Bhojpuri/Maithili (Targets 50k each balanced)
    df_bh = _get_balanced_lang(df[df["language"] == "bhojpuri"], 50_000, "Bhojpuri")
    df_mai = _get_balanced_lang(df[df["language"] == "maithili"], 50_000, "Maithili")
    
    # 4. Multilingual (100k balanced if possible)
    df_multi = _get_balanced_lang(df[df["language"] == "multilingual"], 100_000, "Multilingual")
    
    other = df[~df["language"].isin(["english", "hindi", "bhojpuri", "maithili", "multilingual"])]
    
    balanced = pd.concat([df_eng, df_hi, df_bh, df_mai, df_multi, other], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

def _load_all_raw() -> pd.DataFrame:
    """Walk through all subdirectories in datasets/raw and load CSVs."""
    all_frames = []
    
    if not RAW_DIR.exists():
        logger.warning(f"[!] {RAW_DIR} not found. Skipping raw data load.")
        return pd.DataFrame()
        
    # 1. Check for CSVs in raw core folders
    for lang_dir in RAW_DIR.iterdir():
        if not lang_dir.is_dir(): continue
        
        for csv_file in lang_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df["language"] = lang_dir.name
                df["source"] = csv_file.stem
                all_frames.append(df)
                logger.info(f"Loaded {len(df)} samples from {csv_file}")
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
                
    if not all_frames:
        return pd.DataFrame()
        
    return pd.concat(all_frames, ignore_index=True)

def prepare_dataset(force_rebuild: bool = False, limit: int = 2_000_000, mode: str = "balanced"):
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if FINAL_PATH.exists() and not force_rebuild:
        logger.info(f"Dataset already exists at {FINAL_PATH}")
        return FINAL_PATH

    logger.info("Building 1M+ Multilingual Dataset...")
    raw_combined = _load_all_raw()
    
    if raw_combined.empty:
        logger.warning("No raw data found. Please run scripts/download_datasets.py first.")
        return None

    processed = _normalize_frame(raw_combined)
    
    # Apply Language Balancing (Phases 3 & 4)
    processed = balance_languages(processed, mode=mode)
        
    logger.info(f"Final {mode} dataset size: {len(processed)} samples")
    print(processed["language"].value_counts())
    processed.to_csv(FINAL_PATH, index=False)
    
    # Create scientific subset for QCNN testing
    subset = _balance_classes(processed)
    if len(subset) > 100_000:
        subset = subset.sample(n=100_000, random_state=42)
    subset.to_csv(SUBSET_PATH, index=False)
    
    return FINAL_PATH

def load_processed(path: Path | None = None, mode: str = "balanced") -> pd.DataFrame:
    target = path or FINAL_PATH
    if not target.exists():
        prepare_dataset(force_rebuild=False, mode=mode)
    return pd.read_csv(target)

def load_fixed_split(language: str, split_type: str = "train", max_samples: int = None) -> pd.DataFrame:
    """
    PhD-Grade Data Loader: Retrieves isolated splits created in Phase 1.
    Usage: load_fixed_split('hindi', 'val', max_samples=3000)
    """
    path = SPLITS_DIR / language.lower() / f"{split_type}.csv"
    if not path.exists():
        raise FileNotFoundError(f"[DATA-MISSING]: Split {path} not found. Run Phase 1 first.")
    
    df = pd.read_csv(path)
    if max_samples and len(df) > max_samples:
        logger.info(f"[ELITE-CAP]: Stratified pruning {language} from {len(df)} -> {max_samples}")
        df = _stratified_cap(df, max_samples)
        
    logger.info(f"[LOAD]: {language} | {split_type} | N={len(df)}")
    return df

class OmegaMultilingualDataset(Dataset):
    def __init__(self, x, y, languages):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.lang_ids = torch.tensor([LANG_MAP.get(l.lower(), 0) for l in languages], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lang_ids[idx]

class BalancedMultilingualSampler(Sampler):
    """
    SOTA Omega-Sentinel Sampler: Guarantees every batch contains an interleaved 
    balance of English, Hindi, Bhojpuri, and Maithili.
    """
    def __init__(self, languages, batch_size=32):
        self.batch_size = batch_size
        self.indices_per_lang = {0: [], 1: [], 2: [], 3: []}
        for i, lang in enumerate(languages):
            l_id = LANG_MAP.get(lang.lower(), 0)
            if l_id in self.indices_per_lang:
                self.indices_per_lang[l_id].append(i)
        
        # Determine number of batches: minimum lang pool size / (batch_size // 4)
        samples_per_lang_per_batch = batch_size // 4
        self.num_batches = min(len(indices) for indices in self.indices_per_lang.values()) // samples_per_lang_per_batch
        
    def __iter__(self):
        # Shuffle indices for each language
        shuffled = {l: random.sample(idx, len(idx)) for l, idx in self.indices_per_lang.items()}
        samples_per_lang = self.batch_size // 4
        
        for b in range(self.num_batches):
            batch = []
            for l in range(4):
                start = b * samples_per_lang
                end = (b + 1) * samples_per_lang
                batch.extend(shuffled[l][start:end])
            yield batch

    def __len__(self):
        return self.num_batches

if __name__ == "__main__":
    import random # Ensure random is available for Sampler
    path = prepare_dataset(force_rebuild=True)
    if path:
        print(f"Dataset ready at: {path}")
