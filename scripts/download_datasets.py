from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CI Detection and Volume Contol
IS_CI = os.getenv("CI", "false").lower() == "true"
CI_LIMIT = 100 # Tiny subset for automated runs

RAW_DIR = Path("datasets/raw")
ENGLISH_DIR = RAW_DIR / "english"
HINDI_DIR = RAW_DIR / "hindi"
MULTILINGUAL_DIR = RAW_DIR / "multilingual"
MAITHILI_DIR = RAW_DIR / "maithili"

def ensure_dirs():
    for d in [ENGLISH_DIR, HINDI_DIR, MULTILINGUAL_DIR, MAITHILI_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def download_english():
    logger.info("Downloading English datasets...")
    
    # 1. Sentiment140
    if not (ENGLISH_DIR / "sentiment140.csv").exists():
        try:
            logger.info("Fetching Sentiment140...")
            ds = load_dataset("stanfordnlp/sentiment140", split="train", trust_remote_code=True)
            limit = CI_LIMIT if IS_CI else min(200_000, len(ds))
            df = ds.to_pandas().sample(n=limit, random_state=42)
            # Sentiment140: sentiment (0=Neg, 4=Pos, sometimes 2=Neu -> Map to 0, 2, 1)
            df = df.rename(columns={"text": "text", "sentiment": "label"})
            df["label"] = df["label"].map({0: 0, 2: 1, 4: 2})
            df[["text", "label"]].to_csv(ENGLISH_DIR / "sentiment140.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to download Sentiment140: {e}")
    else: logger.info("Sentiment140 exists. Skipping.")

    # 2. IMDB
    if not (ENGLISH_DIR / "imdb.csv").exists():
        try:
            logger.info("Fetching IMDB...")
<<<<<<< HEAD
            ds = load_dataset("stanfordnlp/imdb", split="train", trust_remote_code=True)
=======
            ds = load_dataset("stanfordnlp/imdb", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
            df = ds.to_pandas()
            # IMDB: text, label (0=Neg, 1=Pos -> Map to 0, 2)
            df = df.rename(columns={"text": "text", "label": "label"})
            df["label"] = df["label"].map({0: 0, 1: 2})
            df[["text", "label"]].to_csv(ENGLISH_DIR / "imdb.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to download IMDB: {e}")
    else: logger.info("IMDB exists. Skipping.")

    # 3. Amazon Polarity
    if not (ENGLISH_DIR / "amazon_polarity.csv").exists():
        try:
            logger.info("Fetching Amazon Polarity...")
<<<<<<< HEAD
            ds = load_dataset("amazon_polarity", split="train", trust_remote_code=True)
=======
            ds = load_dataset("amazon_polarity", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
            limit = CI_LIMIT if IS_CI else min(100_000, len(ds))
            df = ds.to_pandas().sample(n=limit, random_state=42)
            # Amazon: label (1=Neg, 2=Pos -> Map to 0, 2), content
            text_col = "content" if "content" in df.columns else "text"
            df = df.rename(columns={text_col: "text", "label": "label"})
            df["label"] = df["label"].map({0: 0, 1: 0, 2: 2}) # Common maps for Amazon
            df[["text", "label"]].dropna().to_csv(ENGLISH_DIR / "amazon_polarity.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to download Amazon: {e}")
    else: logger.info("Amazon exists. Skipping.")

    # 4. Yelp Polarity
    if not (ENGLISH_DIR / "yelp_polarity.csv").exists():
        try:
            logger.info("Fetching Yelp Polarity...")
<<<<<<< HEAD
            ds = load_dataset("yelp_polarity", split="train", trust_remote_code=True)
=======
            ds = load_dataset("yelp_polarity", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
            limit = CI_LIMIT if IS_CI else min(100_000, len(ds))
            df = ds.to_pandas().sample(n=limit, random_state=42)
            # Yelp: label (1=Neg, 2=Pos -> Map to 0, 2), text
            df = df.rename(columns={"text": "text", "label": "label"})
            df["label"] = df["label"].map({0: 0, 1: 0, 2: 2})
            df[["text", "label"]].dropna().to_csv(ENGLISH_DIR / "yelp_polarity.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to download Yelp: {e}")
    else: logger.info("Yelp exists. Skipping.")

    # 5. Tweet Eval (Critical for Neutral)
    if not (ENGLISH_DIR / "tweet_eval.csv").exists():
        try:
            logger.info("Fetching Tweet Eval (Sentiment)...")
<<<<<<< HEAD
            ds = load_dataset("tweet_eval", "sentiment", split="train", trust_remote_code=True)
=======
            ds = load_dataset("tweet_eval", "sentiment", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
            df = ds.to_pandas()
            # TweetEval: label (0=Neg, 1=Neu, 2=Pos -> Map to 0, 1, 2)
            df = df.rename(columns={"text": "text", "label": "label"})
            # Already mapped to 0, 1, 2, but we ensure
            df[["text", "label"]].to_csv(ENGLISH_DIR / "tweet_eval.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to download TweetEval: {e}")
    else: logger.info("Tweet Eval exists. Skipping.")

def download_hindi():
    logger.info("Downloading Hindi datasets...")
    
    # 1. IndicSentiment (Hindi)
    try:
        logger.info("Fetching IndicSentiment (Hindi)...")
<<<<<<< HEAD
        ds = load_dataset("ai4bharat/IndicSentiment", "translation-hi", trust_remote_code=True)
=======
        ds = load_dataset("ai4bharat/IndicSentiment", "translation-hi")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
        all_dfs = [part.to_pandas() for part in ds.values()]
        df = pd.concat(all_dfs, ignore_index=True)
        if IS_CI:
            df = df.sample(n=min(CI_LIMIT, len(df)), random_state=42)
        df.to_csv(HINDI_DIR / "indic_sentiment.csv", index=False)
        logger.info(f"IndicSentiment (Hindi) saved: {len(df)} samples.")
    except Exception as e:
        logger.warn(f"Failed to download IndicSentiment: {e}")

    # 2. 100k Code-Mixed Hindi Sentiment (Nishat)
    try:
        logger.info("Fetching 100k Code-Mixed Hindi Sentiment...")
        # Path: md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset
<<<<<<< HEAD
        ds = load_dataset("md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset", trust_remote_code=True)
=======
        ds = load_dataset("md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
        all_dfs = [part.to_pandas() for part in ds.values()]
        df = pd.concat(all_dfs, ignore_index=True)
        if IS_CI:
            df = df.sample(n=min(CI_LIMIT, len(df)), random_state=42)
        # Standardize labels if needed (assuming 0, 1, 2 or similar)
        # If labels are not 0, 1, 2, loader.py will handle mapping via LABEL_MAP
        df.to_csv(HINDI_DIR / "hindi_100k.csv", index=False)
        logger.info(f"Hindi 100k saved: {len(df)} samples.")
    except Exception as e:
        logger.error(f"Failed to download Hindi 100k: {e}")

    # 3. Tweet Sentiment Multilingual (Hindi)
    try:
        logger.info("Fetching Tweet Sentiment Multilingual (Hindi)...")
<<<<<<< HEAD
        ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "hindi", split="train", trust_remote_code=True)
=======
        ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "hindi", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
        df = ds.to_pandas()
        df.to_csv(HINDI_DIR / "tweet_hindi.csv", index=False)
        logger.info(f"Tweet Hindi saved: {len(df)} samples.")
    except Exception as e:
        logger.warn(f"Failed to download Tweet Hindi: {e}")

def download_multilingual():
    logger.info("Downloading Multilingual datasets...")
    
    # 1. Tweet Sentiment Multilingual
    try:
        logger.info("Fetching CardiffNLP Multilingual Sentiment...")
<<<<<<< HEAD
        ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "all", split="train", trust_remote_code=True)
=======
        ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "all", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
        df = ds.to_pandas()
        df.to_csv(MULTILINGUAL_DIR / "tweet_multilingual.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to download Multilingual Tweets: {e}")

    # 2. Go Emotions (Optional expansion)
    try:
        logger.info("Fetching Go Emotions...")
<<<<<<< HEAD
        ds = load_dataset("go_emotions", split="train", trust_remote_code=True)
=======
        ds = load_dataset("go_emotions", split="train")
>>>>>>> origin/audit-nexus-loop-5782051324856096483
        df = ds.to_pandas()
        df.to_csv(MULTILINGUAL_DIR / "go_emotions.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to download Go Emotions: {e}")

def extract_maithili():
    logger.info("Extracting Maithili data (Placeholder/Heuristic)...")
    # Rule: This usually requires filtering from a large corpus like AI4Bharat/samanantar or similar.
    # For now, we'll try to find any Maithili-specific sentiment if available or log guidance.
    try:
        # Example: Filter from IndicCorp or similar if possible.
        # Since Samanantar is parallel, we'd need sentiment labels which it lacks.
        # We will create a placeholder or try to find a specific source.
        pass
    except Exception as e:
        logger.error(f"Maithili extraction failed: {e}")

if __name__ == "__main__":
    ensure_dirs()
    download_english()
    download_hindi()
    if not IS_CI:
        download_multilingual()
        extract_maithili()
    else:
        logger.info("CI MODE: Skipping large Multilingual and Maithili extraction.")
    logger.info("Download process complete.")
