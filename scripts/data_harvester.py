"""
Data Harvester v8.5 - ELITE MULTILINGUAL & XNLI HINDI 
- Fixes small Hindi datasets by using massive XNLI Hindi (392k rows) translated data.
- Fixes multilingual by adding Arabic, French, German, Spanish, Italian, and Portuguese directly via CardiffNLP parquets.
- Bhojpuri: `abhiprd20/Bhojpuri-Behavioral-Corpus-8K`
- Maithili: `abhiprd20/Maithili_Sentiment_8K`
- English: `stanfordnlp/sst2`
"""

import os
import csv
from pathlib import Path
from datasets import load_dataset

DATASETS_DIR = Path("datasets")
MAX_SAMPLES = 50000      # Cap per language
UNLABELED_RATIO = 0.3    # ~30% goes to unlabeled pool


def setup_dirs(lang):
    p = DATASETS_DIR / lang
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_and_save(lang, hf_id, split, text_col, label_col, config_name=None, label_map=None, append=False):
    out_dir = setup_dirs(lang)
    train_path = out_dir / "train.csv"
    unlabeled_path = out_dir / "unlabeled.csv"

    existing = 0
    if append and train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            existing = sum(1 for _ in f) - 1
            
    if existing >= MAX_SAMPLES:
        print(f"⏭  [{lang.upper()}]: Already at cap ({existing} rows). Skipping {hf_id}.")
        return

    print(f"📥 [{lang.upper()}]: Downloading '{hf_id}' (config: {config_name}, split: {split})...")

    try:
        if config_name:
            ds = load_dataset(hf_id, config_name, split=split)
        else:
            ds = load_dataset(hf_id, split=split)
    except Exception as e:
        print(f"❌ [{lang.upper()}]: Error downloading '{hf_id}': {e}")
        return

    mode = "a" if append else "w"
    train_count = 0
    unlabeled_count = 0
    
    remaining = MAX_SAMPLES - existing
    
    try:
        with open(train_path, mode, newline="", encoding="utf-8") as tf, \
             open(unlabeled_path, mode, newline="", encoding="utf-8") as uf:

            tw = csv.writer(tf)
            uw = csv.writer(uf)
            if not append:
                tw.writerow(["text", "label"])
                uw.writerow(["text", "label"])

            for i, row in enumerate(ds):
                if train_count + unlabeled_count >= remaining:
                    break

                text = str(row.get(text_col, "") or "").strip()
                if not text:
                    continue

                raw_label = row.get(label_col)
                if label_map and raw_label is not None:
                    label = label_map.get(str(raw_label), 1)
                elif raw_label is not None:
                    try:
                        label = int(raw_label)
                    except (ValueError, TypeError):
                        label = 1
                else:
                    label = 1

                if (i % 10) < (UNLABELED_RATIO * 10):
                    uw.writerow([text, label])
                    unlabeled_count += 1
                else:
                    tw.writerow([text, label])
                    train_count += 1

            print(f"  ✅ [{lang.upper()}] +{train_count:,} train | +{unlabeled_count:,} unlabeled from '{hf_id}'")

    except Exception as e:
        print(f"❌ [{lang.upper()}]: Write error: {e}")


def harvest():
    print("\n🌐 [REAL-HARVEST v8.5]: Elite Datasets (Massive Hindi & 6-Lang Multilingual).\n")
    print("=" * 65)

    # 1. ENGLISH 
    print("\n[1/6] 🇬🇧 ENGLISH")
    download_and_save("english", "stanfordnlp/sst2", "train", "sentence", "label", label_map={"0": 0, "1": 2})

    # 2. HINDI (Massive XNLI Hindi text 392k mapped: contradiction=0, neutral=1, entailment=2)
    print("\n[2/6] 🇮🇳 HINDI")
    download_and_save("hindi", "xnli", "train", "premise", "label", config_name="hi", label_map={0: 2, 1: 1, 2: 0})

    # 3. BHOJPURI
    print("\n[3/6] 🇮🇳 BHOJPURI")
    download_and_save("bhojpuri", "abhiprd20/Bhojpuri-Behavioral-Corpus-8K", "train", "bhojpuri", "label", label_map={"Negative": 0, "Neutral": 1, "Positive": 2, 0: 0, 1: 1, 2: 2})

    # 4. MAITHILI
    print("\n[4/6] 🇮🇳 MAITHILI")
    download_and_save("maithili", "abhiprd20/Maithili_Sentiment_8K", "train", "text", "label", label_map={"Negative": 0, "Neutral": 1, "Positive": 2, 0: 0, 1: 1, 2: 2})

    # 5. MULTILINGUAL (Add 6 different new distinct languages for cross-lingual fusion)
    print("\n[5/6] 🌐 MULTILINGUAL (Adding Many Languages)")
    langs = ["arabic", "french", "german", "spanish", "italian", "portuguese"]
    first = True
    for lg in langs:
        download_and_save("multilingual", "cardiffnlp/tweet_sentiment_multilingual", "train", "text", "label", config_name=lg, label_map={0: 0, 1: 1, 2: 2}, append=not first)
        first = False
    # Top off with the remaining English/Hindi capacity
    download_and_save("multilingual", "stanfordnlp/sst2", "train", "sentence", "label", label_map={"0": 0, "1": 2}, append=True)

    # Final Summary
    print("\n" + "=" * 65)
    print("📊 [HARVEST v8.5 COMPLETE]: Dataset Status")
    print("=" * 65)
    total_rows = 0
    for lang in ["english", "hindi", "bhojpuri", "maithili", "multilingual"]:
        p = DATASETS_DIR / lang / "train.csv"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1
            size_mb = p.stat().st_size / 1e6
            print(f"  ✅ {lang:<14}: {rows:>7,} rows  ({size_mb:.1f} MB)")
            total_rows += rows
        else:
            print(f"  ❌ {lang:<14}: MISSING")
    print(f"\n  🏆 TOTAL: {total_rows:,} training samples across all languages")
    print("  → Ready to launch train_v2.py!\n")


if __name__ == "__main__":
    harvest()
