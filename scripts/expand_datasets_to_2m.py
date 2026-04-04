from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASETS_DIR = Path("datasets")


@dataclass(frozen=True)
class SourceSpec:
    repo: str
    split: str = "train"
    config: str | None = None
    text_keys: tuple[str, ...] = ("text",)
    label_key: str | None = "label"
    label_map: dict[Any, int] | None = None
    max_rows: int | None = None


LANGUAGE_SOURCES: dict[str, list[SourceSpec]] = {
    "english": [
        SourceSpec("stanfordnlp/sentiment140", text_keys=("text",), label_map={0: 0, 2: 1, 4: 2}),
        SourceSpec("stanfordnlp/imdb", text_keys=("text",), label_map={0: 0, 1: 2}),
        SourceSpec("tweet_eval", config="sentiment", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("amazon_polarity", text_keys=("content", "text"), label_map={0: 0, 1: 2, 2: 2}),
        SourceSpec("yelp_polarity", text_keys=("text",), label_map={0: 0, 1: 2, 2: 2}),
        SourceSpec("financial_phrasebank", config="sentences_allagree", split="train", text_keys=("sentence",), label_map={0: 0, 1: 1, 2: 2}),
    ],
    "hindi": [
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="hindi", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("ai4bharat/IndicSentiment", config="translation-hi", split="train", text_keys=("text", "sentence", "review", "premise"), label_map={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2, 0: 0, 1: 1, 2: 2}),
        SourceSpec("xnli", config="hi", split="train", text_keys=("premise",), label_map={0: 2, 1: 1, 2: 0}),
        SourceSpec("md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset", split="train", text_keys=("text", "sentence", "tweet"), label_map={"negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
    ],
    "bhojpuri": [
        SourceSpec("abhiprd20/Bhojpuri-Behavioral-Corpus-8K", split="train", text_keys=("bhojpuri", "text", "sentence"), label_map={"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
        SourceSpec("abhiprd20/Bhojpuri-Behavioral-Corpus-8K", split="test", text_keys=("bhojpuri", "text", "sentence"), label_map={"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
        SourceSpec("abhiprd20/Bhojpuri-Behavioral-Corpus-8K", split="validation", text_keys=("bhojpuri", "text", "sentence"), label_map={"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
    ],
    "maithili": [
        SourceSpec("abhiprd20/Maithili_Sentiment_8K", split="train", text_keys=("text", "sentence"), label_map={"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
        SourceSpec("abhiprd20/Maithili_Sentiment_8K", split="test", text_keys=("text", "sentence"), label_map={"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
        SourceSpec("abhiprd20/Maithili_Sentiment_8K", split="validation", text_keys=("text", "sentence"), label_map={"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2, 0: 0, 1: 1, 2: 2}),
    ],
    "multilingual": [
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="all", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="english", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="hindi", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="arabic", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="french", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="german", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="spanish", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="italian", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("cardiffnlp/tweet_sentiment_multilingual", config="portuguese", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("tweet_eval", config="sentiment", text_keys=("text",), label_map={0: 0, 1: 1, 2: 2}),
        SourceSpec("xnli", config="en", split="train", text_keys=("premise",), label_map={0: 2, 1: 1, 2: 0}),
        SourceSpec("xnli", config="hi", split="train", text_keys=("premise",), label_map={0: 2, 1: 1, 2: 0}),
    ],
}


def extract_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def normalize_label(raw_label: Any, label_map: dict[Any, int] | None) -> int | None:
    if label_map is not None:
        if raw_label in label_map:
            return label_map[raw_label]
        raw_as_str = str(raw_label)
        if raw_as_str in label_map:
            return label_map[raw_as_str]
        return None

    if raw_label is None:
        return None
    try:
        label = int(raw_label)
    except (TypeError, ValueError):
        return None
    return label if label in (0, 1, 2) else None


def dataset_iter(spec: SourceSpec):
    kwargs: dict[str, Any] = {"split": spec.split, "streaming": True}
    if spec.config:
        ds = load_dataset(spec.repo, spec.config, **kwargs)
    else:
        ds = load_dataset(spec.repo, **kwargs)
    return iter(ds)


def ensure_csv(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])


def read_existing_hashes(path: Path, limit: int | None = None) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            hashes.add(hashlib.sha1(text.encode("utf-8")).hexdigest())
            if limit is not None and i >= limit:
                break
    return hashes


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def append_from_local_sources(language: str, train_writer: csv.writer, unl_writer: csv.writer, seen: set[str],
                              train_count: int, unl_count: int, target_train: int, target_unl: int,
                              unlabeled_ratio: float) -> tuple[int, int]:
    if language == "multilingual":
        local_langs = ["english", "hindi", "bhojpuri", "maithili"]
    else:
        local_langs = [language]

    for local_lang in local_langs:
        source_path = DATASETS_DIR / local_lang / "train.csv"
        if not source_path.exists():
            continue

        with open(source_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if train_count >= target_train and unl_count >= target_unl:
                    return train_count, unl_count

                text = (row.get("text") or "").strip()
                label_raw = row.get("label")
                if not text:
                    continue
                try:
                    label = int(label_raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if label not in (0, 1, 2):
                    continue

                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)

                send_to_unl = unl_count < target_unl and ((train_count + unl_count) % int(1 / max(unlabeled_ratio, 0.05)) == 0)
                if send_to_unl:
                    unl_writer.writerow([text, label])
                    unl_count += 1
                elif train_count < target_train:
                    train_writer.writerow([text, label])
                    train_count += 1

    return train_count, unl_count


def expand_language(language: str, target_train: int, unlabeled_ratio: float, max_existing_hashes: int | None) -> None:
    lang_dir = DATASETS_DIR / language
    train_path = lang_dir / "train.csv"
    unl_path = lang_dir / "unlabeled.csv"

    ensure_csv(train_path)
    ensure_csv(unl_path)

    train_count = count_rows(train_path)
    target_unl = int(target_train * unlabeled_ratio)
    unl_count = count_rows(unl_path)

    seen = read_existing_hashes(train_path, max_existing_hashes)
    seen.update(read_existing_hashes(unl_path, max_existing_hashes))

    print(f"\n[{language.upper()}] Starting with train={train_count:,}, unlabeled={unl_count:,}, unique_seen={len(seen):,}")
    print(f"[{language.upper()}] Targets: train={target_train:,}, unlabeled={target_unl:,}")

    with open(train_path, "a", newline="", encoding="utf-8") as train_f, open(unl_path, "a", newline="", encoding="utf-8") as unl_f:
        train_writer = csv.writer(train_f)
        unl_writer = csv.writer(unl_f)

        train_count, unl_count = append_from_local_sources(
            language,
            train_writer,
            unl_writer,
            seen,
            train_count,
            unl_count,
            target_train,
            target_unl,
            unlabeled_ratio,
        )

        for spec in LANGUAGE_SOURCES.get(language, []):
            if train_count >= target_train and unl_count >= target_unl:
                break

            source_name = f"{spec.repo}:{spec.config or '-'}:{spec.split}"
            print(f"[{language.upper()}] Pulling source {source_name}")
            written_from_source = 0

            try:
                for row in dataset_iter(spec):
                    if train_count >= target_train and unl_count >= target_unl:
                        break
                    if spec.max_rows is not None and written_from_source >= spec.max_rows:
                        break

                    text = extract_text(row, spec.text_keys)
                    if len(text) < 8:
                        continue

                    raw_label = row.get(spec.label_key) if spec.label_key else None
                    label = normalize_label(raw_label, spec.label_map)
                    if label is None:
                        continue

                    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)

                    send_to_unl = unl_count < target_unl and ((train_count + unl_count) % int(1 / max(unlabeled_ratio, 0.05)) == 0)
                    if send_to_unl:
                        unl_writer.writerow([text, label])
                        unl_count += 1
                    elif train_count < target_train:
                        train_writer.writerow([text, label])
                        train_count += 1

                    written_from_source += 1
                    if written_from_source % 10000 == 0:
                        print(
                            f"[{language.upper()}] {source_name} progress: +{written_from_source:,} rows | "
                            f"train={train_count:,} unlabeled={unl_count:,}"
                        )

                print(f"[{language.upper()}] Completed {source_name}: +{written_from_source:,}")
            except Exception as ex:
                print(f"[{language.upper()}] Failed source {source_name}: {ex}")

    print(f"[{language.upper()}] FINAL: train={train_count:,}, unlabeled={unl_count:,}")
    if train_count < target_train:
        print(
            f"[{language.upper()}] NOTE: Could not reach target train rows from available open sources. "
            f"Need additional public datasets for this language."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand datasets toward 2M+ rows per language from open-source sources.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["english", "hindi", "bhojpuri", "maithili", "multilingual"],
        help="Languages to expand.",
    )
    parser.add_argument("--target-train", type=int, default=2_000_000, help="Target row count for train.csv")
    parser.add_argument("--unlabeled-ratio", type=float, default=0.30, help="unlabeled.csv target as a fraction of train target")
    parser.add_argument(
        "--max-existing-hashes",
        type=int,
        default=None,
        help="Optional cap while reading existing hashes to reduce RAM use.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    args = parse_args()
    selected = [l.lower() for l in args.languages]

    for language in selected:
        if language not in LANGUAGE_SOURCES:
            print(f"Skipping unknown language: {language}")
            continue
        expand_language(
            language=language,
            target_train=args.target_train,
            unlabeled_ratio=args.unlabeled_ratio,
            max_existing_hashes=args.max_existing_hashes,
        )


if __name__ == "__main__":
    main()
