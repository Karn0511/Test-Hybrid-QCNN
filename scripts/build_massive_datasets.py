from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASETS_DIR = Path("datasets")


@dataclass(frozen=True)
class Source:
    repo: str
    split: str
    config: str | None = None
    text_keys: tuple[str, ...] = ("text",)
    label_key: str = "label"
    label_map: dict[Any, int] | None = None


def label_from_stars(value: Any) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v <= 2:
        return 0
    if v == 3:
        return 1
    if v >= 4:
        return 2
    return None


def normalize_label(raw: Any, label_map: dict[Any, int] | None) -> int | None:
    if label_map is None:
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return None
        return v if v in (0, 1, 2) else None

    if raw in label_map:
        return label_map[raw]

    raw_s = str(raw)
    if raw_s in label_map:
        return label_map[raw_s]

    if "__star__" in label_map:
        return label_from_stars(raw)

    return None


def pick_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            t = str(row[k]).strip()
            if t:
                return t
    return ""


def ensure_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])


def read_hashes(path: Path, max_rows: int = 1_000_000) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = (row.get("text") or "").strip()
            if text:
                out.add(hashlib.md5(text.encode("utf-8")).hexdigest())
            if i >= max_rows:
                break
    return out


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def language_sources(language: str) -> list[Source]:
    star = {"__star__": 1}
    if language == "english":
        return [
            Source("stanfordnlp/sentiment140", "train[:1200000]", text_keys=("text",), label_key="sentiment", label_map={0: 0, 2: 1, 4: 2}),
            Source("amazon_polarity", "train[:800000]", text_keys=("content", "text"), label_key="label", label_map={0: 0, 1: 2, 2: 2}),
            Source("yelp_polarity", "train[:400000]", text_keys=("text",), label_key="label", label_map={0: 0, 1: 2, 2: 2}),
            Source("tweet_eval", "train", config="sentiment", text_keys=("text",), label_key="label", label_map={0: 0, 1: 1, 2: 2}),
            Source("imdb", "train", text_keys=("text",), label_key="label", label_map={0: 0, 1: 2}),
        ]

    if language == "hindi":
        return [
            Source("xnli", "train", config="hi", text_keys=("premise",), label_key="label", label_map={0: 2, 1: 1, 2: 0}),
            Source("cardiffnlp/tweet_sentiment_multilingual", "train", config="hindi", text_keys=("text",), label_key="label", label_map={0: 0, 1: 1, 2: 2}),
            Source("ai4bharat/IndicSentiment", "validation", config="translation-hi", text_keys=("text", "sentence", "review", "premise"), label_key="label", label_map={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2, 0: 0, 1: 1, 2: 2}),
            Source("ai4bharat/IndicSentiment", "test", config="translation-hi", text_keys=("text", "sentence", "review", "premise"), label_key="label", label_map={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2, 0: 0, 1: 1, 2: 2}),
        ]

    if language == "bhojpuri":
        return [
            Source("abhiprd20/Bhojpuri-Behavioral-Corpus-8K", "train", text_keys=("bhojpuri", "text", "sentence"), label_key="label", label_map={"negative": 0, "neutral": 1, "positive": 2, "Negative": 0, "Neutral": 1, "Positive": 2, 0: 0, 1: 1, 2: 2}),
        ]

    if language == "maithili":
        return [
            Source("abhiprd20/Maithili_Sentiment_8K", "train", text_keys=("text", "sentence"), label_key="label", label_map={"negative": 0, "neutral": 1, "positive": 2, "Negative": 0, "Neutral": 1, "Positive": 2, 0: 0, 1: 1, 2: 2}),
        ]

    if language == "multilingual":
        srcs = [
            Source("cardiffnlp/tweet_sentiment_multilingual", "train", config="all", text_keys=("text",), label_key="label", label_map={0: 0, 1: 1, 2: 2}),
            Source("tweet_eval", "train", config="sentiment", text_keys=("text",), label_key="label", label_map={0: 0, 1: 1, 2: 2}),
            Source("xnli", "train", config="en", text_keys=("premise",), label_key="label", label_map={0: 2, 1: 1, 2: 0}),
            Source("xnli", "train", config="hi", text_keys=("premise",), label_key="label", label_map={0: 2, 1: 1, 2: 0}),
        ]
        # XNLI has many language configs and remains available on HF.
        for cfg in ["ar", "bg", "de", "el", "es", "fr", "ru", "sw", "th", "tr", "ur", "vi", "zh"]:
            srcs.append(Source("xnli", "train", config=cfg, text_keys=("premise",), label_key="label", label_map={0: 2, 1: 1, 2: 0}))
        return srcs

    return []


def expand_one(language: str, target_train: int, unlabeled_ratio: float, dedupe: bool) -> None:
    train_path = DATASETS_DIR / language / "train.csv"
    unl_path = DATASETS_DIR / language / "unlabeled.csv"
    ensure_csv(train_path)
    ensure_csv(unl_path)

    train_rows = row_count(train_path)
    unl_rows = row_count(unl_path)
    target_unl = int(target_train * unlabeled_ratio)

    seen: set[str] = set()
    if dedupe:
        seen = read_hashes(train_path)
        seen.update(read_hashes(unl_path))

    print(f"\n[{language.upper()}] start train={train_rows:,} unlabeled={unl_rows:,} seen={len(seen):,}")
    print(f"[{language.upper()}] target train={target_train:,} unlabeled={target_unl:,}")

    with open(train_path, "a", newline="", encoding="utf-8") as tf, open(unl_path, "a", newline="", encoding="utf-8") as uf:
        tw = csv.writer(tf)
        uw = csv.writer(uf)

        for src in language_sources(language):
            if train_rows >= target_train and unl_rows >= target_unl:
                break

            name = f"{src.repo}:{src.config or '-'}:{src.split}"
            print(f"[{language.upper()}] source {name}")

            try:
                if src.config:
                    ds = load_dataset(src.repo, src.config, split=src.split, streaming=True)
                else:
                    ds = load_dataset(src.repo, split=src.split, streaming=True)
            except Exception as stream_ex:
                print(f"[{language.upper()}] streaming failed for {name}: {stream_ex}")
                try:
                    ds = load_dataset(src.repo, src.config, split=src.split) if src.config else load_dataset(src.repo, split=src.split)
                except Exception as ex:
                    print(f"[{language.upper()}] failed loading {name}: {ex}")
                    continue

            added = 0
            for row in ds:
                if train_rows >= target_train and unl_rows >= target_unl:
                    break

                text = pick_text(row, src.text_keys)
                if len(text) < 8:
                    continue

                label = normalize_label(row.get(src.label_key), src.label_map)
                if label is None:
                    continue

                if dedupe:
                    h = hashlib.md5(text.encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)

                if unl_rows < target_unl and ((train_rows + unl_rows) % 4 == 0):
                    uw.writerow([text, label])
                    unl_rows += 1
                elif train_rows < target_train:
                    tw.writerow([text, label])
                    train_rows += 1

                added += 1
                if added % 20000 == 0:
                    print(f"[{language.upper()}] +{added:,} from {name} | train={train_rows:,} unlabeled={unl_rows:,}")

            print(f"[{language.upper()}] done {name}, added={added:,}")

    print(f"[{language.upper()}] final train={train_rows:,} unlabeled={unl_rows:,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build large language datasets from open sources.")
    p.add_argument("--languages", nargs="+", default=["english", "hindi", "bhojpuri", "maithili", "multilingual"])
    p.add_argument("--target-train", type=int, default=2_000_000)
    p.add_argument("--unlabeled-ratio", type=float, default=0.30)
    p.add_argument("--dedupe", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for lang in [x.lower() for x in args.languages]:
        expand_one(lang, args.target_train, args.unlabeled_ratio, args.dedupe)


if __name__ == "__main__":
    main()
