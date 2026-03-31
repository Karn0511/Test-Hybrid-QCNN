from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline


class EmbeddingProvider(Protocol):
    def fit(self, texts: list[str]) -> 'EmbeddingProvider': ...
    def transform(self, texts: list[str]) -> np.ndarray: ...
    def fit_transform(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class TfidfEmbeddingProvider:
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def fit(self, texts: list[str]) -> 'TfidfEmbeddingProvider':
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        sparse_matrix = self.vectorizer.transform(texts)
        return np.asarray(sparse_matrix.todense(), dtype=np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse_matrix = self.vectorizer.fit_transform(texts)
        return np.asarray(sparse_matrix.todense(), dtype=np.float32)

    def save(self, path: str) -> None:
        dump(self.vectorizer, path)


def build_logistic_regression(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[('model', LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced', random_state=random_state))],
        memory=None,
    )


def build_sgd_classifier(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[('model', SGDClassifier(loss='log_loss', class_weight='balanced', random_state=random_state))],
        memory=None,
    )


def build_random_forest(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[(
            'model',
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=-1,
                class_weight='balanced_subsample',
                random_state=random_state,
            ),
        )],
        memory=None,
    )
