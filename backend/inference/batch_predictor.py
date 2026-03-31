from __future__ import annotations

from backend.inference.predictor import PredictionResult, SentimentPredictor


class BatchSentimentPredictor:
    def __init__(self, predictor: SentimentPredictor | None = None) -> None:
        self.predictor = predictor or SentimentPredictor()

    def predict_many(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predictor.predict(text) for text in texts]
