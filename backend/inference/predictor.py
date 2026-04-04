from __future__ import annotations

from dataclasses import dataclass

import torch

from backend.inference.model_registry import ModelRegistry
from backend.preprocessing.text_cleaning import clean_text
from backend.utils.hardware_detection import detect_hardware


@dataclass(slots=True)
class PredictionResult:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    language: str


class SentimentPredictor:
    """
    Production Predictor for Intelligence Platform v3.
    Uses 8-Qubit QCNN + Sentence Transformer.
    """
    def __init__(self) -> None:
        self.device = detect_hardware().device
        
        self.registry = ModelRegistry()
        self.registry.load_all()
        
        self.labels = self.registry.get_labels()
        self.model = self.registry.get_model('qcnn')
        self.embedder = self.registry.get_embedder('transformer')

    def predict(self, text: str) -> PredictionResult:
        if not self.model:
            return PredictionResult("Offline", 0.0, {}, "unknown")
            
        cleaned = clean_text(text)
        features = self.embedder.get_embeddings([cleaned.cleaned_text])
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            
        best_index = int(torch.tensor(probabilities).argmax().item())
        probability_map = {label: round(float(score), 4) for label, score in zip(self.labels, probabilities)}
        
        return PredictionResult(
            sentiment=self.labels[best_index],
            confidence=round(float(probabilities[best_index]), 4),
            probabilities=probability_map,
            language=cleaned.language,
        )
