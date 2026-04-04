import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.models.standardized import build_model
from backend.features.embedding import EmbeddingPipeline
from backend.utils.logger import get_logger

logger = get_logger("KAGGE-BENCHMARK")

try:
    import kbench
except ImportError:
    logger.warning("[!] kaggle-benchmarks SDK not found. Using Mock kbench for local validation.")
    class MockKBench:
        def task(self, func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
    kbench = MockKBench()

class HybridQCNNSentimentModel:
    """
    Wrapper for Hybrid QCNN to match Kaggle Benchmarks SDK expectations.
    """
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.config["training"].get("model_path", "models/best_qcnn.pt")
        
        # Build and load model
        self.model_wrapper = build_model(self.config["training"])
        if Path(self.model_path).exists():
            self.model_wrapper.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"[v] Loaded weights from {self.model_path}")
        else:
            logger.warning(f"[!] No weights found at {self.model_path}. Benchmarking with random initialization!")
            
        self.embedder = EmbeddingPipeline()
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def generate(self, prompt: str) -> str:
        """
        KBench standard 'generate' method. 
        Extracts raw text from prompt and returns a sentiment label.
        """
        # Simple extraction logic for the prompt
        # Usually prompts look like: "Sentiment of '...':"
        text = prompt.strip()
        if "'" in text:
            parts = text.split("'")
            if len(parts) >= 3:
                text = parts[1]
        
        # Embed and Predict
        x = self.embedder.fit_transform([text])
        y_pred = self.model_wrapper.predict(x)[0]
        
        return self.label_map.get(int(y_pred), "neutral")

@kbench.task
def evaluate_sentiment(model, text: str, expected: str):
    """
    Kaggle Benchmark Task Definition.
    """
    prompt = f"Analyze the sentiment of this text: '{text}'. Respond with only 'positive', 'negative', or 'neutral'."
    prediction = model.generate(prompt)
    
    # KBench logs assertions for the leaderboard
    assert prediction.lower() == expected.lower(), f"Expected {expected}, got {prediction}"

def run_benchmark():
    config_path = "configs/master_config.yaml"
    model = HybridQCNNSentimentModel(config_path)
    
    # In a real Kaggle environment, you would use:
    # kbench.run_dataset(dataset_path, task=evaluate_sentiment, model=model)
    
    logger.info("[*] Benchmark logic initialized. Ready for Kaggle deployment.")
    
    # Mock Run for local verification
    samples = [
        ("This is amazing!", "positive"),
        ("I hate this.", "negative"),
        ("The weather is okay.", "neutral")
    ]
    
    logger.info("--- Local Mock Validation ---")
    for text, exp in samples:
        pred = model.generate(text)
        status = "PASS" if pred == exp else "FAIL"
        logger.info(f"[{status}] Text: {text[:20]}... | Pred: {pred} | Exp: {exp}")

if __name__ == "__main__":
    run_benchmark()
