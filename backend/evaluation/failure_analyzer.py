import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from backend.utils.logger import get_logger

logger = get_logger("FAILURE-ANALYZER")

class FailureAnalyzer:
    """
    Elite diagnostic suite for sentiment analysis failure modes.
    Focuses on Neutral-confusion bottlenecks and embedding hotspots.
    """
    def __init__(self, output_dir: str = "evaluation/latest/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_neutral_confusion(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """
        Deep-dive into Neutral (1) class confusion with Positive (2) and Negative (0).
        Calculates 'ambiguity score' for misclassified neutral samples.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Neutral as True but predicted as others
        neutral_mask = (y_true == 1)
        misclassified_neutrals = (y_true == 1) & (y_pred != 1)
        
        stats = {
            "neutral_total": int(np.sum(neutral_mask)),
            "neutral_correct": int(cm[1, 1]),
            "neutral_to_negative": int(cm[1, 0]),
            "neutral_to_positive": int(cm[1, 2]),
            "ambiguity_avg": float(np.mean(y_prob[misclassified_neutrals].max(axis=1))) if any(misclassified_neutrals) else 0.0
        }
        
        with open(self.output_dir / "neutral_confusion.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        logger.info(f"📊 [Neutral Analysis]: Accuracy on Neutral class: {stats['neutral_correct'] / stats['neutral_total']:.2%}")
        return stats

    def cluster_errors(self, texts: list[str], y_true: np.ndarray, y_pred: np.ndarray, embeddings: np.ndarray, n_clusters: int = 5):
        """
        Groups failed samples into thematic clusters to identify 'failure hotspots'.
        Uses K-Means on the frozen embeddings.
        """
        error_mask = (y_true != y_pred)
        if not any(error_mask):
            logger.info("✅ [ZERO FAILURES]: Skipping error clustering.")
            return []

        error_embeddings = embeddings[error_mask]
        error_texts = [texts[i] for i, m in enumerate(error_mask) if m]
        
        # [v11.5] Memory Safety: Cap samples for clustering to avoid OOM
        MAX_ERROR_SAMPLES = 1000
        if len(error_texts) > MAX_ERROR_SAMPLES:
            indices = np.random.choice(len(error_texts), MAX_ERROR_SAMPLES, replace=False)
            error_embeddings = error_embeddings[indices]
            error_texts = [error_texts[i] for i in indices]
            logger.info(f"📊 [Memory Guard]: Capped error clustering to {MAX_ERROR_SAMPLES} samples.")

        # Safe-guard n_clusters
        n_clusters = min(n_clusters, len(error_texts))
        if n_clusters < 2: return []

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(error_embeddings)
        
        # Summarize clusters
        cluster_summary = []
        for i in range(n_clusters):
            c_mask = (clusters == i)
            c_texts = [error_texts[j] for j, m in enumerate(c_mask) if m]
            
            # Simple heuristic for cluster theme (avg sentence length or keywords)
            avg_len = np.mean([len(t.split()) for t in c_texts])
            
            cluster_summary.append({
                "cluster_id": i,
                "size": int(np.sum(c_mask)),
                "avg_words": float(avg_len),
                "samples": c_texts[:3] # Representative samples
            })
            
        with open(self.output_dir / "error_clusters.json", "w") as f:
            json.dump(cluster_summary, f, indent=4)
            
        logger.info(f"🧬 [Error Clustering]: Identified {n_clusters} failure hotspots.")
        return cluster_summary

    def run_full_diagnostics(self, results: dict):
        """
        Entry point to process raw 'train_and_predict' output.
        """
        y_true = np.array(results["y_true"])
        y_pred = np.array(results["y_pred"])
        y_prob = np.array(results["y_prob"])
        texts = results["texts"]
        
        # We need embeddings for clustering. 
        # In a real run, these come from the embedding pipeline.
        # If not provided, we skip clustering.
        embeddings = results.get("embeddings")
        
        self.analyze_neutral_confusion(y_true, y_pred, y_prob)
        
        if embeddings is not None:
            self.cluster_errors(texts, y_true, y_pred, embeddings)
        else:
            logger.warning("[!] No embeddings found in results. Skipping failure clustering.")

if __name__ == "__main__":
    # Mock data for local verification
    analyzer = FailureAnalyzer()
    y_t = np.array([1, 1, 0, 2])
    y_p = np.array([0, 1, 0, 1]) # Missed one neutral, one neutral as positive
    y_pb = np.array([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
    analyzer.analyze_neutral_confusion(y_t, y_p, y_pb)
    print("✅ Failure Analyzer (Mock) Finished.")
