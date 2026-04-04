from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def cluster_errors(y_true: list[int], y_pred: list[int], texts: list[str], n_clusters: int = 3) -> dict:
    """
    Addition: Error Clustering (PhD Insight Section).
    Identifies common themes among misclassified samples using KMeans on TF-IDF.
    """
    errors = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    if len(errors) < n_clusters:
        return {"clusters": []}
    
    error_texts = [texts[i] for i in errors]
    
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X = tfidf.fit_transform(error_texts)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    results = []
    terms = tfidf.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :5]]
        cluster_samples = [error_texts[j] for j, c in enumerate(clusters) if c == i]
        results.append({
            "cluster_id": i,
            "top_terms": top_terms,
            "sample_count": len(cluster_samples),
            "representative_sample": cluster_samples[0] if cluster_samples else ""
        })
        
    return {"clusters": results}

def analyze_neutral_failures(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Addition: Neutral Class Failure Analysis (Research Gold).
    Identifies the confusion rate specifically for the neutral class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    neutral_mask = (y_true == 1)
    if not any(neutral_mask):
        return {"neutral_accuracy": 1.0, "major_confusion": "none"}
    
    neutral_correct = (y_pred[neutral_mask] == 1).sum()
    neutral_total = neutral_mask.sum()
    neutral_acc = neutral_correct / neutral_total
    
    confused_with = y_pred[neutral_mask & (y_pred != 1)]
    if len(confused_with) > 0:
        counts = pd.Series(confused_with).value_counts()
        major_label = "negative" if counts.index[0] == 0 else "positive"
        confusion_rate = float(counts.iloc[0] / neutral_total)
    else:
        major_label = "none"
        confusion_rate = 0.0
        
    return {
        "neutral_accuracy": float(neutral_acc),
        "major_confusion": major_label,
        "confusion_rate": confusion_rate
    }
