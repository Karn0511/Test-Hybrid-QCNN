import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def cluster_misclassifications(y_true: list[int], y_pred: list[int], texts: list[str], n_clusters: int = 5) -> dict:
    """
    Addition 17: Error Clustering.
    Clusters misclassified samples to identify common patterns (e.g., sarcasm, ambiguity).
    """
    logger.info(f"Clustering misclassified samples into {n_clusters} groups...")
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    mis_idx = np.where(y_true_arr != y_pred_arr)[0]
    if len(mis_idx) < n_clusters:
        logger.warning("Not enough misclassified samples to cluster.")
        return {}
        
    mis_texts = [texts[i] for i in mis_idx]
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(mis_texts)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Identify top terms per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_report = {}
    for i in range(n_clusters):
        c_idx = np.where(clusters == i)[0]
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        
        examples = []
        for j in c_idx[:3]:
            examples.append({
                "text": mis_texts[j],
                "true": int(y_true_arr[mis_idx[j]]),
                "pred": int(y_pred_arr[mis_idx[j]])
            })
            
        cluster_report[f"cluster_{i}"] = {
            "size": len(c_idx),
            "top_terms": top_terms,
            "examples": examples
        }
        
    return cluster_report

def save_error_clusters(results: dict, output_path: Path = Path("analysis/error_clusters.json")):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Error clusters saved to {output_path}")
