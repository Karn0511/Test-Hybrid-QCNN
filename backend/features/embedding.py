from __future__ import annotations
import os
import torch
import numpy as np
from pathlib import Path
from joblib import Memory
import hashlib
from backend.utils.logger import get_logger

# v36.0 Strategic Compatibility Shim: 
# Newer transformers may expect register_pytree_node, while torch 2.x exposes only _register_pytree_node.
if hasattr(torch, "utils") and hasattr(torch.utils, "_pytree"):
    _pytree = torch.utils._pytree
    if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
        def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
            return _pytree._register_pytree_node(
                typ, flatten_fn, unflatten_fn,
                to_dumpable_context=kwargs.get("to_dumpable_context"),
                from_dumpable_context=kwargs.get("from_dumpable_context"),
            )
        _pytree.register_pytree_node = _register_pytree_node_compat

logger = get_logger(__name__)

# Initialize joblib cache
_CACHE_DIR = Path("datasets/processed/embeddings")
_memory = Memory(_CACHE_DIR, verbose=0)

def _cached_encode(texts: list[str], model_name: str, batch_size: int = 128) -> np.ndarray:
    """
    v36.0 Xeon-Sentinel Build: Optimized Multi-Dialect BERT Core.
    Abolishes multi-process encoding on Windows to prevent CPU idling/hangs.
    Saturates the 32-core Xeon workstation via high-threading.
    """
    import torch
    import os
    from sentence_transformers import SentenceTransformer
    
    # 1. Threading Optimization for 32-core Xeon Gold 5218
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)
    
    try:
        # Load from disk cache to prevent internet check latency
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        logger.warning("⚠️ [OFFLINE-FAIL]: Local model not found. Attempting internet download...")
        model = SentenceTransformer(model_name, local_files_only=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Sequential Strike (Reliable & High-Throughput for Windows)
    logger.info(f"🧬 [XEON-IGNITION]: Encoding {len(texts)} samples on {device.upper()}...")
    emb = model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=True, 
        batch_size=batch_size,
        device=device
    )
    
    return emb

class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dim: int = 384, batch_size: int = 128):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None
        self._tfidf = None
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        if self._model is not None or self._tfidf is not None:
            return
        from sentence_transformers import SentenceTransformer
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
            except Exception as e:
                logger.warning(f"⚠️ [TF-IDF Fallback]: Transformer failed. Error: {e}")
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf = TfidfVectorizer(max_features=self.dim, ngram_range=(1, 2))

    def fit_transform(self, texts: list[str], dataset_hash: str = None) -> np.ndarray:
        """v31.5 Quicksilver: Direct NPY-Matrix Caching."""
        if dataset_hash:
            cache_file = _CACHE_DIR / f"{dataset_hash}_{self.dim}.npy"
            if cache_file.exists():
                loaded_emb = np.load(cache_file)
                if len(loaded_emb) >= len(texts):
                    logger.info("⚡ [QUICKSILVER-Bypass]: Loading %d samples from cache.", len(texts))
                    return loaded_emb[:len(texts)]
        
        emb = _cached_encode(texts, self.model_name, self.batch_size)
        
        if emb is not None:
            result = self._fix_dim(emb)
            if dataset_hash:
                cache_file = _CACHE_DIR / f"{dataset_hash}_{self.dim}.npy"
                np.save(cache_file, result)
            return result
        
        self._load_model()
        if self._tfidf:
            mat = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
            return self._fix_dim(mat)
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        # Simplified for prediction phase
        emb = _cached_encode(texts, self.model_name, self.batch_size)
        if emb is not None:
            return self._fix_dim(emb)
        self._load_model()
        if self._tfidf:
            mat = self._tfidf.transform(texts).toarray().astype(np.float32)
            return self._fix_dim(mat)
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def _fix_dim(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] == self.dim:
            return x.astype(np.float32)
        if x.shape[1] > self.dim:
            return x[:, : self.dim].astype(np.float32)
        pad = np.zeros((x.shape[0], self.dim - x.shape[1]), dtype=np.float32)
        return np.hstack([x.astype(np.float32), pad])

if __name__ == "__main__":
    pipe = EmbeddingPipeline()
    test_texts = ["Hello world", "नमस्ते दुनिया", "का हाल बा"]
    embs = pipe.transform(test_texts)
    print(f"✅ Success. Semantic shape: {embs.shape}")
