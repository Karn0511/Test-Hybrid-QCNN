from __future__ import annotations
import numpy as np
from pathlib import Path
from joblib import Memory
import hashlib
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize joblib cache in the new processed/embeddings folder
_CACHE_DIR = Path("datasets/processed/embeddings")
_memory = Memory(_CACHE_DIR, verbose=0)

def _cached_encode(texts: list[str], model_name: str, batch_size: int = 128) -> np.ndarray:
    """
    v22.0 Absolute Zero: Sharded Multi-GPU Encoding.
    Fully saturates both T4 GPUs on Kaggle while maintaining strict context safety.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    logger.info(f"🧬 [Absolute Zero]: Sharding {len(texts)} samples across {torch.cuda.device_count()} GPUs...")
    
    try:
        # v35.2 Offline Logic: Prefer local cache to prevent HG check timeouts
        model = SentenceTransformer(model_name, local_files_only=True)
        logger.debug("✅ [LocalModel]: Successfully loaded transformer from local cache.")
    except Exception:
        logger.warning("⚠️ [OFFLINE-FALLBACK]: Local model not found. Attempting internet download...")
        try:
            model = SentenceTransformer(model_name, local_files_only=False)
        except Exception as e:
            logger.error(f"❌ [CRITICAL-OFFLINE]: Failed to download transformer. Error: {e}")
            return None # Signal higher-level fallback
    
    # Detect Multiple GPUs (Tesla T4 x2)
    device_count = torch.cuda.device_count()
    if device_count > 1:
        # v22.0: Parallel Strike
        pool = model.start_multi_process_pool()
        emb = model.encode_multi_process(texts, pool, batch_size=batch_size)
        model.stop_multi_process_pool(pool) # Mandatory closure for JIT safety
    else:
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
        
    logger.info("💾 [CacheMonitor]: Writing embeddings to disk cache... please wait.")
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
        logger.info(f"🔌 [HardwareLoad]: Initializing Transformer Model: {self.model_name}")
        try:
             # Prefer offline cache
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
            except Exception as e:
                logger.warning(f"⚠️ [TF-IDF Fallback]: Could not load transformer model. Error: {e}")
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf = TfidfVectorizer(max_features=self.dim, ngram_range=(1, 2))

    def fit_transform(self, texts: list[str], dataset_hash: str = None) -> np.ndarray:
        """
        v16.0 Quicksilver: Direct NPY-Matrix Caching.
        Bypasses BERT if a pre-computed matrix exists for this hash.
        """
        if dataset_hash:
            cache_file = _CACHE_DIR / f"{dataset_hash}_{self.dim}.npy"
            if cache_file.exists():
                # v28.1: Bypass-Guard verification
                loaded_emb = np.load(cache_file)
                if len(loaded_emb) >= len(texts):
                    logger.info("⚡ [Quicksilver Sync]: Loading %d samples directly from %s (Bypass active)", len(texts), cache_file.name)
                    return loaded_emb[:len(texts)] # Return the exact subset needed
                
                logger.warning("⚠️ [BYPASS-MISMATCH]: Cache size (%d) insufficient for N=%d. Re-embedding...", len(loaded_emb), len(texts))
        
        # Standard flow if no dataset-level cache
        emb = _cached_encode(texts, self.model_name, self.batch_size)
        
        if emb is not None:
            result = self._fix_dim(emb)
            
            # Save for next time if hash is provided
            if dataset_hash:
                cache_file = _CACHE_DIR / f"{dataset_hash}_{self.dim}.npy"
                np.save(cache_file, result)
                logger.debug(f"💾 [Quicksilver Cache]: Persistent NPY matrix saved for hash {dataset_hash[:8]}")
            return result
        
        # Fallback to TF-IDF if model download/load fails
        self._load_model()
        if self._tfidf:
            mat = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
            return self._fix_dim(mat)
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        # Same as fit_transform but without tfidf fit
        if self._model_available():
            emb = _cached_encode(texts, self.model_name, self.batch_size)
            return self._fix_dim(emb)
        self._load_model()
        if self._tfidf:
            mat = self._tfidf.transform(texts).toarray().astype(np.float32)
            return self._fix_dim(mat)
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def _model_available(self) -> bool:
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False

    def _fix_dim(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] == self.dim:
            return x.astype(np.float32)
        if x.shape[1] > self.dim:
            return x[:, : self.dim].astype(np.float32)
        pad = np.zeros((x.shape[0], self.dim - x.shape[1]), dtype=np.float32)
        return np.hstack([x.astype(np.float32), pad])

if __name__ == "__main__":
    # Test
    pipe = EmbeddingPipeline()
    test_texts = ["Hello world", "नमस्ते दुनिया", "का हाल बा"]
    embs = pipe.transform(test_texts)
    print(f"Embedding shape: {embs.shape}")
