import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.utils.logger import get_logger
from backend.utils.config import get_settings
from pathlib import Path
from tqdm import tqdm

logger = get_logger(__name__)

class EmbeddingEngine:
    def __init__(self, model_name: str = None, device: str = None):
        settings = get_settings()
        self.model_name = model_name or settings.transformer_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing EmbeddingEngine with {self.model_name} on {self.device}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.cache_dir = settings.processed_dataset_dir / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_embeddings(self, texts: list[str], cache_key: str = None) -> np.ndarray:
        """
        Extracts embeddings for a list of texts. Uses cache if cache_key is provided.
        """
        if cache_key:
            cache_path = self.cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                logger.info(f"Loading embeddings from cache: {cache_path}")
                return np.load(cache_path)

        logger.info(f"Extracting embeddings for {len(texts)} samples...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Normalize embeddings for better QCNN performance
        logger.info("Normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        if cache_key:
            np.save(cache_path, embeddings)
            logger.info(f"Saved embeddings to cache: {cache_path}")

        return embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

def load_or_create_embeddings(df, cache_name: str = "final_merged_embeddings"):
    """Helper to get embeddings for the main dataset."""
    engine = EmbeddingEngine()
    texts = df['text'].tolist()
    return engine.get_embeddings(texts, cache_key=cache_name)
