from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKEND_ROOT = DEFAULT_PROJECT_ROOT / 'backend'
DEFAULT_RAW_DATASET_DIR = DEFAULT_PROJECT_ROOT / 'datasets' / 'raw'
DEFAULT_PROCESSED_DATASET_DIR = DEFAULT_PROJECT_ROOT / 'datasets' / 'processed'
DEFAULT_EXPERIMENTS_DIR = DEFAULT_PROJECT_ROOT / 'experiments'
DEFAULT_RESULTS_DIR = DEFAULT_EXPERIMENTS_DIR / 'results'
DEFAULT_MODEL_DIR = DEFAULT_PROJECT_ROOT / 'models'
DEFAULT_CHECKPOINTS_DIR = DEFAULT_EXPERIMENTS_DIR / 'models'


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).resolve()


@dataclass(slots=True)
class Settings:
    """Central application configuration loaded from environment variables."""

    app_name: str = os.getenv('QCNN_APP_NAME', 'AI QCNN Sentiment Platform')
    environment: Literal['development', 'staging', 'production'] = os.getenv('QCNN_ENVIRONMENT', 'development')  # type: ignore[assignment]
    api_host: str = os.getenv('QCNN_API_HOST', '0.0.0.0')
    api_port: int = int(os.getenv('QCNN_API_PORT', '8000'))

    project_root: Path = _env_path('QCNN_PROJECT_ROOT', DEFAULT_PROJECT_ROOT)
    backend_root: Path = _env_path('QCNN_BACKEND_ROOT', DEFAULT_BACKEND_ROOT)
    raw_dataset_dir: Path = _env_path('QCNN_RAW_DATASET_DIR', DEFAULT_RAW_DATASET_DIR)
    processed_dataset_dir: Path = _env_path('QCNN_PROCESSED_DATASET_DIR', DEFAULT_PROCESSED_DATASET_DIR)
    experiments_dir: Path = _env_path('QCNN_EXPERIMENTS_DIR', DEFAULT_EXPERIMENTS_DIR)
    results_dir: Path = _env_path('QCNN_RESULTS_DIR', DEFAULT_RESULTS_DIR)
    model_dir: Path = _env_path('QCNN_MODEL_DIR', DEFAULT_MODEL_DIR)
    checkpoints_dir: Path = _env_path('QCNN_CHECKPOINTS_DIR', DEFAULT_CHECKPOINTS_DIR)

    transformer_model_name: str = os.getenv('QCNN_TRANSFORMER_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embedding_backend: Literal['tfidf', 'transformer'] = os.getenv('QCNN_EMBEDDING_BACKEND', 'transformer')  # type: ignore[assignment]
    qcnn_qubits: int = int(os.getenv('QCNN_QCNN_QUBITS', '4'))
    qcnn_layers: int = int(os.getenv('QCNN_QCNN_LAYERS', '2'))
    qcnn_output_dim: int = int(os.getenv('QCNN_QCNN_OUTPUT_DIM', '3'))
    max_sequence_length: int = int(os.getenv('QCNN_MAX_SEQUENCE_LENGTH', '256'))
    batch_size: int = int(os.getenv('QCNN_BATCH_SIZE', '16'))
    num_workers: int = int(os.getenv('QCNN_NUM_WORKERS', '0'))
    learning_rate: float = float(os.getenv('QCNN_LEARNING_RATE', '0.001'))
    weight_decay: float = float(os.getenv('QCNN_WEIGHT_DECAY', '0.00001'))
    max_epochs: int = int(os.getenv('QCNN_MAX_EPOCHS', '8'))
    early_stopping_patience: int = int(os.getenv('QCNN_EARLY_STOPPING_PATIENCE', '3'))
    train_split_ratio: float = float(os.getenv('QCNN_TRAIN_SPLIT_RATIO', '0.8'))
    validation_split_ratio: float = float(os.getenv('QCNN_VALIDATION_SPLIT_RATIO', '0.1'))
    random_seed: int = int(os.getenv('QCNN_RANDOM_SEED', '42'))
    qcnn_checkpoint_name: str = os.getenv('QCNN_QCNN_CHECKPOINT_NAME', 'qcnn_model.pt')
    vectorizer_name: str = os.getenv('QCNN_VECTORIZER_NAME', 'vectorizer.pkl')
    model_bundle_name: str = os.getenv('QCNN_MODEL_BUNDLE_NAME', 'model_bundle.joblib')

    enable_dataset_streaming: bool = os.getenv('QCNN_ENABLE_DATASET_STREAMING', 'true').lower() == 'true'
    enable_multiprocessing_preprocessing: bool = os.getenv('QCNN_ENABLE_MULTIPROCESSING_PREPROCESSING', 'true').lower() == 'true'
    cache_transformer_embeddings: bool = os.getenv('QCNN_CACHE_TRANSFORMER_EMBEDDINGS', 'true').lower() == 'true'

    frontend_api_base_url: str = os.getenv('QCNN_FRONTEND_API_BASE_URL', 'http://localhost:8000/api')

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dataset_dir,
            self.processed_dataset_dir,
            self.results_dir,
            self.model_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
