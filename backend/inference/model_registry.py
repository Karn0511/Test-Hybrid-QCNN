from __future__ import annotations

import joblib
import torch
from typing import Any

from backend.models.embedding_engine import EmbeddingEngine
from backend.models.hybrid_qcnn import HybridQCNN
from backend.utils.config import get_settings
from backend.utils.hardware_detection import detect_hardware
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Centralized model registry for loading and caching all production models.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.settings = get_settings()
        self.device = detect_hardware().device
        
        self.models: dict[str, Any] = {}
        self.baselines: dict[str, Any] = {}
        self.embedders: dict[str, Any] = {}
        self.bundle_data: dict[str, Any] = {}
        self.baseline_metrics: list[dict[str, Any]] = []
        
        self.initialized = True
        logger.info("ModelRegistry initialized on device: %s", self.device)

    def load_all(self):
        """Loads the production QCNN, baselines, and embedding models."""
        self._load_bundle()
        self._load_qcnn()
        self._load_baselines()
        
        if 'transformer' not in self.embedders:
            logger.info("Loading EmbeddingEngine...")
            self.embedders['transformer'] = EmbeddingEngine(device=self.device)
            
        logger.info("ModelRegistry successfully loaded all components.")

    def _load_bundle(self):
        bundle_path = self.settings.model_dir / self.settings.model_bundle_name
        if bundle_path.exists():
            self.bundle_data = joblib.load(bundle_path)
            logger.info("Loaded model bundle from %s", bundle_path)
        else:
            logger.warning("Model bundle not found at %s", bundle_path)
            self.bundle_data = {
                'labels': ['negative', 'neutral', 'positive'],
                'embedding_backend': 'transformer'
            }

    def _load_qcnn(self):
        checkpoint_path = self.settings.model_dir / "qcnn_production_v3.pth"
        if not checkpoint_path.exists():
            checkpoint_path = self.settings.model_dir / self.settings.qcnn_checkpoint_name
            
        if checkpoint_path.exists():
            logger.info("Loading QCNN model from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Use labels from bundle, or default
            labels = self.bundle_data.get('labels', ['negative', 'neutral', 'positive'])

            model = HybridQCNN(input_dim=384, n_qubits=8, n_layers=4, n_classes=len(labels)).to(self.device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            self.models['qcnn'] = model
        else:
            logger.warning("QCNN checkpoint not found. QCNN is offline.")

    def _load_baselines(self):
        baselines_dir = self.settings.model_dir / 'baselines'
        if not baselines_dir.exists():
            return
            
        for path in baselines_dir.glob('*.joblib'):
            model_name = path.stem
            try:
                self.baselines[model_name] = joblib.load(path)
                logger.info("Loaded baseline: %s", model_name)
            except Exception as e:
                logger.warning('Failed to load baseline %s: %s', model_name, e)
        
        metrics_path = self.settings.results_dir / 'baseline_benchmarks.json'
        if metrics_path.exists():
            try:
                import json
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.baseline_metrics = json.load(f)
            except Exception as e:
                logger.warning("Failed to load baseline metrics: %s", e)

    def get_model(self, name: str) -> Any:
        return self.models.get(name) or self.baselines.get(name)

    def get_embedder(self, name: str) -> Any:
        return self.embedders.get(name)

    def get_labels(self) -> list[str]:
        return self.bundle_data.get('labels', ['negative', 'neutral', 'positive'])
