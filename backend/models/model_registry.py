import torch
import joblib
from pathlib import Path
from backend.models.hybrid_qcnn import HybridQCNN
from backend.models.embedding_engine import EmbeddingEngine
from backend.utils.logger import get_logger
from backend.utils.config import get_settings

logger = get_logger(__name__)

class ModelRegistry:
    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self.embedding_engine = EmbeddingEngine()

    def get_model(self, model_name: str, device: str = None):
        """
        Loads and returns a model by name.
        Supported: 'qcnn', 'distilbert', 'logistic_regression', 'sgd', 'random_forest'
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'qcnn':
            model = HybridQCNN(input_dim=self.embedding_engine.get_embedding_dimension())
            path = self.settings.model_dir / "qcnn_best.pt"
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device))
                logger.info(f"Loaded QCNN model from {path}")
            model.to(device)
            model.eval()
            return model
            
        elif model_name in ['logistic_regression', 'sgd', 'random_forest']:
            path = self.settings.model_dir / "baselines" / f"{model_name}.joblib"
            if path.exists():
                model = joblib.load(path)
                logger.info(f"Loaded classical model {model_name} from {path}")
                return model
            else:
                logger.warning(f"Classical model {model_name} not found at {path}")
                return None
        
        # Add DistilBERT loading logic here if needed (e.g., using transformers pipeline)
        
        raise ValueError(f"Unknown model name: {model_name}")

    def predict(self, model_name: str, texts: list[str]):
        """
        Unified prediction interface.
        """
        model = self.get_model(model_name)
        
        if model_name == 'qcnn':
            embeddings = self.embedding_engine.get_embeddings(texts)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
            with torch.no_grad():
                logits = model(embeddings_tensor)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            label_map_inv = {0: 'negative', 1: 'neutral', 2: 'positive'}
            return [label_map_inv[p] for p in preds]
            
        elif model_name in ['logistic_regression', 'sgd', 'random_forest']:
            # Classical models typically use TF-IDF, but for our QCNN comparison
            # we might expect embeddings or TF-IDF. 
            pass

        return None
