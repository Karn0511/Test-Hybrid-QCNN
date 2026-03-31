import sys
import os
from pathlib import Path

# Fix path to root
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

def test_backend_imports():
    print("🧪 Running Backend Import Integrity Test...")
    try:
        from backend.models.hybrid_qcnn import HybridQCNN
        from backend.training.train import train_and_predict
        from backend.data.loader import load_processed
        from backend.features.embedding import EmbeddingPipeline
        print("✅ Backend core modules imported successfully.")
    except ImportError as e:
        print(f"❌ Structural Regression Detected: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_backend_imports()
