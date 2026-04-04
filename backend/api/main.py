from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
from backend.models.standardized import build_model
from backend.features.embedding import EmbeddingPipeline
from backend.features.data_loader import LanguageIDRouter
from backend.utils.logger import get_logger

logger = get_logger("AI-CORE-API")

# --- Milestone 6: The Emotional Intelligence Microservice ---

app = FastAPI(title="Hybrid QCNN Elite v2.0", description="Multi-Stream Quantum-Classical Sentiment Engine")

# Global instances (Loaded into RAM)
model_wrapper = None
embedder = None
router = None

class SentimentRequest(BaseModel):
    text: str = Field(..., max_length=5000)

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    detected_language: str
    expert_stream: str

@app.on_event("startup")
async def startup_event():
    """Initializes the SOTA Fusion Engine in memory."""
    global model_wrapper, embedder, router
    logger.info("⚡ [API-WARMUP]: Loading 5 expert streams and 12-qubit QSAM circuit...")
    
    config = {
        "use_fusion": True,
        "n_qubits": 12,
        "n_layers": 4,
        "num_classes": 3
    }
    
    # Load model and embedder
    model_wrapper = build_model(config)
    embedder = EmbeddingPipeline()
    router = LanguageIDRouter()
    
    logger.info("✅ [API-READY]: Emotional Intelligence Microservice is online.")

@app.post("/api/v1/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text input.")
    
    # 1. Language Routing (CPU-Only)
    lang = router.detect_language(request.text)
    
    # 2. Embedding (Transformer)
    x = embedder.transform([request.text])
    
    # 3. Decision Fusion Inference (GPU/Quantum)
    with torch.no_grad():
        # Note: In production, we'd pass the detected lang to MultiStreamFusion
        # for priority weighting.
        probs = model_wrapper.predict_proba(x)
        idx = np.argmax(probs[0])
        conf = float(probs[0][idx])
        
    sentiments = ["negative", "neutral", "positive"]
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiments[idx],
        confidence=conf,
        detected_language=lang,
        expert_stream=lang if lang in ["english", "hindi", "bhojpuri", "maithili"] else "multilingual"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
