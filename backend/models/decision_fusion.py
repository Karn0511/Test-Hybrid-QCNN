import torch
import torch.nn as nn
from backend.models.hybrid_qcnn import HybridQCNN
from backend.utils.logger import get_logger

logger = get_logger("DECISION-FUSION")

class MultiStreamFusion(nn.Module):
    """
    v2.1 Fusion Engine: [5-Expert Streams | Attention-Soft-Voting]
    """
    def __init__(self, input_dim: int = 384, n_qubits: int = 12, n_layers: int = 4, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes
        
        # Expert Streams
        self.experts = nn.ModuleDict({
            "english":      HybridQCNN(input_dim, n_qubits, n_layers, n_classes),
            "hindi":        HybridQCNN(input_dim, n_qubits, n_layers, n_classes),
            "bhojpuri":     HybridQCNN(input_dim, n_qubits, n_layers, n_classes),
            "maithili":     HybridQCNN(input_dim, n_qubits, n_layers, n_classes),
            "multilingual": HybridQCNN(input_dim, n_qubits, n_layers, n_classes)
        })
        
        # v4.0 Supreme: Language-Aware Gating (LAG)
        # Allows the meta-classifier to know the 'Native' language of the input
        self.lang_embedding = nn.Embedding(5, 16) 
        
        self.meta_attention = nn.Sequential(
            nn.Linear(5 * n_classes + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax(dim=1)
        )
        
        logger.info("[FUSION-CORE-v2.1]: 5 experts initialized (Independent Weight Manifold).")

    def forward(self, x, lang_ids=None):
        batch_size = x.size(0)
        
        # 1. Collect logits from all experts
        stream_logits = []
        for name in ["english", "hindi", "bhojpuri", "maithili", "multilingual"]:
            logits = self.experts[name](x) 
            stream_logits.append(logits)
        
        stacked_logits = torch.stack(stream_logits, dim=1)
        flat_logits = stacked_logits.view(batch_size, -1)
        
        # 2. Supreme LAG: Incorporate Language Context if available
        if lang_ids is not None:
            # lang_ids should be (batch_size,) long tensors
            lang_context = self.lang_embedding(lang_ids)
        else:
            # Fallback to zero context if unknown
            lang_context = torch.zeros((batch_size, 16), device=x.device)
            
        combined_features = torch.cat([flat_logits, lang_context], dim=1)
        
        # 3. Compute Attention with Context
        attn_weights = self.meta_attention(combined_features)
        
        # 4. Decision Fusion (Attention-Soft-Voting)
        fused_logits = torch.bmm(attn_weights.unsqueeze(1), stacked_logits)
        return fused_logits.squeeze(1)
