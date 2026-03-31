import numpy as np
import torch
import torch.nn as nn
from backend.quantum.layers import PennylaneQuantumLayer
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class HybridQCNN(nn.Module):
    """
    Hybrid QCNN (v4 Omega-Sentinel): Universal Multilingual Research Architecture.
    Optimized for RTX 3050 (4GB VRAM) and i7 CPU.
    
    Features:
      1.  Language Embedding (16-dim) for dialect-aware sentiment mapping.
      2.  High-Fidelity MLP Bridge (384 -> 128 -> 32) with GELU.
      3.  Residual Fusion of Classical (128d) and Quantum (12d) features.
      4.  Asymmetric Skip-Connections for gradient preservation.
    """
    def __init__(
        self,
        input_dim: int = 384,
        n_qubits: int = 12,
        n_layers: int = 10,
        n_classes: int = 3,
        use_qcnn: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_qcnn = use_qcnn
        self.n_qubits = n_qubits
        
        logger.info(
            "⚡ INITIALIZING OMEGA-SENTINEL (v4): [Qubits: %d | Layers: %d | Mixed-Precision Ready]",
            n_qubits, n_layers
        )

        # 1. Language Embedding Layer (Dialect Anchor)
        self.lang_embed = nn.Embedding(4, 16) # [0:en, 1:hi, 2:bh, 3:mai]

        # 2. MLP Bridge: 384 -> 128 (Classical Representation) -> 32 (Quantum Input)
        self.bridge_classical = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.LayerNorm(128)
        )
        
        self.bridge_quantum = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, n_qubits),
            nn.Tanh() # Squeeze to [-1, 1] for Quantum Encoding
        )

        # 3. 12-Qubit Quantum Engine
        if self.use_qcnn:
            self.quantum_layer = PennylaneQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        else:
            self.quantum_layer = nn.Identity()

        # 4. Global Fusion Layer (Residual Bridge)
        # 128 (Classical) + n_qubits (Quantum) + 16 (Lang)
        fusion_dim = 128 + (n_qubits if use_qcnn else 0) + 16
        
        self.fused_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x, lang_ids=None):
        """
        Args:
            x (Tensor): BERT embeddings (B, 384)
            lang_ids (Tensor, optional): Language indices (B,). Defaults to 0 (English).
        """
        # A. Classical Latent Extraction (128d)
        c_features = self.bridge_classical(x)
        
        # B. Quantum Path
        if self.use_qcnn:
            # Squeeze to Quantum Latent (e.g. 12d)
            q_latent = self.bridge_quantum(c_features)
            
            # Quantum Encoding [0, pi]
            q_input = (q_latent + 1.0) * (np.pi / 2.0)
            
            # Quantum Processing
            q_out = self.quantum_layer(q_input) # (B, 12)
        else:
            q_out = None

        # C. Language Context (16d)
        if lang_ids is None:
            # Default to English (Index 0)
            lang_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        l_features = self.lang_embed(lang_ids)

        # D. Residual Fusion (Omega Style)
        if self.use_qcnn:
            final_features = torch.cat([c_features, q_out, l_features], dim=1)
        else:
            final_features = torch.cat([c_features, l_features], dim=1)

        # E. Final Sentiment Inference
        return self.fused_head(final_features)

def get_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total_params, "trainable_params": trainable_params}
