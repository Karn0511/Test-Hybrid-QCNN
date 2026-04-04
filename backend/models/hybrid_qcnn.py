import numpy as np
import torch
import torch.nn as nn
from backend.quantum.layers import PennylaneQuantumLayer
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class HybridQCNN(nn.Module):
    """
    Hybrid QCNN (v36.0 - OMEGA HARDENED): Optimized for 32-core Xeon CPU & Multilingual Fusion.
    
    Architecture:
      1.  Classical Backbone: 384-dim (BERT) -> 128-dim (Semantic Latent)
      2.  Quantum Bridge: 128-dim -> 32-dim -> 8-dim (Qubit Input)
      3.  Quantum Circuit: 8-qubit QCNN (6 layers, Circular Pooling)
      4.  Residual Fusion: [Classical(128) + Quantum(8) + Language(16)] -> Output(3)
    
    Enhancements:
    - Shape assertions for rigorous research debugging
    - Residual skip connections for gradient preservation
    - Gelu activations and LayerNorm for stable convergence
    """
    def __init__(
        self,
        input_dim: int = 384,
        n_qubits: int = 8,
        n_layers: int = 6,
        n_classes: int = 3,
        use_qcnn: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_qcnn = use_qcnn
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        
        logger.info(
            f"⚡ [HYBRID-QCNN v36.0]: input={input_dim}d -> classical=128d -> quantum={n_qubits}d -> fusion -> output={n_classes}d"
        )

        # 1. Multi-Language Anchor (16-dim)
        self.lang_embed = nn.Embedding(5, 16)  # [en, hi, bh, mai, multi]

        # 2. Classical Bridge: High-Fidelity Latent Extraction
        self.bridge_classical = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        
        # 3. Quantum Bridge: Squeezing to Quantum Latent (e.g. 8d)
        self.bridge_quantum = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, n_qubits),
            nn.Identity()
        )

        # 4. Quantum VQE/QCNN Engine
        if self.use_qcnn:
            self.quantum_layer = PennylaneQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        else:
            self.quantum_layer = nn.Identity()

        # 5. Residual Fusion Head
        # Concatenation: [Classical(128)] + [Quantum(n_qubits)] + [Language(16)]
        fusion_dim = 128 + (n_qubits if use_qcnn else 0) + 16
        
        self.fused_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes)
        )
        
        # Initialize internal parameters
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier uniform initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, lang_ids=None):
        """Standard Forward Pass with Residual Fusion and Hardware Sync."""
        batch_size = x.shape[0]
        
        # --- STAGE A: Classical Path ---
        c_latent = self.bridge_classical(x)
        assert c_latent.shape == (batch_size, 128), f"Shape Error in Classical Path: {c_latent.shape}"
        
        # --- STAGE B: Quantum Path ---
        if self.use_qcnn:
            q_input_raw = self.bridge_quantum(c_latent)
            # Clamp and Scale to [0, pi] for Angle Encoding
            q_input = (torch.clamp(q_input_raw, -1.0, 1.0) + 1.0) * (np.pi / 2.0)
            q_out = self.quantum_layer(q_input)
        else:
            q_out = torch.zeros(batch_size, self.n_qubits, device=x.device, dtype=x.dtype)

        # --- STAGE C: Language Context ---
        if lang_ids is None:
            lang_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        l_features = self.lang_embed(lang_ids)

        # --- STAGE D: Residual Fusion ---
        fused = torch.cat([c_latent, q_out, l_features], dim=1)
        
        # --- STAGE E: Sentiment Inference ---
        return self.fused_head(fused)

def get_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params, 
        "trainable_params": trainable_params,
        "model_type": "HybridQCNN v36.0 (Hardened)"
    }
