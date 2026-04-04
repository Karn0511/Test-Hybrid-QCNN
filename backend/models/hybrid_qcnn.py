import numpy as np
import torch
import torch.nn as nn
from backend.quantum.layers import PennylaneQuantumLayer
from backend.quantum.tiered_adaptation import TierConfig, TieredQuantumAdapter
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class HybridQCNN(nn.Module):
    """
    Hybrid QCNN (v6 Phase 4 - HARDENED): Optimized for 32-core Xeon CPU & Multilingual Fusion.
    
    Architecture (PhD Mandate):
      1.  Classical Backbone: 384-dim (BERT) -> 128-dim (Semantic Latent)
      2.  Quantum Bridge: 128-dim -> 32-dim -> 8-dim (Qubit Input)
      3.  Quantum Circuit: 8-qubit QCNN (2-3 layers, reduced from 6 for speed)
      4.  Residual Fusion: residual(classical) + quantum + language -> logits
    
    Phase 4 Enhancements:
    - Explicit shape assertions at each layer
    - Residual skip connection for gradient flow
    - Reduced QCNN depth to 2-3 layers
    - Better parameter initialization
    """
    def __init__(
        self,
        input_dim: int = 384,
        n_qubits: int = 8,      # Fixed to 8 qubits
        n_layers: int = 1,      # REDUCED from 6 to 1 for speed (Phase 4)
        n_classes: int = 3,
        use_qcnn: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_qcnn = use_qcnn
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        
        logger.info(
            f"[HYBRID-QCNN v6 HARDENED]: "
            f"input={input_dim}d -> classical=128d -> quantum={n_qubits}d -> fusion -> output={n_classes}d"
        )

        # 1. Multi-Language Anchor (16-dim)
        self.lang_embed = nn.Embedding(5, 16)  # [en, hi, bh, mai, multi]
        logger.info("- Language embedding: 5 languages -> 16-dim")

        # 2. Classical Bridge (384 → 128)
        self.bridge_classical = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        logger.info("- Classical bridge: 384d -> 128d with LayerNorm")
        
        # 3. Quantum Bridge (128 → qubit_input)
        self.bridge_quantum = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, n_qubits),
            nn.Identity()
        )
        logger.info(f"- Quantum bridge: 128d -> 32d -> {n_qubits}d (phase input)")

        # 4. Quantum VQE/QCNN Engine
        if self.use_qcnn:
            self.quantum_layer = PennylaneQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
            logger.info(f"- Quantum layer: {n_qubits} qubits, {n_layers} layers (XY-embedding + entangling)")
        else:
            self.quantum_layer = nn.Identity()
            logger.info("- Quantum layer: DISABLED (classical baseline mode)")

        # 5. Residual Fusion Head
        # Concatenation: [Classical(128) RESIDUAL] + Quantum(n_qubits) + Language(16)
        fusion_dim = 128 + (n_qubits if use_qcnn else 0) + 16
        logger.info(f"- Fusion dimension: {fusion_dim}d (128 classical + {n_qubits if use_qcnn else 0} quantum + 16 lang)")
        
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
        logger.info(f"- Fusion head: {fusion_dim}d -> 128d -> 64d -> {n_classes}d")
        
        # Initialize parameters
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier uniform initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        logger.info("- Weights initialized with Xavier uniform")
        
    def forward(self, x, lang_ids=None):
        """
        Research-Grade Forward Pass with Shape Assertions and Residual Fusion.
        
        Args:
            x: (batch_size, input_dim) - BERT embeddings
            lang_ids: (batch_size,) - language indices
            
        Returns:
            logits: (batch_size, n_classes)
        """
        batch_size = x.shape[0]
        
        # --- STAGE A: Classical Path (Residual) ---
        c_latent = self.bridge_classical(x)
        assert c_latent.shape == (batch_size, 128), \
            f"Classical latent shape error: expected ({batch_size}, 128), got {c_latent.shape}"
        
        # --- STAGE B: Quantum Path ---
        if self.use_qcnn:
            q_input_raw = self.bridge_quantum(c_latent)
            assert q_input_raw.shape == (batch_size, self.n_qubits), \
                f"Quantum bridge output error: expected ({batch_size}, {self.n_qubits}), got {q_input_raw.shape}"
            
            # Clamp to a stable embedding range before angle encoding
            q_input = torch.clamp(q_input_raw, min=-1.0, max=1.0)
            q_input = (q_input + 1.0) * (np.pi / 2.0)
            q_out = self.quantum_layer(q_input)
            assert q_out.shape == (batch_size, self.n_qubits), \
                f"Quantum output error: expected ({batch_size}, {self.n_qubits}), got {q_out.shape}"
        else:
            # Classical baseline: zero-padding for quantum
            q_out = torch.zeros(batch_size, self.n_qubits, device=x.device, dtype=x.dtype)

        # --- STAGE C: Language Context ---
        if lang_ids is None:
            lang_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        l_features = self.lang_embed(lang_ids)
        assert l_features.shape == (batch_size, 16), \
            f"Language embedding error: expected ({batch_size}, 16), got {l_features.shape}"

        # --- STAGE D: Residual Fusion ---
        # Key improvement: classical latent goes directly into fusion (residual connection)
        fused = torch.cat([c_latent, q_out, l_features], dim=1)
        assert fused.shape == (batch_size, 128 + self.n_qubits + 16), \
            f"Fusion shape error: expected ({batch_size}, {128 + self.n_qubits + 16}), got {fused.shape}"

        # --- STAGE E: Final Classification Head ---
        logits = self.fused_head(fused)
        assert logits.shape == (batch_size, 3), \
            f"Output logits error: expected ({batch_size}, 3), got {logits.shape}"
        
        return logits

def get_model_summary(model):
    """Return model parameter count and architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_type": "HybridQCNN v6 (Phase 4 Hardened)"
    }
