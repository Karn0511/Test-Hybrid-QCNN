import torch
import torch.nn as nn
from backend.quantum.layers import PennylaneQuantumLayer
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class MarketVQC_BERT(nn.Module):
    """
    Standard Variational Quantum Circuit (VQC) + BERT Baseline.
    Represents the 2022-2023 'Gold Standard' as seen in PennyLane research.
    
    Architecture:
      Input (384) -> Linear(384->8) -> VQC(8q, 4L) -> Linear(8->3)
    """
    def __init__(self, input_dim: int = 384, n_qubits: int = 8, n_layers: int = 4, n_classes: int = 3):
        super().__init__()
        self.projection = nn.Linear(input_dim, n_qubits)
        self.quantum_layer = PennylaneQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.decoder = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # 1. Simple linear projection (Market 2022 style)
        x = torch.tanh(self.projection(x))
        # 2. Map to [0, pi]
        x = (x + 1.0) * (3.14159 / 2.0)
        # 3. Standard VQC
        x = self.quantum_layer(x)
        # 4. Decode
        return self.decoder(x)


class MarketQVAE_QCNN_2024(nn.Module):
    """
    Quantum Variational Autoencoder (Q-VAE) + Tuned QCNN.
    Represents the June 2024 SOTA (IJISAE 97.64% Hindi model).
    
    Architecture:
      Input (384) -> Q-VAE Encoder (384->8) -> Tuned QCNN (8q) -> Linear(8->3)
    """
    def __init__(self, input_dim: int = 384, n_qubits: int = 8, n_layers: int = 4, n_classes: int = 3):
        super().__init__()
        # Q-VAE style Encoder (Multi-layer bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_qubits),
            nn.Tanh()
        )
        # Tuned QCNN (using our existing PennylaneQuantumLayer which has QCNN pooling)
        self.quantum_layer = PennylaneQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.decoder = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # 1. Q-VAE Encoder
        x = self.encoder(x)
        # 2. Map to [0, pi]
        x = (x + 1.0) * (3.14159 / 2.0)
        # 3. Tuned QCNN
        x = self.quantum_layer(x)
        # 4. Decode
        return self.decoder(x)


class MarketQLSTM_2023(nn.Module):
    """
    Hybrid QLSTM (Quantum-Inspired LSTM).
    Represents the 2023-2024 MDPI / SciELO low-resource benchmarks.
    
    In this simplified version for the benchmark, we use a classical LSTM 
    layer followed by a quantum classification head.
    """
    def __init__(self, input_dim: int = 384, n_qubits: int = 8, n_layers: int = 4, n_classes: int = 3):
        super().__init__()
        # Quantum LSTM cell approximation
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_qubits, batch_first=True)
        self.quantum_layer = PennylaneQuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.decoder = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # x is (B, D) -> need (B, 1, D) for LSTM
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0) # (B, 8)
        # Map to [0, pi]
        x = torch.tanh(x)
        x = (x + 1.0) * (3.14159 / 2.0)
        # Quantum Head
        x = self.quantum_layer(x)
        return self.decoder(x)
