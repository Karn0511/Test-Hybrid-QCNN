import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def create_quantum_layer(n_qubits: int = 8, n_layers: int = 4):
    """
    Creates a high-expressibility QCNN circuit (v35.9 Elite).
    
    Architecture:
      1. XY-Expressive Encoding (Full Bloch Sphere projection)
      2. StronglyEntanglingLayers (Hardware-efficient connectivity)
      3. Global Circular Parametric Pooling (Learnable per-qubit)
      4. PauliZ measurements on all qubits
    """
    # v36.0 Hyper-Drive: Aggressive Hardware Dispatch
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 2.0:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            qnode_diff = "adjoint" 
            logger.info(f"🚀 [HYPER-DRIVE]: {torch.cuda.get_device_name(0)} (Lightning.GPU) locked.")
        else:
            # v36.0 Xeon-Turbo: Optimized for 32-core Xeon configurations
            dev = qml.device("lightning.qubit", wires=n_qubits)
            qnode_diff = "adjoint" 
            logger.info("⚡ [XEON-TURBO]: 32-Core Parallel Simulation active.")
    except Exception as e:
        dev = qml.device("default.qubit", wires=n_qubits)
        qnode_diff = "best"
        logger.warning(f"⚠️ [LOW-POWER]: Default Qubit Simulation active ({str(e)}).")

    @qml.qnode(dev, interface="torch", diff_method=qnode_diff)
    def circuit(inputs, weights_x, weights_y, weights_entangle, weights_pool):
        # === Stage 1: XY-Expressive Encoding ===
        qml.AngleEmbedding(inputs * weights_x, wires=range(n_qubits), rotation='X')
        qml.AngleEmbedding(inputs * weights_y, wires=range(n_qubits), rotation='Y')

        # === Stage 2: Variational Entanglement ===
        qml.StronglyEntanglingLayers(weights_entangle, wires=range(n_qubits))

        # === Stage 3: Circular Parametric Pooling ===
        for i in range(n_qubits):
            qml.CRZ(weights_pool[i], wires=[i, (i + 1) % n_qubits])
        
        # Cross-wire entanglement for deeper fusion
        for i in range(0, n_qubits, 3):
            qml.CNOT(wires=[i, (i + 2) % n_qubits])

        # === Stage 4: Measurement ===
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit

class PennylaneQuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 8, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        weight_shapes = {
            "weights_x":        (n_qubits,),                # Learnable X-embedding scale
            "weights_y":        (n_qubits,),                # Learnable Y-embedding scale
            "weights_entangle": (n_layers, n_qubits, 3),    # StronglyEntangling params
            "weights_pool":     (n_qubits,),                # Learnable QCNN Pooling angles
        }

        self.q_circuit = create_quantum_layer(n_qubits, n_layers)
        self.q_layer = qml.qnn.TorchLayer(self.q_circuit, weight_shapes)
        self._initialize_quantum_parameters()
        
        logger.info(
            "QuantumLayer v36.0 initialized: %d qubits, %d layers, XY-embedded SOTA",
            n_qubits, n_layers
        )

    def _initialize_quantum_parameters(self) -> None:
        """Keep initial quantum parameters small-but-nonzero to avoid flat starts."""
        for name, param in self.q_layer.named_parameters():
            if not param.requires_grad:
                continue
            if "pool" in name:
                nn.init.constant_(param, 0.01)
            else:
                nn.init.uniform_(param, a=-0.05, b=0.05)

        with torch.no_grad():
            total_norm = sum(p.norm().item() for p in self.q_layer.parameters())
            logger.info("🧠 [Q-INIT]: Total Quantum Norm %.6f", total_norm)

    def forward(self, x):
        return self.q_layer(x)
