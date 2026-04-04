import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import torch.optim as optim
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Addition 14: Expected Calibration Error (ECE).
    Measures the difference between confidence and accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Confidence is max probability per sample
    confidences = np.max(y_prob, axis=1)
    # Predictions
    predictions = np.argmax(y_prob, axis=1)
    # Correctness
    accuracies = (predictions == y_true)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, output_path: str, n_bins: int = 10):
    """
    Addition 14: Reliability Diagram.
    Visualizes how well the model's confidence reflects reality.
    """
    # For multi-class, we often plot the 'predicted class' reliability
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = []
    bin_confs = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            
    plt.figure(figsize=(10, 8))
    plt.bar(bin_boundaries[:-1], bin_accs, width=1.0/n_bins, align='edge', alpha=0.7, label='Accuracy')
    plt.plot([0, 1], [0, 1], '--', color='red', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram (Calibration)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Reliability diagram saved to {output_path}")
def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Addition 15: Brier Score (Mean Squared Error of probabilities)."""
    n_classes = y_prob.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_one_hot)**2, axis=1)))

class TemperatureScaler(nn.Module):
    """
    Addition 15: Temperature Scaling for Calibration.
    Optimizes a single parameter 'T' to minimize NLL on a validation set.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        """Optimizes temperature T on validation logits."""
        self.train()
        logits_pt = torch.tensor(logits, dtype=torch.float32)
        labels_pt = torch.tensor(y_true, dtype=torch.long)
        
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_pt), labels_pt)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        logger.info(f"Optimized temperature: {self.temperature.item():.4f}")
        return self.temperature.item()
