from __future__ import annotations
import os
import sys
import hashlib
from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from backend.models.hybrid_qcnn import HybridQCNN
from backend.models.decision_fusion import MultiStreamFusion
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (v35.8 Elite)
    Designed for high-variance multi-dialect sentiment analysis.
    """
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

class PyTorchEstimator:
    """
    Standardized PyTorch Estimator for the Omega-Sentinel Research Engine.
    Implements v35.6 Elite Persistence with Atomic Checkpoints.
    """
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = config.get("epochs", 20)
        self.patience = config.get("patience", 10)
        self.best_state = None
        self.swa_model = None
        self.swa_scheduler = None
        
        # State Indicators
        self.self_learner = None
        self.x_train_pool = None
        self.y_train_pool = None
        self.texts_train_pool = None
        self.langs_train_pool = None
        
        # v35.7: Automated Parameter Accounting
        self.param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, x_train, y_train, x_val=None, y_val=None, languages=None, languages_val=None, texts=None, texts_val=None):
        import torch.optim.swa_utils
        from backend.training.self_learner import ContinuousSelfLearner
        from sklearn.utils.class_weight import compute_class_weight
        from backend.data.loader import OmegaMultilingualDataset, BalancedMultilingualSampler

        self.model.train()
        self.best_state = None
        
        # v35.6 Sync: Atomic attribute lock
        self.self_learner = ContinuousSelfLearner(self, None, self.config)
        self.x_train_pool = x_train
        self.y_train_pool = y_train
        self.texts_train_pool = texts
        self.langs_train_pool = languages

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        # v35.8: Using Focal Loss for dialect-aware imbalance handling
        criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 1e-3), weight_decay=0.01)
        
        # SWA Setup
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=self.config.get("lr", 1e-3) * 0.1)
        
        dataset = OmegaMultilingualDataset(x_train, y_train, languages)
        sampler = BalancedMultilingualSampler(dataset)
        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32), sampler=sampler)
        
        if x_val is not None:
            val_dataset = OmegaMultilingualDataset(x_val, y_val, languages_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config.get("batch_size", 32))

        history = []
        best_loss = float('inf')
        patience_counter = 0

        # Rich HUD Implementation
        from rich.live import Live
        from rich.table import Table
        from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
        
        telemetry_table = Table(title="CYBER-SUPREMACY TELEMETRY ENGINE (v16.0)", border_style="cyan")
        telemetry_table.add_column("Epoch", justify="center")
        telemetry_table.add_column("Loss", justify="center")
        telemetry_table.add_column("Val Loss", justify="center")
        telemetry_table.add_column("Status", justify="center")

        with Live(telemetry_table, refresh_per_second=4):
            for epoch in range(self.epochs):
                running_loss = 0.0
                for i, (inputs, targets, _) in enumerate(loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(loader)
                history.append(avg_loss)
                val_loss = 0.0
                status = "Epoch Synced"
                
                if x_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, targets, _ in val_loader:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            outputs = self.model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                    val_loss /= len(val_loader)
                    self.model.train()

                if epoch > self.epochs // 2:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                    status = "SWA Active"

                if epoch <= self.epochs // 2:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        status = "Best Model Saved"
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            status = "Early Stopping"
                            telemetry_table.add_row(f"{epoch+1}", f"{avg_loss:.4f}", f"{val_loss:.4f}", status)
                            break
                            
                telemetry_table.add_row(f"{epoch+1}", f"{avg_loss:.4f}", f"{val_loss:.4f}", status)

        # Phase 10: Final Weight Synchronization (v35.6 Fail-Safe)
        try:
            if self.swa_model is not None:
                logger.info("🧬 [SWA-SYNC]: Finalizing weight averaging and synchronizing BN layers...")
                torch.optim.swa_utils.update_bn(loader, self.swa_model, device=self.device)
                self.model = self.swa_model.module
        except (RuntimeError, ValueError, Exception) as e:
            logger.warning("⚠️ [SWA-FAIL]: Weight synchronization failed (%s). Reverting to best checkpoint.", e)
            if self.best_state:
                self.model.load_state_dict(self.best_state)
        
        # [NEW]: Atomic Save Persistence
        if self.config.get("save_model"):
            target_path = Path(self.config.get("model_path", "models/best_model.pt"))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(".tmp")
            torch.save(self.model.state_dict(), tmp_path)
            if target_path.exists(): os.remove(target_path)
            os.rename(tmp_path, target_path)
            logger.info("💾 [SOTA-SAVED]: Absolute persistent weights at %s", target_path)

    def predict(self, x_test: np.ndarray, lang_ids: np.ndarray | None = None) -> np.ndarray:
        probs = self.predict_proba(x_test, lang_ids=lang_ids)
        return np.argmax(probs, axis=1)

    def predict_proba(self, x_test: np.ndarray, lang_ids: np.ndarray | None = None) -> np.ndarray:
        self.model.eval()
        device = self.device
        x_tensor = torch.from_numpy(x_test).to(device)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32))
        
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

class SklearnEstimator:
    """Standardized Sklearn Wrapper for classical baselines."""
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.param_count = 0 # Not applicable

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train)

    def predict(self, x_test, **kwargs):
        return self.model.predict(x_test)

    def predict_proba(self, x_test, **kwargs):
        return self.model.predict_proba(x_test)

def build_model(config: dict, seed: int = 42) -> PyTorchEstimator | SklearnEstimator:
    """Factory function for the Elite-Tier Hybrid QCNN Engine."""
    torch.manual_seed(seed)
    model_id = config.get("id", "qcnn").lower()
    
    # v35.7 Elite Switch: Use Boolean flags instead of substring matching
    if config.get("use_fusion", False):
        raw_model = MultiStreamFusion(
            input_dim=config.get("input_dim", 384),
            n_qubits=config.get("n_qubits", 12),
            n_layers=config.get("n_layers", 4),
            n_classes=config.get("num_classes", 3)
        )
        return PyTorchEstimator(raw_model, config)
    
    if config.get("use_qcnn", False):
        raw_model = HybridQCNN(
            input_dim=config.get("input_dim", 384),
            n_classes=config.get("num_classes", 3),
            n_qubits=config.get("n_qubits", 12),
            n_layers=config.get("n_layers", 10)
        )
        return PyTorchEstimator(raw_model, config)
    
    elif "vqc" in model_id:
        from backend.models.market_baselines import MarketVQC_BERT
        raw_model = MarketVQC_BERT(n_qubits=8)
        return PyTorchEstimator(raw_model, config)
    elif "qvae" in model_id:
        from backend.models.market_baselines import MarketQVAE_QCNN_2024
        raw_model = MarketQVAE_QCNN_2024(n_qubits=6)
        return PyTorchEstimator(raw_model, config)
    elif "qlstm" in model_id:
        from backend.models.market_baselines import MarketQLSTM_2023
        raw_model = MarketQLSTM_2023(n_qubits=4)
        return PyTorchEstimator(raw_model, config)
    else:
        # Default Logistic Regression wrapper (v1.0 baseline)
        from sklearn.linear_model import LogisticRegression
        raw_model = LogisticRegression(max_iter=1000)
        return SklearnEstimator(raw_model, config)
