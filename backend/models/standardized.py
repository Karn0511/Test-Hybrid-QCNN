from __future__ import annotations
import os
import sys
import json
import time
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
        if isinstance(self.model, nn.Module):
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
        from backend.data.loader import OmegaMultilingualDataset, BalancedMultilingualSampler, build_tempered_weights

        self.model.train()
        self.best_state = None
        
        # v35.6 Sync: Atomic attribute lock
        self.self_learner = ContinuousSelfLearner(self, None, self.config)
        self.x_train_pool = x_train
        self.y_train_pool = y_train
        self.texts_train_pool = texts
        self.langs_train_pool = languages

        # v35.8: Using Elite Tempered Focal Loss for dialect-aware imbalance handling
        class_weights = build_tempered_weights(y_train, num_classes=3).to(self.device)
        criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 1e-3), weight_decay=0.01)
        
        # Legacy-ready AMP Setup (v2.1)
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        
        # Elite v36.0: Gradient Accumulation (Simulating larger batches for CPU/RTX stability)
        accumulation_steps = self.config.get("accumulation_steps", 8)
        
        # SWA Setup (v36.0: Pickle-Safe Guard for Lightning Devices)
        try:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=self.config.get("lr", 1e-3) * 0.1)
        except (TypeError, Exception):
            self.swa_model = None
            self.swa_scheduler = None
        
        dataset = OmegaMultilingualDataset(x_train, y_train, languages)
        
        # v31.5 Dynamic Dispatch: Use Balanced Sampler ONLY for Multi-Language Fusion
        unique_langs = set(languages) if languages is not None else set()
        if len(unique_langs) > 1:
            sampler = BalancedMultilingualSampler(languages, batch_size=self.config.get("batch_size", 32))
            loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32), sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32), shuffle=True)
        
        if x_val is not None:
            val_dataset = OmegaMultilingualDataset(x_val, y_val, languages_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config.get("batch_size", 32))

        history = []
        best_loss = float('inf')
        patience_counter = 0

        # v4.4 Neural Pulse: Instant Mirroring
        self.pulse_dir = Path("evaluation/latest/pulses")
        self.pulse_dir.mkdir(parents=True, exist_ok=True)
        l_prefix = languages[0] if (languages and len(languages)>0) else "expert"
        seed_val = self.config.get("seed", 0)
        m_id_str = self.config.get("id", "qcnn").upper()
        self.pulse_id = f"{l_prefix}_{m_id_str}_s{seed_val}"

        try:
            with open(self.pulse_dir / f"{self.pulse_id}.json", "w") as f:
                json.dump({"model": m_id_str, "epoch": 0, "batch": 0, "total_batches": 1, "status": "Warming Up...", "timestamp": time.time()}, f)
        except Exception: pass

        try:
            for epoch in range(self.epochs):
                running_loss = 0.0
                for i, (inputs, targets, lang_ids) in enumerate(loader):
                    inputs, targets, lang_ids = inputs.to(self.device), targets.to(self.device), lang_ids.to(self.device)
                    
                    optimizer.zero_grad()
                    with autocast(enabled=(self.device.type == 'cuda')):
                        outputs = self.model(inputs, lang_ids=lang_ids)
                        loss = criterion(outputs, targets) / accumulation_steps
                    
                    if scaler:
                        scaler.scale(loss).backward()
                        if (i + 1) % accumulation_steps == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss.backward()
                        if (i + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    running_loss += loss.item() * accumulation_steps
                    
                    # Pulse for Overwatch dashboard
                    if i % 5 == 0:
                        preds = torch.argmax(outputs, dim=1)
                        batch_acc = (preds == targets).float().mean().item()
                        try:
                            pulse_data = {
                                "model": m_id_str,
                                "epoch": epoch + 1,
                                "batch": i + 1,
                                "total_batches": len(loader),
                                "acc": batch_acc,
                                "loss": loss.item() * accumulation_steps,
                                "status": "Crunching...",
                                "timestamp": time.time()
                            }
                            with open(self.pulse_dir / f"{self.pulse_id}.json", "w") as f:
                                json.dump(pulse_data, f)
                        except Exception: pass
                
                avg_loss = running_loss / len(loader)
                history.append(avg_loss)
                val_loss = 0.0
                status = "Epoch Synced"
                
                # Evaluation pulse
                try:
                    with open(self.pulse_dir / f"{self.pulse_id}.json", "w") as f:
                        json.dump({"model": m_id_str, "epoch": epoch + 1, "batch": len(loader), "total_batches": len(loader), "status": "Evaluating...", "timestamp": time.time()}, f)
                except Exception: pass

                if x_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, targets, l_ids in val_loader:
                            inputs, targets, l_ids = inputs.to(self.device), targets.to(self.device), l_ids.to(self.device)
                            outputs = self.model(inputs, lang_ids=l_ids)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                    val_loss /= len(val_loader)
                    self.model.train()

                if epoch > self.epochs // 2 and self.swa_model is not None:
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
                            break
                            
                logger.info("Expert Pulse: Epoch %d/%d | Loss: %.4f | Val: %.4f | %s", epoch+1, self.epochs, avg_loss, val_loss, status)
        except Exception as e:
            logger.error(f"[CRITICAL-TRAINER-FAIL]: {e}")
            raise e

        # Final SWA Sync
        try:
            if self.swa_model is not None:
                logger.info("🧬 [SWA-SYNC]: Finalizing weight averaging...")
                torch.optim.swa_utils.update_bn(loader, self.swa_model, device=self.device)
                self.model = self.swa_model.module
        except Exception as e:
            logger.warning("⚠️ [SWA-FAIL]: %s", e)
            if self.best_state: self.model.load_state_dict(self.best_state)
        
        # Atomic Save
        if self.config.get("save_model", True):
            target_path = Path(self.config.get("model_path", "models/best_model.pt"))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), target_path)
            logger.info("💾 [SOTA-SAVED]: Weights at %s", target_path)

    def predict(self, x_test: np.ndarray, lang_ids: np.ndarray | None = None) -> np.ndarray:
        probs = self.predict_proba(x_test, lang_ids=lang_ids)
        return np.argmax(probs, axis=1)

    def predict_proba(self, x_test: np.ndarray, lang_ids: np.ndarray | None = None) -> np.ndarray:
        self.model.eval()
        device = self.device
        batch_size = self.config.get("eval_batch_size", 512) 
        
        x_tensor = torch.from_numpy(x_test).to(device)
        if lang_ids is not None:
            l_tensor = torch.from_numpy(lang_ids).to(device)
            dataset = TensorDataset(x_tensor, l_tensor)
        else:
            dataset = TensorDataset(x_tensor)
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                l_ids = batch[1] if len(batch) > 1 else None
                outputs = self.model(inputs, lang_ids=l_ids)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

class SklearnEstimator:
    """Standardized Sklearn Wrapper."""
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.param_count = 0

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
    
    if config.get("use_fusion", False):
        raw_model = MultiStreamFusion(
            input_dim=config.get("input_dim", 384),
            n_qubits=config.get("n_qubits", 12),
            n_layers=config.get("n_layers", 4),
            n_classes=config.get("num_classes", 3)
        )
        return PyTorchEstimator(raw_model, config)
    
    if config.get("use_qcnn", False) or "qcnn" in model_id:
        from backend.models.hybrid_qcnn import HybridQCNN
        raw_model = HybridQCNN(
            input_dim=config.get("input_dim", 384),
            n_classes=config.get("num_classes", 3),
            n_qubits=config.get("n_qubits", 8),
            n_layers=config.get("n_layers", 6),
            dropout=config.get("dropout", 0.2)
        )
        return PyTorchEstimator(raw_model, config)
    
    # Jules Baselines
    elif "vqc" in model_id:
        from backend.models.market_baselines import MarketVQC_BERT
        return PyTorchEstimator(MarketVQC_BERT(n_qubits=8), config)
    elif "qvae" in model_id:
        from backend.models.market_baselines import MarketQVAE_QCNN_2024
        return PyTorchEstimator(MarketQVAE_QCNN_2024(n_qubits=6), config)
    elif "qlstm" in model_id:
        from backend.models.market_baselines import MarketQLSTM_2023
        return PyTorchEstimator(MarketQLSTM_2023(n_qubits=4), config)
    else:
        return SklearnEstimator(LogisticRegression(max_iter=1000), config)
