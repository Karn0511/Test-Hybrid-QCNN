import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from backend.utils.logger import get_logger
from backend.features.data_loader import get_multi_stream_loaders, GradientAccumulator, MultiStreamDataset
from backend.models.standardized import build_model
from evaluation.metrics.evaluator import evaluate_predictions

logger = get_logger("TRAINER-v2")

class ActiveSelfLearningTrainer:
    """
    v2.1 OMEGA-TRAINER: [Entropy-Driven | Pseudo-Labeling | Xeon-Optimized]
    """
    def __init__(self, model_wrapper, config):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(exist_ok=True)

    def calculate_entropy(self, probs):
        """H(X) for mathematical uncertainty."""
        return -np.sum(probs * np.log2(probs + 1e-12), axis=1)

    def train_epoch(self, loader, optimizer, criterion, accum):
        self.model.train()
        total_loss = 0
        for i, (bx, by, bl) in enumerate(loader): # Sync: Passing lang_ids (bl)
            bx, by, bl = bx.to(self.device), by.to(self.device), bl.to(self.device)
            
            optimizer.zero_grad()
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = self.model(bx, lang_ids=bl)
                    loss = criterion(outputs, by)
            else:
                with torch.amp.autocast('cpu'):
                    outputs = self.model(bx, lang_ids=bl)
                    loss = criterion(outputs, by)
            
            loss = loss / accum.steps
            loss.backward()
            
            if accum.should_step():
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accum.steps
        return total_loss / len(loader)

    def save_checkpoint(self, epoch, optimizer, loss):
        """Standard Epoch Persistence."""
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

def train_v2_fusion(config):
    """v36.1 Xeon-Unbound entry point."""
    loaders, router = get_multi_stream_loaders(batch_size=config.get("batch_size", 32)) 
    model_wrapper = build_model(config)
    trainer = ActiveSelfLearningTrainer(model_wrapper, config)
    accum = GradientAccumulator(steps=config.get("grad_accum_steps", 8))
    epochs = config.get("epochs", 5)
    optimizer = torch.optim.AdamW(model_wrapper.model.parameters(), lr=config.get("lr", 1e-4))
    from backend.models.standardized import FocalLoss
    criterion = FocalLoss(gamma=2.5) 
    
    for epoch in range(epochs):
        for stream, loader in loaders.items():
            loss = trainer.train_epoch(loader, optimizer, criterion, accum)
            logger.info(f"Epoch {epoch+1} | Stream {stream} | Loss: {loss:.4f}")
        trainer.save_checkpoint(epoch + 1, optimizer, loss)
    return model_wrapper

if __name__ == "__main__":
    config = { "use_fusion": True, "batch_size": 32, "grad_accum_steps": 8, "epochs": 1, "lr": 1e-4 }
    train_v2_fusion(config)
