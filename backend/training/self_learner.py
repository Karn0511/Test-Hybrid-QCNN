import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from backend.utils.logger import get_logger
from backend.training.hard_negative_miner import extract_hard_negatives
from backend.data.loader import LANG_MAP
from backend.models.standardized import FocalLoss

logger = get_logger("SELF-LEARNER")

class ContinuousSelfLearner:
    """
    v34.0 Nexus-Sentinel (v2): High-Fidelity Autonomous Refinement.
    1. Confidence Filtering: Only mines hard samples with < 0.7 confidence.
    2. Omega-Trace: Persistent history tracking for research reproducibility.
    3. Mixed-Precision: CUDA-accelerated refinement for 4GB VRAM.
    """
    def __init__(self, model_wrapper, embedder, config: dict):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = getattr(model_wrapper, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.embedder = embedder
        self.config = config
        self.history = []
        
    def auto_correct(self, x_pool: np.ndarray, y_pool: np.ndarray, texts_pool: list[str], langs_pool: list[str] = None, max_iterations: int = None):
        """
        Runs the autonomous refinement loop with confidence filtering and Omega-Trace.
        """
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.table import Table
        
        console = Console()
        current_model = self.model_wrapper
        max_iters = max_iterations if max_iterations is not None else 7
        
        # Initialize Trace Path within the specific run directory
        run_id = self.config.get("run_id", "default_nexus")
        trace_dir = Path("runs") / run_id / "nexus_trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "omega_trace.json"

        for iteration in range(max_iters):
            console.print(f"\n[bold magenta]Nexus v2: Deep Diagnostic {iteration+1}/{max_iters}[/]")
            
            with console.status(f"[bold cyan]Scanning Network Layers...[/]", spinner="dots"):
                # Layer 1: Latent Space Scan
                latent_centroids = []
                for c in range(3):
                    idx = np.where(y_pool == c)[0]
                    if len(idx) > 0: latent_centroids.append(x_pool[idx].mean(axis=0))
                
                separation_score = np.mean([np.linalg.norm(latent_centroids[i] - latent_centroids[j]) for i, j in [(0,1), (1,2), (0,2)]]) if len(latent_centroids) == 3 else 0.0
                
                # Layer 2: Output Probability Scan
                # v35.8 Sync: Pass language IDs for correct multilingual expert auditing
                y_prob = current_model.predict_proba(x_pool, lang_ids=np.array([LANG_MAP.get(l.lower(), 0) for l in langs_pool]) if langs_pool else None)
                y_pred = np.argmax(y_prob, axis=1)
                entropy = -np.sum(y_prob * np.log(y_prob + 1e-9), axis=1).mean()
                
                # Layer 3: Matrix Profiling
                cm = confusion_matrix(y_pool, y_pred, labels=[0, 1, 2])
                class_accs = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
                mean_acc = class_accs.mean()
                
                # Omega-Trace Telemetry
                trace_entry = {
                    "iteration": iteration + 1,
                    "accuracy": float(mean_acc),
                    "entropy": float(entropy),
                    "separation": float(separation_score),
                    "distribution": np.bincount(y_pred, minlength=3).tolist()
                }
                self.history.append(trace_entry)
                with open(trace_path, "w") as f:
                    json.dump(self.history, f, indent=4)
                
                # Diagnostic HUD (v35.0 Elite Polish)
                diag_table = Table(title="Nexus-Sentinel Elite Report", header_style="bold magenta", border_style="bright_blue")
                diag_table.add_column("Layer", style="cyan")
                diag_table.add_column("Metric", style="yellow")
                diag_table.add_column("State", style="bold")
                
                sep_state = "[bold green]Distinct" if separation_score > 0.8 else "[bold yellow]Muddled"
                entropy_state = "[bold green]Sharp" if entropy < 0.4 else "[bold red]Blurred"
                acc_state = "[bold blue]SOTA" if mean_acc > 0.94 else "[bold yellow]Refining"
                
                diag_table.add_row("Latent Space", f"Sep: {separation_score:.4f}", sep_state)
                diag_table.add_row("Decision Boundary", f"Entropy: {entropy:.4f}", entropy_state)
                diag_table.add_row("Final Matrix", f"Mean: {mean_acc:.2%}", acc_state)
                
                # Confusion Matrix
                cm_table = Table(title="Sentinel Matrix (v2)")
                cm_table.add_column("Label \\ Pred", justify="center", style="cyan")
                cm_table.add_column("Neg", justify="center")
                cm_table.add_column("Neu", justify="center")
                cm_table.add_column("Pos", justify="center")
                for i in range(3):
                    cm_table.add_row(["Negative", "Neutral", "Positive"][i], *[str(val) for val in cm[i]])
            
            console.log(diag_table)
            console.log(cm_table)
            
            if mean_acc > 0.95 and iteration > 0:
                console.print("[bold l_green][Target] v2 Consensus Reached: 95%+ Target Locked.[/]")
                break
                
            # Mining Hard Negatives with Confidence Filtering (< 0.7)
            top_k = 500 + (iteration * 250)
            # Filter pool based on confidence threshold
            confidences = y_prob.max(axis=1)
            hard_mask = confidences < 0.7
            
            if not np.any(hard_mask) and mean_acc > 0.90:
                console.print("[bold green][Done] No low-confidence samples remaining.[/]")
                break

            hard_df = extract_hard_negatives(
                y_pool.tolist(), 
                y_prob.tolist(), 
                texts_pool, 
                languages=langs_pool,
                top_k=min(top_k, len(x_pool))
            )
            
            # [v34.1 Fix]: Handle SOTA scenarios where zero hard negatives exist
            if not hard_df.empty and "confidence" in hard_df.columns:
                # Apply strict confidence filter to our mined candidates
                hard_df = hard_df[hard_df["confidence"] < 0.7]
            
            if hard_df.empty:
                logger.info("--- [SOTA LOCK]: Zero hard negatives found. Pool is mastered. ---")
                break
                
            with console.status("[bold blue]Slicing Nexus Pool v2...[/]", spinner="dots"):
                hard_idx = hard_df["idx"].tolist()
                hard_x = x_pool[hard_idx]
                label_map = {"negative": 0, "neutral": 1, "positive": 2}
                hard_y = np.array([label_map[l] for l in hard_df["label"].tolist()])
                
                # Capture Language IDs for v4 model compat
                if langs_pool:
                    hard_langs = np.array([LANG_MAP.get(langs_pool[i].lower(), 0) for i in hard_idx])
                else:
                    hard_langs = np.zeros(len(hard_idx), dtype=int)
                
                # Augment
                aug = 2
                hard_x_aug = np.vstack([hard_x]*aug)
                hard_y_aug = np.concatenate([hard_y]*aug)
                hard_l_aug = np.concatenate([hard_langs]*aug)
            
            epochs = 3
            self.model.train()
            refinement_lr = 5.5e-4 * (0.8 ** iteration) # More conservative LR
            optimizer = optim.AdamW(self.model.parameters(), lr=refinement_lr, weight_decay=0.01)
            
            loader = DataLoader(
                TensorDataset(torch.tensor(hard_x_aug, dtype=torch.float32), 
                              torch.tensor(hard_y_aug, dtype=torch.long),
                              torch.tensor(hard_l_aug, dtype=torch.long)),
                batch_size=32, shuffle=True
            )
            
            pulse_weights = torch.tensor([1.2, 8.0, 1.2]).to(self.device, non_blocking=True).float() # Multi-Dialect Neutral Anchor
            # v35.0 Elite: Switch to Focal Loss for high-precision boundary adjustment
            criterion = FocalLoss(weight=pulse_weights, gamma=2.5) 
            scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
            
            progress = Progress(SpinnerColumn("dots"), TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=30), TaskProgressColumn())
            with progress:
                pulse_task = progress.add_task(f"[bold blue]Pulse L{iteration+1} (AMP)...[/]", total=epochs)
                for epoch in range(epochs):
                    t_loss = 0.0
                    for bx, by, bl in loader:
                        optimizer.zero_grad()
                        bx = bx.to(self.device, non_blocking=True)
                        by = by.to(self.device, non_blocking=True)
                        bl = bl.to(self.device, non_blocking=True)
                        
                        if scaler:
                            with torch.amp.autocast('cuda'):
                                loss = criterion(self.model(bx, lang_ids=bl), by)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss = criterion(self.model(bx, lang_ids=bl), by)
                            loss.backward()
                            optimizer.step()
                        t_loss += loss.item()
                    progress.update(pulse_task, advance=1, description=f"[bold blue]Pulse Loss: {t_loss/len(loader):.4f}[/]")
            
            current_model.model = self.model
            
        return self.model_wrapper
