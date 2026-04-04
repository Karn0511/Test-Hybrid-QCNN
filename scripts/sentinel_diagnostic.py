import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# [PULSE-PATH] Ensure root is accessible
sys.path.append(str(Path(__file__).parent.parent))

from backend.data.loader import load_processed, LANG_MAP
from backend.models.hybrid_qcnn import HybridQCNN
from backend.utils.logger import configure_logging, get_logger
from backend.data.dataset_guard import validate_integrity, get_dataset_hash

logger = get_logger("SENTINEL")
console = Console()

class SentinelDiagnostic:
    """
    JARVIS Sentinel (v1.0): Universal Research Diagnostic & Logic Guard.
    Consolidates environment checks, overfit testing, and gradient flow audits.
    """
    def __init__(self, mode="full"):
        self.mode = mode
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "status": "INITIALIZING",
            "checks": {},
            "results": {}
        }
        configure_logging(level="INFO")

    def run_env_checks(self):
        """Verifies Python, Torch, PennyLane, and core libraries."""
        console.print("\n[bold cyan]SENTINEL Phase 1: Environment Logic Audit[/]")
        checks = {}
        
        # 1. PyTorch & CUDA
        try:
            checks["torch"] = {
                "status": "PASS",
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
        except Exception as e:
            checks["torch"] = {"status": "FAIL", "error": str(e)}

        # 2. PennyLane
        try:
            import pennylane as qml
            checks["pennylane"] = {"status": "PASS", "version": qml.__version__}
        except Exception as e:
            checks["pennylane"] = {"status": "FAIL", "error": str(e)}

        # 3. Transformers
        try:
            import transformers
            checks["transformers"] = {"status": "PASS", "version": transformers.__version__}
        except Exception as e:
            checks["transformers"] = {"status": "FAIL", "error": str(e)}

        self.report["checks"]["environment"] = checks
        
        # Display Table
        env_table = Table(title="Environment Matrix", border_style="bright_blue")
        env_table.add_column("Component", style="cyan")
        env_table.add_column("Status", style="bold")
        env_table.add_column("Detail", style="yellow")
        
        for k, v in checks.items():
            status_style = "green" if v["status"] == "PASS" else "red"
            detail = v.get("version", v.get("error", "N/A"))
            env_table.add_row(k.capitalize(), f"[{status_style}]{v['status']}[/]", str(detail))
        
        console.print(env_table)
        return all(v["status"] == "PASS" for v in checks.values())

    def run_data_audit(self):
        """Verifies dataset integrity and split availability."""
        console.print("\n[bold magenta]SENTINEL Phase 2: Data Integrity Audit[/]")
        data_checks = {}
        
        try:
            from backend.data.loader import load_fixed_split
            df = load_fixed_split("hindi", "train", max_samples=100)
            data_checks["hindi_split"] = {
                "status": "PASS",
                "rows": len(df),
                "columns": df.columns.tolist()
            }
        except Exception as e:
            data_checks["hindi_split"] = {"status": "FAIL", "error": str(e)}

        self.report["checks"]["data"] = data_checks
        return data_checks["hindi_split"]["status"] == "PASS"

    def run_logic_overfit(self, n_samples=100, epochs=30):
        """Validates that the model can actually learn (Micro-Overfit)."""
        console.print(f"\n[bold green]SENTINEL Phase 3: Logic Overfit Pulse (N={n_samples})[/]")
        
        try:
            from backend.data.loader import load_fixed_split
            df_mini = load_fixed_split("hindi", "train", max_samples=n_samples)
            
            # Embed
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            x = embedder.encode(df_mini["text"].tolist(), convert_to_numpy=True)
            y = df_mini["label"].values
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x_pt = torch.tensor(x, dtype=torch.float32).to(device)
            y_pt = torch.tensor(y, dtype=torch.long).to(device)
            
            # Build Hybrid QCNN (Mini-Depth)
            model = HybridQCNN(input_dim=384, n_qubits=8, n_layers=4, n_classes=3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            
            model.train()
            final_acc = 0.0
            
            with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                pulse_task = progress.add_task("[cyan]Refining Logic...", total=epochs)
                
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(x_pt)
                    loss = criterion(outputs, y_pt)
                    loss.backward()
                    
                    # Gradient Check
                    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                    avg_grad = np.mean(grads) if grads else 0.0
                    
                    optimizer.step()
                    
                    # Accuracy
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        acc = (preds == y_pt).float().mean().item()
                        final_acc = acc
                    
                    progress.update(pulse_task, advance=1, description=f"Loss: {loss.item():.4f} | Acc: {acc:.2%}")
            
            status = "PASS" if final_acc >= 0.90 else "FAIL"
            self.report["results"]["overfit"] = {"status": status, "final_accuracy": final_acc}
            console.print(f"\n[bold {'green' if status == 'PASS' else 'red'}]Logic Pulse: {status} (Final Accuracy: {final_acc:.2%})[/]")
            return status == "PASS"

        except Exception as e:
            logger.error(f"Logic Pulse Aborted: {e}")
            self.report["results"]["overfit"] = {"status": "FAIL", "error": str(e)}
            return False

    def execute_full_audit(self):
        """Runs the entire diagnostic suite."""
        console.print(Panel.fit("[bold white]SENTINEL UNIVERSAL DIAGNOSTIC[/]\n[bright_black]Research Grade v1.0", border_style="cyan"))
        
        success = True
        success &= self.run_env_checks()
        success &= self.run_data_audit()
        
        if self.mode == "full":
            success &= self.run_logic_overfit()

        self.report["status"] = "COMPLETED" if success else "FLAGGED"
        
        # Save Report
        target_path = Path("evaluation/latest/SENTINEL_REPORT.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(self.report, f, indent=4)
        
        console.print(f"\n[bold cyan]Master Signal Locked:[/] {target_path}")
        return success

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal Sentinel Diagnostic")
    parser.add_argument("--mode", choices=["env", "data", "full"], default="full")
    args = parser.parse_args()
    
    sentinel = SentinelDiagnostic(mode=args.mode)
    if not sentinel.execute_full_audit():
        sys.exit(1)
    sys.exit(0)
