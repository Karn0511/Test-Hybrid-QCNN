import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Ensure root path is accessible
sys.path.append(str(Path(__file__).parent.parent))

from backend.data.loader import load_processed
from backend.models.hybrid_qcnn import HybridQCNN
from backend.features.embedding import EmbeddingPipeline
from sklearn.metrics import accuracy_score

def run_micro_overfit():
    print("🚀 INITIALIZING ENHANCED MICRO-OVERFIT VALIDATION TASK")
    print("-" * 50)
    
    # 1. DATA
    print("[1/5] Loading 100 balanced samples...")
    df = load_processed()
    df_eng = df[df['language'] == 'english'].copy()
    
    neg = df_eng[df_eng['label'] == 0].sample(n=33, random_state=42)
    neu = df_eng[df_eng['label'] == 1].sample(n=33, random_state=42)
    pos = df_eng[df_eng['label'] == 2].sample(n=34, random_state=42)
    df_mini = pd.concat([neg, neu, pos]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("[2/5] Vectorizing Embeddings (Cache Bypass Active)...")
    from sentence_transformers import SentenceTransformer
    raw_embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X = raw_embedder.encode(df_mini['text'].tolist(), convert_to_numpy=True)
    y = df_mini['label'].values
    
    X_pt_base = torch.tensor(X, dtype=torch.float32)
    y_pt_base = torch.tensor(y, dtype=torch.long)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_pt, y_pt = X_pt_base.to(device), y_pt_base.to(device)
    
    # SWEEP MATRIX
    lrs = [1e-2, 1e-3, 5e-4]
    class_weight_modes = [False, True]
    epochs = 50
    batch_size = 16
    
    results_log = {}
    
    for c_weights in class_weight_modes:
        for lr in lrs:
            run_id = f"LR_{lr}_Weights_{c_weights}"
            print(f"\n[!] Starting Ablation Run: {run_id}")
            
            model = HybridQCNN(input_dim=384, n_qubits=8, n_classes=3, use_qcnn=False, dropout=0.0).to(device)
            
            if c_weights:
                criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device))
            else:
                criterion = nn.CrossEntropyLoss()
                
            optimizer = optim.Adam(model.parameters(), lr=lr)
            dataset = torch.utils.data.TensorDataset(X_pt, y_pt)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            history_log = []
            
            model.train()
            final_acc = 0.0
            for epoch in range(epochs):
                epoch_loss = 0.0
                grad_norm_clf = 0.0
                epoch_preds = []
                
                for bx, by in loader:
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if 'fc' in name or 'classifier' in name:
                                grad_norm_clf += param.grad.data.norm(2).item() ** 2
                                
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    epoch_preds.extend(preds)
                    
                avg_loss = epoch_loss / len(loader)
                grad_norm_clf = grad_norm_clf ** 0.5
                dist = np.bincount(epoch_preds, minlength=3)
                
                model.eval()
                with torch.no_grad():
                    preds_full = torch.argmax(model(X_pt), dim=1)
                    acc = float(accuracy_score(y_pt.cpu().numpy(), preds_full.cpu().numpy()))
                    final_acc = acc
                model.train()
                
                if (epoch + 1) % 25 == 0 or epoch == 0:
                     print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | GN: {grad_norm_clf:.4f} | Dist: {dist.tolist()}")
                     
                history_log.append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": acc,
                    "grad_norm_clf": grad_norm_clf,
                    "pred_dist": dist.tolist()
                })
                
            results_log[run_id] = {
                "final_accuracy": final_acc,
                "trace": history_log
            }
            if final_acc >= 0.95:
                 print(f"-> 🟢 OVERFIT CONFIRMED for {run_id}")
            else:
                 print(f"-> 🔴 OVERFIT FAILED for {run_id}")
                 
    Path("evaluation/latest").mkdir(parents=True, exist_ok=True)
    with open("evaluation/latest/micro_overfit_log.json", "w") as f:
        json.dump(results_log, f, indent=4)
    print("\n[5/5] MATRIX COMPLETE: Full telemetry saved to evaluation/latest/micro_overfit_log.json")

if __name__ == "__main__":
    run_micro_overfit()
