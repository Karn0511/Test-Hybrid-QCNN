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

def run_single_batch_overfit():
    print("🚀 INITIALIZING DIAGNOSTIC 2: SINGLE BATCH OVERFIT TEST")
    print("-" * 50)
    
    # 1. DATA
    print("[1/5] Loading 16 balanced samples (Exact 1 Batch)...")
    df = load_processed()
    df_eng = df[df['language'] == 'english']
    
    neg = df_eng[df_eng['label'] == 0].sample(n=5, random_state=42)
    neu = df_eng[df_eng['label'] == 1].sample(n=5, random_state=42)
    pos = df_eng[df_eng['label'] == 2].sample(n=6, random_state=42)
    df_mini = pd.concat([neg, neu, pos]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("[2/5] Vectorizing Embeddings...")
    pipeline = EmbeddingPipeline(dim=384)
    X = pipeline.fit_transform(df_mini['text'].tolist())
    y = df_mini['label'].values
    
    X_pt = torch.tensor(X, dtype=torch.float32)
    y_pt = torch.tensor(y, dtype=torch.long)
    
    # 2. MODEL CONFIG
    print("[3/5] Instantiating Pure PyTorch...")
    model = HybridQCNN(
        input_dim=384,
        n_qubits=8,
        n_classes=3,
        use_qcnn=False,  
        dropout=0.0      
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_pt, y_pt = X_pt.to(device), y_pt.to(device)
    
    # 3. TRAINING
    epochs = 100
    lr = 0.01
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"[4/5] Training 1 Batch Protocol: Epochs={epochs}, LR={lr}\n")
    
    model.train()
    final_acc = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_pt)
        loss = criterion(logits, y_pt)
        loss.backward()
        optimizer.step()
        
        # Calculate Epoch Accuracy
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_pt), dim=1)
            acc = accuracy_score(y_pt.cpu().numpy(), preds.cpu().numpy())
            final_acc = float(acc)
        model.train()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")
            
        if final_acc >= 0.99 and epoch > 10:
            print(f"-> 🛑 Early stop hit 100% convergence at epoch {epoch+1}")
            break
        
    print("-" * 50)
    
    # 5. ASSERT
    if final_acc >= 0.99:
        print("[5/5] SINGLE BATCH TEST: PASS ✅")
        print(f"Model converged perfectly to {final_acc:.2%}.")
    else:
        print(f"[5/5] SINGLE BATCH TEST: FAIL ❌ (Accuracy={final_acc:.2%})")
        
if __name__ == "__main__":
    run_single_batch_overfit()
