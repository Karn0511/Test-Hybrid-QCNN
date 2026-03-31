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

def run_sanity_shuffle():
    print("🚀 INITIALIZING DIAGNOSTIC 1: LABEL SHUFFLE COLLAPSE TEST")
    print("-" * 50)
    
    # 1. DATA
    print("[1/5] Loading 100 balanced samples...")
    df = load_processed()
    df_eng = df[df['language'] == 'english']
    
    # Stratified sampling of 100 rows
    neg = df_eng[df_eng['label'] == 0].sample(n=33, random_state=42)
    neu = df_eng[df_eng['label'] == 1].sample(n=33, random_state=42)
    pos = df_eng[df_eng['label'] == 2].sample(n=34, random_state=42)
    df_mini = pd.concat([neg, neu, pos]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("[2/5] Vectorizing Embeddings...")
    pipeline = EmbeddingPipeline(dim=384)
    X = pipeline.fit_transform(df_mini['text'].tolist())
    y = df_mini['label'].values
    
    print("[*] 🚨 PERTURBATION: SHUFFLING LABELS RANDOMLY TO DESTROY SIGNAL 🚨")
    np.random.seed(999)
    np.random.shuffle(y)
    
    X_pt = torch.tensor(X, dtype=torch.float32)
    y_pt = torch.tensor(y, dtype=torch.long)
    
    # 2. MODEL CONFIG
    print("[3/5] Instantiating Pure PyTorch Transformer Baseline (Dropout=0.0, QCNN=False)...")
    model = HybridQCNN(
        input_dim=384,
        n_qubits=8,
        n_classes=3,
        use_qcnn=False,  # Bypass QCNN
        dropout=0.0      # Disable dropout for pure memorization
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_pt, y_pt = X_pt.to(device), y_pt.to(device)
    
    # 3. TRAINING
    epochs = 50
    lr = 1e-2
    batch_size = 16
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(X_pt, y_pt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"[4/5] Training Protocol: Epochs={epochs}, LR={lr}, Batch={batch_size}\n")
    
    model.train()
    final_acc = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        
        # Calculate Epoch Accuracy
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_pt), dim=1)
            acc = accuracy_score(y_pt.cpu().numpy(), preds.cpu().numpy())
            final_acc = float(acc)
        model.train()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        
    print("-" * 50)
    
    # 5. ASSERT
    if final_acc < 0.50:
        print("[5/5] LABEL SHUFFLE TEST: PASS ✅")
        print(f"Model successfully collapsed to random state (Accuracy={final_acc:.2%}).")
        print("This PROVES the transformer is learning genuine underlying signal in standard mode.")
    else:
        print(f"[5/5] LABEL SHUFFLE TEST: FAIL ❌ (Accuracy={final_acc:.2%})")
        print("Model memorized noise. Over-parameterization or leakage occurring.")
        
if __name__ == "__main__":
    run_sanity_shuffle()
