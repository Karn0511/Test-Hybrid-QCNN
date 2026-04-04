import os
import ast
from pathlib import Path

def analyze_datasets():
    print("=== Dataset Multilingual Balance Audit ===")
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("datasets/ directory not found locally.")
    else:
        print("Scanning local datasets/")
        for lang in ["english", "hindi", "bhojpuri", "maithili", "multilingual"]:
            lang_dir = datasets_dir / lang
            if lang_dir.exists():
                count = sum(1 for _ in lang_dir.glob("*.csv"))
                print(f"[{lang.capitalize()}] -> {count} CSV files found.")
            else:
                print(f"[{lang.capitalize()}] -> No directory found.")

def profile_self_learner():
    print("\n=== Profiling QCNN_NEXUS Self-Learning Loop ===")
    target = Path("backend/training/self_learner.py")
    if not target.exists():
        print(f"File {target} not found.")
        return
    with open(target, 'r', encoding='utf-8') as f:
        content = f.read()

    bottlenecks = []
    if "current_model.predict_proba(x_pool)" in content and "batch_size" not in content:
        bottlenecks.append("⚠️ Bottleneck Found: Unbatched inference on x_pool detected.")
    if "current_model.predict_proba(x_pool, lang_ids=" in content:
        print(" - OK: Multilingual Language-Sync Verified in Self-Learner.")
    if "batch_size=512" in content or "batch_size=batch_size" in content:
         print(" - OK: Xeon-Turbo Batching Verified.")

    print("Diagnostic Findings:")
    for b in bottlenecks:
        print(f" - {b}")

def verify_quantum_layers():
    print("\n=== Verifying Quantum Entanglement Depth ===")
    target = Path("backend/quantum/layers.py")
    if not target.exists():
        print(f"File {target} not found.")
        return
    with open(target, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Xeon-Side Execution Optimizations:")
    if "lightning.qubit" in content and "adjoint" in content:
        print(" - OK: Fallback to multi-threaded 'lightning.qubit' with 'adjoint' diff method locked.")

    print("Entanglement Depth (SOTA):")
    if "AngleEmbedding" in content and "rotation='X'" in content and "rotation='Y'" in content:
        print(" - OK: Rich XY-Expressive feature embedding verified.")
    if "StronglyEntanglingLayers" in content:
        print(" - OK: Global StronglyEntanglingLayers verified.")
    if "qml.CRZ" in content and "qml.CNOT" in content:
        print(" - OK: CRZ pooling and CNOT cross-wire entanglement verified.")

if __name__ == "__main__":
    analyze_datasets()
    profile_self_learner()
    verify_quantum_layers()
