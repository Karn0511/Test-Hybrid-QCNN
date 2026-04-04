import os
import sys
import json
import torch
import numpy as np
import psutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- MASTER PATH INJECTION (v6.24) ---
# Ensure the parent directory is in the path for cloud-native execution
script_dir = Path(__file__).parent.absolute()
root_dir = script_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# --------------------------------------

from backend.training.train import train_and_predict, set_seed, prepare_dataset
from backend.utils.logger import get_logger, configure_logging
from evaluation.visualizer import plot_radar_metrics, plot_elite_ranking

logger = get_logger(__name__)

# Ensure datasets directory exists (v6.32)
os.makedirs("datasets/raw", exist_ok=True)
os.makedirs("datasets/processed", exist_ok=True)

# QUANTUM SPEEDUP: SET OMP THREADS FOR C++ LIGHTNING BACKEND
cores = psutil.cpu_count(logical=True)
os.environ["OMP_NUM_THREADS"] = str(cores)
logger.info(f"⚡ Utilizing {cores} OMP threads for Quantum Simulation")

logger = get_logger(__name__)

# Standardized Output Paths
OUTPUT_DIR = Path("evaluation/latest/global_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    {"id": "Classical_Baseline", "baseline": "logistic", "use_qcnn": False},
    {"id": "Market_VQC_BERT_2022", "baseline": "market_vqc", "use_qcnn": True, "n_qubits": 8},
    {"id": "Market_QVAE_QCNN_2024", "baseline": "market_qvae", "use_qcnn": True, "n_qubits": 8},
    {"id": "Market_QLSTM_2023", "baseline": "market_qlstm", "use_qcnn": True, "n_qubits": 8},
    {"id": "Hybrid_QCNN_Model (Ours)", "baseline": None, "use_qcnn": True, "n_qubits": 12}
]

def run_benchmark(n_rows: int = 1000, seed: int = 42):
    logger.info(f"🚀 Starting Global Ground-Base Benchmark (N={n_rows}, Seed={seed})")
    set_seed(seed)
    
    results = []
    for cfg in MODEL_CONFIGS:
        logger.info(f"--- Training Model: {cfg['id']} ---")
        config = {
            "id": cfg["id"],
            "baseline": cfg.get("baseline"),
            "use_qcnn": cfg["use_qcnn"],
            "n_qubits": cfg.get("n_qubits", 16),
            "max_rows": n_rows,
            "epochs": 2 if n_rows <= 500 else 15, # Deep balanced training
            "batch_size": 32, # Fast local go: updates logs every few seconds
            "learning_rate": 0.003
        }
        
        try:
            res = train_and_predict(config, seed=seed)
            metrics = res["metrics"]
            results.append({
                "Model": cfg["id"],
                "Accuracy": metrics["accuracy"],
                "F1_Macro": metrics.get("f1-score", metrics.get("f1_macro", 0.0)),
                "ECE": metrics.get("ece", 0.0),
                "Params": res.get("param_count", 0)
            })
            logger.info(f"✅ {cfg['id']} Accuracy: {metrics['accuracy']:.4f}")
            
            # Progressive Checkpoint
            pd.DataFrame(results).to_csv(OUTPUT_DIR / f"results_N{n_rows}_latest.csv", index=False)
        except Exception as e:
            logger.error(f"❌ Failed to train {cfg['id']}: {e}")
            
    # Save Results
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(OUTPUT_DIR / f"results_N{n_rows}.csv", index=False)
        
        # Professional Elite-Tier Plots (v4.5)
        plot_elite_ranking(df, n_rows)
        plot_radar_metrics(df)
        
        logger.info(f"📊 Elite Ranking & Radar plots saved to {OUTPUT_DIR}")
    else:
        logger.warning("⚠️ No results to plot! All models failed to train.")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, choices=["fast", "full"], default="fast")
    args = parser.parse_args()
    
    # CI VOLUME CONTROL: If running in CI, use a tiny subset (20 rows) for sub-minute turnaround
    IS_CI = os.getenv("CI", "false").lower() == "true"
    if IS_CI:
        n_rows = 20
        logger.info("📡 CI MODE DETECTED: Setting benchmark size to 20 samples.")
    else:
        n_rows = 100 if args.size == "fast" else 2500
        
    run_benchmark(n_rows=n_rows)
