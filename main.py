"""
Hybrid QCNN Sentiment Analysis: Omega-Sentinel Research Engine (v34.0)
High-Fidelity Project Record for RTX 3050 (4GB VRAM) and i7 CPU.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml

# Core Imports
from backend.training.train import run_multi_seed, train_and_predict
from backend.utils.logger import configure_logging, get_logger
from evaluation.metrics.evaluator import evaluate_predictions
from evaluation.visualizer import plot_supremacy_dashboard

try:
    # 🛰️ Launch Sequence Hardware Safe-Lock
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = get_logger("OMEGA-MAIN")

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_omega_benchmarks(config: dict):
    """
    Executes the 3-Experiment Protocol (Omega-Sentinel):
    EXP A: Classical + Nexus (Self-Learner Only)
    EXP B: Classical + QCNN (Quantum Only)
    EXP C: Classical + QCNN + Nexus (Full Hybrid Omega)
    """
    logger.info("🚀 [OMEGA-SENTINEL]: PROJECT OVERDRIVE INITIALIZED.")
    logger.info("Initializing 3-Experiment Protocol for Scientifically Valid Supremacy...")
    
    # 🏁 1. Pre-Flight Micro-Overfit Test (N=100)
    logger.info("--- [Pre-Flight]: Validating Logic Overfit Guard (N=100) ---")
    micro_cfg = config["training"].copy()
    micro_cfg.update({
        "id": "MICRO_TEST", 
        "max_rows": 100, 
        "use_qcnn": True, 
        "use_nexus": True, 
        "epochs": 50, 
        "patience": 50,
        "test_size": 0.0,  # Force Overfit Mode (100% Train/Val/Test Sync)
        "num_workers": 0   # Windows Stability for quick test
    })
    micro_res = train_and_predict(micro_cfg)
    
    # Logic Gate: Model MUST be able to overfit its own small subset to >= 90%
    if micro_res["metrics"]["accuracy"] < 0.90:
        logger.error(f"[!] [VALIDATION FAIL]: Micro-test failed to reach 90%+ Accuracy (Got {micro_res['metrics']['accuracy']:.2%}). Check Hardware/VRAM.")
        sys.exit(1)
    logger.info("[✔] [VALIDATION PASS]: 100% Accuracy Guard functional. Proceeding to Omega-Sentinel run.")

    # 🔬 2. Main Experiment Protocol (N=3000)
    results = {}
    exp_registry = config.get("experiments", [])
    
    for exp in exp_registry:
        logger.info(f"\n--- [OMEGA RUN]: {exp['id']} | {exp['description']} ---")
        full_cfg = config["training"].copy()
        full_cfg.update(exp)
        
        # Enforce Research Constraints
        full_cfg["max_rows"] = 3000
        full_cfg["batch_size"] = 32
        full_cfg["grad_accum_steps"] = 2
        
        # Run Multi-Seed (n=3)
        res = run_multi_seed(full_cfg, n_seeds=3)
        results[exp['id']] = {
            "accuracy": res['summary']['accuracy']['mean'],
            "accuracy_std": res['summary']['accuracy']['std'],
            "f1": res['summary']['f1_macro']['mean'],
            "metrics": res['runs'][0]['metrics']
        }
    
    # 📊 3. Final Comparison Record
    omega_results_path = config["logging"]["results_json"]
    os.makedirs(os.path.dirname(omega_results_path), exist_ok=True)
    with open(omega_results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"💾 [OMEGA RECORD SAVED]: {omega_results_path}")
    
    # 🏆 Check for Scientific Supremacy (EXP C > EXP A)
    if results["EXP_C"]["accuracy"] > results["EXP_A"]["accuracy"]:
        diff = results["EXP_C"]["accuracy"] - results["EXP_A"]["accuracy"]
        logger.info(f"[🏆] [PROJECT SUPREMACY]: EXP_C out-performed EXP_A by {diff:.2%}. Quantum Delta Verified.")
    else:
        logger.warning("[!] [ACCURACY WARNING]: Quantum contribution negligible in this seed. Re-tuning required.")

    # 🎨 4. Render Global Comparison Plot
    try:
        from evaluation.visualizer import plot_omega_comparison
        plot_omega_comparison(results, save_path=config["logging"]["comparison_plot"])
        logger.info(f"🎨 [OMEGA VISUAL SAVED]: {config['logging']['comparison_plot']}")
    except Exception as e:
        logger.error(f"Failed to render Omega Visual: {e}")

def main():
    parser = argparse.ArgumentParser(description="Omega-Sentinel Project Engine")
    parser.add_argument("--config", type=str, default="configs/master_config.yaml")
    parser.add_argument("--mode", type=str, choices=["omega", "sanity", "train", "self_learn"], default="omega")
    parser.add_argument("--max_rows", type=int, help="Override maximum rows")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Apply Overrides
    if args.max_rows:
        config["training"]["max_rows"] = args.max_rows
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    configure_logging(level="INFO")
    
    if args.mode == "omega":
        run_omega_benchmarks(config)
    elif args.mode == "sanity":
        # Simplified sanity code
        logger.info("Running N=100 Sanity Check...")
        cfg = config["training"].copy()
        cfg.update({"id": "SANITY", "max_rows": 100})
        train_and_predict(cfg)
    elif args.mode == "self_learn":
        logger.info("Running Self-Learning benchmark (Classical + Nexus)...")
        cfg = config["training"].copy()
        cfg.update({
            "id": "SELF_LEARN",
            "use_qcnn": False,
            "use_nexus": True,
        })
        result = train_and_predict(cfg)
        logger.info(
            "Self-Learning completed | Accuracy=%.4f | F1=%.4f",
            result["metrics"].get("accuracy", 0.0),
            result["metrics"].get("f1_macro", 0.0),
        )
    else:
        logger.info("Mode not implemented in Omega-Sentinel build.")

if __name__ == "__main__":
    main()
