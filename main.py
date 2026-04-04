"""
Hybrid QCNN Sentiment Analysis: Omega-Sentinel Research Engine (v36.1)
Unified Master Entry Point for 32-Core Xeon Workstations.
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
from backend.training.orchestrator import UniversalOrchestrator

try:
    # 🛰️ Launch Sequence Hardware Safe-Lock
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = get_logger("OMEGA-MAIN")

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_expert_mode(args, config: dict):
    """
    Unified Expert Training Interface (v36.1 Elite).
    Targeted training for specific languages with optional Nexus-V3 Self-Learning.
    """
    orchestrator = UniversalOrchestrator()
    target_langs = [args.lang] if args.lang else ["english", "hindi", "bhojpuri", "maithili"]
    models = ["qcnn_nexus" if args.nexus else "qcnn"]
    seeds = [42, 43, 44, 45, 46] # PhD-Standard Seed Matrix
    
    logger.info(f"🚀 [EXPERT-LAUNCH]: Target={target_langs} | Nexus={args.nexus} | 12-Job Parallel Matrix Active.")
    
    # Broadcast Metadata for Sentinel Watchdog
    pulse_data = {
        "run_id": orchestrator.run_id,
        "start_time": datetime.now().isoformat(),
        "status": "EXPERT_MATRIX_ACTIVE",
        "languages": target_langs,
        "models": models,
        "config": config["training"]
    }
    with open(orchestrator.pulse_path, "w") as f:
        json.dump(pulse_data, f, indent=4)
        
    orchestrator.run_matrix(target_langs, models, seeds, config["training"])
    logger.info("🏁 [OMEGA-COMPLETE]: Expert series locked in models/experts/")

def main():
    parser = argparse.ArgumentParser(description="Omega-Sentinel Unified Project Engine")
    parser.add_argument("--config", type=str, default="configs/master_config.yaml")
    parser.add_argument("--mode", type=str, choices=["omega", "expert", "matrix", "sanity"], default="omega")
    parser.add_argument("--lang", type=str, help="Research Target (hindi|english|bhojpuri|maithili)")
    parser.add_argument("--nexus", action="store_true", help="Enable Nexus-V3 Autonomous Refinement")
    parser.add_argument("--max_rows", type=int, help="Override maximum rows")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Master Override Synchronization
    if args.max_rows: config["training"]["max_rows"] = args.max_rows
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    
    configure_logging(level="INFO")
    
    if args.mode == "expert":
        run_expert_mode(args, config)
    elif args.mode == "matrix":
        orchestrator = UniversalOrchestrator()
        logger.info("🚀 [GLOBAL-MATRIX]: Initializing 12-job zero-queuing parallel suite.")
        orchestrator.run_matrix(["english", "hindi", "bhojpuri", "maithili"], ["qcnn", "qcnn_nexus"], [42, 43, 44, 45, 46], config["training"])
    elif args.mode == "omega":
        from backend.training.train import run_multi_seed
        logger.info("🚀 [LEGACY-OMEGA]: Initializing 3-Experiment Protocol (N=3000).")
        pass # Placeholder for research compatibility
    elif args.mode == "sanity":
        logger.info("Running N=100 Sanity Check...")
        cfg = config["training"].copy()
        cfg.update({"id": "SANITY", "max_rows": 100})
        train_and_predict(cfg)
    else:
        logger.info("Mode not implemented. Use --mode expert for targeted language training.")

if __name__ == "__main__":
    main()
