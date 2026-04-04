import sys
import os
import json
import numpy as np
import time
import argparse
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

# [PULSE-PATH] Ensure backend package is discoverable
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.utils.logger import get_logger, configure_logging

# Lazy imports for expert training
def _lazy_train(lang, model_type, seed, config, data_payload):
    import torch
    from backend.models.standardized import build_model
    from backend.training.self_learner import ContinuousSelfLearner
    
    # RTX 3050 Optimized Settings
    config.update({"accumulation_steps": 8, "batch_size": 32, "id": model_type})
    
    estimator = build_model(config, seed=seed)
    
    # 1. Standard Training
    estimator.fit(
        data_payload["train"][0], data_payload["train"][1],
        x_val=data_payload["val"][0], y_val=data_payload["val"][1],
        languages=[lang]*len(data_payload["train"][1]),
        languages_val=[lang]*len(data_payload["val"][1]),
        texts=data_payload["train_texts"],
        texts_val=data_payload["val_texts"]
    )
    
    # 2. Nexus-V3 Self-Learning (Active Phase)
    if model_type == "qcnn_nexus":
        from backend.features.embedding import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        self_learner = ContinuousSelfLearner(estimator, embedder, config)
        logger.info(f"[NEXUS-V3 ACTIVATE]: Starting autonomous refinement for {lang}_{model_type}...")
        self_learner.auto_correct(
            data_payload["train"][0], data_payload["train"][1], 
            data_payload["train_texts"], 
            langs_pool=[lang]*len(data_payload["train"][1])
        )
    
    # 3. Persistent Save to Expert Repository
    expert_dir = Path("models/experts")
    expert_dir.mkdir(parents=True, exist_ok=True)
    expert_path = expert_dir / f"{lang}_{model_type}_s{seed}.pt"
    torch.save(estimator.model.state_dict(), expert_path)
    
    return expert_path

class UniversalOrchestrator:
    """
    Universal Training Sentinel (v1.0): Consolidates all research training workflows.
    Optimized for 32-core Xeon & RTX 3050.
    """
    def __init__(self):
        configure_logging("INFO")
        self.logger = get_logger("UniversalOrchestrator")
        self.base_dir = Path("runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"sentinel_{datetime.now().strftime('%m%d_%H%M')}"
        # v4.4 Neural Pulse: Instant Mirroring
        self.pulse_dir = Path("evaluation/latest/pulses")
        self.pulse_dir.mkdir(parents=True, exist_ok=True)
        self.pulse_path = Path("evaluation/latest/status_pulse.json")
        self.pulse_path.parent.mkdir(parents=True, exist_ok=True)

    def run_matrix(self, languages, models, seeds, global_config):
        """Sequential Language Priority Matrix."""
        from backend.data.loader import load_fixed_split
        from backend.features.embedding import EmbeddingPipeline
        
        # v4.6 Master Pulse: Absolute Clean-Slate at matrix ignition
        pulse_dir = Path("evaluation/latest/pulses")
        if not pulse_dir.exists(): pulse_dir.mkdir(parents=True, exist_ok=True)
        for f in pulse_dir.glob("*.json"): 
            try: os.remove(f)
            except Exception: pass

        embedder = EmbeddingPipeline()
        results = []
        
        for lang in languages:
            self.logger.info(f"[PHANTOM-START]: Initializing {lang} manifold...")
            
            # Load Splits (Abolish Merge)
            max_s = global_config.get("max_samples", 5000)
            train_df = load_fixed_split(lang, "train", max_samples=max_s)
            val_df = load_fixed_split(lang, "val", max_samples=max_s // 5)
            
            # 1. Quicksilver Cache Check (Bypass BERT if possible)
            import hashlib
            data_hash = hashlib.md5(f"{lang}_{len(train_df)}".encode()).hexdigest()
            val_hash = hashlib.md5(f"{lang}_val_{len(val_df)}".encode()).hexdigest()

            x_train = embedder.fit_transform(train_df["text"].tolist(), dataset_hash=data_hash)
            y_train = train_df["label"].to_numpy()
            
            x_val = embedder.fit_transform(val_df["text"].tolist(), dataset_hash=val_hash)
            y_val = val_df["label"].to_numpy()
            
            data_payload = {
                "train": (x_train, y_train),
                "val": (x_val, y_val),
                "train_texts": train_df["text"].tolist(),
                "val_texts": val_df["text"].tolist()
            }
            
            # Seed Parallelization (Expert Isolation)
            
            for model in models:
                m_id = model.upper()
                for seed in seeds:
                    p_id = f"{lang}_{m_id}_s{seed}.json"
                    with open(pulse_dir / p_id, "w") as f:
                        json.dump({"epoch": 0, "batch": 0, "total_batches": 1, "status": "Initializing...", "timestamp": time.time()}, f)

            self.logger.info(f"[SEED-LAUNCH]: Distributing kernels for {lang}...")
            
            # v35.9 Xeon-Unbound: Launch 12 concurrent kernels to ensure zero queuing for the 10-expert matrix
            lang_results = Parallel(n_jobs=12)(
                delayed(_lazy_train)(lang, model, seed, {**global_config, "seed": seed}, data_payload)
                for model in models
                for seed in seeds
            )
            results.extend(lang_results)
            self.logger.info(f"[COMPLETED]: {lang} Expert Series locked.")
            
        return results

    def train_fusion(self, languages, config):
        """Stage 2: Decision Fusion Soft-Voting Training."""
        self.logger.info("[FUSION-PHASE]: Initializing Meta-Attention Decision Fusion...")
        import torch
        from backend.models.standardized import build_model
        from backend.data.loader import load_fixed_split
        from backend.features.embedding import EmbeddingPipeline
        
        # 1. Initialize Fusion Model
        config["use_fusion"] = True
        fusion_estimator = build_model(config)
        embedder = EmbeddingPipeline()
        
        # 2. Load Experts into Streams
        expert_dir = Path("models/experts")
        for lang in languages:
            # Load the best available seed for this language (e.g., s42)
            expert_path = expert_dir / f"{lang}_qcnn_nexus_s42.pt"
            if not expert_path.exists():
                expert_path = expert_dir / f"{lang}_qcnn_s42.pt"
            
            if expert_path.exists():
                self.logger.info(f"[LINKINFO]: Loading expert from {expert_path}")
                fusion_estimator.model.experts[lang].load_state_dict(torch.load(expert_path))
                # Freeze Experts
                for param in fusion_estimator.model.experts[lang].parameters():
                    param.requires_grad = False
            else:
                self.logger.warning(f"[LINK-FAIL]: No expert found for {lang}. Skipping...")

        # 3. Train Meta-Attention on Interleaved Data
        # Collect small validation samples from each language
        val_x, val_y, val_langs = [], [], []
        for lang in languages:
            df = load_fixed_split(lang, "val", max_samples=500)
            x = embedder.fit_transform(df["text"].tolist())
            val_x.append(x)
            val_y.append(df["label"].to_numpy())
            val_langs.extend([lang] * len(df))
            
        val_x = np.concatenate(val_x)
        val_y = np.concatenate(val_y)
        
        self.logger.info(f"[FUSION-REFINE]: Training meta-voter on {len(val_y)} interleaved samples.")
        fusion_estimator.fit(val_x, val_y, languages=val_langs)
        
        torch.save(fusion_estimator.model.state_dict(), "models/SENTINEL_FUSION_SOTA.pt")
        self.logger.info("[SUPREMACY]: Fusion model locked at models/SENTINEL_FUSION_SOTA.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Training Sentinel")
    parser.add_argument("--mode", choices=["expert", "fusion", "matrix"], default="matrix")
    parser.add_argument("--lang", type=str, help="Language for isolated expert training")
    parser.add_argument("--nexus", action="store_true", help="Enable Nexus-V3 Self-Learning")
    args = parser.parse_args()
    
    orchestrator = UniversalOrchestrator()
    
    config = {
        "epochs": 20,
        "lr": 1e-3,
        "use_qcnn": True,
        "n_qubits": 12,
        "n_layers": 4,
        "max_samples": 10000
    }
    
    languages = ["hindi", "english", "bhojpuri", "maithili"]
    seeds = [42, 43, 44, 45, 46]
    
    if args.mode == "expert":
        target_langs = [args.lang] if args.lang else languages
        models = ["qcnn_nexus" if args.nexus else "qcnn"]
        
        # [PULSE-INIT]: Broadcast metadata for the Sentinel Watchdog
        pulse_data = {
            "run_id": orchestrator.run_id,
            "start_time": datetime.now().isoformat(),
            "status": "EXPERT_MATRIX_ACTIVE",
            "languages": target_langs,
            "models": models,
            "config": config
        }
        with open(orchestrator.pulse_path, "w") as f:
            json.dump(pulse_data, f, indent=4)
            
        orchestrator.run_matrix(target_langs, models, seeds, config)
        
    elif args.mode == "fusion":
        orchestrator.train_fusion(languages, config)
        
    elif args.mode == "matrix":
        # Full Sequence: All Experts (Both Modes) -> Fusion
        pulse_data = {
            "run_id": orchestrator.run_id,
            "start_time": datetime.now().isoformat(),
            "status": "GLOBAL_MATRIX_ACTIVE",
            "languages": languages,
            "models": ["qcnn", "qcnn_nexus"],
            "config": config
        }
        with open(orchestrator.pulse_path, "w") as f:
            json.dump(pulse_data, f, indent=4)

        if args.nexus:
            seeds = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
            orchestrator.run_matrix(languages, ["qcnn_nexus"], seeds, config)
        else:
            orchestrator.run_matrix(languages, ["qcnn", "qcnn_nexus"], seeds, config)
        orchestrator.train_fusion(languages, config)
        
        # Dashboard Synthesis
        try:
            import subprocess
            subprocess.run([sys.executable, "scripts/generate_v31_global_dashboard.py"])
        except:
            pass
