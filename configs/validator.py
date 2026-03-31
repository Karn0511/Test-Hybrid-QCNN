import yaml
from pathlib import Path
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def validate_config(config: dict) -> bool:
    """
    Enforces scientific consistency constraints on a model configuration.
    Addition H: Enforce valid parameter ranges and QCNN meaningful differences.
    """
    required_keys = ["id", "use_qcnn"]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: '{key}'")
            return False
            
    # Check for hybrid consistency
    use_qcnn = config.get("use_qcnn", False)
    if use_qcnn:
        layers = config.get("layers", 0)
        n_qubits = config.get("n_qubits", 0)
        if layers <= 0:
            logger.error(f"Config '{config['id']}': use_qcnn is True but layers ({layers}) <= 0.")
            return False
        if n_qubits <= 0:
            logger.error(f"Config '{config['id']}': use_qcnn is True but n_qubits ({n_qubits}) <= 0.")
            return False
            
    # Check parameter ranges
    epochs = config.get("epochs", 1)
    if epochs < 1:
        logger.error(f"Config '{config['id']}': epochs ({epochs}) must be >= 1.")
        return False
        
    lr = config.get("learning_rate", 0.001)
    if lr <= 0 or lr > 1.0:
        logger.error(f"Config '{config['id']}': invalid learning_rate ({lr}).")
        return False
        
    return True

def validate_all_configs(config_dir: Path) -> dict:
    """Scan all YAML configs and check for duplicates or sanity failures."""
    configs = []
    seen_ids = set()
    failed = []
    
    for path in config_dir.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            try:
                payload = yaml.safe_load(f)
                if not payload:
                    failed.append((path.name, "Empty file"))
                    continue
                    
                cfg_id = payload.get("id")
                if cfg_id in seen_ids:
                    failed.append((path.name, f"Duplicate ID: {cfg_id}"))
                seen_ids.add(cfg_id)
                
                if not validate_config(payload):
                    failed.append((path.name, "Validation failed (check logs)"))
                else:
                    configs.append(payload)
                    
            except Exception as e:
                failed.append((path.name, f"Unmarshal error: {str(e)}"))
                
    return {
        "valid_count": len(configs),
        "failed": failed,
        "configs": configs
    }

if __name__ == "__main__":
    res = validate_all_configs(Path("configs"))
    print(f"Validated {res['valid_count']} configs.")
    if res["failed"]:
        print(f"FAILURES: {res['failed']}")
