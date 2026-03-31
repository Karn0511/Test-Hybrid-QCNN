import torch
import torch.nn as nn
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def audit_gradient_flow(model: nn.Module) -> bool:
    """
    Addition: Checks gradients for all parameters after a backward pass.
    Ensures all layers (including QCNN) are actually training.
    """
    logger.info("--- Gradient Flow Audit ---")
    dead_layers = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            if param.grad is None:
                logger.warning(f"NO GRAD: {name}")
                dead_layers.append(name)
            elif torch.all(param.grad == 0):
                logger.warning(f"ZERO GRAD: {name}")
                dead_layers.append(name)
    
    if dead_layers:
        logger.error(f"Audit Failed: {len(dead_layers)}/{total_params} layers are dead.")
        return False
        
    logger.info("Audit Passed: All trainable layers have active gradients.")
    return True

def monitor_gradient_norms(model: nn.Module) -> dict:
    """Returns a dict of gradient norms per layer."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms
