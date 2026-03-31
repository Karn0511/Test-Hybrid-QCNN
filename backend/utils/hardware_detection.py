from __future__ import annotations

from dataclasses import dataclass, asdict

import psutil
import torch


@dataclass(slots=True)
class HardwareProfile:
    device: str
    accelerator: str
    cpu_cores: int
    total_memory_gb: float
    cuda_available: bool
    cuda_device_count: int
    cuda_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
    return HardwareProfile(
        device='cuda' if cuda_available else 'cpu',
        accelerator='CUDA GPU' if cuda_available else 'CPU',
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        total_memory_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
        cuda_name=cuda_name,
    )
