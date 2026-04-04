from __future__ import annotations
import os
import logging
from rich.logging import RichHandler

def configure_logging(level: str = 'INFO') -> None:
    """
    v36.0 Strategic Logger: Auto-switches between Cinematic UI and Stream-Ready modes.
    Ensures zero-latency log-handshake with Sentinel Watchdog.
    """
    if os.environ.get("PYTHONUNBUFFERED") == "1":
        # Sentinel Handshake: Use plain stream for rapid IPC parsing
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
    else:
        # Standard Lab Work: Use Rich high-fidelity formatting
        handler = RichHandler(rich_tracebacks=True, markup=True)
        
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler]
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
