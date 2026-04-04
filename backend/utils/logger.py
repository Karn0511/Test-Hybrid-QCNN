import os
import logging
from rich.logging import RichHandler

def configure_logging(level: str = 'INFO') -> None:
    # v4.1 Master Sync: Use plain stream when piped to Watchdog (UNBUFFERED)
    if os.environ.get("PYTHONUNBUFFERED") == "1":
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
    else:
        handler = RichHandler(rich_tracebacks=True, markup=True)
        
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler]
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
