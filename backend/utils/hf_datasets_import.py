from __future__ import annotations

import importlib
import sys
from pathlib import Path


def import_hf_datasets():
    """
    Import HuggingFace `datasets` even when a local `datasets/` folder exists.
    """
    project_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)

    def _is_project_root_path(path_entry: str) -> bool:
        try:
            resolved = Path(path_entry or ".").resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        return resolved == project_root

    try:
        sys.path = [entry for entry in original_sys_path if not _is_project_root_path(entry)]
        return importlib.import_module("datasets")
    finally:
        sys.path = original_sys_path
