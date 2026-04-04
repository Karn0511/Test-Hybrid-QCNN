from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def export_latest_summary(summary: dict[str, Any], output_path: Path | str = "evaluation/latest_summary.json") -> Path:
    """
    Persist a canonical run summary to a single contract file.
    Uses atomic replacement to avoid partial writes.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": "1.0",
        **summary,
    }

    tmp_path = target.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    tmp_path.replace(target)
    return target
