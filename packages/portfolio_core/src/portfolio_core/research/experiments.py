from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_experiment_record(
    *,
    data_dir: Path,
    features: list[str],
    parameters: dict[str, object],
    training_window: dict[str, object],
    validation_folds: list[int],
    metrics: dict[str, object],
    notes: list[str] | None = None,
) -> Path:
    out_dir = data_dir / "gold" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    payload = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "features": features,
        "parameters": parameters,
        "training_window": training_window,
        "validation_folds": validation_folds,
        "metrics": metrics,
        "notes": notes or [],
    }
    path = out_dir / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
