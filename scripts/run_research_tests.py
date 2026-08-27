from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from portfolio_core.research.verification import RESEARCH_TEST_PATHS, source_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered research tests and write a gate manifest")
    parser.add_argument("--manifest", default="data/gold/test_manifests/research_tests.json")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    command = ["pytest", "-q", *RESEARCH_TEST_PATHS]
    completed = subprocess.run(command, cwd=repository_root, check=False)
    manifest_path = repository_root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "test_paths": list(RESEARCH_TEST_PATHS),
        "source_fingerprint": source_fingerprint(repository_root),
        "command": command,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
