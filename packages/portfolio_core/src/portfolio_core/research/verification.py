from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESEARCH_TEST_PATHS = (
    "packages/portfolio_core/tests",
    "services/api/tests",
)
FINGERPRINT_PATHS = (
    "packages/portfolio_core/src",
    "packages/portfolio_core/tests",
    "services/api/app",
    "services/api/tests",
    "scripts",
)


def source_fingerprint(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_root in FINGERPRINT_PATHS:
        root = repository_root / relative_root
        if not root.exists():
            continue
        for path in sorted(
            item for item in root.rglob("*.py") if item.is_file() and "__pycache__" not in item.parts
        ):
            digest.update(path.relative_to(repository_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def verify_test_manifest(path: Path, *, repository_root: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing test manifest: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid test manifest: {exc}"
    if payload.get("status") != "passed" or int(payload.get("exit_code", -1)) != 0:
        return False, "test manifest does not record a passing run"
    if tuple(payload.get("test_paths", ())) != RESEARCH_TEST_PATHS:
        return False, "test manifest does not cover the registered suites"
    if payload.get("source_fingerprint") != source_fingerprint(repository_root):
        return False, "test manifest is stale for the current source tree"
    return True, "verified passing manifest for current source tree"
