"""Environment provenance for reproducible runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_environment(*, root: str | Path = ".") -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"], root=root),
        "git_branch": _git_value(["git", "branch", "--show-current"], root=root),
    }


def _git_value(command: list[str], *, root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None
