"""Small config loader for the repository's simple YAML manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the limited YAML subset used by dataset manifests.

    The parser supports nested mappings by indentation and scalar values. It is
    intentionally small so tests and scripts do not require PyYAML.
    """

    config_path = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(
                f"{config_path}:{line_number} uses odd indentation"
            )
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"{config_path}:{line_number} expected key: value")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{config_path}:{line_number} empty key")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root
