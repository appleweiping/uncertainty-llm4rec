"""Small YAML config loader and resolved-config writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = _load_yaml_subset(config_path)
    data["_config_path"] = str(config_path)
    return data


def save_resolved_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(config), encoding="utf-8")


def dump_yaml(value: Any, *, indent: int = 0) -> str:
    lines: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            item = value[key]
            prefix = " " * indent + f"{key}:"
            if isinstance(item, dict):
                lines.append(prefix)
                lines.append(dump_yaml(item, indent=indent + 2).rstrip("\n"))
            elif isinstance(item, list):
                lines.append(prefix + " " + _format_scalar(item))
            else:
                lines.append(prefix + " " + _format_scalar(item))
    else:
        lines.append(" " * indent + _format_scalar(value))
    return "\n".join(lines) + "\n"


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"{path}:{line_number} uses odd indentation")
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number} expected key: value")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value.strip())
    return root


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
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
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


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    text = str(value)
    if text == "" or any(char in text for char in ":#[]{}") or text.strip() != text:
        return repr(text)
    return text
