from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl as load_jsonl_rows
from src.utils.io import save_jsonl as save_jsonl_rows


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_yaml_without_pyyaml(path: str | Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any, str | None]] = [(-1, root, None)]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        idx += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if isinstance(parent, list):
                parent.append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped or not isinstance(parent, dict):
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            next_is_list = False
            lookahead = idx
            while lookahead < len(lines):
                nxt = lines[lookahead]
                if not nxt.strip() or nxt.lstrip().startswith("#"):
                    lookahead += 1
                    continue
                next_indent = len(nxt) - len(nxt.lstrip(" "))
                next_is_list = next_indent > indent and nxt.strip().startswith("- ")
                break
            child: Any = [] if next_is_list else {}
            parent[key] = child
            stack.append((indent, child, key))
        elif value == ">":
            collected: list[str] = []
            while idx < len(lines):
                follow = lines[idx]
                follow_indent = len(follow) - len(follow.lstrip(" "))
                if follow.strip() and follow_indent <= indent:
                    break
                if follow.strip():
                    collected.append(follow.strip())
                idx += 1
            parent[key] = " ".join(collected)
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return _load_yaml_without_pyyaml(path)


class FunctionPromptBuilderAdapter:
    def __init__(self, fn, template_path: str | Path):
        self.fn = fn
        self.template_path = str(template_path)

    def build_pointwise_prompt(self, sample: dict[str, Any], candidate: dict[str, Any]) -> str:
        try:
            return self.fn(sample, candidate, template_path=self.template_path)
        except TypeError:
            return self.fn(sample, candidate)


def get_prompt_builder(prompt_path: str | Path):
    try:
        from src.llm.prompt_builder import PromptBuilder

        return PromptBuilder(template_path=str(prompt_path))
    except Exception:
        try:
            from src.llm.prompt_builder import build_pointwise_prompt

            return FunctionPromptBuilderAdapter(build_pointwise_prompt, template_path=prompt_path)
        except Exception as exc:
            raise ImportError("Cannot find a usable prompt builder in src/llm/prompt_builder.py") from exc


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(path)
    if max_samples is not None and max_samples > 0:
        return rows[:max_samples]
    return rows


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    save_jsonl_rows(records, path)
