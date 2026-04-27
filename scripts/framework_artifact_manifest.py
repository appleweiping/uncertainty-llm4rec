from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = [
    Path("data_done"),
    Path("data_done_lora"),
    Path("output-repaired/framework"),
    Path("artifacts/lora"),
    Path("configs/framework"),
    Path("prompts/framework"),
]

JSON_OUT = Path("data_done/framework_artifact_manifest_server.json")
MD_OUT = Path("data_done/framework_artifact_manifest_server.md")
SHA_LIMIT = 50 * 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _category(path: Path, size: int) -> str:
    s = path.as_posix()
    if s.startswith("artifacts/lora"):
        return "adapter_artifact"
    if path.suffix == ".jsonl":
        if "output-repaired/framework" in s:
            return "prediction_jsonl"
        return "data_jsonl"
    if size >= SHA_LIMIT:
        return "large_data"
    if path.suffix in {".py", ".yaml", ".yml", ".md", ".csv", ".json", ".txt"}:
        return "code_config_report"
    return "ignored"


def _policy(category: str, path: Path) -> dict[str, Any]:
    s = path.as_posix()
    should_commit = category == "code_config_report"
    if path.suffix == ".json" and "schema_validation" in s:
        should_commit = True
    if "framework_artifact_manifest_server" in s:
        should_commit = True
    should_sync = category in {"prediction_jsonl", "data_jsonl", "large_data"} or category == "adapter_artifact"
    never_commit = category in {"prediction_jsonl", "data_jsonl", "large_data", "adapter_artifact"}
    notes = ""
    if category == "adapter_artifact":
        notes = "Record metadata only; do not commit adapter weights/checkpoints."
    elif category == "prediction_jsonl":
        notes = "Prediction artifact; sync only when local audit needs it."
    elif category == "data_jsonl":
        notes = "Generated training/eval data; do not commit large JSONL."
    elif category == "code_config_report":
        notes = "Lightweight code/config/report candidate for Git."
    return {
        "should_commit_to_git": should_commit and not never_commit,
        "should_sync_to_local": should_sync,
        "should_never_commit": never_commit,
        "notes": notes,
    }


def build_manifest(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                if path.as_posix().startswith("artifacts/lora"):
                    try:
                        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                    except OSError:
                        mtime = ""
                    rows.append(
                        {
                            "path": path.as_posix(),
                            "size_bytes": 0,
                            "mtime": mtime,
                            "sha256": "",
                            "category": "adapter_artifact",
                            **_policy("adapter_artifact", path),
                        }
                    )
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            category = _category(path, st.st_size)
            sha = _sha256(path) if st.st_size < SHA_LIMIT and category != "adapter_artifact" else ""
            rows.append(
                {
                    "path": path.as_posix(),
                    "size_bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "sha256": sha,
                    "category": category,
                    **_policy(category, path),
                }
            )
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    lines = [
        "# Framework Artifact Manifest",
        "",
        "This manifest records framework-stage files visible from the current machine. Server runs should regenerate it after each experiment.",
        "",
        "## Category Counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: `{counts[key]}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Commit code/config/prompt/lightweight reports only.",
            "- Do not commit `data_done*.jsonl`, prediction JSONL, adapter weights, checkpoints, or large archives.",
            "- Sync prediction JSONL only when a local audit explicitly needs it.",
            "",
            "## Files",
            "",
            "| path | category | size_bytes | commit | sync | never_commit |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['path']}` | `{row['category']}` | {row['size_bytes']} | "
            f"`{row['should_commit_to_git']}` | `{row['should_sync_to_local']}` | `{row['should_never_commit']}` |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework artifact manifest.")
    parser.add_argument("--roots", nargs="*", default=[str(p) for p in DEFAULT_ROOTS])
    args = parser.parse_args()
    rows = build_manifest([Path(x) for x in args.roots])
    write_outputs(rows)
    print(json.dumps({"files": len(rows), "json": str(JSON_OUT), "md": str(MD_OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
