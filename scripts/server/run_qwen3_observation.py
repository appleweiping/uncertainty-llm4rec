"""Prepare or run Qwen3 server observation.

Default behavior is plan-only. Actual inference requires --execute-server and
is intended for approved server hardware, not local Codex runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.server import (  # noqa: E402
    default_qwen_server_output_dir,
    load_qwen_server_config,
    run_qwen_server_observation,
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server/qwen3_8b_observation.yaml")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--run-label")
    parser.add_argument("--run-stage", default="server")
    parser.add_argument("--execute-server", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)

    config_path = _resolve(args.config)
    input_jsonl = _resolve(args.input_jsonl)
    if not input_jsonl.exists():
        raise SystemExit(f"Observation input JSONL not found: {input_jsonl}")

    output_dir = _resolve(args.output_dir) if args.output_dir else None
    if output_dir is None:
        config = load_qwen_server_config(config_path)
        output_dir = default_qwen_server_output_dir(
            input_jsonl=input_jsonl,
            model_alias=str(config["model_alias"]),
            root=ROOT,
        )

    manifest = run_qwen_server_observation(
        config_path=config_path,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        max_examples=args.max_examples,
        execute_server=args.execute_server,
        resume=args.resume,
        run_label=args.run_label,
        run_stage=args.run_stage,
        root=ROOT,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
