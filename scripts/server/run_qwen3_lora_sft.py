"""Prepare Qwen3-8B LoRA/SFT training plan.

Default behavior is plan-only. Local Codex must not start heavy training.
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

from storyflow.training import run_qwen_lora_training  # noqa: E402


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server/qwen3_8b_lora_sft.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--execute-server", action="store_true")
    args = parser.parse_args(argv)

    manifest = run_qwen_lora_training(
        config_path=_resolve(args.config),
        output_dir=_resolve(args.output_dir) if args.output_dir else None,
        execute_server=args.execute_server,
        root=ROOT,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
