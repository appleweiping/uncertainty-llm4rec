"""Check DeepSeek/API pilot gates without making a network call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.providers import check_api_pilot_readiness  # noqa: E402


def _resolve(path: str | Path) -> Path:
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-config", default="configs/providers/deepseek.yaml")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--stage", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--approved-provider")
    parser.add_argument("--approved-model")
    parser.add_argument("--approved-rate-limit", type=int)
    parser.add_argument("--approved-budget-label")
    parser.add_argument("--execute-api-intended", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    manifest = check_api_pilot_readiness(
        provider_config_path=_resolve(args.provider_config),
        input_jsonl=_resolve(args.input_jsonl),
        sample_size=args.sample_size,
        stage=args.stage,
        approved_provider=args.approved_provider,
        approved_model=args.approved_model,
        approved_rate_limit=args.approved_rate_limit,
        approved_budget_label=args.approved_budget_label,
        execute_api_intended=args.execute_api_intended,
    )
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "api_readiness" / str(manifest["provider"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "readiness_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "ready_for_execute_api" else 2


if __name__ == "__main__":
    raise SystemExit(main())
