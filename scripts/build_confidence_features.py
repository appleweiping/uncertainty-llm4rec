"""Build CURE/TRUCE feature JSONL from grounded observation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.confidence import build_confidence_features  # noqa: E402


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _default_output_dir(grounded_jsonl: Path) -> Path:
    try:
        relative_parent = grounded_jsonl.resolve().parent.relative_to(ROOT / "outputs")
        return ROOT / "outputs" / "confidence_features" / relative_parent
    except ValueError:
        return ROOT / "outputs" / "confidence_features" / grounded_jsonl.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-jsonl", required=True)
    parser.add_argument("--input-jsonl")
    parser.add_argument("--catalog-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--manifest-json")
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args(argv)

    grounded_jsonl = _resolve(args.grounded_jsonl)
    if grounded_jsonl is None or not grounded_jsonl.exists():
        raise SystemExit(f"Grounded predictions not found: {grounded_jsonl}")
    input_jsonl = _resolve(args.input_jsonl)
    if input_jsonl is not None and not input_jsonl.exists():
        raise SystemExit(f"Observation input JSONL not found: {input_jsonl}")
    catalog_csv = _resolve(args.catalog_csv)
    if catalog_csv is not None and not catalog_csv.exists():
        raise SystemExit(f"Catalog CSV not found: {catalog_csv}")

    output_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir(grounded_jsonl)
    assert output_dir is not None
    output_jsonl = _resolve(args.output_jsonl) if args.output_jsonl else output_dir / "features.jsonl"
    manifest_json = (
        _resolve(args.manifest_json) if args.manifest_json else output_dir / "manifest.json"
    )
    assert output_jsonl is not None
    assert manifest_json is not None

    manifest = build_confidence_features(
        grounded_jsonl=grounded_jsonl,
        input_jsonl=input_jsonl,
        catalog_csv=catalog_csv,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        max_examples=args.max_examples,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
