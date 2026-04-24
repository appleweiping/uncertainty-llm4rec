from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.data.noise import apply_noise_to_sample


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_noisy_rows(
    rows: list[dict[str, Any]],
    *,
    history_drop_prob: float,
    text_noise_prob: float,
    label_flip_prob: float,
) -> list[dict[str, Any]]:
    return [
        apply_noise_to_sample(
            row,
            history_drop_prob=history_drop_prob,
            text_noise_prob=text_noise_prob,
            label_flip_prob=label_flip_prob,
        )
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean/noisy valid-test splits for evidence-posterior robustness."
    )
    parser.add_argument(
        "--domain_root",
        type=str,
        default="data/processed/amazon_beauty",
        help="Directory containing clean split JSONL files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["valid", "test"],
        help="Split names to perturb, e.g. valid test.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional output directory. Defaults to --domain_root.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="evidence_noisy_nl10",
        help="Suffix used in output filenames: {split}_{suffix}.jsonl.",
    )
    parser.add_argument("--history_drop_prob", type=float, default=0.1)
    parser.add_argument("--text_noise_prob", type=float, default=0.1)
    parser.add_argument("--label_flip_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional smoke-test cap per split. Omit for full split generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_root = Path(args.domain_root)
    output_root = Path(args.output_root) if args.output_root is not None else domain_root
    output_root.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "domain_root": str(domain_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "output_suffix": args.output_suffix,
        "history_drop_prob": args.history_drop_prob,
        "text_noise_prob": args.text_noise_prob,
        "label_flip_prob": args.label_flip_prob,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "outputs": [],
    }

    for split_idx, split in enumerate(args.splits):
        input_path = domain_root / f"{split}.jsonl"
        output_path = output_root / f"{split}_{args.output_suffix}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Clean split not found: {input_path}")

        random.seed(args.seed + split_idx)
        rows = load_jsonl(input_path, max_samples=args.max_samples)
        noisy_rows = build_noisy_rows(
            rows,
            history_drop_prob=args.history_drop_prob,
            text_noise_prob=args.text_noise_prob,
            label_flip_prob=args.label_flip_prob,
        )
        save_jsonl(noisy_rows, output_path)

        split_metadata = {
            "split": split,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "num_samples": len(noisy_rows),
        }
        metadata["outputs"].append(split_metadata)
        print(f"[{split}] Saved {len(noisy_rows)} noisy rows to: {output_path}")

    metadata_path = output_root / f"{args.output_suffix}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved noisy split metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
