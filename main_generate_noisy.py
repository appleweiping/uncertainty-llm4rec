from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data.noise import apply_noise_to_sample


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input clean jsonl.")
    parser.add_argument("--output_path", type=str, required=True, help="Output noisy jsonl.")
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Optional path to save noise metadata json.",
    )
    parser.add_argument("--history_drop_prob", type=float, default=0.2)
    parser.add_argument("--text_noise_prob", type=float, default=0.5)
    parser.add_argument("--label_flip_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rows = load_jsonl(args.input_path)
    noisy_rows = [
        apply_noise_to_sample(
            row,
            history_drop_prob=args.history_drop_prob,
            text_noise_prob=args.text_noise_prob,
            label_flip_prob=args.label_flip_prob,
        )
        for row in rows
    ]

    save_jsonl(noisy_rows, args.output_path)

    metadata = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "history_drop_prob": args.history_drop_prob,
        "text_noise_prob": args.text_noise_prob,
        "label_flip_prob": args.label_flip_prob,
        "seed": args.seed,
        "num_samples": len(rows),
    }

    metadata_path = (
        Path(args.metadata_path)
        if args.metadata_path is not None
        else Path(args.output_path).with_name(Path(args.output_path).stem + "_noise_metadata.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved noisy dataset to: {args.output_path}")
    print(f"Saved noise metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
