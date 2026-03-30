# scripts/generate_noisy_data.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data.noise import apply_noise_to_sample


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/test.jsonl",
        help="Input clean dataset."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/test_noisy.jsonl",
        help="Output noisy dataset."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/processed/test_noisy_metadata.json",
        help="Path to save noise configuration metadata."
    )
    parser.add_argument(
        "--history_drop_prob",
        type=float,
        default=0.2,
        help="Probability of dropping each history item."
    )
    parser.add_argument(
        "--text_noise_prob",
        type=float,
        default=0.5,
        help="Probability of perturbing each text field."
    )
    parser.add_argument(
        "--label_flip_prob",
        type=float,
        default=0.0,
        help="Optional label noise probability."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    args = parser.parse_args()

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

    metadata_path = Path(args.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved noisy dataset to: {args.output_path}")
    print(f"Saved noise metadata to: {args.metadata_path}")


if __name__ == "__main__":
    main()