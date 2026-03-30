# src/utils/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpPaths:
    exp_name: str
    root: Path
    predictions_dir: Path
    calibrated_dir: Path
    reranked_dir: Path
    figures_dir: Path
    tables_dir: Path


def default_input_path_for_exp(
    exp_name: str,
    data_root: str | Path = "data/processed",
) -> Path:
    data_root = Path(data_root)

    if exp_name == "clean":
        return data_root / "test.jsonl"
    if exp_name == "noisy":
        return data_root / "test_noisy.jsonl"

    return data_root / f"test_{exp_name}.jsonl"


def build_exp_paths(
    exp_name: str,
    output_root: str | Path = "outputs",
) -> ExpPaths:
    root = Path(output_root) / exp_name
    return ExpPaths(
        exp_name=exp_name,
        root=root,
        predictions_dir=root / "predictions",
        calibrated_dir=root / "calibrated",
        reranked_dir=root / "reranked",
        figures_dir=root / "figures",
        tables_dir=root / "tables",
    )


def ensure_exp_dirs(
    exp_name: str,
    output_root: str | Path = "outputs",
) -> ExpPaths:
    paths = build_exp_paths(exp_name=exp_name, output_root=output_root)
    for path in [
        paths.root,
        paths.predictions_dir,
        paths.calibrated_dir,
        paths.reranked_dir,
        paths.figures_dir,
        paths.tables_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_compare_dirs(
    compare_name: str,
    output_root: str | Path = "outputs",
) -> Path:
    root = Path(output_root) / "robustness" / compare_name
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root