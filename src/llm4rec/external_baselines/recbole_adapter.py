"""RecBole adapter contract for paper-grade external baselines."""

from __future__ import annotations

import importlib.util
import csv
import json
import time
from pathlib import Path
from typing import Any

from llm4rec.external_baselines.base import ExternalBaselineConfig, MissingExternalDependencyError
from llm4rec.io.artifacts import read_jsonl


def ensure_recbole_available() -> None:
    if importlib.util.find_spec("recbole") is None:
        raise MissingExternalDependencyError(
            "RecBole is not installed. Install optional baselines dependencies with "
            "`py -3 -m pip install -e .[baselines]` in an environment compatible with RecBole."
        )


def build_recbole_config(config: ExternalBaselineConfig, *, exported_dataset_dir: str | Path) -> dict[str, Any]:
    """Build a RecBole config dictionary without importing RecBole."""

    dataset_dir = Path(exported_dataset_dir)
    recbole_config = {
        "model": config.model_name,
        "dataset": config.dataset_name,
        "data_path": str(dataset_dir.parent),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "eval_args": {"split": {"LS": "valid_and_test"}, "group_by": "user", "order": "TO", "mode": "full"},
        "train_neg_sample_args": None,
        "epochs": int(config.training_config.get("epochs", 100)),
        "train_batch_size": int(config.training_config.get("train_batch_size", 2048)),
        "eval_batch_size": int(config.training_config.get("eval_batch_size", 4096)),
        "learning_rate": float(config.training_config.get("learning_rate", 0.001)),
        "metrics": config.training_config.get("metrics", ["Recall", "NDCG", "MRR"]),
        "topk": config.training_config.get("topk", [10]),
        "valid_metric": str(config.training_config.get("valid_metric", "NDCG@10")),
        "stopping_step": int(config.training_config.get("stopping_step", 10)),
        "seed": int(config.seed),
        "reproducibility": True,
        "show_progress": bool(config.training_config.get("show_progress", False)),
        "device": str(config.training_config.get("device", "cpu")),
        "checkpoint_dir": str(config.output_dir / "checkpoints" / f"{config.dataset_name}_{config.name}_seed{config.seed}"),
    }
    if config.model_name.lower() == "lightgcn":
        recbole_config["benchmark_filename"] = ["train", "valid", "test"]
        recbole_config["eval_args"] = {"split": {"RS": [8, 1, 1]}, "group_by": "user", "order": "TO", "mode": "full"}
        recbole_config["train_neg_sample_args"] = {"distribution": "uniform", "sample_num": 1}
    elif _is_sequential_recbole_model(config.model_name):
        recbole_config["benchmark_filename"] = ["sasrec_train", "sasrec_valid", "sasrec_test"]
        recbole_config["load_col"] = {"inter": ["user_id", "item_id", "item_id_list", "timestamp"]}
        recbole_config["eval_args"] = {"split": {"RS": [8, 1, 1]}, "group_by": "user", "order": "TO", "mode": "full"}
        recbole_config["alias_of_item_id"] = ["item_id_list"]
    for key, value in config.training_config.items():
        if key not in recbole_config:
            recbole_config[key] = value
    return recbole_config


def write_recbole_config(path: str | Path, config: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return out


def run_recbole_training(config: ExternalBaselineConfig, *, exported_dataset_dir: str | Path) -> dict[str, Any]:
    """Train a RecBole model on explicit TRUCE benchmark splits."""

    ensure_recbole_available()
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.transform import construct_transform
    from recbole.utils import get_model, get_trainer, init_logger, init_seed
    from recbole.utils.utils import get_environment, get_flops

    recbole_config = build_recbole_config(config, exported_dataset_dir=exported_dataset_dir)
    checkpoint_dir = Path(str(recbole_config["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in checkpoint_dir.glob("*.pth")}
    started = time.perf_counter()
    original_torch_load = torch.load
    torch.load = _trusted_torch_load(original_torch_load)
    try:
        rb_config = Config(model=config.model_name, dataset=config.dataset_name, config_dict=recbole_config)
        init_seed(rb_config["seed"], rb_config["reproducibility"])
        init_logger(rb_config)
        dataset = create_dataset(rb_config)
        train_data, valid_data, test_data = data_preparation(rb_config, dataset)
        init_seed(rb_config["seed"] + rb_config["local_rank"], rb_config["reproducibility"])
        model = get_model(rb_config["model"])(rb_config, train_data._dataset).to(rb_config["device"])
        transform = construct_transform(rb_config)
        get_flops(model, dataset, rb_config["device"], None, transform)
        trainer = get_trainer(rb_config["MODEL_TYPE"], rb_config["model"])(rb_config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=rb_config["show_progress"],
        )
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=rb_config["show_progress"])
        get_environment(rb_config)
        recbole_result = {
            "best_valid_score": best_valid_score,
            "valid_score_bigger": rb_config["valid_metric_bigger"],
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        }
    finally:
        torch.load = original_torch_load
    elapsed = time.perf_counter() - started
    after = sorted(
        (path for path in checkpoint_dir.glob("*.pth") if path.resolve() not in before),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not after:
        after = sorted(checkpoint_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    checkpoint_path = after[0] if after else None
    return {
        "recbole_result": recbole_result,
        "checkpoint_path": str(checkpoint_path or ""),
        "training_seconds": elapsed,
        "recbole_config": recbole_config,
        "_rb_config": rb_config,
        "_model": model,
        "_dataset": train_data._dataset,
    }


def score_recbole_candidates(
    *,
    config: ExternalBaselineConfig,
    checkpoint_path: str | Path,
    examples_path: str | Path,
    output_path: str | Path,
    split: str = "test",
    batch_size: int = 8192,
    rb_config: Any | None = None,
    model: Any | None = None,
    dataset: Any | None = None,
) -> dict[str, Any]:
    """Score the exact TRUCE candidate sets with a trained RecBole checkpoint."""

    ensure_recbole_available()
    import torch
    from recbole.data.interaction import Interaction
    from recbole.quick_start import load_data_and_model

    started = time.perf_counter()
    if rb_config is None or model is None or dataset is None:
        original_torch_load = torch.load
        torch.load = _trusted_torch_load(original_torch_load)
        try:
            rb_config, model, dataset, *_ = load_data_and_model(str(checkpoint_path))
        finally:
            torch.load = original_torch_load
    device = rb_config["device"]
    model.eval()
    examples = [
        row for row in read_jsonl(examples_path)
        if _normalize_split(row.get("split")) == _normalize_split(split)
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "user_id", "item_id", "score"])
        writer.writeheader()
        model_name = str(rb_config["model"]).lower()
        if _is_sequential_recbole_model(model_name) and hasattr(model, "full_sort_predict"):
            rows_written = _write_sasrec_full_sort_scores(
                writer=writer,
                model=model,
                dataset=dataset,
                rb_config=rb_config,
                examples=examples,
                device=device,
                batch_size=batch_size,
            )
        elif hasattr(model, "full_sort_predict"):
            rows_written = _write_general_full_sort_scores(
                writer=writer,
                model=model,
                dataset=dataset,
                examples=examples,
                device=device,
                batch_size=batch_size,
            )
        else:
            for ex in examples:
                candidate_items = [str(item) for item in ex.get("candidates") or ex.get("candidate_items") or []]
                if not candidate_items:
                    continue
                scores = _predict_scores(
                    model=model,
                    dataset=dataset,
                    rb_config=rb_config,
                    ex=ex,
                    candidate_items=candidate_items,
                    device=device,
                    batch_size=batch_size,
                )
                example_id = _example_id(ex)
                user_id = str(ex.get("user_id") or "")
                for item_id, score in zip(candidate_items, scores):
                    writer.writerow({"example_id": example_id, "user_id": user_id, "item_id": item_id, "score": float(score)})
                    rows_written += 1
    return {
        "scores": str(output),
        "example_count": len(examples),
        "score_count": rows_written,
        "scoring_seconds": time.perf_counter() - started,
        "split": split,
        "model_name": config.model_name,
    }


def _predict_scores(
    *,
    model: Any,
    dataset: Any,
    rb_config: Any,
    ex: dict[str, Any],
    candidate_items: list[str],
    device: Any,
    batch_size: int,
) -> list[float]:
    import torch

    user_id = str(ex.get("user_id") or "")
    model_name = str(rb_config["model"]).lower()
    all_scores: list[float] = []
    for start in range(0, len(candidate_items), batch_size):
        batch_items = candidate_items[start:start + batch_size]
        try:
            item_ids = dataset.token2id(dataset.iid_field, batch_items)
        except ValueError:
            item_ids = [_safe_token2id(dataset, dataset.iid_field, item) for item in batch_items]
        if _is_sequential_recbole_model(model_name):
            interaction = _sequential_interaction(dataset, rb_config, ex, item_ids)
        else:
            from recbole.data.interaction import Interaction

            user_internal = _safe_token2id(dataset, dataset.uid_field, user_id)
            interaction = Interaction({
                dataset.uid_field: torch.full((len(batch_items),), int(user_internal), dtype=torch.long),
                dataset.iid_field: torch.tensor(item_ids, dtype=torch.long),
            })
        with torch.no_grad():
            scores = model.predict(interaction.to(device)).detach().cpu().tolist()
        all_scores.extend(float(score) for score in scores)
    return all_scores


def _write_sasrec_full_sort_scores(
    *,
    writer: csv.DictWriter,
    model: Any,
    dataset: Any,
    rb_config: Any,
    examples: list[dict[str, Any]],
    device: Any,
    batch_size: int,
) -> int:
    import torch
    from recbole.data.interaction import Interaction

    max_len = int(rb_config["MAX_ITEM_LIST_LENGTH"] or 50)
    rows_written = 0
    example_batch_size = max(1, min(batch_size, 256))
    for start in range(0, len(examples), example_batch_size):
        batch = examples[start:start + example_batch_size]
        item_seq_rows: list[list[int]] = []
        seq_lengths: list[int] = []
        candidate_internal_rows: list[list[int]] = []
        candidate_token_rows: list[list[str]] = []
        for ex in batch:
            history = [str(item) for item in ex.get("history") or ex.get("history_item_ids") or []]
            history_ids = [_safe_token2id(dataset, dataset.iid_field, item) for item in history if item]
            history_ids = history_ids[-max_len:]
            seq_len = len(history_ids)
            padded = [0] * (max_len - seq_len) + [int(item) for item in history_ids]
            candidates = [str(item) for item in ex.get("candidates") or ex.get("candidate_items") or []]
            candidate_ids = [_safe_token2id(dataset, dataset.iid_field, item) for item in candidates]
            item_seq_rows.append(padded)
            seq_lengths.append(seq_len)
            candidate_internal_rows.append(candidate_ids)
            candidate_token_rows.append(candidates)
        interaction = Interaction({
            rb_config["LIST_SUFFIX"].join([dataset.iid_field, ""]): torch.tensor(item_seq_rows, dtype=torch.long),
            rb_config["ITEM_LIST_LENGTH_FIELD"]: torch.tensor(seq_lengths, dtype=torch.long),
        })
        with torch.no_grad():
            full_scores = model.full_sort_predict(interaction.to(device)).detach().cpu()
        for offset, ex in enumerate(batch):
            example_id = _example_id(ex)
            user_id = str(ex.get("user_id") or "")
            for item_id, internal_id in zip(candidate_token_rows[offset], candidate_internal_rows[offset]):
                score = float(full_scores[offset, int(internal_id)].item()) if int(internal_id) >= 0 else 0.0
                writer.writerow({"example_id": example_id, "user_id": user_id, "item_id": item_id, "score": score})
                rows_written += 1
    return rows_written


def _write_general_full_sort_scores(
    *,
    writer: csv.DictWriter,
    model: Any,
    dataset: Any,
    examples: list[dict[str, Any]],
    device: Any,
    batch_size: int,
) -> int:
    import torch
    from recbole.data.interaction import Interaction

    rows_written = 0
    example_batch_size = max(1, min(batch_size, 1024))
    item_num = int(getattr(dataset, "item_num", 0) or getattr(model, "n_items", 0))
    for start in range(0, len(examples), example_batch_size):
        batch = examples[start:start + example_batch_size]
        user_ids = [str(ex.get("user_id") or "") for ex in batch]
        user_internal = [_safe_token2id(dataset, dataset.uid_field, user) for user in user_ids]
        candidate_token_rows = [
            [str(item) for item in ex.get("candidates") or ex.get("candidate_items") or []]
            for ex in batch
        ]
        candidate_internal_rows = [
            [_safe_token2id(dataset, dataset.iid_field, item) for item in candidates]
            for candidates in candidate_token_rows
        ]
        interaction = Interaction({
            dataset.uid_field: torch.tensor(user_internal, dtype=torch.long),
        })
        with torch.no_grad():
            raw_scores = model.full_sort_predict(interaction.to(device)).detach().cpu()
        if raw_scores.dim() == 1:
            full_scores = raw_scores.view(len(batch), item_num)
        else:
            full_scores = raw_scores
        for offset, ex in enumerate(batch):
            example_id = _example_id(ex)
            user_id = user_ids[offset]
            for item_id, internal_id in zip(candidate_token_rows[offset], candidate_internal_rows[offset]):
                score = float(full_scores[offset, int(internal_id)].item()) if int(internal_id) >= 0 else 0.0
                writer.writerow({"example_id": example_id, "user_id": user_id, "item_id": item_id, "score": score})
                rows_written += 1
    return rows_written


def _trusted_torch_load(original_load: Any) -> Any:
    def trusted_local_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    return trusted_local_load


def _sequential_interaction(dataset: Any, rb_config: Any, ex: dict[str, Any], item_ids: Any) -> Any:
    import torch
    from recbole.data.interaction import Interaction

    max_len = int(rb_config["MAX_ITEM_LIST_LENGTH"] or 50)
    history = [str(item) for item in ex.get("history") or ex.get("history_item_ids") or []]
    history_ids = [_safe_token2id(dataset, dataset.iid_field, item) for item in history if item]
    history_ids = history_ids[-max_len:]
    seq_len = len(history_ids)
    padded = [0] * (max_len - seq_len) + [int(item) for item in history_ids]
    batch_size = len(item_ids)
    return Interaction({
        dataset.iid_field: torch.tensor(item_ids, dtype=torch.long),
        rb_config["ITEM_LIST_LENGTH_FIELD"]: torch.tensor([seq_len] * batch_size, dtype=torch.long),
        rb_config["LIST_SUFFIX"].join([dataset.iid_field, ""]): torch.tensor([padded] * batch_size, dtype=torch.long),
    })


def _safe_token2id(dataset: Any, field: str, token: str) -> int:
    try:
        return int(dataset.token2id(field, token))
    except ValueError:
        return 0


def _example_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("example_id") or meta.get("example_id") or row.get("user_id") or "")


def _normalize_split(value: Any) -> str:
    split = str(value or "").lower()
    if split in {"val", "validation"}:
        return "valid"
    return split


def _is_sequential_recbole_model(model_name: Any) -> bool:
    return str(model_name or "").lower() in {"sasrec", "bert4rec"}
