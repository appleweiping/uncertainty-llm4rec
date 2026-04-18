from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.llm import build_backend_from_config
from src.llm.inference import run_pointwise_inference
from src.llm.local_backend import LocalHFBackend
from src.llm.base import normalize_generation_result
from src.llm.parser import parse_ranking_response
from src.utils.paths import default_input_path_for_exp, ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_llm_backend(model_cfg_path: str | Path):
    model_cfg = load_yaml(model_cfg_path)
    backend_name = str(model_cfg.get("backend_name", "")).strip().lower()

    if backend_name != "local_hf":
        return build_backend_from_config(model_cfg_path)

    generation_cfg = model_cfg.get("generation", {}) or {}
    local_cfg = model_cfg.get("local", {}) or {}

    model_name = str(model_cfg.get("model_name") or generation_cfg.get("model_name") or "local-model")
    model_path = local_cfg.get("model_path") or model_cfg.get("model_path")
    tokenizer_path = local_cfg.get("tokenizer_path") or model_cfg.get("tokenizer_path")
    if not model_path:
        raise ValueError(f"model_path is required for local_hf config: {model_cfg_path}")

    return LocalHFBackend(
        model_name=model_name,
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        provider=str(model_cfg.get("provider") or "local_hf"),
        dtype=str(local_cfg.get("dtype") or model_cfg.get("dtype") or "auto"),
        device_map=local_cfg.get("device_map", model_cfg.get("device_map", "auto")),
        max_tokens=int(generation_cfg.get("max_tokens", model_cfg.get("max_tokens", 300))),
        temperature=float(generation_cfg.get("temperature", model_cfg.get("temperature", 0.0))),
        top_p=float(generation_cfg.get("top_p", model_cfg.get("top_p", 1.0))),
        do_sample=local_cfg.get("do_sample", generation_cfg.get("do_sample", model_cfg.get("do_sample"))),
        use_chat_template=bool(local_cfg.get("use_chat_template", model_cfg.get("use_chat_template", True))),
        trust_remote_code=bool(local_cfg.get("trust_remote_code", model_cfg.get("trust_remote_code", False))),
    )


def describe_backend(backend) -> dict[str, Any]:
    details = {
        "backend_class": type(backend).__name__,
        "backend_type": getattr(backend, "backend_type", None),
        "provider": getattr(backend, "provider", None),
        "model_name": getattr(backend, "model_name", None),
    }
    model_path = getattr(backend, "model_path", None)
    tokenizer_path = getattr(backend, "tokenizer_path", None)
    if model_path is not None:
        details["model_path"] = model_path
    if tokenizer_path is not None:
        details["tokenizer_path"] = tokenizer_path
    return details


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict]:
    df = pd.read_json(path, lines=True)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(path, orient="records", lines=True, force_ascii=False)

class FunctionPromptBuilderAdapter:
    def __init__(self, fn, template_path: str | Path):
        self.fn = fn
        self.template_path = str(template_path)

    def build_pointwise_prompt(self, sample: dict, candidate: dict) -> str:
        try:
            return self.fn(sample, candidate, template_path=self.template_path)
        except TypeError:
            return self.fn(sample, candidate)

    def build_candidate_ranking_prompt(self, sample: dict, ranking_mode: str = "score_list") -> str:
        raise NotImplementedError(
            "Fallback function prompt builder does not support candidate_ranking. "
            "Please use PromptBuilder class from src/llm/prompt_builder.py."
        )


def get_prompt_builder(prompt_path: str | Path):
    try:
        from src.llm.prompt_builder import PromptBuilder

        return PromptBuilder(template_path=str(prompt_path))
    except Exception:
        try:
            from src.llm.prompt_builder import build_pointwise_prompt

            return FunctionPromptBuilderAdapter(build_pointwise_prompt, template_path=prompt_path)
        except Exception as e:
            raise ImportError("Cannot find a usable prompt builder in src/llm/prompt_builder.py") from e


def infer_prediction_filename(input_path: str | Path, task_type: str = "pointwise_yesno") -> str:
    stem = Path(input_path).stem.lower()
    suffix = "_raw.jsonl" if task_type == "pointwise_yesno" else "_ranking_raw.jsonl"
    if stem.startswith("valid"):
        return f"valid{suffix}"
    if stem.startswith("test"):
        return f"test{suffix}"
    if stem.startswith("train"):
        return f"train{suffix}"
    return f"{stem}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Experiment config path.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Input pointwise jsonl path.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional default data directory.")
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--prompt_path", type=str, default=None, help="Prompt template path.")
    parser.add_argument("--model_config", type=str, default=None, help="Model config path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--output_path", type=str, default=None, help="Prediction output jsonl path.")
    parser.add_argument("--split_name", type=str, default=None, help="Optional split name, e.g. train/valid/test.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Task type: pointwise_yesno or candidate_ranking.",
    )
    parser.add_argument(
        "--ranking_mode",
        type=str,
        default=None,
        help="Ranking parser/prompt mode: score_list or rank_topk.",
    )
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    merged = {
        "exp_name": args.exp_name if args.exp_name is not None else cfg.get("exp_name", "clean"),
        "input_path": args.input_path if args.input_path is not None else cfg.get("input_path"),
        "data_root": args.data_root if args.data_root is not None else cfg.get("data_root"),
        "output_root": args.output_root if args.output_root is not None else cfg.get("output_root", "outputs"),
        "prompt_path": args.prompt_path if args.prompt_path is not None else cfg.get("prompt_path", "prompts/pointwise_yesno.txt"),
        "model_config": args.model_config if args.model_config is not None else cfg.get("model_config"),
        "max_samples": args.max_samples if args.max_samples is not None else cfg.get("max_samples"),
        "output_path": args.output_path if args.output_path is not None else cfg.get("output_path"),
        "split_name": args.split_name if args.split_name is not None else cfg.get("split_name"),
        "overwrite": bool(args.overwrite or cfg.get("overwrite", False)),
        "seed": args.seed if args.seed is not None else cfg.get("seed"),
        "task_type": args.task_type if args.task_type is not None else cfg.get("task_type", "pointwise_yesno"),
        "ranking_mode": args.ranking_mode if args.ranking_mode is not None else cfg.get("ranking_mode"),
    }
    return merged


def build_candidate_ranking_samples(
    input_path: str | Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    df = pd.read_json(input_path, lines=True)
    rows = df.to_dict(orient="records")

    user_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        user_id = str(row.get("user_id", "")).strip()
        if not user_id:
            continue
        user_groups.setdefault(user_id, []).append(row)

    samples: list[dict[str, Any]] = []
    for user_id, group_rows in user_groups.items():
        first = group_rows[0]
        target_item_id = str(
            first.get("target_item_id")
            or next(
                (
                    row.get("candidate_item_id")
                    for row in group_rows
                    if int(row.get("label", 0)) == 1 and row.get("candidate_item_id")
                ),
                "",
            )
        ).strip()

        candidates: list[dict[str, Any]] = []
        for row in group_rows:
            candidate_item_id = str(row.get("candidate_item_id", "")).strip()
            if not candidate_item_id:
                continue
            candidates.append(
                {
                    "item_id": candidate_item_id,
                    "title": str(row.get("candidate_title") or candidate_item_id).strip(),
                    "meta": str(
                        row.get("candidate_meta")
                        or row.get("candidate_description")
                        or row.get("candidate_text")
                        or ""
                    ).strip(),
                    "label": int(row.get("label", 0)),
                }
            )

        samples.append(
            {
                "user_id": user_id,
                "history": first.get("history", []),
                "history_items": first.get("history_items", []),
                "target_item_id": target_item_id,
                "target_popularity_group": first.get("target_popularity_group", "unknown"),
                "candidates": candidates,
            }
        )

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    return samples


def run_candidate_ranking_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
    ranking_mode: str = "score_list",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running candidate ranking inference"):
        prompt = prompt_builder.build_candidate_ranking_prompt(sample, ranking_mode=ranking_mode)
        generation = normalize_generation_result(
            llm_backend.generate(prompt),
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        raw_text = generation["raw_text"]
        parsed = parse_ranking_response(raw_text, ranking_mode=ranking_mode)

        candidate_payload = [
            {
                "item_id": str(candidate.get("item_id", "")).strip(),
                "title": str(candidate.get("title", "")).strip(),
                "label": int(candidate.get("label", 0)),
            }
            for candidate in sample.get("candidates", [])
        ]

        results.append(
            {
                "task_type": "candidate_ranking",
                "ranking_mode": parsed.get("ranking_mode", ranking_mode),
                "user_id": sample.get("user_id", ""),
                "target_item_id": sample.get("target_item_id", ""),
                "target_popularity_group": sample.get("target_popularity_group", "unknown"),
                "candidate_count": len(candidate_payload),
                "candidates": candidate_payload,
                "prompt": prompt,
                "raw_response": raw_text,
                "selected_item_id": parsed.get("selected_item_id", ""),
                "top_k_item_ids": parsed.get("top_k_item_ids", []),
                "ranked_item_ids": parsed.get("ranked_item_ids", []),
                "candidate_scores": parsed.get("candidate_scores", []),
                "reason": parsed.get("reason", ""),
                "response_latency": generation.get("latency", 0.0),
                "response_model_name": generation.get("model_name", ""),
                "response_provider": generation.get("provider", ""),
                "response_backend_type": generation.get("backend_type"),
                "response_usage": generation.get("usage", {}),
            }
        )

    return results


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    exp_name = str(cfg["exp_name"])
    output_root = cfg["output_root"]
    input_path = cfg["input_path"]
    data_root = cfg["data_root"]
    prompt_path = cfg["prompt_path"]
    model_config = cfg["model_config"]
    max_samples = cfg["max_samples"]
    output_path = cfg["output_path"]
    split_name = cfg["split_name"]
    overwrite = cfg["overwrite"]
    seed = cfg["seed"]
    task_type = str(cfg["task_type"]).strip().lower()
    ranking_mode = cfg["ranking_mode"]

    set_global_seed(seed)

    if model_config is None:
        raise ValueError("model_config must be provided via config or CLI.")
    if task_type not in {"pointwise_yesno", "candidate_ranking"}:
        raise ValueError(f"Unsupported task_type: {task_type}")
    if ranking_mode is None:
        ranking_mode = "rank_topk" if "rank_topk" in str(prompt_path).lower() else "score_list"
    ranking_mode = str(ranking_mode).strip().lower()

    paths = ensure_exp_dirs(exp_name, output_root)

    if input_path is None:
        if data_root is None:
            raise ValueError("input_path or data_root must be provided via config or CLI.")
        input_path = default_input_path_for_exp(exp_name, data_root)

    input_path = Path(input_path)
    if output_path is not None:
        output_path = Path(output_path)
    elif split_name is not None:
        suffix = "_raw.jsonl" if task_type == "pointwise_yesno" else "_ranking_raw.jsonl"
        output_path = paths.predictions_dir / f"{str(split_name).strip().lower()}{suffix}"
    else:
        output_path = paths.predictions_dir / infer_prediction_filename(input_path, task_type=task_type)

    print(f"[{exp_name}] Input path: {input_path}")
    print(f"[{exp_name}] Output path: {output_path}")
    print(f"[{exp_name}] Model config: {model_config}")
    print(f"[{exp_name}] Task type: {task_type}")
    if task_type == "candidate_ranking":
        print(f"[{exp_name}] Ranking mode: {ranking_mode}")
    if seed is not None:
        print(f"[{exp_name}] Seed: {seed}")

    if not input_path.exists():
        raise FileNotFoundError(f"[{exp_name}] Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        print(f"[{exp_name}] Output already exists: {output_path}")
        print(f"[{exp_name}] Use overwrite=true in config or --overwrite to rerun.")
        return

    if task_type == "pointwise_yesno":
        samples = load_jsonl(input_path, max_samples=max_samples)
    else:
        samples = build_candidate_ranking_samples(input_path, max_samples=max_samples)
    print(f"[{exp_name}] Loaded {len(samples)} samples.")

    llm_backend = build_llm_backend(model_config)
    backend_details = describe_backend(llm_backend)
    print(f"[{exp_name}] Resolved backend: {backend_details}")
    prompt_builder = get_prompt_builder(prompt_path)

    if task_type == "pointwise_yesno":
        predictions = run_pointwise_inference(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
        )
    else:
        predictions = run_candidate_ranking_inference(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
            ranking_mode=ranking_mode,
        )

    save_jsonl(predictions, output_path)
    print(f"[{exp_name}] Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
