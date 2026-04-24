from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml


def _shell_join(parts: list[str]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def _exp_prefix(domain: str, variant: str, scenario: str) -> str:
    return f"{domain}_qwen3_{variant}_{scenario}"


def _variant_list(cfg: dict[str, Any], selected: str) -> list[str]:
    if selected == "all":
        return sorted((cfg.get("variants") or {}).keys())
    return [item.strip() for item in selected.split(",") if item.strip()]


def _domain_list(scenario_cfg: dict[str, Any], selected: str) -> list[str]:
    domains = scenario_cfg.get("domains") or {}
    if selected == "all":
        return sorted(domains.keys())
    return [item.strip() for item in selected.split(",") if item.strip()]


def _max_arg(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def build_commands(
    cfg: dict[str, Any],
    *,
    scenario: str,
    variants: list[str],
    domains: list[str],
    include_noisy: bool,
) -> list[str]:
    scenario_cfg = cfg["scenarios"][scenario]
    output_root = str(cfg.get("output_root", "outputs"))
    seed = str(cfg.get("seed", 42))
    topk = str(cfg.get("default_topk", 10))
    max_new_tokens = str(cfg.get("default_rank_max_new_tokens", 96))
    rerank_variant = str(cfg.get("default_rerank_variant", "nonlinear_structured_risk_rerank"))
    lambda_penalty = str(cfg.get("default_lambda_penalty", 0.5))
    commands: list[str] = ["cd ~/projects/uncertainty-llm4rec", "mkdir -p outputs/logs outputs/summary"]

    for domain in domains:
        domain_cfg = scenario_cfg["domains"][domain]
        rank_exp = f"{domain}_qwen3_shadow_rank_{scenario}"
        commands.append(
            _shell_join(
                [
                    "python main_rank.py",
                    "--exp_name", rank_exp,
                    "--input_path", domain_cfg["ranking_test_path"],
                    "--model_config", domain_cfg["rank_model_config"],
                    "--prompt_path prompts/candidate_ranking.txt",
                    "--output_root", output_root,
                    "--topk", topk,
                    "--max_new_tokens", max_new_tokens,
                    *_max_arg("--max_samples", domain_cfg.get("max_rank_samples")),
                    "--resume_partial",
                    "--seed", seed,
                ]
            )
        )

        noisy_rank_exp = f"{domain}_qwen3_shadow_rank_{scenario}_noisy_nl10"
        if include_noisy:
            commands.append(
                _shell_join(
                    [
                        "python main_rank.py",
                        "--exp_name", noisy_rank_exp,
                        "--input_path", domain_cfg["noisy_ranking_test_path"],
                        "--model_config", domain_cfg["rank_model_config"],
                        "--prompt_path prompts/candidate_ranking.txt",
                        "--output_root", output_root,
                        "--topk", topk,
                        "--max_new_tokens", max_new_tokens,
                        *_max_arg("--max_samples", domain_cfg.get("max_rank_samples")),
                        "--resume_partial",
                        "--seed", seed,
                    ]
                )
            )

        for variant in variants:
            prompt_path = cfg["variants"][variant]["prompt_path"]
            pointwise_exp = f"{_exp_prefix(domain, variant, scenario)}_pointwise"
            commands.extend(
                [
                    _shell_join(
                        [
                            "python main_infer.py",
                            "--exp_name", pointwise_exp,
                            "--input_path", domain_cfg["pointwise_valid_path"],
                            "--split_name valid",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    _shell_join(
                        [
                            "python main_infer.py",
                            "--exp_name", pointwise_exp,
                            "--input_path", domain_cfg["pointwise_test_path"],
                            "--split_name test",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    f"python main_eval_shadow.py --exp_name {pointwise_exp} --output_root {output_root} --score_col shadow_score --seed {seed}",
                    f"python main_calibrate_shadow.py --exp_name {pointwise_exp} --shadow_variant {variant} --output_root {output_root} --score_col shadow_score --method isotonic",
                ]
            )

            rerank_exp = f"{_exp_prefix(domain, variant, scenario)}_structured_risk"
            commands.append(
                _shell_join(
                    [
                        "python main_rank_rerank.py",
                        "--exp_name", rank_exp,
                        "--new_exp_name", rerank_exp,
                        "--uncertainty_exp_name", pointwise_exp,
                        "--uncertainty_input_path", f"{output_root}/{pointwise_exp}/calibrated/test_calibrated.jsonl",
                        "--uncertainty_col shadow_uncertainty",
                        "--uncertainty_confidence_col shadow_calibrated_score",
                        "--output_root", output_root,
                        "--rerank_variant", rerank_variant,
                        "--lambda_penalty", lambda_penalty,
                        "--k", topk,
                        "--seed", seed,
                    ]
                )
            )

            if not include_noisy:
                continue
            noisy_pointwise_exp = f"{pointwise_exp}_noisy_nl10"
            commands.extend(
                [
                    _shell_join(
                        [
                            "python main_infer.py",
                            "--exp_name", noisy_pointwise_exp,
                            "--input_path", domain_cfg["noisy_valid_path"],
                            "--split_name valid",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    _shell_join(
                        [
                            "python main_infer.py",
                            "--exp_name", noisy_pointwise_exp,
                            "--input_path", domain_cfg["noisy_test_path"],
                            "--split_name test",
                            "--model_config", domain_cfg["model_config"],
                            "--prompt_path", prompt_path,
                            "--output_root", output_root,
                            "--response_schema shadow",
                            "--shadow_variant", variant,
                            *_max_arg("--max_samples", domain_cfg.get("max_pointwise_samples")),
                            "--resume_partial",
                            "--checkpoint_every_batches 1",
                            "--seed", seed,
                        ]
                    ),
                    f"python main_eval_shadow.py --exp_name {noisy_pointwise_exp} --output_root {output_root} --score_col shadow_score --seed {seed}",
                    f"python main_calibrate_shadow.py --exp_name {noisy_pointwise_exp} --shadow_variant {variant} --output_root {output_root} --score_col shadow_score --method isotonic",
                ]
            )
            noisy_rerank_exp = f"{rerank_exp}_noisy_nl10"
            commands.append(
                _shell_join(
                    [
                        "python main_rank_rerank.py",
                        "--exp_name", noisy_rank_exp,
                        "--new_exp_name", noisy_rerank_exp,
                        "--uncertainty_exp_name", noisy_pointwise_exp,
                        "--uncertainty_input_path", f"{output_root}/{noisy_pointwise_exp}/calibrated/test_calibrated.jsonl",
                        "--uncertainty_col shadow_uncertainty",
                        "--uncertainty_confidence_col shadow_calibrated_score",
                        "--output_root", output_root,
                        "--rerank_variant", rerank_variant,
                        "--lambda_penalty", lambda_penalty,
                        "--k", topk,
                        "--seed", seed,
                    ]
                )
            )

    commands.append(
        f"python main_compare_shadow_line.py --scenario {scenario} --variants {','.join(variants)} --domains {','.join(domains)}"
    )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/shadow/week7_9_shadow_runtime.yaml")
    parser.add_argument("--scenario", choices=["small_prior", "full_replay", "formal_full_domains"], default="small_prior")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--include_noisy", action="store_true")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.manifest)
    scenario_cfg = cfg["scenarios"][args.scenario]
    variants = _variant_list(cfg, args.variants)
    domains = _domain_list(scenario_cfg, args.domains)
    commands = build_commands(
        cfg,
        scenario=args.scenario,
        variants=variants,
        domains=domains,
        include_noisy=args.include_noisy,
    )
    text = "\n".join(commands) + "\n"
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
