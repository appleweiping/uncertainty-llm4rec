"""Day13 LLMEmb candidate score export adapter.

The adapter is deliberately conservative: it exports real LLMEmb model scores
only when the external repository has the required Beauty handled data,
embedding files, and checkpoint. It never fabricates backbone scores.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_REPO = WORKSPACE / "external" / "LLMEmb"
DEFAULT_OUTPUT = WORKSPACE / "output-repaired" / "backbone" / "llmemb_beauty_100" / "candidate_scores.csv"
SUMMARY_DIR = WORKSPACE / "output-repaired" / "summary"


def _repo_head(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = repo / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else text
    return text


def _required_paths(repo: Path, dataset: str, model_name: str, check_path: str, inter_file: str, llm_emb_file: str) -> dict[str, Path]:
    handled = repo / "data" / dataset / "handled"
    return {
        "repo": repo,
        "handled_dir": handled,
        "interaction_file": handled / f"{inter_file}.txt",
        "id_map": handled / "id_map.json",
        "llm_embedding": handled / f"{llm_emb_file}.pkl",
        "srs_embedding": handled / "itm_emb_sasrec.pkl",
        "checkpoint": repo / "saved" / dataset / model_name / check_path / "pytorch_model.bin",
    }


def _write_alignment_diagnosis(repo: Path, paths: dict[str, Path], missing: list[str]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    current_files = {
        "data/processed/amazon_beauty/test.jsonl": WORKSPACE / "data" / "processed" / "amazon_beauty" / "test.jsonl",
        "data/processed/amazon_beauty/ranking_test.jsonl": WORKSPACE / "data" / "processed" / "amazon_beauty" / "ranking_test.jsonl",
        "Day9 evidence": WORKSPACE / "output-repaired" / "beauty_deepseek_relevance_evidence_full" / "calibrated" / "relevance_evidence_posterior_test.jsonl",
    }
    report = f"""# Day13 LLMEmb Alignment Diagnosis

## Status

Blocked before real candidate-score export.

Missing LLMEmb requirements:

{chr(10).join(f"- `{name}`: `{path}`" for name, path in paths.items() if name in missing)}

## 1. LLMEmb Required Data Format

LLMEmb reads sequential data from:

`external/LLMEmb/data/<dataset>/handled/<inter_file>.txt`

Each line is expected to contain integer-mapped user and item ids:

`user_id item_id`

The repository README also expects:

- `data/<dataset>/handled/id_map.json`
- `data/<dataset>/handled/<llm_emb_file>.pkl`
- `data/<dataset>/handled/itm_emb_sasrec.pkl`
- `saved/<dataset>/<model_name>/<check_path>/pytorch_model.bin`

The evaluation loader constructs each test candidate pool as:

`item_indices = [positive_item] + sampled_negative_items`

and the true score is produced by:

`self.model.predict(**inputs)`

inside `external/LLMEmb/trainers/sequence_trainer.py`.

## 2. Our Current Beauty Processed Data

Available local project files:

{chr(10).join(f"- `{label}`: `{path}` exists={path.exists()}" for label, path in current_files.items())}

Our Day9 evidence uses raw Amazon-style ids such as `user_id` and `candidate_item_id`. LLMEmb internally expects contiguous integer ids starting from 1 after its own preprocessing. Therefore, a direct join is only safe if `id_map.json` exposes a reversible mapping between raw ids and mapped integer ids, or if the LLMEmb export writes both raw and mapped ids.

## 3. User/Item Alignment

Current state:

- LLMEmb repo is cloned at `{repo}`.
- LLMEmb Beauty handled data is not present.
- LLMEmb checkpoint is not present.
- The external repo cannot currently produce real backbone scores locally.

Required Day13/Day14 conversion plan:

1. Build `external/LLMEmb/data/beauty/handled/inter.txt` from a Beauty split whose raw user/item ids can be mapped back to our Day9 ids.
2. Preserve `raw_user_id -> mapped_user_id` and `raw_item_id -> mapped_item_id`.
3. Obtain or generate `0722_avg_pca.pkl` and `itm_emb_sasrec.pkl`, or run a base non-LLMEmb backbone first if the goal is only score-export smoke.
4. Train/load LLMEmb checkpoint at `external/LLMEmb/saved/beauty/llmemb_sasrec/llmemb/pytorch_model.bin`.
5. Export candidate scores with both raw and mapped ids.
6. Join with Day9 evidence on raw `user_id + candidate_item_id`.

## 4. Candidate Pool Compatibility

LLMEmb's default evaluation uses one positive plus `test_neg` sampled negatives. Our Day9 evidence already covers the Beauty candidate rows used by our project. To avoid low join coverage, the preferred path is to make LLMEmb evaluate the same candidate pools as Day9, not random new negatives. If using LLMEmb's random negatives, the generated negative candidate ids may not exist in Day9 evidence and join coverage will drop.

## 5. Can We Join Without Regenerating Day9 Evidence?

Yes, but only if the LLMEmb candidate export uses the same raw user/item ids or a reversible mapping back to them. If the LLMEmb export only contains integer ids, Day9 evidence cannot be joined safely.
"""
    (SUMMARY_DIR / "day13_llmemb_alignment_diagnosis.md").write_text(report, encoding="utf-8")


def _write_score_export_report(repo: Path, paths: dict[str, Path], missing: list[str], output_path: Path, status: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# Day13 LLMEmb Score Export Report

## Status

`{status}`

LLMEmb repo:

`{repo}`

Commit hash:

`{_repo_head(repo)}`

Target score export:

`{output_path}`

## Required Files

| Requirement | Path | Exists |
|---|---|---|
{chr(10).join(f"| {name} | `{path}` | {path.exists()} |" for name, path in paths.items())}

## Located Score Entrypoint

Training/evaluation entry:

`external/LLMEmb/main.py`

Trainer:

`external/LLMEmb/trainers/sequence_trainer.py`

Candidate score generation:

```python
inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
pred_logits = -self.model.predict(**inputs)
```

The model-level candidate score comes from each model's `predict` method. For SASRec-style models, this computes:

```python
item_embs = self._get_embedding(item_indices)
logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
```

Metric computation:

`external/LLMEmb/utils/utils.py::metric_report`

LLMEmb currently keeps only the positive item's rank for HR@10/NDCG@10. To export a plug-in candidate table, the adapter must write every candidate in `item_indices`, not only the final top-10.

## Missing Blockers

{chr(10).join(f"- `{name}`: `{paths[name]}`" for name in missing) if missing else "- None"}

## Outcome

No synthetic scores were generated. If status is `blocked`, the next step is to prepare handled Beauty data, checkpoint, and reversible id mapping before running the exporter again.
"""
    (SUMMARY_DIR / "day13_llmemb_score_export_report.md").write_text(report, encoding="utf-8")


def _blocked(repo: Path, paths: dict[str, Path], missing: list[str], output_path: Path) -> None:
    _write_alignment_diagnosis(repo, paths, missing)
    _write_score_export_report(repo, paths, missing, output_path, "blocked")
    print("Blocked: missing LLMEmb data/checkpoint requirements.")
    for name in missing:
        print(f"- {name}: {paths[name]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument("--model_name", default="llmemb_sasrec")
    parser.add_argument("--check_path", default="llmemb")
    parser.add_argument("--inter_file", default="inter")
    parser.add_argument("--llm_emb_file", default="0722_avg_pca")
    parser.add_argument("--max_users", type=int, default=100)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _export_scores(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        import torch

        from generators.generator import Generator, Seq2SeqGenerator
        from trainers.sequence_trainer import SeqTrainer
        from utils.argument import get_main_arguments, get_model_arguments, get_train_arguments
        from utils.logger import Logger
        from utils.utils import set_seed

        parser = argparse.ArgumentParser()
        parser = get_main_arguments(parser)
        parser = get_model_arguments(parser)
        parser = get_train_arguments(parser)
        llm_args = parser.parse_args(
            [
                "--dataset",
                args.dataset,
                "--model_name",
                args.model_name,
                "--hidden_size",
                "128",
                "--train_batch_size",
                "128",
                "--max_len",
                "200",
                "--num_workers",
                "0",
                "--check_path",
                args.check_path,
                "--freeze_emb",
                "--llm_emb_file",
                args.llm_emb_file,
                "--inter_file",
                args.inter_file,
                "--no_cuda",
            ]
        )
        set_seed(llm_args.seed)
        llm_args.output_dir = os.path.join(llm_args.output_dir, llm_args.dataset)
        llm_args.pretrain_dir = os.path.join(llm_args.output_dir, llm_args.pretrain_dir)
        llm_args.output_dir = os.path.join(llm_args.output_dir, llm_args.model_name)
        llm_args.output_dir = os.path.join(llm_args.output_dir, llm_args.check_path)
        llm_args.llm_emb_path = os.path.join("data/" + llm_args.dataset + "/handled/", f"{llm_args.llm_emb_file}.pkl")

        logger_manager = Logger(llm_args)
        logger, writer = logger_manager.get_logger()
        device = torch.device("cpu")
        generator_cls = Seq2SeqGenerator if llm_args.model_name in ["sasrec_seq", "llmemb_sasrec"] else Generator
        generator = generator_cls(llm_args, logger, device)
        trainer = SeqTrainer(llm_args, logger, writer, device, generator)
        checkpoint = torch.load(os.path.join(llm_args.output_dir, "pytorch_model.bin"), map_location=device)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.to(device)
        trainer.model.eval()

        rows = []
        user_counter = 0
        with torch.no_grad():
            for batch in trainer.test_loader:
                batch = tuple(t.to(device) for t in batch)
                inputs = trainer._prepare_eval_inputs(batch)
                item_indices = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs["item_indices"] = item_indices
                scores = (-trainer.model.predict(**inputs)).detach().cpu()
                item_ids = item_indices.detach().cpu()
                for row_idx in range(scores.shape[0]):
                    user_counter += 1
                    if user_counter > args.max_users:
                        break
                    user_id = str(user_counter)
                    per_scores = scores[row_idx].numpy()
                    per_items = item_ids[row_idx].numpy()
                    order = per_scores.argsort()[::-1]
                    ranks = {int(per_items[idx]): rank + 1 for rank, idx in enumerate(order)}
                    for col_idx, item_id in enumerate(per_items):
                        item_id = int(item_id)
                        rows.append(
                            {
                                "user_id": user_id,
                                "candidate_item_id": str(item_id),
                                "backbone_score": float(per_scores[col_idx]),
                                "label": 1 if col_idx == 0 else 0,
                                "backbone_rank": ranks[item_id],
                                "mapped_user_id": user_id,
                                "mapped_item_id": str(item_id),
                                "mapping_success": False,
                            }
                        )
                if user_counter >= args.max_users:
                    break

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_path, index=False)
        _write_score_export_report(repo, _required_paths(repo, args.dataset, args.model_name, args.check_path, args.inter_file, args.llm_emb_file), [], args.output_path, "exported")
        print(f"Exported {len(rows)} candidate rows to {args.output_path}")
    finally:
        os.chdir(old_cwd)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    paths = _required_paths(repo, args.dataset, args.model_name, args.check_path, args.inter_file, args.llm_emb_file)
    missing = [name for name, path in paths.items() if name != "id_map" and not path.exists()]
    if not repo.exists():
        missing = ["repo"]
    if missing:
        _blocked(repo, paths, missing, args.output_path)
        return
    _export_scores(args)


if __name__ == "__main__":
    main()
