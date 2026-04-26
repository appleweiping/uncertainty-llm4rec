from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch may be absent in local dry-run
    class Dataset:  # type: ignore
        pass

from src.framework.prompt_formatters import format_sample


SUPPORTED_TASKS = {"candidate_ranking_listwise", "candidate_relevance_pointwise"}


@dataclass
class FormattedSample:
    sample_id: str
    domain: str
    task: str
    candidate_pool_setting: str
    prompt: str
    target: str
    text_missing: bool
    text_fallback_used: bool


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _metadata(sample: dict[str, Any], prompt: str, target: str) -> FormattedSample:
    meta = sample.get("metadata", {})
    task = sample.get("task", "")
    if task == "candidate_ranking":
        candidates = sample.get("input", {}).get("candidate_pool", [])
        text_missing = any(c.get("candidate_text_missing") for c in candidates)
        text_fallback = any(c.get("candidate_text_fallback_used") for c in candidates)
    else:
        candidate = sample.get("input", {}).get("candidate_item", {})
        text_missing = bool(candidate.get("candidate_text_missing") or meta.get("text_missing"))
        text_fallback = bool(candidate.get("candidate_text_fallback_used") or meta.get("text_fallback_used"))
    return FormattedSample(
        sample_id=str(sample.get("sample_id", "")),
        domain=str(sample.get("domain", "")),
        task=task,
        candidate_pool_setting=str(meta.get("candidate_pool_setting", "")),
        prompt=prompt,
        target=target,
        text_missing=text_missing,
        text_fallback_used=text_fallback,
    )


class QwenRecommendationDataset(Dataset):
    """JSONL dataset for Qwen recommendation baseline LoRA.

    This loader intentionally ignores CEP/confidence/evidence fields as training
    targets. It only trains raw recommendation ranking/relevance behavior.
    """

    def __init__(
        self,
        data_file: str | Path,
        task_type: str,
        prompt_template: str | Path,
        tokenizer: Any | None = None,
        max_seq_len: int = 4096,
        max_samples: int | None = None,
        train_on_inputs: bool = False,
    ) -> None:
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task_type={task_type}. Expected one of {sorted(SUPPORTED_TASKS)}")
        self.data_file = Path(data_file)
        self.task_type = task_type
        self.prompt_template = Path(prompt_template)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_on_inputs = train_on_inputs
        raw_samples = list(_iter_jsonl(self.data_file))
        if max_samples is not None and max_samples > 0:
            raw_samples = raw_samples[:max_samples]
        self.samples: list[FormattedSample] = []
        for sample in raw_samples:
            prompt, target = format_sample(sample, task_type=task_type, template_path=self.prompt_template)
            self.samples.append(_metadata(sample, prompt, target))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, prompt: str, target: str) -> dict[str, Any]:
        if self.tokenizer is None:
            return {
                "prompt": prompt,
                "target": target,
                "input_char_len": len(prompt),
                "output_char_len": len(target),
                "approx_token_len": max(1, (len(prompt) + len(target)) // 4),
            }
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            target_ids = target_ids + [eos]
        input_ids = prompt_ids + target_ids
        labels = input_ids[:] if self.train_on_inputs else ([-100] * len(prompt_ids) + target_ids)
        input_ids = input_ids[-self.max_seq_len :]
        labels = labels[-self.max_seq_len :]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        encoded = self._encode(sample.prompt, sample.target)
        encoded["metadata"] = {
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "task": sample.task,
            "candidate_pool_setting": sample.candidate_pool_setting,
            "text_missing": sample.text_missing,
            "text_fallback_used": sample.text_fallback_used,
        }
        return encoded

    def sample_prompts(self, n: int = 5) -> list[dict[str, Any]]:
        rows = []
        for sample in self.samples[:n]:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "domain": sample.domain,
                    "task": sample.task,
                    "candidate_pool_setting": sample.candidate_pool_setting,
                    "prompt": sample.prompt,
                    "target": sample.target,
                    "text_missing": sample.text_missing,
                    "text_fallback_used": sample.text_fallback_used,
                }
            )
        return rows

    def stats(self, split: str, tokenizer_available: bool) -> dict[str, Any]:
        input_chars = [len(s.prompt) for s in self.samples]
        output_chars = [len(s.target) for s in self.samples]
        if self.tokenizer is None:
            token_lens = [max(1, (len(s.prompt) + len(s.target)) // 4) for s in self.samples]
        else:
            token_lens = [len(self._encode(s.prompt, s.target)["input_ids"]) for s in self.samples]
        n = len(self.samples)
        return {
            "domain": self.samples[0].domain if self.samples else "",
            "task_type": self.task_type,
            "split": split,
            "num_samples": n,
            "avg_input_chars": mean(input_chars) if input_chars else 0,
            "avg_output_chars": mean(output_chars) if output_chars else 0,
            "avg_token_len": mean(token_lens) if token_lens else 0,
            "max_token_len": max(token_lens) if token_lens else 0,
            "text_missing_rate": sum(1 for s in self.samples if s.text_missing) / n if n else 0,
            "text_fallback_used_rate": sum(1 for s in self.samples if s.text_fallback_used) / n if n else 0,
            "tokenizer_available": tokenizer_available,
        }

