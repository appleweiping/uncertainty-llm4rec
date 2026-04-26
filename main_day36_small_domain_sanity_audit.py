"""Day36 small-domain cross-domain sanity audit.

This script is intentionally local-only. It inventories existing small-domain
processed data and previous outputs, diagnoses cold-candidate rates, checks
metric readiness, prepares config templates, and writes a route plan.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(".")
SUMMARY_DIR = Path("output-repaired/summary")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = {
    "books": [
        Path("data/processed/books_small"),
        Path("data/processed/amazon_books_small"),
    ],
    "electronics": [
        Path("data/processed/electronics_small"),
        Path("data/processed/amazon_electronics_small"),
    ],
    "movies": [
        Path("data/processed/movies_small"),
        Path("data/processed/amazon_movies_small"),
    ],
}

MODEL_NAMES = ["deepseek", "qwen", "glm", "doubao", "kimi"]
PRIMARY_METRICS = "NDCG@10;MRR;HR@1;HR@3;NDCG@3;NDCG@5"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_item(row: dict) -> str:
    return str(row.get("candidate_item_id", "")).strip()


ITEM_RE = re.compile(r"Item ID:\s*([A-Za-z0-9._:-]+)")


def history_items(row: dict) -> list[str]:
    out: list[str] = []
    history = row.get("history") or []
    if not isinstance(history, list):
        return out
    for item in history:
        text = str(item).strip()
        match = ITEM_RE.search(text)
        out.append(match.group(1).strip() if match else text)
    return [x for x in out if x]


def pick_domain_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if (path / "train.jsonl").exists() and (path / "valid.jsonl").exists() and (path / "test.jsonl").exists():
            return path
    for path in paths:
        if path.exists():
            return path
    return None


def detect_outputs(domain: str) -> tuple[bool, bool, str]:
    candidates = []
    for root in [Path("outputs"), Path("output-repaired")]:
        if root.exists():
            candidates.extend([p for p in root.rglob(f"*{domain}*small*")])
            candidates.extend([p for p in root.rglob(f"*{domain}_small*")])
    has_llm = False
    has_relevance = False
    notes = []
    for path in candidates:
        text = str(path).lower()
        if any(model in text for model in MODEL_NAMES):
            if path.is_dir() or path.name in {"diagnostic_metrics.csv", "calibration_comparison.csv"}:
                has_llm = True
        if "relevance" in text and (path.suffix in {".jsonl", ".csv", ".md"} or path.is_dir()):
            has_relevance = True
    for model in MODEL_NAMES:
        if Path(f"outputs/{domain}_small_{model}").exists():
            has_llm = True
            notes.append(f"outputs/{domain}_small_{model}")
    return has_llm, has_relevance, "; ".join(notes[:5])


def split_stats(rows: list[dict]) -> dict:
    users: dict[str, int] = defaultdict(int)
    items = set()
    positives = 0
    for row in rows:
        user = str(row.get("user_id", "")).strip()
        if user:
            users[user] += 1
        item = candidate_item(row)
        if item:
            items.add(item)
        if int(row.get("label", 0)) == 1:
            positives += 1
    counts = list(users.values())
    return {
        "rows": len(rows),
        "num_users": len(users),
        "num_items": len(items),
        "positive_rows": positives,
        "negative_rows": len(rows) - positives,
        "candidate_pool_size_mean": mean(counts) if counts else 0,
        "candidate_pool_size_min": min(counts) if counts else 0,
        "candidate_pool_size_max": max(counts) if counts else 0,
    }


def schema_flags(rows: list[dict]) -> dict:
    sample = rows[0] if rows else {}
    return {
        "has_history": "history" in sample,
        "has_candidate_item_id": "candidate_item_id" in sample,
        "has_candidate_text": "candidate_text" in sample,
        "has_label": "label" in sample,
    }


def vocab_from_train(train_rows: list[dict]) -> dict[str, set[str]]:
    candidate_vocab = {candidate_item(row) for row in train_rows if candidate_item(row)}
    history_vocab: set[str] = set()
    for row in train_rows:
        history_vocab.update(history_items(row))
    return {
        "train_candidate_vocab": candidate_vocab,
        "train_history_vocab": history_vocab,
        "train_backbone_vocab": candidate_vocab | history_vocab,
    }


def cold_rate_rows(domain: str, split: str, rows: list[dict], vocabs: dict[str, set[str]]) -> list[dict]:
    out = []
    for name, vocab in vocabs.items():
        positive = [row for row in rows if int(row.get("label", 0)) == 1]
        negative = [row for row in rows if int(row.get("label", 0)) == 0]
        pos_items = {candidate_item(row) for row in positive if candidate_item(row)}
        neg_items = {candidate_item(row) for row in negative if candidate_item(row)}
        all_items = {candidate_item(row) for row in rows if candidate_item(row)}
        pos_cold = [row for row in positive if candidate_item(row) not in vocab]
        neg_cold = [row for row in negative if candidate_item(row) not in vocab]
        all_cold = [row for row in rows if candidate_item(row) not in vocab]
        out.append(
            {
                "domain": domain,
                "split": split,
                "vocab_definition": name,
                "train_vocab_size": len(vocab),
                "num_rows": len(rows),
                "num_positive_rows": len(positive),
                "num_negative_rows": len(negative),
                "positive_cold_rows": len(pos_cold),
                "positive_cold_rate": len(pos_cold) / len(positive) if positive else "",
                "negative_cold_rows": len(neg_cold),
                "negative_cold_rate": len(neg_cold) / len(negative) if negative else "",
                "all_candidate_cold_rows": len(all_cold),
                "all_candidate_cold_rate": len(all_cold) / len(rows) if rows else "",
                "unique_candidate_items": len(all_items),
                "unique_cold_candidate_items": len({candidate_item(row) for row in all_cold}),
                "unique_positive_items": len(pos_items),
                "unique_positive_cold_items": len({candidate_item(row) for row in pos_cold}),
                "unique_negative_items": len(neg_items),
                "unique_negative_cold_items": len({candidate_item(row) for row in neg_cold}),
            }
        )
    return out


def yaml_text_for_exp(domain: str, path: Path) -> str:
    return f"""exp_name: {domain}_small_deepseek_relevance_evidence
domain: {domain}
train_input_path: {path.as_posix()}/train.jsonl
split_input_paths:
  valid: {path.as_posix()}/valid.jsonl
  test: {path.as_posix()}/test.jsonl
prompt_path: prompts/candidate_relevance_evidence.txt
output_root: output-repaired
output_dir: output-repaired/{domain}_small_deepseek_relevance_evidence
model_config: configs/model/deepseek.yaml
output_schema: relevance_evidence
method_variant: candidate_relevance_evidence_small_sanity
resume: true
concurrent: true
max_workers: 4
requests_per_minute: 120
max_retries: 3
retry_backoff_seconds: 2.0
checkpoint_every: 1
max_samples: null
notes: Day36 template only. Do not start API until Day37 decision.
"""


def yaml_text_for_backbone(domain: str, path: Path) -> str:
    return f"""backbone_name: sasrec
domain: {domain}
stage: small_cross_domain_sanity
train_input_path: {path.as_posix()}/train.jsonl
valid_input_path: {path.as_posix()}/valid.jsonl
test_input_path: {path.as_posix()}/test.jsonl
score_output_path: output-repaired/backbone/sasrec_{domain}_small/candidate_scores.csv
evidence_table:
  path: output-repaired/{domain}_small_deepseek_relevance_evidence/calibrated/relevance_evidence_posterior_test.jsonl
  join_keys:
    - user_id
    - candidate_item_id
  fields:
    - relevance_probability
    - calibrated_relevance_probability
    - evidence_risk
    - ambiguity
    - missing_information
    - abs_evidence_margin
    - positive_evidence
    - negative_evidence
rerank:
  top_k: 10
  normalizations:
    - minmax
    - zscore
  lambdas:
    - 0.0
    - 0.05
    - 0.1
    - 0.2
    - 0.5
  alphas:
    - 0.5
    - 0.75
    - 0.9
  metric_note: small domains are sanity settings. If candidate pool <= 10, HR@10 is trivial and not a primary claim.
"""


def main() -> None:
    inventory_rows: list[dict] = []
    cold_rows: list[dict] = []
    metric_rows: list[dict] = []
    cep_rows: list[dict] = []
    domain_summaries: dict[str, dict] = {}

    for domain, paths in DOMAINS.items():
        path = pick_domain_path(paths)
        if not path:
            inventory_rows.append({"domain": domain, "path": "", "notes": "no small processed path found"})
            continue

        train = read_jsonl(path / "train.jsonl")
        valid = read_jsonl(path / "valid.jsonl")
        test = read_jsonl(path / "test.jsonl")
        valid_stats = split_stats(valid)
        test_stats = split_stats(test)
        flags = schema_flags(test or valid or train)
        has_llm, has_rel, output_notes = detect_outputs(domain)
        candidate_mean = test_stats["candidate_pool_size_mean"] or valid_stats["candidate_pool_size_mean"]
        inventory_rows.append(
            {
                "domain": domain,
                "path": path.as_posix(),
                "has_train_jsonl": bool(train),
                "has_valid_jsonl": bool(valid),
                "has_test_jsonl": bool(test),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "num_users_valid": valid_stats["num_users"],
                "num_users_test": test_stats["num_users"],
                "avg_candidates_per_user": candidate_mean,
                "candidate_pool_size_mean": candidate_mean,
                "candidate_pool_size_min": test_stats["candidate_pool_size_min"] or valid_stats["candidate_pool_size_min"],
                "candidate_pool_size_max": test_stats["candidate_pool_size_max"] or valid_stats["candidate_pool_size_max"],
                "has_history": flags["has_history"],
                "has_candidate_item_id": flags["has_candidate_item_id"],
                "has_candidate_text": flags["has_candidate_text"],
                "has_label": flags["has_label"],
                "has_existing_llm_diagnostics": has_llm,
                "has_existing_relevance_evidence": has_rel,
                "notes": output_notes,
            }
        )

        vocabs = vocab_from_train(train)
        for split_name, rows in [("valid", valid), ("test", test)]:
            cold_rows.extend(cold_rate_rows(domain, split_name, rows, vocabs))

        hr10_trivial = (test_stats["candidate_pool_size_max"] <= 10) or (candidate_mean <= 10)
        metric_rows.append(
            {
                "domain": domain,
                "candidate_pool_size_mean": candidate_mean,
                "hr10_trivial_flag": hr10_trivial,
                "recommended_primary_metrics": PRIMARY_METRICS,
            }
        )

        test_backbone = [row for row in cold_rows if row["domain"] == domain and row["split"] == "test" and row["vocab_definition"] == "train_backbone_vocab"][0]
        valid_backbone = [row for row in cold_rows if row["domain"] == domain and row["split"] == "valid" and row["vocab_definition"] == "train_backbone_vocab"][0]
        positive_cold = max(float(valid_backbone["positive_cold_rate"]), float(test_backbone["positive_cold_rate"]))
        all_cold = max(float(valid_backbone["all_candidate_cold_rate"]), float(test_backbone["all_candidate_cold_rate"]))
        if has_rel:
            action = "use_existing_outputs_only"
            reason = "existing relevance evidence-like outputs were found; verify schema before reuse"
        elif positive_cold < 0.2 and all_cold < 0.3:
            action = "id_backbone_plugin_ready"
            reason = "small split has comparatively warm candidates and can support ID-backbone sanity after evidence is generated"
        elif len(valid) + len(test) <= 5000:
            action = "run_relevance_evidence_small"
            reason = "small split is low-cost; use as sanity/calibration, not regular-domain main claim"
        else:
            action = "calibration_only"
            reason = "schema exists but cold rate or size makes ID-backbone claim weak"
        cep_rows.append(
            {
                "domain": domain,
                "has_raw_confidence_outputs": has_llm,
                "has_relevance_evidence_outputs": has_rel,
                "valid_rows": len(valid),
                "test_rows": len(test),
                "total_rows_if_run_deepseek": len(valid) + len(test),
                "recommended_action": action,
                "reason": reason,
            }
        )

        domain_summaries[domain] = {
            "path": path,
            "candidate_mean": candidate_mean,
            "hr10_trivial": hr10_trivial,
            "valid_cold": valid_backbone,
            "test_cold": test_backbone,
            "has_llm": has_llm,
            "has_rel": has_rel,
            "action": action,
        }

        if train and valid and test:
            exp_path = Path(f"configs/exp/{domain}_small_deepseek_relevance_evidence.yaml")
            backbone_path = Path(f"configs/external_backbone/{domain}_small_sasrec_plugin.yaml")
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            backbone_path.parent.mkdir(parents=True, exist_ok=True)
            exp_path.write_text(yaml_text_for_exp(domain, path), encoding="utf-8")
            backbone_path.write_text(yaml_text_for_backbone(domain, path), encoding="utf-8")

    write_csv(
        SUMMARY_DIR / "day36_small_domains_inventory.csv",
        inventory_rows,
        [
            "domain",
            "path",
            "has_train_jsonl",
            "has_valid_jsonl",
            "has_test_jsonl",
            "train_rows",
            "valid_rows",
            "test_rows",
            "num_users_valid",
            "num_users_test",
            "avg_candidates_per_user",
            "candidate_pool_size_mean",
            "candidate_pool_size_min",
            "candidate_pool_size_max",
            "has_history",
            "has_candidate_item_id",
            "has_candidate_text",
            "has_label",
            "has_existing_llm_diagnostics",
            "has_existing_relevance_evidence",
            "notes",
        ],
    )
    write_csv(
        SUMMARY_DIR / "day36_small_domains_cold_rate_diagnostics.csv",
        cold_rows,
        [
            "domain",
            "split",
            "vocab_definition",
            "train_vocab_size",
            "num_rows",
            "num_positive_rows",
            "num_negative_rows",
            "positive_cold_rows",
            "positive_cold_rate",
            "negative_cold_rows",
            "negative_cold_rate",
            "all_candidate_cold_rows",
            "all_candidate_cold_rate",
            "unique_candidate_items",
            "unique_cold_candidate_items",
            "unique_positive_items",
            "unique_positive_cold_items",
            "unique_negative_items",
            "unique_negative_cold_items",
        ],
    )
    write_csv(SUMMARY_DIR / "day36_small_domains_metric_readiness.csv", metric_rows, ["domain", "candidate_pool_size_mean", "hr10_trivial_flag", "recommended_primary_metrics"])
    write_csv(
        SUMMARY_DIR / "day36_small_domains_cep_readiness.csv",
        cep_rows,
        ["domain", "has_raw_confidence_outputs", "has_relevance_evidence_outputs", "valid_rows", "test_rows", "total_rows_if_run_deepseek", "recommended_action", "reason"],
    )

    report_lines = [
        "# Day36 Small-Domain Sanity Plan",
        "",
        "## 1. Why Consider Small Domains",
        "",
        "Small domains are useful as lightweight cross-domain sanity / continuity experiments. They should not replace regular medium domains as the realistic cross-domain setting.",
        "",
        "## 2. Small vs Regular Medium",
        "",
        "Regular medium domains preserve more realistic cold-start/content-carrier behavior. Small domains can answer whether the older cross-domain pipeline and ID-based backbone sanity checks are technically feasible under lower cost.",
        "",
        "## 3. Schema / Candidate Pool / Cold-Rate",
        "",
    ]
    cold_report_lines = [
        "# Day36 Small-Domain Cold-Rate Report",
        "",
        "Cold rate is computed with train_candidate_vocab, train_history_vocab, and train_backbone_vocab. The route decision uses train_backbone_vocab.",
        "",
    ]
    for domain, summary in domain_summaries.items():
        valid = summary["valid_cold"]
        test = summary["test_cold"]
        report_lines.append(
            f"- {domain}: path `{summary['path'].as_posix()}`, candidate pool mean `{float(summary['candidate_mean']):.2f}`, "
            f"HR@10 trivial `{summary['hr10_trivial']}`, valid all-cold `{float(valid['all_candidate_cold_rate']):.4f}`, "
            f"test all-cold `{float(test['all_candidate_cold_rate']):.4f}`."
        )
        cold_report_lines.append(
            f"- {domain}: valid pos/neg/all cold `{float(valid['positive_cold_rate']):.4f}` / "
            f"`{float(valid['negative_cold_rate']):.4f}` / `{float(valid['all_candidate_cold_rate']):.4f}`; "
            f"test pos/neg/all cold `{float(test['positive_cold_rate']):.4f}` / `{float(test['negative_cold_rate']):.4f}` / "
            f"`{float(test['all_candidate_cold_rate']):.4f}`."
        )
    report_lines.extend(
        [
            "",
            "## 4. ID-Based Backbone Suitability",
            "",
            "Use small domains for ID-backbone sanity only if cold rates are materially lower than regular medium and candidate coverage is healthy. Otherwise, treat them as calibration sanity rather than backbone evidence.",
            "",
            "## 5. Existing LLM Confidence Observation",
            "",
        ]
    )
    for domain, summary in domain_summaries.items():
        report_lines.append(f"- {domain}: existing raw-confidence diagnostics `{summary['has_llm']}`, existing relevance evidence `{summary['has_rel']}`.")
    report_lines.extend(
        [
            "",
            "## 6. New DeepSeek Relevance Evidence",
            "",
            "No API was launched in Day36. Config templates were prepared only for domains with train/valid/test jsonl.",
            "",
            "## 7. Day37 Recommendation",
            "",
            "If the goal is a low-cost cross-domain continuity check, run one small domain first with DeepSeek relevance evidence and keep the claim as sanity-only. If the goal is realistic cross-domain behavior, continue with regular medium cold-style content carriers rather than forcing ID-based backbones.",
            "",
            "Small domains provide a lightweight cross-domain sanity setting, while regular medium domains provide more realistic cold-start/content-carrier analysis.",
        ]
    )
    cold_report_lines.extend(
        [
            "",
            "If small-domain cold rates are low, they can support ID-based backbone sanity. If they are still cold, they should be used for calibration sanity or content-carrier checks only.",
        ]
    )
    (SUMMARY_DIR / "day36_small_domain_sanity_plan.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (SUMMARY_DIR / "day36_small_domains_cold_rate_report.md").write_text("\n".join(cold_report_lines) + "\n", encoding="utf-8")
    print("Wrote Day36 small-domain audit, readiness tables, config templates, and report.")


if __name__ == "__main__":
    main()
