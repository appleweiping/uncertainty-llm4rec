"""Export R3b conservative-gate summary tables from completed run directories."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import read_jsonl

RUN_PREFIX = "r3b_movielens_1m_conservative_gate_cache_replay_"
HIGH_CONF = 0.7
LOW_CONF = 0.5


def _parse_run_dir(name: str) -> tuple[str, int] | None:
    if not name.startswith(RUN_PREFIX) or "_seed" not in name:
        return None
    head, seed_part = name.rsplit("_seed", 1)
    try:
        seed = int(seed_part)
    except ValueError:
        return None
    method = head[len(RUN_PREFIX) :]
    if not method:
        return None
    return method, seed


def _dedupe_top10(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = str(item)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 10:
            break
    return out


def _is_correct(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    if isinstance(meta.get("is_grounded_hit"), bool):
        return bool(meta["is_grounded_hit"])
    pred = [str(x) for x in row.get("predicted_items") or []]
    return bool(pred and pred[0] == str(row.get("target_item")))


def _confidence(row: dict[str, Any]) -> float | None:
    meta = row.get("metadata") or {}
    c = meta.get("confidence")
    if isinstance(c, (int, float)):
        return float(c)
    return None


def _decision(row: dict[str, Any]) -> str | None:
    meta = row.get("metadata") or {}
    d = meta.get("uncertainty_decision")
    return str(d) if d else None


def _example_key(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    eid = meta.get("example_id")
    if eid:
        return str(eid)
    return f"{row.get('user_id')}:{row.get('target_item')}"


def _metrics_flat(metrics: dict[str, Any]) -> dict[str, Any]:
    agg = metrics.get("aggregate") or {}
    conf = agg.get("confidence") or {}
    cal = agg.get("calibration") or {}
    out = {
        "recall@10": float(agg.get("recall@10") or 0.0),
        "ndcg@10": float(agg.get("ndcg@10") or 0.0),
        "mrr@10": float(agg.get("mrr@10") or 0.0),
        "validity": float(agg.get("validity_rate") or 0.0),
        "hallucination": float(agg.get("hallucination_rate") or 0.0),
        "parse_success": float(agg.get("parse_success_rate") or 0.0),
        "grounding_success": float(agg.get("grounding_success_rate") or 0.0),
        "mean_confidence": float(conf.get("mean_confidence") or 0.0) if conf.get("mean_confidence") is not None else 0.0,
        "ece": float(cal.get("ece") or 0.0),
        "brier": float(cal.get("brier_score") or 0.0),
    }
    return out


def _prediction_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high_wrong = 0
    low_correct = 0
    decisions: dict[str, int] = defaultdict(int)
    for row in rows:
        conf = _confidence(row)
        correct = _is_correct(row)
        if conf is not None and conf >= HIGH_CONF and not correct:
            high_wrong += 1
        if conf is not None and conf < LOW_CONF and correct:
            low_correct += 1
        d = _decision(row)
        if d:
            decisions[d] += 1
    return {
        "high_confidence_wrong_count": high_wrong,
        "low_confidence_correct_count": low_correct,
        "decisions": dict(decisions),
    }


def _load_cost_latency(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "cost_latency.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def discover_r3b_runs(runs_root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    if not runs_root.is_dir():
        return found
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_run_dir(child.name)
        if parsed is None:
            continue
        method, seed = parsed
        found[(method, seed)] = child
    return found


def _is_r3b_run_complete(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").is_file() and (run_dir / "predictions.jsonl").is_file()


def export_r3b_tables(*, runs_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_index = discover_r3b_runs(runs_dir)
    if not runs_index:
        raise FileNotFoundError(
            f"no R3b run directories matching {RUN_PREFIX!r} under {runs_dir}. "
            "Run scripts/run_all.py with configs/experiments/r3b_movielens_1m_conservative_gate_cache_replay.yaml first."
        )

    skipped_incomplete: list[str] = []
    complete_index: dict[tuple[str, int], Path] = {}
    for key, run_dir in runs_index.items():
        if _is_r3b_run_complete(run_dir):
            complete_index[key] = run_dir
        else:
            skipped_incomplete.append(str(run_dir))

    if not complete_index:
        raise FileNotFoundError(
            "no complete R3b runs (each needs metrics.json and predictions.jsonl). "
            f"Incomplete directories skipped ({len(skipped_incomplete)}): {skipped_incomplete[:20]}"
            + (" ..." if len(skipped_incomplete) > 20 else "")
        )

    by_method: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for (method, seed), path in complete_index.items():
        by_method[method].append((seed, path))

    # Per (method, seed) scalars and predictions
    per_run: dict[tuple[str, int], dict[str, Any]] = {}
    predictions_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (method, seed), run_dir in complete_index.items():
        mpath = run_dir / "metrics.json"
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
        preds = read_jsonl(run_dir / "predictions.jsonl")
        flat = _metrics_flat(metrics)
        pstats = _prediction_stats(preds)
        flat.update(
            {
                "high_confidence_wrong_count": pstats["high_confidence_wrong_count"],
                "low_confidence_correct_count": pstats["low_confidence_correct_count"],
                **_load_cost_latency(run_dir),
            }
        )
        per_run[(method, seed)] = flat
        predictions_cache[(method, seed)] = preds

    def mean_std(method: str, key: str) -> tuple[float, float]:
        vals: list[float] = []
        for s, _ in sorted(by_method[method]):
            tup = (method, s)
            if tup not in per_run:
                continue
            raw = per_run[tup].get(key)
            if raw is None or raw == "":
                continue
            vals.append(float(raw))
        if not vals:
            return 0.0, 0.0
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    methods = sorted(by_method.keys())
    fallback_recall_by_seed: dict[int, float] = {}
    for seed, run_dir in by_method.get("bm25", []):
        fallback_recall_by_seed[seed] = float(per_run[("bm25", seed)]["recall@10"])

    # top-10 identical vs bm25 for each (method, seed) where method uses LLM routing
    identical_rate: dict[tuple[str, int], float] = {}
    for method in methods:
        if method in {"bm25", "llm_generative_real"}:
            continue
        for seed, _ in by_method[method]:
            ours_rows = predictions_cache.get((method, seed), [])
            base_rows = predictions_cache.get(("bm25", seed), [])
            if not ours_rows or not base_rows:
                identical_rate[(method, seed)] = 0.0
                continue
            base_map = {_example_key(r): _dedupe_top10([str(x) for x in r.get("predicted_items") or []]) for r in base_rows}
            same = 0
            total = 0
            for row in ours_rows:
                key = _example_key(row)
                if key not in base_map:
                    continue
                total += 1
                o10 = _dedupe_top10([str(x) for x in row.get("predicted_items") or []])
                if o10 == base_map[key]:
                    same += 1
            identical_rate[(method, seed)] = (same / total) if total else 0.0

    main_rows: list[dict[str, Any]] = []
    for method in methods:
        m_recall, s_recall = mean_std(method, "recall@10")
        m_ndcg, s_ndcg = mean_std(method, "ndcg@10")
        m_mrr, s_mrr = mean_std(method, "mrr@10")
        delta_vals = [
            per_run[(method, s)]["recall@10"] - fallback_recall_by_seed[s]
            for s, _ in sorted(by_method[method])
            if s in fallback_recall_by_seed and (method, s) in per_run
        ]
        delta_mean = statistics.mean(delta_vals) if delta_vals else 0.0
        id_rates = [identical_rate.get((method, s), 0.0) for s, _ in sorted(by_method[method])]
        id_mean = statistics.mean(id_rates) if id_rates else 0.0
        row = {
            "method": method,
            "seeds": ",".join(str(s) for s, _ in sorted(by_method[method])),
            "recall@10_mean": round(m_recall, 6),
            "recall@10_std": round(s_recall, 6),
            "ndcg@10_mean": round(m_ndcg, 6),
            "ndcg@10_std": round(s_ndcg, 6),
            "mrr@10_mean": round(m_mrr, 6),
            "mrr@10_std": round(s_mrr, 6),
            "validity_mean": round(mean_std(method, "validity")[0], 6),
            "hallucination_mean": round(mean_std(method, "hallucination")[0], 6),
            "parse_success_mean": round(mean_std(method, "parse_success")[0], 6),
            "grounding_success_mean": round(mean_std(method, "grounding_success")[0], 6),
            "mean_confidence_mean": round(mean_std(method, "mean_confidence")[0], 6),
            "ece_mean": round(mean_std(method, "ece")[0], 6),
            "brier_mean": round(mean_std(method, "brier")[0], 6),
            "high_confidence_wrong_count_mean": round(mean_std(method, "high_confidence_wrong_count")[0], 2),
            "low_confidence_correct_count_mean": round(mean_std(method, "low_confidence_correct_count")[0], 2),
            "delta_recall@10_vs_bm25_mean": round(delta_mean, 6),
            "top10_identical_to_bm25_rate_mean": round(id_mean, 6),
            "cache_hit_requests_mean": round(mean_std(method, "cache_hit_requests")[0], 2),
            "live_provider_requests_mean": round(mean_std(method, "live_provider_requests")[0], 2),
            "replay_latency_seconds_sum_mean": round(mean_std(method, "replay_latency_seconds_sum")[0], 4),
            "effective_cost_usd_mean": round(mean_std(method, "effective_cost_usd")[0], 6),
        }
        main_rows.append(row)

    _write_csv(output_dir / "r3b_conservative_gate_main.csv", main_rows)

    ours_like = [
        m
        for m in methods
        if m.startswith("ours_") or m == "llm_generative_real"
    ]
    ablation_rows = [r for r in main_rows if r["method"] in ours_like]
    _write_csv(output_dir / "r3b_conservative_gate_ablation.csv", ablation_rows)

    decision_rows: list[dict[str, Any]] = []
    for method in methods:
        if not method.startswith("ours_"):
            continue
        for seed, _ in sorted(by_method[method]):
            if (method, seed) not in predictions_cache:
                continue
            stats = _prediction_stats(predictions_cache[(method, seed)])
            for dec, cnt in sorted(stats["decisions"].items()):
                decision_rows.append({"method": method, "seed": seed, "decision": dec, "count": cnt})
            if not stats["decisions"]:
                decision_rows.append({"method": method, "seed": seed, "decision": "none", "count": 0})
    _write_csv(output_dir / "r3b_conservative_gate_decision_stats.csv", decision_rows)

    fail_rows: list[dict[str, Any]] = []
    for method in methods:
        for seed, _ in sorted(by_method[method]):
            if (method, seed) not in per_run:
                continue
            preds = predictions_cache[(method, seed)]
            parse_fail = sum(1 for r in preds if not (r.get("metadata") or {}).get("parse_success", True))
            fail_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "parse_failures": parse_fail,
                    "high_confidence_wrong": per_run[(method, seed)]["high_confidence_wrong_count"],
                    "low_confidence_correct": per_run[(method, seed)]["low_confidence_correct_count"],
                    "hallucination_rate": per_run[(method, seed)]["hallucination"],
                }
            )
    _write_csv(output_dir / "r3b_observation_failures.csv", fail_rows)

    return {
        "run_count_discovered": len(runs_index),
        "run_count_complete": len(complete_index),
        "skipped_incomplete_runs": skipped_incomplete,
        "outputs": {
            "main": str(output_dir / "r3b_conservative_gate_main.csv"),
            "ablation": str(output_dir / "r3b_conservative_gate_ablation.csv"),
            "decisions": str(output_dir / "r3b_conservative_gate_decision_stats.csv"),
            "failures": str(output_dir / "r3b_observation_failures.csv"),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
