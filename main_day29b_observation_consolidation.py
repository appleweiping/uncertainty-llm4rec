"""Day29b local-only observation consolidation.

This script reads existing diagnostic/calibration outputs and produces
observation tables for the paper motivation: raw LLM confidence/relevance is
informative but miscalibrated. It intentionally does not call any API or train
any model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SUMMARY = ROOT / "output-repaired" / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

MODELS = {
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "glm": "GLM",
    "doubao": "Doubao",
    "kimi": "Kimi",
}

DOMAIN_LABELS = {
    "beauty": "Beauty",
    "books_small": "Books-small",
    "electronics_small": "Electronics-small",
    "movies_small": "Movies-small",
}


def safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_domain_model(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    exp = None
    for part in parts:
        lower = part.lower()
        if lower.startswith(("beauty_", "books_small_", "electronics_small_", "movies_small_")):
            exp = lower
            break
    if exp is None:
        return None, None

    domain = None
    for prefix, label in DOMAIN_LABELS.items():
        if exp.startswith(prefix + "_"):
            domain = label
            model_key = exp[len(prefix) + 1 :].split("_")[0]
            return domain, MODELS.get(model_key)
    return None, None


def correctness_from_record(record: dict[str, Any]) -> bool | None:
    label = record.get("label")
    recommend = str(record.get("recommend", "")).strip().lower()
    if label is None or recommend not in {"yes", "no", "true", "false", "1", "0"}:
        return None
    pred = 1 if recommend in {"yes", "true", "1"} else 0
    try:
        return pred == int(label)
    except (TypeError, ValueError):
        return None


def raw_prediction_stats(exp_dir: Path) -> dict[str, float | None]:
    path = exp_dir / "predictions" / "test_raw.jsonl"
    if not path.exists():
        return {
            "std_confidence": None,
            "high_conf_error_rate": None,
            "prediction_notes": "test_raw.jsonl missing; std/high-conf error unavailable",
        }

    confidences: list[float] = []
    high_conf_correct: list[bool] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            conf = safe_float(record.get("confidence"))
            if conf is None:
                continue
            confidences.append(conf)
            correct = correctness_from_record(record)
            if conf >= 0.8 and correct is not None:
                high_conf_correct.append(correct)

    std_conf = float(pd.Series(confidences).std(ddof=0)) if confidences else None
    high_conf_error = (
        float(1.0 - pd.Series(high_conf_correct).mean()) if high_conf_correct else None
    )
    return {
        "std_confidence": std_conf,
        "high_conf_error_rate": high_conf_error,
        "prediction_notes": "computed from predictions/test_raw.jsonl",
    }


def collect_diagnostic_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"outputs[\\/](beauty|books_small|electronics_small|movies_small)_"
        r"(deepseek|qwen|glm|doubao|kimi)[\\/]tables[\\/]diagnostic_metrics\.csv$",
        re.IGNORECASE,
    )

    for path in sorted(OUTPUTS.rglob("diagnostic_metrics.csv")):
        rel = str(path.relative_to(ROOT))
        if not pattern.search(rel):
            continue
        domain, model = infer_domain_model(path)
        if not domain or not model:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive reporting
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "setting": "raw_confidence_diagnostic",
                    "source_file": rel,
                    "notes": f"failed to read csv: {exc}",
                }
            )
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        exp_dir = path.parents[1]
        pred_stats = raw_prediction_stats(exp_dir)
        rows.append(
            {
                "model": model,
                "domain": domain,
                "setting": "raw_confidence_diagnostic",
                "raw_diagnostic_ece": safe_float(row.get("ece")),
                "raw_brier": safe_float(row.get("brier_score")),
                "raw_auroc": safe_float(row.get("auroc")),
                "high_conf_error_rate": pred_stats["high_conf_error_rate"],
                "mean_confidence": safe_float(row.get("avg_confidence")),
                "std_confidence": pred_stats["std_confidence"],
                "num_samples": safe_float(row.get("num_samples")),
                "source_file": rel,
                "notes": pred_stats["prediction_notes"],
            }
        )
    return pd.DataFrame(rows)


def collect_calibration_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"outputs[\\/](beauty|books_small|electronics_small|movies_small)_"
        r"(deepseek|qwen|glm|doubao|kimi)[\\/]tables[\\/]calibration_comparison\.csv$",
        re.IGNORECASE,
    )

    for path in sorted(OUTPUTS.rglob("calibration_comparison.csv")):
        rel = str(path.relative_to(ROOT))
        if not pattern.search(rel):
            continue
        domain, model = infer_domain_model(path)
        if not domain or not model:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "source_file": rel,
                    "notes": f"failed to read csv: {exc}",
                }
            )
            continue
        test = df[df["split"].astype(str).str.lower() == "test"] if "split" in df else df
        metric_values: dict[str, tuple[float | None, float | None]] = {}
        for metric in ["ece", "brier_score", "auroc"]:
            metric_df = test[test["metric"].astype(str).str.lower() == metric]
            if metric_df.empty:
                metric_values[metric] = (None, None)
                continue
            record = metric_df.iloc[0]
            metric_values[metric] = (safe_float(record.get("before")), safe_float(record.get("after")))

        raw_ece, calibrated_ece = metric_values["ece"]
        raw_brier, calibrated_brier = metric_values["brier_score"]
        raw_auroc, calibrated_auroc = metric_values["auroc"]
        rows.append(
            {
                "model": model,
                "domain": domain,
                "raw_ece": raw_ece,
                "calibrated_ece": calibrated_ece,
                "delta_ece": None if raw_ece is None or calibrated_ece is None else calibrated_ece - raw_ece,
                "raw_brier": raw_brier,
                "calibrated_brier": calibrated_brier,
                "delta_brier": None
                if raw_brier is None or calibrated_brier is None
                else calibrated_brier - raw_brier,
                "raw_auroc": raw_auroc,
                "calibrated_auroc": calibrated_auroc,
                "delta_auroc": None
                if raw_auroc is None or calibrated_auroc is None
                else calibrated_auroc - raw_auroc,
                "calibration_method": "valid_set_calibration",
                "valid_fit_test_eval_protocol": "fit calibration on valid split; evaluate reported values on test split",
                "source_file": rel,
                "notes": "delta = calibrated - raw; negative ECE/Brier means improvement",
            }
        )
    return pd.DataFrame(rows)


def collect_relevance_rows() -> pd.DataFrame:
    path = SUMMARY / "beauty_day9_relevance_evidence_full_calibration_comparison.csv"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "score_type": "missing",
                    "ECE": None,
                    "Brier": None,
                    "AUROC": None,
                    "high_conf_error_rate": None,
                    "notes": f"source missing: {path}",
                }
            ]
        )

    df = pd.read_csv(path)
    test = df[df["split"].astype(str).str.lower() == "test"].copy()
    desired = [
        "raw_relevance_probability",
        "calibrated_relevance_probability",
        "evidence_posterior_relevance_minimal",
        "evidence_posterior_relevance_full",
    ]
    for score_type in desired:
        subset = test[test["variant"].astype(str) == score_type] if "variant" in test else pd.DataFrame()
        if subset.empty:
            rows.append(
                {
                    "score_type": score_type,
                    "ECE": None,
                    "Brier": None,
                    "AUROC": None,
                    "high_conf_error_rate": None,
                    "notes": f"not found in {path.relative_to(ROOT)}",
                }
            )
            continue
        record = subset.iloc[0]
        rows.append(
            {
                "score_type": score_type,
                "ECE": safe_float(record.get("ece")),
                "Brier": safe_float(record.get("brier_score")),
                "AUROC": safe_float(record.get("auroc")),
                "high_conf_error_rate": safe_float(record.get("high_conf_error_rate")),
                "notes": f"test split from {path.relative_to(ROOT)}; valid split was used for fitting",
            }
        )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_observation_report(raw_df: pd.DataFrame, cal_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
    beauty = raw_df[raw_df["domain"] == "Beauty"].sort_values("model")
    beauty_lines = [
        f"- {row.model}: diagnostic ECE={fmt(row.raw_diagnostic_ece)}, "
        f"Brier={fmt(row.raw_brier)}, AUROC={fmt(row.raw_auroc)}, "
        f"mean confidence={fmt(row.mean_confidence)}"
        for row in beauty.itertuples(index=False)
    ]

    raw_mean_ece = beauty["raw_diagnostic_ece"].mean()
    cal_test = cal_df[cal_df["domain"] == "Beauty"]
    ece_improvements = (cal_test["raw_ece"] - cal_test["calibrated_ece"]).dropna()
    brier_improvements = (cal_test["raw_brier"] - cal_test["calibrated_brier"]).dropna()
    auroc_delta = cal_test["delta_auroc"].dropna()

    rel_lookup = {row.score_type: row for row in rel_df.itertuples(index=False)}
    raw_rel = rel_lookup.get("raw_relevance_probability")
    cal_rel = rel_lookup.get("calibrated_relevance_probability")
    full_rel = rel_lookup.get("evidence_posterior_relevance_full")

    report = f"""# Day29b Observation: Raw LLM Confidence Is Informative but Miscalibrated

## 1. Motivation

The starting question is whether an LLM's verbalized confidence in a recommendation task can be used directly as a decision signal. The consolidated evidence says no: the signal is useful, but its probability scale is not reliable enough to use without calibration.

## 2. Raw Confidence Is Informative but Miscalibrated

Raw confidence is not pure noise. Across Beauty diagnostics, AUROC is often meaningfully above chance, indicating that confidence has a relationship with correctness. At the same time, ECE, Brier score, and high-confidence error behavior show that the raw confidence values should not be interpreted as calibrated probabilities.

## 3. Multi-Model Evidence

Beauty raw confidence diagnostics across five LLMs:

{chr(10).join(beauty_lines)}

Mean Beauty diagnostic ECE across the five model runs is {fmt(raw_mean_ece)}. This supports the observation that miscalibration is not a single-model artifact. The broader output table also includes Books-small, Electronics-small, and Movies-small where available.

## 4. Calibration Helps but Does Not Create Ranking Ability

Valid-set calibration usually reduces probability-scale error. On Beauty, the mean ECE reduction across the five model runs is {fmt(ece_improvements.mean())}, and the mean Brier reduction is {fmt(brier_improvements.mean())}. AUROC changes are much smaller on average ({fmt(auroc_delta.mean())}), which is expected: calibration mainly repairs probability scale rather than training a new ranker.

This distinction matters for the paper framing. CEP should not be described as a standalone ranking model. Its core role is to turn informative but miscalibrated LLM signals into calibrated posterior scores that can support downstream decisions.

## 5. Relevance Setting Evidence

The same phenomenon appears after moving from yes/no confidence to candidate relevance scoring. On full Beauty candidate relevance evidence, raw relevance probability has ECE={fmt(getattr(raw_rel, 'ECE', None))}, Brier={fmt(getattr(raw_rel, 'Brier', None))}, and AUROC={fmt(getattr(raw_rel, 'AUROC', None))}. Calibrated relevance probability reduces this to ECE={fmt(getattr(cal_rel, 'ECE', None))}, Brier={fmt(getattr(cal_rel, 'Brier', None))}, and AUROC={fmt(getattr(cal_rel, 'AUROC', None))}. The full evidence posterior variant reports ECE={fmt(getattr(full_rel, 'ECE', None))}, Brier={fmt(getattr(full_rel, 'Brier', None))}, and AUROC={fmt(getattr(full_rel, 'AUROC', None))}.

## 6. Why Scheme 4 / CEP Is Needed

The consolidated conclusion is: raw LLM confidence or relevance signal is informative but unreliable as a probability. Therefore, before using it in recommendation decisions, we need an evidence-grounded calibrated posterior. Scheme 4 / CEP supplies that bridge by combining relevance probability, positive/negative evidence, ambiguity, missing information, and valid-set calibration.

## 7. Connection to External Backbone Plug-In

The later Day20/Day23/Day25 results should be read through this lens. The method does not replace SASRec, GRU4Rec, or Bert4Rec with raw LLM confidence. Instead, it plugs a calibrated relevance posterior into external sequential backbones, with evidence risk acting as a secondary regularizer.

## Local-Only Execution Note

This Day29b consolidation used only existing files under `outputs/` and `output-repaired/summary/`. It did not call DeepSeek, did not train a backbone, did not change prompts/parsers/formulas, and did not touch the running Day29 Movies inference process.
"""
    (SUMMARY / "day29b_observation_raw_llm_confidence_miscalibration_report.md").write_text(
        report, encoding="utf-8"
    )


def write_paper_snippet(raw_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
    beauty = raw_df[raw_df["domain"] == "Beauty"]
    ece_values = ", ".join(
        f"{row.model}: {fmt(row.raw_diagnostic_ece)}" for row in beauty.sort_values("model").itertuples(index=False)
    )
    raw_rel = rel_df[rel_df["score_type"] == "raw_relevance_probability"].iloc[0]
    cal_rel = rel_df[rel_df["score_type"] == "calibrated_relevance_probability"].iloc[0]

    snippet = f"""# Paper Motivation Snippet

The starting question is whether LLM verbalized confidence in recommendation can be directly used as a decision signal. Our multi-model diagnostics show that this is not the case. On Beauty, raw confidence is informative but substantially miscalibrated across multiple LLMs, with diagnostic ECE values of {ece_values}. This indicates that the signal carries information about correctness, but its numerical scale should not be interpreted as a calibrated probability.

This issue persists when moving from yes/no recommendation confidence to candidate-level relevance probability. In the full Beauty relevance setting, raw relevance probability has ECE={fmt(raw_rel.ECE)} and Brier={fmt(raw_rel.Brier)}, while valid-set calibration reduces these values to ECE={fmt(cal_rel.ECE)} and Brier={fmt(cal_rel.Brier)}. The improvement is mainly a probability-quality repair rather than a guarantee of standalone ranking superiority, since calibration does not fundamentally retrain the ranker.

These observations motivate CEP: instead of directly using raw LLM confidence or relevance scores, we convert evidence-grounded LLM outputs into a calibrated relevance posterior before downstream decision making. In later plug-in experiments, this calibrated posterior is combined with external sequential recommendation backbones rather than replacing them.
"""
    (SUMMARY / "day29b_paper_motivation_snippet.md").write_text(snippet, encoding="utf-8")


def main() -> None:
    raw_df = collect_diagnostic_rows()
    cal_df = collect_calibration_rows()
    rel_df = collect_relevance_rows()

    raw_out = SUMMARY / "day29b_beauty_multimodel_raw_confidence_diagnostics.csv"
    cal_out = SUMMARY / "day29b_beauty_multimodel_calibration_effect.csv"
    rel_out = SUMMARY / "day29b_beauty_relevance_probability_diagnostics.csv"

    raw_df.to_csv(raw_out, index=False)
    cal_df.to_csv(cal_out, index=False)
    rel_df.to_csv(rel_out, index=False)

    write_observation_report(raw_df, cal_df, rel_df)
    write_paper_snippet(raw_df, rel_df)

    print(f"Wrote {raw_out.relative_to(ROOT)} ({len(raw_df)} rows)")
    print(f"Wrote {cal_out.relative_to(ROOT)} ({len(cal_df)} rows)")
    print(f"Wrote {rel_out.relative_to(ROOT)} ({len(rel_df)} rows)")
    print("Wrote Day29b observation report and paper motivation snippet")


if __name__ == "__main__":
    main()
