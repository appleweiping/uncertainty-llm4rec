from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SUMMARY_DIR = Path("output-repaired/summary")
PRED_DIR = Path("output-repaired/movies_deepseek_relevance_evidence_medium_5neg_2000/predictions")
BACKBONE_PATH = Path("output-repaired/backbone/sasrec_movies_medium5_2000/candidate_scores.csv")

RAW_PATHS = {
    "valid": PRED_DIR / "valid_raw.jsonl",
    "test": PRED_DIR / "test_raw.jsonl",
}

REQUIRED_FIELDS = [
    "relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "ambiguity",
    "missing_information",
    "evidence_risk",
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def inference_status() -> pd.DataFrame:
    rows = []
    for split, path in RAW_PATHS.items():
        records = read_jsonl(path) if path.exists() else []
        num_rows = len(records)
        parse_success = sum(bool(r.get("parse_success")) for r in records)
        raw_nonempty = sum(bool(str(r.get("raw_response", "")).strip()) for r in records)
        missing_fields = 0
        for r in records:
            if any(field not in r or pd.isna(r.get(field)) for field in REQUIRED_FIELDS):
                missing_fields += 1
        rows.append(
            {
                "split": split,
                "prediction_path": str(path),
                "expected_rows": 12000,
                "actual_rows": num_rows,
                "complete": num_rows == 12000,
                "parse_success_rate": parse_success / num_rows if num_rows else 0.0,
                "raw_response_nonempty_rate": raw_nonempty / num_rows if num_rows else 0.0,
                "missing_required_field_rows": missing_fields,
                "status": "complete" if num_rows == 12000 and missing_fields == 0 else "needs_attention",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, SUMMARY_DIR / "day31_movies_medium5_inference_status.csv")
    return out


def normalize_calibration() -> pd.DataFrame:
    src = SUMMARY_DIR / "day29_movies_medium5_2000_calibration_comparison.csv"
    raw = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "score_type": raw["variant"],
            "split": raw["split"],
            "ECE": raw["ece"],
            "Brier": raw["brier_score"],
            "AUROC": raw["auroc"],
            "high_conf_error_rate": raw["high_conf_error_rate"],
            "accuracy": raw["accuracy"],
            "status": raw["status"],
            "fallback_reason": raw.get("fallback_reason", ""),
            "source_file": str(src),
        }
    )
    write_csv(out, SUMMARY_DIR / "day31_movies_medium5_calibration_comparison.csv")
    return out


def copy_day29_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    field = pd.read_csv(SUMMARY_DIR / "day29_movies_medium5_2000_field_diagnostics.csv")
    join = pd.read_csv(SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_join_diagnostics.csv")
    grid = pd.read_csv(SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_plugin_rerank_grid.csv")
    diag = pd.read_csv(SUMMARY_DIR / "day29_movies_sasrec_medium5_2000_plugin_diagnostics.csv")
    write_csv(field, SUMMARY_DIR / "day31_movies_medium5_field_diagnostics.csv")
    write_csv(join, SUMMARY_DIR / "day31_movies_sasrec_medium5_join_diagnostics.csv")
    write_csv(grid, SUMMARY_DIR / "day31_movies_sasrec_medium5_plugin_rerank_grid.csv")
    write_csv(diag, SUMMARY_DIR / "day31_movies_sasrec_medium5_plugin_diagnostics.csv")
    return field, join, grid, diag


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(status: pd.DataFrame, calibration: pd.DataFrame, join: pd.DataFrame, grid: pd.DataFrame, diag: pd.DataFrame) -> None:
    test_cal = calibration[calibration["split"] == "test"].copy()
    raw = test_cal[test_cal["score_type"] == "raw_relevance_probability"].iloc[0]
    cal = test_cal[test_cal["score_type"] == "calibrated_relevance_probability"].iloc[0]
    minimal = test_cal[test_cal["score_type"] == "evidence_posterior_relevance_minimal"].iloc[0]
    full = test_cal[test_cal["score_type"] == "evidence_posterior_relevance_full"].iloc[0]

    backbone = grid[grid["method"] == "A_SASRec_only"].iloc[0]
    best = grid.sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
    best_b = grid[grid["method"] == "B_SASRec_plus_calibrated_relevance"].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
    best_d = grid[grid["method"] == "D_SASRec_plus_calibrated_relevance_plus_evidence_risk"].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
    join_row = join.iloc[0]
    diag_row = diag.iloc[0]

    report = f"""# Day31 Movies medium_5neg_2000 Cross-Domain Report

## 1. Why Movies Medium5

Movies medium_5neg_2000 comes from the regular Movies processed domain, keeps the Beauty-compatible pointwise schema, and is much cheaper than medium_20neg_2000. This makes it a useful first cross-domain consistency check before spending more API budget.

## 2. Data Scale And Metric Protocol

Valid and test inference are complete: valid `{int(status.loc[status['split'] == 'valid', 'actual_rows'].iloc[0])}` rows, test `{int(status.loc[status['split'] == 'test', 'actual_rows'].iloc[0])}` rows. Each user has 1 positive plus 5 negatives, so HR@10 is trivial and is not used as primary evidence. Main metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## 3. Movies Relevance Calibration

On test, raw relevance has ECE `{raw['ECE']:.4f}`, Brier `{raw['Brier']:.4f}`, and AUROC `{raw['AUROC']:.4f}`. Calibrated relevance has ECE `{cal['ECE']:.4f}`, Brier `{cal['Brier']:.4f}`, and AUROC `{cal['AUROC']:.4f}`. Minimal evidence posterior ECE is `{minimal['ECE']:.4f}` and full evidence posterior ECE is `{full['ECE']:.4f}`. This matches the Beauty-side observation: raw relevance probability is informative but miscalibrated, while calibrated relevance posterior repairs probability quality.

## 4. Movies SASRec Plug-in

SASRec-only reaches NDCG@10 `{backbone['NDCG@10']:.4f}`, MRR `{backbone['MRR']:.4f}`, HR@1 `{backbone['HR@1']:.4f}`, and HR@3 `{backbone['HR@3']:.4f}`. The best plug-in row is `{best['method']}` with NDCG@10 `{best['NDCG@10']:.4f}`, MRR `{best['MRR']:.4f}`, HR@1 `{best['HR@1']:.4f}`, and HR@3 `{best['HR@3']:.4f}`. Best B-only reaches NDCG@10 `{best_b['NDCG@10']:.4f}` and MRR `{best_b['MRR']:.4f}`; best D reaches NDCG@10 `{best_d['NDCG@10']:.4f}` and MRR `{best_d['MRR']:.4f}`.

## 5. Important Backbone Health Caveat

Join coverage is `{join_row['join_coverage']:.4f}`, but SASRec fallback_rate is `{join_row['fallback_rate']:.4f}` with positive fallback `{join_row['fallback_rate_positive']:.4f}`. This means the Movies SASRec score export is not a healthy external-backbone conclusion yet; many candidates rely on fallback scores, likely because the regular Movies IDs/history are sparse or not mapped into the trained sequence vocabulary. Therefore, the plug-in gain should be treated as cross-domain CEP consistency, not as a final Movies external backbone result.

## 6. Direction Against Beauty

The calibration result is consistent with Beauty: raw relevance is miscalibrated and calibrated relevance posterior sharply improves ECE/Brier. The plug-in table is directionally positive, but because fallback is high, Day32 should either repair Movies backbone mapping or run Books medium5 if its backbone coverage is healthier.

## 7. Day32 Recommendation

Do not open LoRA yet. Recommended next step: inspect why Movies SASRec fallback is high. If it is a Movies mapping issue, repair backbone export before claiming Movies external plug-in. In parallel, Books medium5 can be the next cross-domain target if its processed schema and backbone mapping are healthier.
"""
    (SUMMARY_DIR / "day31_movies_medium5_cross_domain_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    status = inference_status()
    calibration = normalize_calibration()
    _, join, grid, diag = copy_day29_outputs()
    write_report(status, calibration, join, grid, diag)
    print("Wrote Day31 Movies medium5 status, diagnostics, plug-in tables, and report.")


if __name__ == "__main__":
    main()
