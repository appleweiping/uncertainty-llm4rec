"""Day38 movies_small fallback sensitivity and validity analysis.

Local-only analysis. Reads Day37 movies_small backbone/evidence outputs and
quantifies whether plug-in gains persist beyond fallback/cold rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main_day15_bprmf_backbone_plugin_smoke import _normalize_per_user


SUMMARY_DIR = Path("output-repaired/summary")
SEED = 42

BACKBONES = {
    "sasrec": {
        "joined": SUMMARY_DIR / "day37_movies_small_sasrec_joined_candidates.csv",
        "grid": SUMMARY_DIR / "day37_movies_small_sasrec_plugin_rerank_grid.csv",
        "method_label": "SASRec",
    },
    "gru4rec": {
        "joined": SUMMARY_DIR / "day37_movies_small_gru4rec_joined_candidates.csv",
        "grid": SUMMARY_DIR / "day37_movies_small_gru4rec_plugin_rerank_grid.csv",
        "method_label": "GRU4Rec",
    },
    "bert4rec": {
        "joined": SUMMARY_DIR / "day37_movies_small_bert4rec_joined_candidates.csv",
        "grid": SUMMARY_DIR / "day37_movies_small_bert4rec_plugin_rerank_grid.csv",
        "method_label": "Bert4Rec",
    },
}


def _fallback_bool(df: pd.DataFrame) -> pd.Series:
    return df["fallback_score"].fillna(0).astype(int) == 1


def _user_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_fallback"] = _fallback_bool(work)
    rows = []
    for user, group in work.groupby("user_id", sort=False):
        pos = group[group["label"].astype(int) == 1]
        positive_fallback = bool(pos["_fallback"].any()) if len(pos) else False
        rows.append(
            {
                "user_id": user,
                "fallback_rate_per_user": float(group["_fallback"].mean()),
                "positive_fallback": positive_fallback,
                "candidate_pool_size": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _select_users(df: pd.DataFrame, stratum: str) -> pd.Index:
    stats = _user_stats(df)
    if stratum == "all_users":
        return stats["user_id"]
    if stratum == "positive_fallback_false":
        return stats.loc[~stats["positive_fallback"], "user_id"]
    if stratum == "positive_fallback_true":
        return stats.loc[stats["positive_fallback"], "user_id"]
    if stratum == "fallback_rate_per_user_eq_0":
        return stats.loc[stats["fallback_rate_per_user"] == 0, "user_id"]
    if stratum == "fallback_rate_per_user_0_to_0.5":
        return stats.loc[(stats["fallback_rate_per_user"] > 0) & (stats["fallback_rate_per_user"] <= 0.5), "user_id"]
    if stratum == "fallback_rate_per_user_gt_0.5":
        return stats.loc[stats["fallback_rate_per_user"] > 0.5, "user_id"]
    raise ValueError(stratum)


def _stratum_stats(backbone: str, df: pd.DataFrame, stratum: str) -> dict[str, Any]:
    users = set(_select_users(df, stratum))
    sub = df[df["user_id"].isin(users)].copy()
    fallback = _fallback_bool(sub)
    pos = sub["label"].astype(int) == 1
    pool = sub.groupby("user_id").size()
    return {
        "backbone": backbone,
        "stratum": stratum,
        "num_users": int(sub["user_id"].nunique()),
        "num_rows": int(len(sub)),
        "positive_rows": int(pos.sum()),
        "negative_rows": int((~pos).sum()),
        "fallback_rate": float(fallback.mean()) if len(sub) else math.nan,
        "positive_fallback_rate": float((fallback & pos).sum() / max(pos.sum(), 1)) if len(sub) else math.nan,
        "negative_fallback_rate": float((fallback & ~pos).sum() / max((~pos).sum(), 1)) if len(sub) else math.nan,
        "candidate_pool_size_mean": float(pool.mean()) if len(pool) else math.nan,
        "notes": "HR@10 trivial for movies_small because candidate pool size is 6.",
    }


def _rank_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    rows = []
    pool_sizes = []
    if df.empty:
        return {
            "NDCG@10": math.nan,
            "MRR": math.nan,
            "HR@1": math.nan,
            "HR@3": math.nan,
            "NDCG@3": math.nan,
            "NDCG@5": math.nan,
            "HR@10": math.nan,
            "hr10_trivial_flag": True,
            "candidate_pool_size_mean": math.nan,
        }
    for _, group in df.groupby("user_id", sort=False):
        ranked = group.sort_values([score_col, "candidate_item_id"], ascending=[False, True]).reset_index(drop=True)
        labels = ranked["label"].astype(int).to_numpy()
        pool_sizes.append(len(labels))

        def hr_at(k: int) -> float:
            return float(labels[: min(k, len(labels))].sum() > 0)

        def ndcg_at(k: int) -> float:
            kk = min(k, len(labels))
            dcg = sum(float(labels[i]) / math.log2(i + 2) for i in range(kk))
            total_pos = int(labels.sum())
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(total_pos, kk)))
            return 0.0 if idcg == 0 else float(dcg / idcg)

        rr = 0.0
        for i, label in enumerate(labels, start=1):
            if label:
                rr = 1.0 / i
                break
        rows.append(
            {
                "NDCG@10": ndcg_at(10),
                "MRR": rr,
                "HR@1": hr_at(1),
                "HR@3": hr_at(3),
                "NDCG@3": ndcg_at(3),
                "NDCG@5": ndcg_at(5),
                "HR@10": hr_at(10),
            }
        )
    out = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    out["candidate_pool_size_mean"] = float(np.mean(pool_sizes))
    out["hr10_trivial_flag"] = bool(max(pool_sizes) <= 10 or np.mean(pool_sizes) <= 10)
    return out


def _best_settings(grid: pd.DataFrame, method_label: str) -> dict[str, dict[str, Any]]:
    prefixes = {
        "A": f"A_{method_label}_only",
        "B": f"B_{method_label}_plus_calibrated_relevance",
        "C": f"C_{method_label}_plus_evidence_risk",
        "D": f"D_{method_label}_plus_calibrated_relevance_plus_evidence_risk",
    }
    settings = {}
    for key, method in prefixes.items():
        rows = grid[grid["method"] == method].copy()
        if rows.empty:
            raise ValueError(f"Missing method {method}")
        settings[key] = rows.sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0].to_dict()
    return settings


def _score_for_setting(df: pd.DataFrame, setting: dict[str, Any], calibrated_col: str = "calibrated_relevance_probability") -> pd.Series:
    method = str(setting["method"])
    if method.startswith("A_"):
        return df["backbone_score"]
    normalization = str(setting["normalization"])
    alpha = float(setting["alpha"])
    beta = float(setting["beta"])
    lam = float(setting["lambda"])
    norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
    norm_calibrated = _normalize_per_user(df[calibrated_col], df["user_id"], normalization)
    norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
    if method.startswith("B_"):
        return alpha * norm_backbone + beta * norm_calibrated
    if method.startswith("C_"):
        return norm_backbone - lam * norm_risk
    if method.startswith("D_"):
        return alpha * norm_backbone + beta * norm_calibrated - lam * norm_risk
    raise ValueError(method)


def _evaluate_settings(df: pd.DataFrame, settings: dict[str, dict[str, Any]], backbone: str, stratum: str) -> list[dict[str, Any]]:
    rows = []
    base_df = df.copy()
    base_df["final_score"] = _score_for_setting(base_df, settings["A"])
    base_metrics = _rank_metrics(base_df, "final_score")
    for key in ["A", "B", "C", "D"]:
        setting = settings[key]
        work = df.copy()
        work["final_score"] = _score_for_setting(work, setting)
        metrics = _rank_metrics(work, "final_score")
        rows.append(
            {
                "backbone": backbone,
                "stratum": stratum,
                "method_group": key,
                "method": setting["method"],
                "lambda": setting["lambda"],
                "alpha": setting["alpha"],
                "beta": setting["beta"],
                "normalization": setting["normalization"],
                **metrics,
                "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                / max(base_metrics["NDCG@10"], 1e-12)
                if np.isfinite(base_metrics["NDCG@10"])
                else math.nan,
                "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12)
                if np.isfinite(base_metrics["MRR"])
                else math.nan,
                "num_users": int(work["user_id"].nunique()),
            }
        )
    return rows


def _sanity_scores(
    df: pd.DataFrame,
    settings: dict[str, dict[str, Any]],
    backbone: str,
    repeat: int,
    sanity_method: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    work = df.copy()
    if sanity_method == "shuffled_calibrated_relevance":
        work["_sanity_calibrated"] = work.groupby("user_id")["calibrated_relevance_probability"].transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)
        )
    elif sanity_method == "random_score_same_distribution":
        values = work["calibrated_relevance_probability"].dropna().to_numpy()
        work["_sanity_calibrated"] = rng.choice(values, size=len(work), replace=True)
    else:
        raise ValueError(sanity_method)

    base = work.copy()
    base["final_score"] = _score_for_setting(base, settings["A"])
    base_metrics = _rank_metrics(base, "final_score")
    rows = []
    for key in ["B", "D"]:
        scored = work.copy()
        scored["final_score"] = _score_for_setting(scored, settings[key], calibrated_col="_sanity_calibrated")
        metrics = _rank_metrics(scored, "final_score")
        rows.append(
            {
                "backbone": backbone,
                "sanity_method": f"{sanity_method}_{key}",
                "repeat": repeat,
                "NDCG@10": metrics["NDCG@10"],
                "MRR": metrics["MRR"],
                "HR@1": metrics["HR@1"],
                "HR@3": metrics["HR@3"],
                "NDCG@3": metrics["NDCG@3"],
                "NDCG@5": metrics["NDCG@5"],
                "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                / max(base_metrics["NDCG@10"], 1e-12),
                "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
            }
        )
    return rows


def _fallback_indicator_baseline(df: pd.DataFrame, backbone: str) -> list[dict[str, Any]]:
    rows = []
    base = df.copy()
    base["final_score"] = base["backbone_score"]
    base_metrics = _rank_metrics(base, "final_score")
    for normalization in ["minmax", "zscore"]:
        norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        fallback = _fallback_bool(df).astype(float)
        for lam in [0.05, 0.1, 0.2, 0.5]:
            scored = df.copy()
            scored["final_score"] = norm_backbone - lam * fallback
            metrics = _rank_metrics(scored, "final_score")
            rows.append(
                {
                    "backbone": backbone,
                    "normalization": normalization,
                    "lambda": lam,
                    "NDCG@10": metrics["NDCG@10"],
                    "MRR": metrics["MRR"],
                    "HR@1": metrics["HR@1"],
                    "HR@3": metrics["HR@3"],
                    "NDCG@3": metrics["NDCG@3"],
                    "NDCG@5": metrics["NDCG@5"],
                    "HR@10": metrics["HR@10"],
                    "hr10_trivial_flag": metrics["hr10_trivial_flag"],
                    "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"])
                    / max(base_metrics["NDCG@10"], 1e-12),
                    "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
                }
            )
    return rows


def _write_report(
    strata: pd.DataFrame,
    strat_metrics: pd.DataFrame,
    warm: pd.DataFrame,
    cold: pd.DataFrame,
    sanity: pd.DataFrame,
    fallback_base: pd.DataFrame,
) -> None:
    lines = [
        "# Day38 Movies Small Fallback Sensitivity Report",
        "",
        "## 1. Motivation",
        "",
        "Day37 movies_small gives positive cross-domain sanity results, but ID-backbone fallback is not low. This analysis checks whether CEP gains are only a fallback artifact.",
        "",
        "## 2. Fallback Distribution",
        "",
    ]
    all_rows = strata[strata["stratum"] == "all_users"]
    for _, row in all_rows.iterrows():
        lines.append(
            f"- {row['backbone']}: fallback `{row['fallback_rate']:.4f}`, positive fallback `{row['positive_fallback_rate']:.4f}`, negative fallback `{row['negative_fallback_rate']:.4f}`."
        )
    lines.extend(["", "## 3. Warm-Positive Subset", ""])
    for backbone in BACKBONES:
        rows = warm[(warm["backbone"] == backbone) & (warm["method_group"].isin(["A", "D"]))]
        if rows.empty:
            continue
        a = rows[rows["method_group"] == "A"].iloc[0]
        d = rows[rows["method_group"] == "D"].iloc[0]
        lines.append(
            f"- {backbone}: warm-positive users `{int(a['num_users'])}`; A NDCG `{a['NDCG@10']:.4f}` / MRR `{a['MRR']:.4f}`, D NDCG `{d['NDCG@10']:.4f}` / MRR `{d['MRR']:.4f}`."
        )
    lines.extend(["", "## 4. Cold-Positive Subset", ""])
    for backbone in BACKBONES:
        rows = cold[(cold["backbone"] == backbone) & (cold["method_group"].isin(["A", "D"]))]
        if rows.empty:
            continue
        a = rows[rows["method_group"] == "A"].iloc[0]
        d = rows[rows["method_group"] == "D"].iloc[0]
        lines.append(
            f"- {backbone}: cold-positive users `{int(a['num_users'])}`; A NDCG `{a['NDCG@10']:.4f}` / MRR `{a['MRR']:.4f}`, D NDCG `{d['NDCG@10']:.4f}` / MRR `{d['MRR']:.4f}`."
        )
    lines.extend(["", "## 5. Signal Sanity", ""])
    sanity_summary = (
        sanity.groupby(["backbone", "sanity_method"])[["NDCG@10", "MRR", "relative_NDCG_vs_backbone", "relative_MRR_vs_backbone"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    for _, row in sanity_summary.iterrows():
        lines.append(
            f"- {row[('backbone', '')]} / {row[('sanity_method', '')]}: rel NDCG mean `{row[('relative_NDCG_vs_backbone', 'mean')]:.4f}`, rel MRR mean `{row[('relative_MRR_vs_backbone', 'mean')]:.4f}`."
        )
    lines.extend(["", "## 6. Fallback Indicator Baseline", ""])
    best_fb = fallback_base.sort_values(["backbone", "NDCG@10", "MRR"], ascending=[True, False, False]).groupby("backbone").head(1)
    for _, row in best_fb.iterrows():
        lines.append(
            f"- {row['backbone']}: best fallback-indicator baseline rel NDCG `{row['relative_NDCG_vs_backbone']:.4f}`, rel MRR `{row['relative_MRR_vs_backbone']:.4f}` at lambda `{row['lambda']}` / `{row['normalization']}`."
        )
    lines.extend(
        [
            "",
            "## 7. Conclusion",
            "",
            "If warm-positive improvements are positive, movies_small supports cross-domain directionality beyond pure positive fallback. If improvements are concentrated in cold-positive users, the correct claim is that CEP helps compensate for weak/cold backbone scores in this small-domain sanity setting. In all cases, do not describe movies_small as a fully healthy external-backbone benchmark.",
            "",
            "## 8. Relation To Main Evidence",
            "",
            "Beauty full three-backbone multi-seed remains the primary performance evidence. Movies_small is cross-domain sanity / continuity evidence and is useful because it reproduces the direction with a different domain while explicitly exposing fallback sensitivity.",
        ]
    )
    (SUMMARY_DIR / "day38_movies_small_fallback_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    strata_rows = []
    strat_metric_rows = []
    warm_rows = []
    cold_rows = []
    sanity_rows = []
    fallback_baseline_rows = []
    rng = np.random.default_rng(SEED)

    for backbone, cfg in BACKBONES.items():
        df = pd.read_csv(cfg["joined"])
        grid = pd.read_csv(cfg["grid"])
        method_label = str(cfg["method_label"])
        settings = _best_settings(grid, method_label)
        strata = [
            "all_users",
            "positive_fallback_false",
            "positive_fallback_true",
            "fallback_rate_per_user_eq_0",
            "fallback_rate_per_user_0_to_0.5",
            "fallback_rate_per_user_gt_0.5",
        ]
        for stratum in strata:
            users = set(_select_users(df, stratum))
            sub = df[df["user_id"].isin(users)].copy()
            strata_rows.append(_stratum_stats(backbone, df, stratum))
            strat_metric_rows.extend(_evaluate_settings(sub, settings, backbone, stratum))

        warm_users = set(_select_users(df, "positive_fallback_false"))
        cold_users = set(_select_users(df, "positive_fallback_true"))
        warm_rows.extend(_evaluate_settings(df[df["user_id"].isin(warm_users)].copy(), settings, backbone, "warm_positive_subset"))
        cold_rows.extend(_evaluate_settings(df[df["user_id"].isin(cold_users)].copy(), settings, backbone, "cold_positive_subset"))

        for repeat in range(1, 6):
            sanity_rows.extend(_sanity_scores(df, settings, backbone, repeat, "shuffled_calibrated_relevance", rng))
            sanity_rows.extend(_sanity_scores(df, settings, backbone, repeat, "random_score_same_distribution", rng))
        fallback_baseline_rows.extend(_fallback_indicator_baseline(df, backbone))

    strata_df = pd.DataFrame(strata_rows)
    strat_df = pd.DataFrame(strat_metric_rows)
    warm_df = pd.DataFrame(warm_rows)
    cold_df = pd.DataFrame(cold_rows)
    sanity_df = pd.DataFrame(sanity_rows)
    fb_df = pd.DataFrame(fallback_baseline_rows)

    strata_df.to_csv(SUMMARY_DIR / "day38_movies_small_fallback_strata_stats.csv", index=False)
    strat_df.to_csv(SUMMARY_DIR / "day38_movies_small_fallback_stratified_plugin_metrics.csv", index=False)
    warm_df.to_csv(SUMMARY_DIR / "day38_movies_small_warm_positive_subset_metrics.csv", index=False)
    cold_df.to_csv(SUMMARY_DIR / "day38_movies_small_cold_positive_subset_metrics.csv", index=False)
    sanity_df.to_csv(SUMMARY_DIR / "day38_movies_small_signal_sanity_check.csv", index=False)
    fb_df.to_csv(SUMMARY_DIR / "day38_movies_small_fallback_indicator_baseline.csv", index=False)
    _write_report(strata_df, strat_df, warm_df, cold_df, sanity_df, fb_df)
    print("Day38 movies_small fallback sensitivity analysis complete.")


if __name__ == "__main__":
    main()
