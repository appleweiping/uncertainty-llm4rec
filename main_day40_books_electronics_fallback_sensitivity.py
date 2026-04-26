"""Day40 books/electronics small fallback sensitivity analysis.

Local-only analysis. Reads Day39 joined candidates, plug-in grids, and
diagnostics for books_small/electronics_small. It mirrors Day38 movies_small
fallback sensitivity and then merges all three small domains into one summary.
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

DOMAINS = ["books_small", "electronics_small"]
BACKBONES = {
    "sasrec": "SASRec",
    "gru4rec": "GRU4Rec",
    "bert4rec": "Bert4Rec",
}
STRATA = [
    "all_users",
    "positive_fallback_false",
    "positive_fallback_true",
    "fallback_rate_per_user_eq_0",
    "fallback_rate_per_user_0_to_0.5",
    "fallback_rate_per_user_gt_0.5",
]


def _fallback_bool(df: pd.DataFrame) -> pd.Series:
    return df["fallback_score"].fillna(0).astype(int) == 1


def _user_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_fallback"] = _fallback_bool(work)
    rows = []
    for user, group in work.groupby("user_id", sort=False):
        pos = group[group["label"].astype(int) == 1]
        rows.append(
            {
                "user_id": user,
                "fallback_rate_per_user": float(group["_fallback"].mean()),
                "positive_fallback": bool(pos["_fallback"].any()) if len(pos) else False,
                "candidate_pool_size": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _select_users(df: pd.DataFrame, stratum: str) -> pd.Series:
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


def _stratum_stats(domain: str, backbone: str, df: pd.DataFrame, stratum: str) -> dict[str, Any]:
    users = set(_select_users(df, stratum))
    sub = df[df["user_id"].isin(users)].copy()
    fallback = _fallback_bool(sub)
    pos = sub["label"].astype(int) == 1
    pool = sub.groupby("user_id").size()
    return {
        "domain": domain,
        "backbone": backbone,
        "stratum": stratum,
        "num_users": int(sub["user_id"].nunique()),
        "num_rows": int(len(sub)),
        "positive_rows": int(pos.sum()),
        "negative_rows": int((~pos).sum()),
        "fallback_rows": int(fallback.sum()),
        "fallback_rate": float(fallback.mean()) if len(sub) else math.nan,
        "positive_fallback_rate": float((fallback & pos).sum() / max(pos.sum(), 1)) if len(sub) else math.nan,
        "negative_fallback_rate": float((fallback & ~pos).sum() / max((~pos).sum(), 1)) if len(sub) else math.nan,
        "candidate_pool_size_mean": float(pool.mean()) if len(pool) else math.nan,
        "notes": "HR@10 trivial for small domains because candidate pool size is 6.",
    }


def _rank_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    if df.empty:
        return {
            "NDCG@10": math.nan,
            "MRR": math.nan,
            "HR@1": math.nan,
            "HR@3": math.nan,
            "NDCG@3": math.nan,
            "NDCG@5": math.nan,
            "HR@10": math.nan,
            "candidate_pool_size_mean": math.nan,
            "hr10_trivial_flag": True,
        }
    rows = []
    pool_sizes = []
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
    names = {
        "A": f"A_{method_label}_only",
        "B": f"B_{method_label}_plus_calibrated_relevance",
        "C": f"C_{method_label}_plus_evidence_risk",
        "D": f"D_{method_label}_plus_calibrated_relevance_plus_evidence_risk",
    }
    settings = {}
    for key, method in names.items():
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
    lamb = float(setting["lambda"])
    norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
    norm_calibrated = _normalize_per_user(df[calibrated_col], df["user_id"], normalization)
    norm_risk = _normalize_per_user(df["evidence_risk"], df["user_id"], normalization)
    if method.startswith("B_"):
        return alpha * norm_backbone + beta * norm_calibrated
    if method.startswith("C_"):
        return norm_backbone - lamb * norm_risk
    if method.startswith("D_"):
        return alpha * norm_backbone + beta * norm_calibrated - lamb * norm_risk
    raise ValueError(method)


def _evaluate_settings(df: pd.DataFrame, settings: dict[str, dict[str, Any]], domain: str, backbone: str, stratum: str) -> list[dict[str, Any]]:
    base_df = df.copy()
    base_df["final_score"] = _score_for_setting(base_df, settings["A"])
    base_metrics = _rank_metrics(base_df, "final_score")
    rows = []
    for group in ["A", "B", "C", "D"]:
        setting = settings[group]
        work = df.copy()
        work["final_score"] = _score_for_setting(work, setting)
        metrics = _rank_metrics(work, "final_score")
        rows.append(
            {
                "domain": domain,
                "backbone": backbone,
                "stratum": stratum,
                "method_group": group,
                "method": setting["method"],
                "lambda": setting["lambda"],
                "alpha": setting["alpha"],
                "beta": setting["beta"],
                "normalization": setting["normalization"],
                **metrics,
                "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12)
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
    domain: str,
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
    for group in ["B", "D"]:
        scored = work.copy()
        scored["final_score"] = _score_for_setting(scored, settings[group], calibrated_col="_sanity_calibrated")
        metrics = _rank_metrics(scored, "final_score")
        rows.append(
            {
                "domain": domain,
                "backbone": backbone,
                "sanity_method": f"{sanity_method}_{group}",
                "repeat": repeat,
                "NDCG@10": metrics["NDCG@10"],
                "MRR": metrics["MRR"],
                "HR@1": metrics["HR@1"],
                "HR@3": metrics["HR@3"],
                "NDCG@3": metrics["NDCG@3"],
                "NDCG@5": metrics["NDCG@5"],
                "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12),
                "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
            }
        )
    return rows


def _fallback_indicator_baseline(df: pd.DataFrame, domain: str, backbone: str) -> list[dict[str, Any]]:
    base = df.copy()
    base["final_score"] = base["backbone_score"]
    base_metrics = _rank_metrics(base, "final_score")
    fallback = _fallback_bool(df).astype(float)
    rows = []
    for normalization in ["minmax", "zscore"]:
        norm_backbone = _normalize_per_user(df["backbone_score"], df["user_id"], normalization)
        for lamb in [0.05, 0.1, 0.2, 0.5]:
            scored = df.copy()
            scored["final_score"] = norm_backbone - lamb * fallback
            metrics = _rank_metrics(scored, "final_score")
            rows.append(
                {
                    "domain": domain,
                    "backbone": backbone,
                    "normalization": normalization,
                    "lambda": lamb,
                    "NDCG@10": metrics["NDCG@10"],
                    "MRR": metrics["MRR"],
                    "HR@1": metrics["HR@1"],
                    "HR@3": metrics["HR@3"],
                    "NDCG@3": metrics["NDCG@3"],
                    "NDCG@5": metrics["NDCG@5"],
                    "HR@10": metrics["HR@10"],
                    "hr10_trivial_flag": metrics["hr10_trivial_flag"],
                    "relative_NDCG_vs_backbone": (metrics["NDCG@10"] - base_metrics["NDCG@10"]) / max(base_metrics["NDCG@10"], 1e-12),
                    "relative_MRR_vs_backbone": (metrics["MRR"] - base_metrics["MRR"]) / max(base_metrics["MRR"], 1e-12),
                }
            )
    return rows


def _load_day38_for_summary() -> pd.DataFrame:
    rows = []
    strata = pd.read_csv(SUMMARY_DIR / "day38_movies_small_fallback_strata_stats.csv")
    strat = pd.read_csv(SUMMARY_DIR / "day38_movies_small_fallback_stratified_plugin_metrics.csv")
    warm = pd.read_csv(SUMMARY_DIR / "day38_movies_small_warm_positive_subset_metrics.csv")
    cold = pd.read_csv(SUMMARY_DIR / "day38_movies_small_cold_positive_subset_metrics.csv")
    sanity = pd.read_csv(SUMMARY_DIR / "day38_movies_small_signal_sanity_check.csv")
    fb = pd.read_csv(SUMMARY_DIR / "day38_movies_small_fallback_indicator_baseline.csv")
    for backbone in ["sasrec", "gru4rec", "bert4rec"]:
        s = strata[(strata["backbone"] == backbone) & (strata["stratum"] == "all_users")].iloc[0]
        all_d = strat[(strat["backbone"] == backbone) & (strat["stratum"] == "all_users") & (strat["method_group"] == "D")].iloc[0]
        warm_d = warm[(warm["backbone"] == backbone) & (warm["method_group"] == "D")].iloc[0]
        cold_d = cold[(cold["backbone"] == backbone) & (cold["method_group"] == "D")].iloc[0]
        sanity_d = sanity[(sanity["backbone"] == backbone) & (sanity["sanity_method"].str.endswith("_D"))]
        shuffled = sanity_d[sanity_d["sanity_method"].str.startswith("shuffled")]["relative_NDCG_vs_backbone"].mean()
        random = sanity_d[sanity_d["sanity_method"].str.startswith("random")]["relative_NDCG_vs_backbone"].mean()
        fb_best = fb[fb["backbone"] == backbone].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
        rows.append(
            {
                "domain": "movies_small",
                "backbone": backbone,
                "fallback_rate": s["fallback_rate"],
                "positive_fallback_rate": s["positive_fallback_rate"],
                "best_method": all_d["method"],
                "all_users_relative_NDCG": all_d["relative_NDCG_vs_backbone"],
                "warm_positive_relative_NDCG": warm_d["relative_NDCG_vs_backbone"],
                "cold_positive_relative_NDCG": cold_d["relative_NDCG_vs_backbone"],
                "shuffled_gap": all_d["relative_NDCG_vs_backbone"] - shuffled,
                "random_gap": all_d["relative_NDCG_vs_backbone"] - random,
                "fallback_indicator_relative_NDCG": fb_best["relative_NDCG_vs_backbone"],
                "interpretation": _interpret(s["fallback_rate"], s["positive_fallback_rate"], warm_d["num_users"], warm_d["relative_NDCG_vs_backbone"], all_d["relative_NDCG_vs_backbone"]),
            }
        )
    return pd.DataFrame(rows)


def _interpret(fallback_rate: float, positive_fallback_rate: float, warm_users: float, warm_rel: float, all_rel: float) -> str:
    if warm_users < 30:
        return "sample_too_small"
    if fallback_rate < 0.2 and positive_fallback_rate < 0.2 and warm_rel > 0:
        return "healthy_directionality"
    if fallback_rate >= 0.5 or positive_fallback_rate >= 0.5:
        return "fallback_heavy_caution" if warm_rel > 0 else "fallback_compensation"
    if all_rel > 0 and warm_rel > 0:
        return "healthy_directionality"
    return "fallback_compensation"


def _summary_rows(
    strata: pd.DataFrame,
    strat: pd.DataFrame,
    warm: pd.DataFrame,
    cold: pd.DataFrame,
    sanity: pd.DataFrame,
    fallback_base: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for backbone in BACKBONES:
            s = strata[(strata["domain"] == domain) & (strata["backbone"] == backbone) & (strata["stratum"] == "all_users")].iloc[0]
            all_d = strat[(strat["domain"] == domain) & (strat["backbone"] == backbone) & (strat["stratum"] == "all_users") & (strat["method_group"] == "D")].iloc[0]
            warm_d = warm[(warm["domain"] == domain) & (warm["backbone"] == backbone) & (warm["method_group"] == "D")].iloc[0]
            cold_d = cold[(cold["domain"] == domain) & (cold["backbone"] == backbone) & (cold["method_group"] == "D")].iloc[0]
            sanity_d = sanity[(sanity["domain"] == domain) & (sanity["backbone"] == backbone) & (sanity["sanity_method"].str.endswith("_D"))]
            shuffled = sanity_d[sanity_d["sanity_method"].str.startswith("shuffled")]["relative_NDCG_vs_backbone"].mean()
            random = sanity_d[sanity_d["sanity_method"].str.startswith("random")]["relative_NDCG_vs_backbone"].mean()
            fb_best = fallback_base[(fallback_base["domain"] == domain) & (fallback_base["backbone"] == backbone)].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]
            rows.append(
                {
                    "domain": domain,
                    "backbone": backbone,
                    "fallback_rate": s["fallback_rate"],
                    "positive_fallback_rate": s["positive_fallback_rate"],
                    "best_method": all_d["method"],
                    "all_users_relative_NDCG": all_d["relative_NDCG_vs_backbone"],
                    "warm_positive_relative_NDCG": warm_d["relative_NDCG_vs_backbone"],
                    "cold_positive_relative_NDCG": cold_d["relative_NDCG_vs_backbone"],
                    "shuffled_gap": all_d["relative_NDCG_vs_backbone"] - shuffled,
                    "random_gap": all_d["relative_NDCG_vs_backbone"] - random,
                    "fallback_indicator_relative_NDCG": fb_best["relative_NDCG_vs_backbone"],
                    "interpretation": _interpret(s["fallback_rate"], s["positive_fallback_rate"], warm_d["num_users"], warm_d["relative_NDCG_vs_backbone"], all_d["relative_NDCG_vs_backbone"]),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    strata: pd.DataFrame,
    warm: pd.DataFrame,
    cold: pd.DataFrame,
    sanity: pd.DataFrame,
    fallback_base: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Day40 Books/Electronics Small Fallback Sensitivity Report",
        "",
        "## 1. Motivation",
        "",
        "Day39 books/electronics small-domain plug-in results were directionally positive, but fallback caveats were clear. This local-only Day40 analysis mirrors Day38 movies_small to check whether gains are fallback artifacts, warm-positive directionality, or cold/fallback compensation.",
        "",
        "## 2. Fallback Distribution",
        "",
    ]
    for _, row in strata[strata["stratum"] == "all_users"].iterrows():
        lines.append(
            f"- {row['domain']} / {row['backbone']}: fallback `{row['fallback_rate']:.4f}`, positive fallback `{row['positive_fallback_rate']:.4f}`, negative fallback `{row['negative_fallback_rate']:.4f}`."
        )
    lines.extend(["", "## 3. Warm-Positive Subset", ""])
    for _, row in warm[warm["method_group"] == "D"].iterrows():
        lines.append(
            f"- {row['domain']} / {row['backbone']}: warm-positive users `{int(row['num_users'])}`, D relative NDCG `{row['relative_NDCG_vs_backbone']:.4f}`, D relative MRR `{row['relative_MRR_vs_backbone']:.4f}`."
        )
    lines.extend(["", "## 4. Cold-Positive Subset", ""])
    for _, row in cold[cold["method_group"] == "D"].iterrows():
        lines.append(
            f"- {row['domain']} / {row['backbone']}: cold-positive users `{int(row['num_users'])}`, D relative NDCG `{row['relative_NDCG_vs_backbone']:.4f}`, D relative MRR `{row['relative_MRR_vs_backbone']:.4f}`."
        )
    lines.extend(["", "## 5. Shuffled/Random Sanity", ""])
    sanity_summary = (
        sanity.groupby(["domain", "backbone", "sanity_method"])[["relative_NDCG_vs_backbone", "relative_MRR_vs_backbone"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    for _, row in sanity_summary.iterrows():
        lines.append(
            f"- {row[('domain', '')]} / {row[('backbone', '')]} / {row[('sanity_method', '')]}: rel NDCG mean `{row[('relative_NDCG_vs_backbone', 'mean')]:.4f}`, rel MRR mean `{row[('relative_MRR_vs_backbone', 'mean')]:.4f}`."
        )
    lines.extend(["", "## 6. Fallback Indicator Baseline", ""])
    best_fb = fallback_base.sort_values(["domain", "backbone", "NDCG@10", "MRR"], ascending=[True, True, False, False]).groupby(["domain", "backbone"]).head(1)
    for _, row in best_fb.iterrows():
        lines.append(
            f"- {row['domain']} / {row['backbone']}: best fallback-indicator baseline rel NDCG `{row['relative_NDCG_vs_backbone']:.4f}`, rel MRR `{row['relative_MRR_vs_backbone']:.4f}` at lambda `{row['lambda']}` / `{row['normalization']}`."
        )
    lines.extend(["", "## 7. Cross-Domain Small Summary", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['domain']} / {row['backbone']}: all-users rel NDCG `{row['all_users_relative_NDCG']:.4f}`, warm rel NDCG `{row['warm_positive_relative_NDCG']:.4f}`, cold rel NDCG `{row['cold_positive_relative_NDCG']:.4f}`, interpretation `{row['interpretation']}`."
        )
    lines.extend(
        [
            "",
            "## 8. Claim Boundary",
            "",
            "Small-domain results are cross-domain sanity / continuity evidence. They should not be described as fully healthy external-backbone benchmarks when fallback is high. Beauty full three-backbone multi-seed remains the primary performance evidence. Day40 helps phrase small-domain gains as either warm-positive directionality or fallback/cold compensation, depending on each domain/backbone.",
        ]
    )
    (SUMMARY_DIR / "day40_books_electronics_fallback_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    strata_rows: list[dict[str, Any]] = []
    strat_rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []
    cold_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for backbone, method_label in BACKBONES.items():
            joined = pd.read_csv(SUMMARY_DIR / f"day39_{domain}_{backbone}_joined_candidates.csv")
            grid = pd.read_csv(SUMMARY_DIR / f"day39_{domain}_{backbone}_plugin_rerank_grid.csv")
            settings = _best_settings(grid, method_label)
            for stratum in STRATA:
                users = set(_select_users(joined, stratum))
                sub = joined[joined["user_id"].isin(users)].copy()
                strata_rows.append(_stratum_stats(domain, backbone, joined, stratum))
                strat_rows.extend(_evaluate_settings(sub, settings, domain, backbone, stratum))

            warm_users = set(_select_users(joined, "positive_fallback_false"))
            cold_users = set(_select_users(joined, "positive_fallback_true"))
            warm_rows.extend(_evaluate_settings(joined[joined["user_id"].isin(warm_users)].copy(), settings, domain, backbone, "warm_positive_subset"))
            cold_rows.extend(_evaluate_settings(joined[joined["user_id"].isin(cold_users)].copy(), settings, domain, backbone, "cold_positive_subset"))

            for repeat in range(1, 6):
                sanity_rows.extend(_sanity_scores(joined, settings, domain, backbone, repeat, "shuffled_calibrated_relevance", rng))
                sanity_rows.extend(_sanity_scores(joined, settings, domain, backbone, repeat, "random_score_same_distribution", rng))
            fallback_rows.extend(_fallback_indicator_baseline(joined, domain, backbone))

    strata = pd.DataFrame(strata_rows)
    strat = pd.DataFrame(strat_rows)
    warm = pd.DataFrame(warm_rows)
    cold = pd.DataFrame(cold_rows)
    sanity = pd.DataFrame(sanity_rows)
    fallback_base = pd.DataFrame(fallback_rows)

    strata.to_csv(SUMMARY_DIR / "day40_books_electronics_small_fallback_strata_stats.csv", index=False)
    strat.to_csv(SUMMARY_DIR / "day40_books_electronics_small_fallback_stratified_plugin_metrics.csv", index=False)
    warm.to_csv(SUMMARY_DIR / "day40_books_electronics_small_warm_positive_subset_metrics.csv", index=False)
    cold.to_csv(SUMMARY_DIR / "day40_books_electronics_small_cold_positive_subset_metrics.csv", index=False)
    sanity.to_csv(SUMMARY_DIR / "day40_books_electronics_small_signal_sanity_check.csv", index=False)
    fallback_base.to_csv(SUMMARY_DIR / "day40_books_electronics_small_fallback_indicator_baseline.csv", index=False)

    summary = pd.concat([_load_day38_for_summary(), _summary_rows(strata, strat, warm, cold, sanity, fallback_base)], ignore_index=True)
    summary.to_csv(SUMMARY_DIR / "day40_small_domains_fallback_sensitivity_summary.csv", index=False)
    _write_report(strata, warm, cold, sanity, fallback_base, summary)
    print("Day40 books/electronics fallback sensitivity analysis complete.")


if __name__ == "__main__":
    main()
