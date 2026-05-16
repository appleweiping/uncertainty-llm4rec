from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

from scripts.build_pony_baseline_comparison import build_tables, eligible_rows
from scripts.import_pony_official_baselines import build_manifest


def _write_tar_with_summary(path: Path, method: str, domain: str, status: str = "completed_result") -> None:
    summary = (
        "baseline_name,domain,status_label,artifact_class,implementation_status,"
        "score_coverage_rate,sample_count,HR@10,NDCG@10,MRR\n"
        f"{method},{domain},same_schema_external_baseline,{status},official_completed,1.0,7,0.4,0.2,0.1\n"
    )
    temp = path.parent / "same_candidate_external_baseline_summary.csv"
    temp.write_text(summary, encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(temp, arcname=f"outputs/{domain}_{method}_same_candidate/tables/same_candidate_external_baseline_summary.csv")
    temp.unlink()


def test_import_manifest_records_hash_and_pending_rows(tmp_path: Path) -> None:
    pony = tmp_path / "pony"
    pony.mkdir()
    package = pony / "llm2rec_beauty_official_qwen3base_evidence_2026-01-01_000000.tar.gz"
    _write_tar_with_summary(package, "llm2rec_official_qwen3base_sasrec", "beauty")

    manifest = build_manifest(
        pony_root=pony,
        output_root=tmp_path / "truce_outputs",
        manifest_path=tmp_path / "manifest.yaml",
        summary_csv=None,
        no_copy=False,
    )

    beauty = next(entry for entry in manifest["entries"] if entry["method_key"] == "llm2rec" and entry["domain"] == "beauty")
    books = next(entry for entry in manifest["entries"] if entry["method_key"] == "llm2rec" and entry["domain"] == "books")
    assert beauty["completion_label"] == "completed_result"
    assert beauty["main_table_eligible"] is True
    assert beauty["evidence_present"] is True
    assert len(beauty["evidence_sha256"]) == 64
    assert beauty["evidence_size_bytes"] > 0
    assert Path(beauty["truce_evidence_tar"]).exists()
    assert books["completion_label"] == "pending_running"
    assert books["main_table_eligible"] is False


def test_build_comparison_excludes_pending_rows(tmp_path: Path) -> None:
    manifest = {
        "entries": [
            {
                "domain": "beauty",
                "display_method": "LLM2Rec",
                "method": "llm2rec_official_qwen3base_sasrec",
                "sample_count": 7,
                "HR@10": "0.4",
                "NDCG@10": "0.2",
                "MRR": "0.1",
                "coverage@10": "0.5",
                "completion_label": "completed_result",
                "status_label": "same_schema_external_baseline",
                "artifact_class": "completed_result",
                "implementation_status": "official_completed",
                "score_coverage_rate": "1.0",
                "evidence_present": True,
                "evidence_tar_readable": True,
                "evidence_sha256": "a" * 64,
                "truce_evidence_tar": "outputs/pony/a.tar.gz",
                "main_table_eligible": True,
            },
            {
                "domain": "movies",
                "display_method": "ProMax",
                "method": "promax_official_qwen3base_profile",
                "sample_count": None,
                "HR@10": None,
                "NDCG@10": None,
                "MRR": None,
                "coverage@10": None,
                "completion_label": "pending_running",
                "status_label": None,
                "artifact_class": None,
                "implementation_status": None,
                "score_coverage_rate": None,
                "evidence_present": False,
                "evidence_tar_readable": False,
                "evidence_sha256": None,
                "truce_evidence_tar": None,
                "main_table_eligible": False,
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = build_tables(manifest_path, tmp_path / "tables", "pony")
    rows = list(csv.DictReader(Path(result["main_csv"]).open(encoding="utf-8")))
    status_rows = list(csv.DictReader(Path(result["status_csv"]).open(encoding="utf-8")))

    assert len(rows) == 1
    assert rows[0]["method"] == "llm2rec_official_qwen3base_sasrec"
    assert rows[0]["HR@10"] == "0.4"
    assert len(status_rows) == 2
    assert any(row["completion_label"] == "pending_running" for row in status_rows)
    assert len(eligible_rows(manifest)) == 1


def test_completed_metrics_without_evidence_package_are_pending_import(tmp_path: Path) -> None:
    pony = tmp_path / "pony"
    pony.mkdir()
    summary = pony / "week8_official_external_qwen3base_multik_comparison.csv"
    summary.write_text(
        "domain,display_method,method,sample_count,status_label,artifact_class\n"
        "beauty,LLM2Rec,llm2rec_official_qwen3base_sasrec,7,same_schema_external_baseline,completed_result\n",
        encoding="utf-8",
    )

    manifest = build_manifest(
        pony_root=pony,
        output_root=tmp_path / "truce_outputs",
        manifest_path=tmp_path / "manifest.yaml",
        summary_csv=None,
        no_copy=True,
    )

    row = next(entry for entry in manifest["entries"] if entry["method_key"] == "llm2rec" and entry["domain"] == "beauty")
    assert row["completion_label"] == "pending_import"
    assert row["evidence_present"] is False
    assert row["main_table_eligible"] is False
