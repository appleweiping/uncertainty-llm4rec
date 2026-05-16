from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_METHODS = {
    "llm2rec": "llm2rec_official_qwen3base_sasrec",
    "llmesr": "llmesr_official_qwen3base_sasrec",
    "llmemb": "llmemb_official_qwen3base",
    "rlmrec": "rlmrec_official_qwen3base_graphcl",
    "irllrec": "irllrec_official_qwen3base_intent",
    "elmrec": "elmrec_official_qwen3base_graph",
    "proex": "proex_official_qwen3base_profile",
    "promax": "promax_official_qwen3base_profile",
}

DOMAINS = ("beauty", "books", "electronics", "movies")
METRIC_FIELDS = (
    "HR@5",
    "NDCG@5",
    "HR@10",
    "NDCG@10",
    "HR@20",
    "NDCG@20",
    "MRR",
    "coverage@5",
    "coverage@10",
    "coverage@20",
    "head_exposure_ratio@10",
    "longtail_coverage@10",
)
REQUIRED_SCORE_SCHEMA = "source_event_id,user_id,item_id,score"
ELIGIBLE_STATUS = "same_schema_external_baseline"
ELIGIBLE_ARTIFACT = "completed_result"
ELIGIBLE_IMPLEMENTATION = "official_completed"


@dataclass(frozen=True)
class EvidencePackage:
    path: Path
    method_key: str
    domain: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Pony/Uncertainty official baseline evidence packages into TRUCE outputs and build a tracked manifest."
    )
    parser.add_argument("--pony-root", default=r"D:\Research\Uncertainty")
    parser.add_argument("--output-root", default="outputs/pony_official_baselines")
    parser.add_argument("--manifest", default="configs/baselines/pony_official_external_baselines.yaml")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--no-copy", action="store_true", help="Build manifest without copying tar.gz packages.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _json_value(str(value))


def write_yaml(data: dict[str, Any], path: Path) -> None:
    def emit(value: Any, indent: int) -> list[str]:
        prefix = " " * indent
        if isinstance(value, dict):
            lines: list[str] = []
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(emit(child, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {_yaml_scalar(child)}")
            return lines
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}-")
                    lines.extend(emit(item, indent + 2))
                elif isinstance(item, list):
                    lines.append(f"{prefix}-")
                    lines.extend(emit(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {_yaml_scalar(item)}")
            return lines
        return [f"{prefix}{_yaml_scalar(value)}"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(emit(data, 0)) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _method_key_from_name(name: str) -> str | None:
    lower = name.lower()
    for key in sorted(DEFAULT_METHODS, key=len, reverse=True):
        if lower.startswith(f"{key}_"):
            return key
    return None


def _domain_from_name(name: str) -> str | None:
    lower = name.lower()
    for domain in DOMAINS:
        if f"_{domain}_" in lower:
            return domain
    return None


def find_evidence_packages(pony_root: Path) -> list[EvidencePackage]:
    packages: list[EvidencePackage] = []
    for path in sorted(pony_root.glob("*official_qwen3base*.tar.gz")):
        method_key = _method_key_from_name(path.name)
        domain = _domain_from_name(path.name)
        if method_key and domain and method_key in DEFAULT_METHODS:
            packages.append(EvidencePackage(path=path, method_key=method_key, domain=domain))
    return packages


def _extract_first_matching_csv(package: Path, pattern: str) -> list[dict[str, str]]:
    try:
        with tarfile.open(package, "r:gz") as archive:
            for member in archive.getmembers():
                if pattern in member.name and member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    text = extracted.read().decode("utf-8-sig")
                    return list(csv.DictReader(text.splitlines()))
    except (tarfile.TarError, EOFError, OSError, UnicodeDecodeError):
        return []
    return []


def _summary_rows_by_domain_method(pony_root: Path, packages: list[EvidencePackage], summary_csv: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    if summary_csv and summary_csv.exists():
        rows.extend(read_csv_rows(summary_csv))
    default_summary = pony_root / "week8_official_external_qwen3base_multik_comparison.csv"
    if default_summary.exists():
        rows.extend(read_csv_rows(default_summary))
    for package in packages:
        rows.extend(_extract_first_matching_csv(package.path, "same_candidate_external_baseline_summary.csv"))

    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        domain = row.get("domain", "").strip()
        method = (row.get("method") or row.get("baseline_name") or "").strip()
        if not domain or not method:
            continue
        current = by_key.get((domain, method))
        candidate_score = int(bool(row.get("implementation_status"))) + int(bool(row.get("score_coverage_rate")))
        current_score = int(bool(current and current.get("implementation_status"))) + int(bool(current and current.get("score_coverage_rate")))
        if current is None or candidate_score >= current_score:
            by_key[(domain, method)] = row
    return by_key


def _copy_package(package: Path, evidence_dir: Path, no_copy: bool) -> Path:
    target = evidence_dir / package.name
    if no_copy:
        return target
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != package.stat().st_size:
        shutil.copy2(package, target)
    return target


def _tar_readable(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as archive:
            archive.getmembers()
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def _completion_label(row: dict[str, str] | None, package: EvidencePackage | None) -> str:
    if row:
        status = row.get("status_label")
        artifact = row.get("artifact_class")
        implementation = row.get("implementation_status") or ELIGIBLE_IMPLEMENTATION
        if status == ELIGIBLE_STATUS and artifact == ELIGIBLE_ARTIFACT and implementation == ELIGIBLE_IMPLEMENTATION:
            return "completed_result" if package is not None else "pending_import"
    if package is not None:
        return "pending_import"
    return "pending_running"


def _is_main_table_eligible(entry: dict[str, Any]) -> bool:
    return (
        entry.get("completion_label") == "completed_result"
        and entry.get("artifact_class") == ELIGIBLE_ARTIFACT
        and entry.get("status_label") == ELIGIBLE_STATUS
        and entry.get("implementation_status") == ELIGIBLE_IMPLEMENTATION
        and entry.get("evidence_present") is True
        and entry.get("evidence_tar_readable") is True
    )


def build_manifest(
    *,
    pony_root: Path,
    output_root: Path,
    manifest_path: Path,
    summary_csv: Path | None,
    no_copy: bool,
) -> dict[str, Any]:
    packages = find_evidence_packages(pony_root)
    package_by_key: dict[tuple[str, str], EvidencePackage] = {}
    for package in packages:
        key = (package.domain, DEFAULT_METHODS[package.method_key])
        current = package_by_key.get(key)
        package_rank = (_tar_readable(package.path), package.path.stat().st_size)
        current_rank = (_tar_readable(current.path), current.path.stat().st_size) if current else (False, -1)
        if current is None or package_rank > current_rank:
            package_by_key[key] = package

    summaries = _summary_rows_by_domain_method(pony_root, packages, summary_csv)
    evidence_dir = output_root / "evidence_packages"
    entries: list[dict[str, Any]] = []

    for method_key, method_name in DEFAULT_METHODS.items():
        for domain in DOMAINS:
            package = package_by_key.get((domain, method_name))
            row = summaries.get((domain, method_name))
            copied_target = _copy_package(package.path, evidence_dir, no_copy) if package else None
            source_size = package.path.stat().st_size if package else None
            source_mtime = (
                datetime.fromtimestamp(package.path.stat().st_mtime, tz=timezone.utc).isoformat()
                if package
                else None
            )
            entry = {
                "method_key": method_key,
                "method": method_name,
                "display_method": (row or {}).get("display_method", method_name),
                "domain": domain,
                "sample_count": int(float((row or {}).get("sample_count", "0") or 0)) if row else None,
                "completion_label": _completion_label(row, package),
                "status_label": (row or {}).get("status_label"),
                "artifact_class": (row or {}).get("artifact_class"),
                "implementation_status": (row or {}).get("implementation_status") or (ELIGIBLE_IMPLEMENTATION if row else None),
                "score_schema": REQUIRED_SCORE_SCHEMA,
                "score_coverage_rate": (row or {}).get("score_coverage_rate"),
                "source_summary_file": (row or {}).get("source_file"),
                "source_evidence_tar": str(package.path) if package else None,
                "truce_evidence_tar": str(copied_target) if copied_target else None,
                "evidence_present": package is not None,
                "evidence_sha256": _sha256(package.path) if package else None,
                "evidence_size_bytes": source_size,
                "evidence_mtime_utc": source_mtime,
                "evidence_tar_readable": _tar_readable(package.path) if package else False,
            }
            for metric in METRIC_FIELDS:
                entry[metric] = (row or {}).get(metric)
            entry["main_table_eligible"] = _is_main_table_eligible(entry)
            entries.append(entry)

    manifest = {
        "schema_version": "pony_official_external_baselines_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pony_root": str(pony_root),
        "truce_output_root": str(output_root),
        "evidence_package_dir": str(evidence_dir),
        "source_policy": "reuse_pony_uncertainty_official_qwen3base_same_candidate_evidence",
        "score_schema": REQUIRED_SCORE_SCHEMA,
        "eligibility_rule": {
            "artifact_class": ELIGIBLE_ARTIFACT,
            "status_label": ELIGIBLE_STATUS,
            "implementation_status": ELIGIBLE_IMPLEMENTATION,
            "pending_rows_enter_main_table": False,
        },
        "methods": list(DEFAULT_METHODS.values()),
        "domains": list(DOMAINS),
        "entries": entries,
    }
    write_yaml(manifest, manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        pony_root=Path(args.pony_root).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        manifest_path=Path(args.manifest).expanduser(),
        summary_csv=Path(args.summary_csv).expanduser() if args.summary_csv else None,
        no_copy=args.no_copy,
    )
    entries = manifest["entries"]
    copied = sum(1 for entry in entries if entry["evidence_present"])
    eligible = sum(1 for entry in entries if entry["main_table_eligible"])
    pending = sum(1 for entry in entries if entry["completion_label"] != "completed_result")
    print(f"wrote_manifest={args.manifest}")
    print(f"evidence_packages_present={copied}")
    print(f"main_table_eligible_rows={eligible}")
    print(f"pending_rows={pending}")


if __name__ == "__main__":
    main()
