"""Download or initialize dataset entries for Storyflow / TRUCE-Rec."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.utils.config import load_simple_yaml

HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("raw_dir", "interim_dir", "processed_dir", "cache_dir"):
        if config.get(key):
            Path(str(config[key])).mkdir(parents=True, exist_ok=True)


def _write_chinese_report(
    *,
    dataset: str,
    title: str,
    failure: str,
    expected_path: str,
    resume_command: str,
    user_action: str,
    report_dir: Path = ROOT / "local_reports",
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{_timestamp()}-{dataset}-download-note.md"
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- 数据集：`{dataset}`",
                f"- 当前状态：{failure}",
                f"- 需要用户操作：{user_action}",
                f"- 期望路径：`{expected_path}`",
                f"- 恢复命令：`{resume_command}`",
                "",
                "本说明由下载脚本自动生成。该脚本没有静默跳过数据集，也没有伪造下载成功。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _urlopen(url: str, *, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(request, timeout=60)


def _download_text(url: str) -> str:
    with _urlopen(url) as response:
        return response.read().decode("utf-8").strip()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, *, resume: bool = True) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    headers: dict[str, str] = {}
    mode = "wb"
    if resume and partial.exists():
        size = partial.stat().st_size
        if size > 0:
            headers["Range"] = f"bytes={size}-"
            mode = "ab"
    try:
        with _urlopen(url, headers=headers) as response, partial.open(mode + "") as handle:
            if headers.get("Range") and response.status == 200:
                handle.seek(0)
                handle.truncate()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.HTTPError as error:
        if headers.get("Range") and error.code == 416:
            pass
        else:
            raise
    partial.replace(destination)


def _download_movielens_1m(config: dict[str, Any], *, force: bool) -> dict[str, Any]:
    _ensure_dirs(config)
    raw_dir = Path(str(config["raw_dir"]))
    interim_dir = Path(str(config["interim_dir"]))
    archive_path = raw_dir / str(config["archive_name"])
    extracted_dir = raw_dir / str(config["expected_archive_dir"])

    expected_md5 = None
    checksum_warning = None
    checksum_url = config.get("checksum_url")
    if checksum_url:
        try:
            checksum_text = _download_text(str(checksum_url))
            expected_md5 = checksum_text.split()[0]
        except Exception as exc:
            checksum_warning = str(exc)

    if archive_path.exists() and not force:
        status = "existing"
    else:
        _download_file(str(config["download_url"]), archive_path, resume=True)
        status = "downloaded"

    observed_md5 = _md5(archive_path)
    if expected_md5 and observed_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"MD5 mismatch for {archive_path}: expected {expected_md5}, got {observed_md5}"
        )

    if force or not extracted_dir.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(raw_dir)

    manifest = {
        "dataset": config["name"],
        "status": status,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "md5": observed_md5,
        "expected_md5": expected_md5,
        "checksum_warning": checksum_warning,
        "extracted_dir": str(extracted_dir),
        "source_url": config.get("source_url"),
        "download_url": config.get("download_url"),
    }
    interim_dir.mkdir(parents=True, exist_ok=True)
    (interim_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _download_hf_entry(config: dict[str, Any], *, allow_large: bool) -> dict[str, Any]:
    _ensure_dirs(config)
    dataset = str(config["name"])
    cache_dir = Path(str(config["cache_dir"]))
    raw_dir = Path(str(config["raw_dir"]))
    interim_dir = Path(str(config["interim_dir"]))
    hf_dataset = str(config["hf_dataset"])
    encoded_dataset = urllib.parse.quote(hf_dataset, safe="")
    parquet_url = f"{HF_DATASETS_SERVER}/parquet?dataset={encoded_dataset}"
    try:
        parquet_index = json.loads(_download_text(parquet_url))
    except Exception as exc:
        report = _write_chinese_report(
            dataset=dataset,
            title="Amazon Reviews 2023 Hugging Face 入口获取失败",
            failure=f"无法读取 Hugging Face Dataset Viewer parquet index：{exc}",
            expected_path=str(raw_dir),
            resume_command=f"python scripts/download_datasets.py --dataset {dataset}",
            user_action=(
                "检查网络/证书环境后重试；如在服务器下载 full 数据，请保留 Hugging Face parquet index、命令和日志。"
            ),
        )
        raise RuntimeError(f"HF parquet index failed; report: {report}") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "hf_parquet_index.json"
    index_path.write_text(
        json.dumps(parquet_index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    instructions = interim_dir / "download_instructions_zh.md"
    instructions.parent.mkdir(parents=True, exist_ok=True)
    instructions.write_text(
        "\n".join(
            [
                f"# {dataset} 下载入口说明",
                "",
                "本轮只建立 Hugging Face 可恢复入口，不在本地下载 full-scale Amazon 数据。",
                "",
                f"- Hugging Face dataset: `{hf_dataset}`",
                f"- Review config: `{config.get('hf_review_config')}`",
                f"- Metadata config: `{config.get('hf_meta_config')}`",
                f"- Split: `{config.get('hf_split')}`",
                f"- Parquet index cache: `{index_path}`",
                f"- 目标 raw 目录: `{raw_dir}`",
                "",
                "如需服务器下载，请在具备磁盘和网络的环境中使用 Hugging Face `datasets` 或 parquet shards，并记录 config、commit、日志和输出路径。",
                "",
                "示例：",
                "",
                "```python",
                "from datasets import load_dataset",
                f"reviews = load_dataset('{hf_dataset}', '{config.get('hf_review_config')}', split='{config.get('hf_split')}', trust_remote_code=True)",
                f"meta = load_dataset('{hf_dataset}', '{config.get('hf_meta_config')}', split='{config.get('hf_split')}', trust_remote_code=True)",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": dataset,
        "status": "hf_entry_cached",
        "hf_dataset": hf_dataset,
        "hf_review_config": config.get("hf_review_config"),
        "hf_meta_config": config.get("hf_meta_config"),
        "hf_split": config.get("hf_split"),
        "parquet_index": str(index_path),
        "instructions": str(instructions),
        "allow_large": allow_large,
    }
    (interim_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    if allow_large:
        report = _write_chinese_report(
            dataset=dataset,
            title="Amazon Reviews 2023 full download not executed locally",
            failure="脚本已缓存 Hugging Face parquet 入口；full download 属于 server-scale，需要用户在服务器环境执行。",
        expected_path=str(raw_dir),
        resume_command=f"python scripts/download_datasets.py --dataset {dataset}",
        user_action="在具备磁盘和网络的服务器环境执行 full download；本地只缓存入口说明。",
    )
        manifest["local_full_download_report"] = str(report)
    return manifest


def _planned_dataset(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_dirs(config)
    dataset = str(config["name"])
    report = _write_chinese_report(
        dataset=dataset,
        title="Planned dataset has no verified automatic downloader",
        failure="该数据集目前只是 planned/server-scale config，尚未确认 official 或 widely accepted source。",
        expected_path=str(config["raw_dir"]),
        resume_command="先更新 configs/datasets/steam.yaml 的 source_url 和下载协议，再运行下载脚本。",
        user_action="先确认 Steam/games 数据的 official 或 widely accepted source、license 和下载路径。",
    )
    return {"dataset": dataset, "status": "planned_only", "report": str(report)}


def _config_path(dataset: str) -> Path:
    return ROOT / "configs" / "datasets" / f"{dataset}.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Acknowledge server-scale dataset size. Amazon full download is still not run locally.",
    )
    args = parser.parse_args(argv)

    path = _config_path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Unknown dataset config: {path}")
    config = load_simple_yaml(path)
    dataset_type = str(config.get("type"))
    try:
        if dataset_type == "movielens_1m":
            manifest = _download_movielens_1m(config, force=args.force)
        elif dataset_type == "amazon_reviews_2023":
            manifest = _download_hf_entry(config, allow_large=args.allow_large)
        elif dataset_type == "planned":
            manifest = _planned_dataset(config)
        else:
            raise SystemExit(f"Unsupported dataset type: {dataset_type}")
    except Exception as exc:
        if dataset_type == "movielens_1m":
            report = _write_chinese_report(
                dataset=str(config.get("name", args.dataset)),
                title="MovieLens 1M 下载失败",
                failure=str(exc),
                expected_path=str(Path(str(config["raw_dir"])) / str(config["archive_name"])),
                resume_command=f"python scripts/download_datasets.py --dataset {args.dataset}",
                user_action=(
                    f"手动从 {config.get('download_url')} 下载 `{config.get('archive_name')}` "
                    f"并放到 `{Path(str(config['raw_dir'])) / str(config['archive_name'])}`，"
                    "然后重新运行下载脚本完成校验/解压。"
                ),
            )
            print(json.dumps({"status": "failed", "report": str(report)}, ensure_ascii=False))
            return 1
        raise
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
