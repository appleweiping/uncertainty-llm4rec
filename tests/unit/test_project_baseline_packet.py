import json
from pathlib import Path

from scripts.prepare_project_baseline_packet import prepare_packet


def _fixture_processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "items.csv").write_text(
        "item_id,title,description,category,brand,domain,raw_text\n"
        "i1,One,,cat,,tiny,One\n"
        "i2,Two,,cat,,tiny,Two\n"
        "i3,Three,,cat,,tiny,Three\n",
        encoding="utf-8",
    )
    rows = [
        {"example_id": "u1:1", "user_id": "u1", "history": ["i1"], "target": "i2", "candidates": ["i2", "i3"], "split": "train", "domain": "tiny"},
        {"example_id": "u1:2", "user_id": "u1", "history": ["i1", "i2"], "target": "i3", "candidates": ["i2", "i3"], "split": "valid", "domain": "tiny"},
        {"example_id": "u1:3", "user_id": "u1", "history": ["i1", "i2", "i3"], "target": "i1", "candidates": ["i1", "i2"], "split": "test", "domain": "tiny"},
    ]
    for name in ["examples.jsonl", "candidate_sets.jsonl"]:
        with (processed / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return processed


def test_tallrec_packet_exports_pairwise_rows(tmp_path: Path) -> None:
    processed = _fixture_processed_dir(tmp_path)
    config = {
        "project": "tallrec",
        "official_repo": "https://github.com/SAI990323/TALLRec",
        "seed": 7,
        "output_dir": str(tmp_path / "packet"),
        "dataset": {"name": "tiny", "processed_dir": str(processed)},
        "candidate": {"protocol": "sampled", "size": 2, "include_target": True},
        "tallrec": {"negatives_per_train_example": 1, "max_history_items": 2},
    }
    manifest = prepare_packet(config=config, config_path=tmp_path / "cfg.yaml")
    packet = Path(manifest["output_dir"])
    assert (packet / "tallrec" / "train.json").exists()
    assert (packet / "tallrec" / "test_row_map.jsonl").exists()
    assert (packet / "tallrec" / "candidate_scores_template.csv").exists()
    train_rows = json.loads((packet / "tallrec" / "train.json").read_text(encoding="utf-8"))
    assert {row["output"] for row in train_rows} == {"Yes.", "No."}
    assert manifest["metric_contract"].startswith("External project")


def test_openp5_packet_exports_sequential_tasks(tmp_path: Path) -> None:
    processed = _fixture_processed_dir(tmp_path)
    config = {
        "project": "openp5",
        "official_repo": "https://github.com/agiresearch/OpenP5",
        "seed": 7,
        "output_dir": str(tmp_path / "packet"),
        "dataset": {"name": "tiny", "processed_dir": str(processed)},
        "candidate": {"protocol": "sampled", "size": 2, "include_target": True},
        "openp5": {"max_history_items": 2},
    }
    manifest = prepare_packet(config=config, config_path=tmp_path / "cfg.yaml")
    packet = Path(manifest["output_dir"])
    assert (packet / "openp5" / "train_sequential_tasks.jsonl").exists()
    assert (packet / "openp5" / "item_id_mapping.csv").exists()
    first = json.loads((packet / "openp5" / "test_sequential_tasks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["candidate_item_ids"] == ["i1", "i2"]
    assert first["target_item_id"] == "i1"


def test_generic_project_packet_exports_candidate_contract(tmp_path: Path) -> None:
    processed = _fixture_processed_dir(tmp_path)
    config = {
        "project": "lc_rec",
        "official_repo": "https://github.com/RUCAIBox/LC-Rec/",
        "seed": 7,
        "output_dir": str(tmp_path / "packet"),
        "dataset": {"name": "tiny", "processed_dir": str(processed)},
        "candidate": {"protocol": "sampled", "size": 2, "include_target": True},
        "lc_rec": {"task": "candidate_ranking", "max_history_items": 2},
    }
    manifest = prepare_packet(config=config, config_path=tmp_path / "cfg.yaml")
    packet = Path(manifest["output_dir"])
    project_dir = packet / "lc_rec"
    assert (project_dir / "project_tasks.jsonl").exists()
    assert (project_dir / "test_candidate_map.jsonl").exists()
    assert (project_dir / "candidate_scores_template.csv").exists()
    first = json.loads((project_dir / "test_project_tasks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["candidate_item_ids"] == ["i1", "i2"]
    assert first["candidate_texts"][0]["item_id"] == "i1"
    assert first["output_contract"].startswith("Score or rank only")
    assert manifest["project_display_name"] == "LC-Rec"
    assert manifest["adapter_contract"] == "generic_candidate_ranking"
    assert "--split test" in manifest["truce_import_command_template"]
