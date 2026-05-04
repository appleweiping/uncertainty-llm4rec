from pathlib import Path

from llm4rec.external_baselines.recbole_adapter import build_recbole_config
from llm4rec.external_baselines.bert4rec_adapter import bert4rec_config
from llm4rec.external_baselines.sasrec_adapter import sasrec_config
from llm4rec.external_baselines.lightgcn_adapter import lightgcn_config


def test_sasrec_recbole_config_contract(tmp_path: Path) -> None:
    cfg = sasrec_config(dataset_name="d", processed_dir=tmp_path, output_dir=tmp_path / "out", seed=13, training_config={"epochs": 3})
    rb = build_recbole_config(cfg, exported_dataset_dir=tmp_path / "out" / "d")
    assert rb["model"] == "SASRec"
    assert rb["dataset"] == "d"
    assert rb["seed"] == 13
    assert rb["epochs"] == 3
    assert rb["eval_args"]["order"] == "TO"
    assert rb["benchmark_filename"] == ["sasrec_train", "sasrec_valid", "sasrec_test"]
    assert "item_id_list" in rb["load_col"]["inter"]
    assert rb["alias_of_item_id"] == ["item_id_list"]


def test_lightgcn_recbole_config_contract(tmp_path: Path) -> None:
    cfg = lightgcn_config(dataset_name="d", processed_dir=tmp_path, output_dir=tmp_path / "out", seed=21)
    rb = build_recbole_config(cfg, exported_dataset_dir=tmp_path / "out" / "d")
    assert rb["model"] == "LightGCN"
    assert rb["dataset"] == "d"
    assert rb["seed"] == 21


def test_bert4rec_recbole_config_contract(tmp_path: Path) -> None:
    cfg = bert4rec_config(dataset_name="d", processed_dir=tmp_path, output_dir=tmp_path / "out", seed=34, training_config={"epochs": 5})
    rb = build_recbole_config(cfg, exported_dataset_dir=tmp_path / "out" / "d")
    assert rb["model"] == "BERT4Rec"
    assert rb["dataset"] == "d"
    assert rb["seed"] == 34
    assert rb["epochs"] == 5
    assert rb["eval_args"]["order"] == "TO"
    assert rb["benchmark_filename"] == ["sasrec_train", "sasrec_valid", "sasrec_test"]
    assert "item_id_list" in rb["load_col"]["inter"]
    assert rb["alias_of_item_id"] == ["item_id_list"]
