from pathlib import Path

from llm4rec.external_baselines.recbole_adapter import build_recbole_config
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


def test_lightgcn_recbole_config_contract(tmp_path: Path) -> None:
    cfg = lightgcn_config(dataset_name="d", processed_dir=tmp_path, output_dir=tmp_path / "out", seed=21)
    rb = build_recbole_config(cfg, exported_dataset_dir=tmp_path / "out" / "d")
    assert rb["model"] == "LightGCN"
    assert rb["dataset"] == "d"
    assert rb["seed"] == 21
