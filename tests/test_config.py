"""Tests for src/config.py: composition, overrides, typo rejection, read-only."""

import pytest
from hydra.errors import HydraException
from omegaconf.errors import ReadonlyConfigError

from src.config import cli_config, config_as_dict, load_config


def test_default_config_has_the_data_settings():
    cfg = load_config()
    assert cfg.data.uniprot.query == "reviewed:true AND ec:*"
    assert cfg.data.filtering.min_class_count == 100
    assert cfg.data.clustering.min_seq_id == 0.30
    assert cfg.data.clustering.min_clusters_per_class == 20
    assert cfg.data.clustering.min_eval_clusters == 3
    assert cfg.data.clustering.seed == 42


def test_override_keeps_its_type():
    cfg = load_config(["data.clustering.seed=7", "data.clustering.min_seq_id=0.4"])
    assert cfg.data.clustering.seed == 7
    assert isinstance(cfg.data.clustering.seed, int)
    assert cfg.data.clustering.min_seq_id == 0.4


def test_misspelled_key_is_an_error():
    with pytest.raises(HydraException, match="min_class_cont"):
        load_config(["data.filtering.min_class_cont=50"])


def test_config_is_read_only():
    cfg = load_config()
    with pytest.raises(ReadonlyConfigError):
        cfg.data.clustering.seed = 1


def test_config_dir_can_be_redirected(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("defaults:\n  - _self_\nonly_key: 1\n")
    monkeypatch.setenv("DECODEC_CONFIG_DIR", str(tmp_path))
    assert config_as_dict(load_config()) == {"only_key": 1}


def test_cli_set_and_typo_message(capsys):
    cfg = cli_config("test", ["--set", "data.clustering.seed=9"])
    assert cfg.data.clustering.seed == 9
    with pytest.raises(SystemExit) as exit_info:
        cli_config("test", ["--set", "data.filtering.min_class_cont=50"])
    message = str(exit_info.value)
    assert message.startswith("Config error:")
    assert "\n" not in message
