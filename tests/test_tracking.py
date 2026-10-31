"""Tests for src/tracking.py, against a temporary SQLite MLflow store."""

import json
import logging

import pytest
from mlflow.tracking import MlflowClient

from src.config import load_config
from src.tracking import flatten, log_run, numeric_metrics, tracking_uri


@pytest.fixture
def cfg(tmp_path):
    return load_config([f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'"])


def client_for(cfg):
    return MlflowClient(tracking_uri=tracking_uri(cfg.tracking.dir))


def test_flatten_and_numeric_metrics():
    assert flatten({"a": {"b": 1, "c": {"d": [1, 2]}}, "e": None}) == {
        "a.b": "1", "a.c.d": "[1, 2]", "e": "None",
    }
    report = {"n": 3, "frac": 0.5, "flag": True, "name": "x", "counts": {"a": 1}}
    assert numeric_metrics(report) == {"n": 3.0, "frac": 0.5}


def test_log_run_records_config_tags_metrics_and_artifacts(cfg, tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"n_records": 10}))

    run_id = log_run(cfg, "decodec-test", "stage", {"n_records": 10}, [report])

    run = client_for(cfg).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["data.clustering.seed"] == "42"
    assert run.data.params["data.clustering.min_clusters_per_class"] == "20"
    assert run.data.metrics == {"n_records": 10.0}
    assert {"git_commit", "dirty"} <= set(run.data.tags)
    assert run.data.tags["dirty"] in {"true", "false"}
    artifacts = [a.path for a in client_for(cfg).list_artifacts(run_id)]
    assert artifacts == ["report.json"]


def test_more_than_100_params_are_all_logged(tmp_path):
    # log_batch takes at most 100 params per call; the config may grow past that
    many = [f"+extra.k{i}={i}" for i in range(150)]
    cfg = load_config([f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'", *many])
    run_id = log_run(cfg, "decodec-test", "stage", {})
    params = client_for(cfg).get_run(run_id).data.params
    assert params["extra.k149"] == "149"


def test_runs_share_one_experiment(cfg):
    first = log_run(cfg, "decodec-test", "a", {"x": 1})
    second = log_run(cfg, "decodec-test", "b", {"x": 2})
    client = client_for(cfg)
    assert (
        client.get_run(first).info.experiment_id
        == client.get_run(second).info.experiment_id
    )


def test_disabled_tracking_logs_nothing(tmp_path):
    cfg = load_config([
        "tracking.enabled=false",
        f"tracking.dir='{(tmp_path / 'mlruns').as_posix()}'",
    ])
    assert log_run(cfg, "decodec-test", "stage", {"x": 1}) is None
    assert not (tmp_path / "mlruns").exists()


def test_tracking_failure_is_a_warning_not_an_error(cfg, tmp_path, caplog):
    missing = tmp_path / "does_not_exist.json"
    with caplog.at_level(logging.WARNING):
        assert log_run(cfg, "decodec-test", "stage", {"x": 1}, [missing]) is None
    assert "MLflow tracking failed" in caplog.text
    # The half-logged run is closed as FAILED, not left running
    runs = client_for(cfg).search_runs(
        [client_for(cfg).get_experiment_by_name("decodec-test").experiment_id]
    )
    assert [r.info.status for r in runs] == ["FAILED"]
