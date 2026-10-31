"""
src/tracking.py

MLflow experiment tracking for every pipeline stage and experiment.

What this module does:
    - Stores runs in a local SQLite database: mlruns/mlflow.db, with
      artifacts in mlruns/artifacts/ (both gitignored)
    - Logs one run per call: params = the full resolved config, flattened;
      tags = git commit and whether tracked files had uncommitted changes;
      metrics = the numbers a stage produced; artifacts = its report files
    - Never costs results: logging happens after a stage has saved its
      outputs, and any tracking error is only a warning

Why SQLite and not the plain ./mlruns folder?
    MLflow 3 deprecates the file store. Browse runs with:
        uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

Why the dirty tag?
    A result produced by uncommitted code or config cannot be traced back to
    a commit. The run is still kept, but marked dirty=true.
"""

import logging
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from src.config import config_as_dict

logger = logging.getLogger(__name__)


def tracking_uri(tracking_dir: str | Path) -> str:
    return f"sqlite:///{(Path(tracking_dir).resolve() / 'mlflow.db').as_posix()}"


def git_state() -> tuple[str, bool]:
    """(short commit hash, True if tracked files have uncommitted changes)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit, bool(changes)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """{"data": {"clustering": {"seed": 42}}} -> {"data.clustering.seed": "42"}."""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def numeric_metrics(report: dict) -> dict[str, float]:
    """The top-level numbers of a report (counts, fractions); bools excluded."""
    return {
        k: float(v) for k, v in report.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def log_run(
    cfg: DictConfig,
    experiment: str,
    run_name: str,
    metrics: dict[str, float],
    artifacts: list[Path] = (),
    tags: dict[str, str] | None = None,
) -> str | None:
    """
    Log one run. Returns the MLflow run ID, or None if tracking is disabled
    or failed (a failure is logged as a warning, never raised).
    """
    if not cfg.tracking.enabled:
        return None
    try:
        # Imported here: mlflow takes a few seconds to import, and stages
        # with tracking disabled should not pay for it
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_dir = Path(cfg.tracking.dir)
        tracking_dir.mkdir(parents=True, exist_ok=True)
        client = MlflowClient(tracking_uri=tracking_uri(tracking_dir))

        found = client.get_experiment_by_name(experiment)
        if found is None:
            experiment_id = client.create_experiment(
                experiment,
                artifact_location=(tracking_dir / "artifacts").resolve().as_uri(),
            )
        else:
            experiment_id = found.experiment_id

        commit, dirty = git_state()
        run = client.create_run(
            experiment_id,
            run_name=run_name,
            tags={"git_commit": commit, "dirty": str(dirty).lower(), **(tags or {})},
        )
        run_id = run.info.run_id
        try:
            params = flatten(config_as_dict(cfg))
            # log_batch accepts at most 100 params per call
            items = sorted(params.items())
            for start in range(0, len(items), 100):
                client.log_batch(run_id, params=[
                    mlflow.entities.Param(k, v) for k, v in items[start:start + 100]
                ])
            for key, value in metrics.items():
                client.log_metric(run_id, key, value)
            for path in artifacts:
                client.log_artifact(run_id, str(path))
            client.set_terminated(run_id, "FINISHED")
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise

        logger.info(f"Logged MLflow run {run_id} ({experiment} / {run_name})")
        return run_id
    except Exception as e:
        logger.warning(f"MLflow tracking failed, results are saved but not tracked: {e}")
        return None
