"""
src/config.py

Loads the project configuration from config/ with Hydra's compose API.

What this module does:
    - Composes config/config.yaml and the groups it lists (data, ...)
    - Applies command-line overrides ("data.clustering.seed=7")
    - Returns a read-only config, so no step can change a setting mid-run

Why the compose API and not @hydra.main?
    @hydra.main takes over sys.argv, changes the working directory and creates
    a run folder per call. The pipeline uses relative paths (data/raw, ...)
    and its own command line, so it composes the config explicitly instead.

Why struct mode?
    A misspelled key in an override ("data.filtering.min_class_cont=50") is an
    error, not a new setting that nothing reads.
"""

import argparse
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf

# config/ at the repository root; DECODEC_CONFIG_DIR can point elsewhere
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def config_dir() -> Path:
    return Path(os.environ.get("DECODEC_CONFIG_DIR", DEFAULT_CONFIG_DIR)).resolve()


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """
    Compose the full config with optional Hydra-style overrides.
    Raises hydra.errors.HydraException on an unknown key or a bad override.
    """
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    OmegaConf.set_readonly(cfg, True)
    return cfg


def config_as_dict(cfg: DictConfig) -> dict:
    """Plain, resolved dict of the config, for reports and experiment tracking."""
    return OmegaConf.to_container(cfg, resolve=True)


def cli_config(description: str, argv: list[str] | None = None) -> DictConfig:
    """
    Parse --set KEY=VALUE options and return the composed config.
    Used by every script's __main__ block.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a setting, e.g. --set data.clustering.seed=7 (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        return load_config(args.overrides)
    except HydraException as e:
        # Only the first line: Hydra's follow-up hint suggests adding the key
        # with "+", which would turn a typo into a new, unused setting
        sys.exit(f"Config error: {str(e).splitlines()[0]}")
