"""Evolutionary & swarm PID tuning for the Geomagic Touch (Phantom Omni)."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "robot_params.yaml"


def load_config(path=None):
    """Load the robot/optimisation YAML config as a nested dict."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
