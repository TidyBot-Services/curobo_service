"""Env-based config. Loaded once at startup."""

import os
from pathlib import Path


class Config:
    HOST = os.environ.get("CUROBO_HOST", "127.0.0.1")
    PORT = int(os.environ.get("CUROBO_PORT", "7000"))
    LOG_LEVEL = os.environ.get("CUROBO_LOG_LEVEL", "info")

    DEVICE = os.environ.get("CUROBO_DEVICE", "cuda:0")
    ROBOT_CFG = os.environ.get("CUROBO_ROBOT_CFG", "franka_tidyverse.yml")

    ASSETS_DIR = Path(os.environ.get(
        "CUROBO_ASSETS_DIR",
        str(Path(__file__).parent / "assets"),
    )).resolve()

    DEFAULT_ENV = os.environ.get("CUROBO_DEFAULT_ENV", "default")
    MAX_ENVS = int(os.environ.get("CUROBO_MAX_ENVS", "8"))
