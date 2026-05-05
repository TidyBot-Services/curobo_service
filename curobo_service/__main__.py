"""Entry point: `python -m curobo_service` starts the HTTP server."""

import os

import uvicorn

from . import apply_patches
from .config import Config


def main():
    # Apply curobo_service-shipped configuration patches to the local cuRobo
    # install before any cuRobo import happens (uvicorn imports server.py
    # which imports planner_core.py which imports curobo). Idempotent: skips
    # files already byte-identical. Set CUROBO_SERVICE_SKIP_PATCHES=1 to opt
    # out (e.g. when running against a pre-patched fork).
    if not os.environ.get("CUROBO_SERVICE_SKIP_PATCHES"):
        apply_patches.apply(verbose=True)

    uvicorn.run(
        "curobo_service.server:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL,
    )


if __name__ == "__main__":
    main()
