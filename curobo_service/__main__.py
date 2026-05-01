"""Entry point: `python -m curobo_service` starts the HTTP server."""

import uvicorn

from .config import Config


def main():
    uvicorn.run(
        "curobo_service.server:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL,
    )


if __name__ == "__main__":
    main()
