"""
Wood Token Inspection System - Main Entry Point
================================================
Starts the FastAPI web server providing:
  - Web UI dashboard at http://localhost:8080
  - REST API for camera/model/training management
  - WebSocket for live camera view
  - AI inference pipeline control
"""

import os
import sys
import yaml
import logging
import logging.handlers
import uvicorn
import signal
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

# Global config accessible by other modules
_config = {}
server = None  # Global server instance

def get_config():
    return _config


def setup_logging(config):
    log_level_str = config.get("general", {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_dir = Path(config.get("general", {}).get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File handler (daily rotation)
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_dir / "app.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def initialize_directories(config):
    dirs = [
        Path(config.get("general", {}).get("data_dir", "data")) / "db",
        Path(config.get("classification", {}).get("reference_dir", "data/references")),
        Path(config.get("classification", {}).get("embeddings_dir", "data/embeddings")),
        Path(config.get("event_logging", {}).get("events_dir", "data/events")),
        Path("data/references/good"),
        Path("data/references/bad"),
        Path("data/models"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_banner():
    banner = """
    ===========================================================
    |     WOOD TOKEN INSPECTION SYSTEM                        |
    |     Anomaly Detection on Conveyor Belt                  |
    |     Powered by YOLO + MobileNetV3 Embedding             |
    ================================e===========================
    """
    print(banner)

async def main():
    global _config, server

    config_path = os.environ.get("CONFIG_PATH", "pipeline_config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

    setup_logging(_config)
    print_banner()

    logger.info("Loading configuration from %s", config_path)
    initialize_directories(_config)

    host = _config.get("web", {}).get("host", "0.0.0.0")
    port = _config.get("web", {}).get("port", 8080)

    logger.info("Web dashboard: http://%s:%d", host, port)
    logger.info("API docs:      http://%s:%d/docs", host, port)

    config = uvicorn.Config(
        "web.app:app",
        host=host,
        port=port,
        log_config=None,
        reload=False,
    )
    server = uvicorn.Server(config)

    # Graceful shutdown handler
    def _signal_handler(signum, frame):
        logger.info("Shutdown signal received — exiting gracefully.")
        if server:
            server.should_exit = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        await server.serve()
    except Exception as e:
        logger.error("Failed to start web server: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server has shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application exiting.")
