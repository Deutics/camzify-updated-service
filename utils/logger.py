#utils/logger.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

# ----------------------------------------------------------------------
# Configuration
_LOG_NAMESPACE = "CamzifyService"
_CONFIGURED = False

# Paths
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

# --- Create directories/files if they don't exist ---
os.makedirs(_LOG_DIR, exist_ok=True)
if not os.path.exists(_LOG_FILE):
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

# Format
_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ----------------------------------------------------------------------
def _configure_logging() -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    # File handler (rotating daily)
    file_handler = TimedRotatingFileHandler(
        _LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_FORMATTER)
    file_handler.suffix = "%Y-%m-%d"

    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_FORMATTER)

    base_logger = logging.getLogger(_LOG_NAMESPACE)
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    # Attach both handlers
    base_logger.addHandler(file_handler)
    base_logger.addHandler(console_handler)

    _CONFIGURED = True




def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger with hierarchical naming.

    Example:
        logger = get_logger(__name__)
        logger.info("something happened")

    Produces:
        2025-10-30 12:00:00 [INFO] [CamzifyService.capture.ForwarderWorker] something happened
    """
    _configure_logging()
    full_name = _LOG_NAMESPACE if not name else f"{_LOG_NAMESPACE}.{name}"
    return logging.getLogger(full_name)


# Default logger (optional)
logger = get_logger()
