import logging
import os
from datetime import datetime

# Create a logs directory under the current working directory
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Default log file path (timestamped)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

_DEFAULT_FORMAT = "[ %(asctime)s ] %(name)s:%(lineno)d - %(levelname)s - %(message)s"

def get_logger(logger_name: str = __name__, log_file_path: str | None = None) -> logging.Logger:
    """Return a configured logger that writes to logs directory and avoids duplicate handlers.

    Args:
        logger_name: Name of the logger to create or retrieve.
        log_file_path: Optional custom path to a log file. Defaults to timestamped file under logs/.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if getattr(logger, "_is_configured", False):
        return logger

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    # File handler
    file_path = log_file_path or LOG_FILE_PATH
    try:
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # As a fallback (e.g., if file path is invalid), at least attach a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Also add a stream handler for console output during development (optional)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Mark as configured to prevent duplicate handlers on repeated imports
    logger._is_configured = True  # type: ignore[attr-defined]

    return logger

__all__ = ["get_logger", "LOGS_DIR", "LOG_FILE_PATH"]