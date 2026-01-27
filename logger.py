import logging
from typing import Optional

_LOGGER_INITIALIZED = False

def setup_file_logging(log_file: str, level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """
    Configure global file logging once.
    Call this exactly once at application startup.
    """
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    format_string = fmt or (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    formatter = logging.Formatter(format_string)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger anywhere in the codebase.
    """
    return logging.getLogger(name)
