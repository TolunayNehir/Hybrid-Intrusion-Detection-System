"""Centralized logging utility for all system modules."""

import logging
import os
import sys
from config.settings import LOG_DIR, LOGGING


def get_logger(name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOGGING["level"]))
    formatter = logging.Formatter(LOGGING["format"])

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(LOG_DIR, "system.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
