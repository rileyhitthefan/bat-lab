"""
This file defines a simple logging utility used throughout the project.

Instead of using print statements everywhere, I use a logger so that messages are
formatted consistently and include timestamps and log levels.

This makes it easier to follow what the code is doing when training or running inference.
"""
import logging

def get_logger(name: str = "batlab") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger
