"""Simple logger for ml package."""
import logging

def get_logger(name: str = "ml") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger
