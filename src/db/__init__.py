"""Database utilities for BatLab.

This package centralises MySQL helpers. Low-level connection handling lives in
`mysql_connection`, and higher-level read/write helpers live in `connection`.
"""

from src.db.mysql_connection import get_connection, get_connection_params
from src.db.connection import (
    get_call_library_data,
    load_training_records_df,
    load_detectors_from_db,
    load_species_from_db,
    save_call_library_entries,
    save_detectors,
    save_species,
    save_training_data,
)

__all__ = [
    "get_connection",
    "get_connection_params",
    "get_call_library_data",
    "load_training_records_df",
    "save_detectors",
    "save_species",
    "save_call_library_entries",
    "save_training_data",
    "load_detectors_from_db",
    "load_species_from_db",
]