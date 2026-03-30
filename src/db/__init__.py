"""Database utilities for BatLab.

This package centralises MySQL helpers. Low-level connection handling lives in
`connection`.
"""

from src.db.connection import (
    get_connection,
    get_connection_params,
    get_call_library_data,
    load_training_records_df,
    load_detectors_from_db,
    load_species_from_db,
    delete_detectors,
    delete_species,
    delete_call_library_data,
    save_call_library_entries,
    save_detectors,
    save_species,
    save_training_data,
    update_call_library_data,
)

__all__ = [
    "get_connection",
    "get_connection_params",
    "get_call_library_data",
    "load_training_records_df",
    "save_detectors",
    "save_species",
    "delete_detectors",
    "delete_species",
    "delete_call_library_data",
    "save_call_library_entries",
    "save_training_data",
    "update_call_library_data",
    "load_detectors_from_db",
    "load_species_from_db",
]