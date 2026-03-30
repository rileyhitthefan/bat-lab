"""High-level MySQL helpers for BatLab.

This module implements the functions imported in ``src.db.__init__`` and used
by ``app.py`` to read/write the MySQL schema defined in ``src/batlab.sql``:

* ``Locations``      – detectors (Detector_name, latitude, longitude)
* ``Bats``           – species   (abbreviation, latin_name, common_name)
* ``Call_Library``   – training calls (files blob, latin_name FK, Detector_name FK)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Tuple

import pandas as pd
import mysql.connector
from mysql.connector import Error


# ---------------------------------------------------------------------------
# LOW-LEVEL CONNECTION
# ---------------------------------------------------------------------------

def get_connection_params() -> dict:
    """Read MySQL connection params from environment or defaults."""
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "root@1234"),
        "database": os.getenv("MYSQL_DATABASE", "batlab_schema"),
    }


def _connect():
    """Create a new MySQL connection using current env config."""
    return mysql.connector.connect(**get_connection_params())


def _close(conn) -> None:
    if conn is not None and hasattr(conn, "is_connected") and conn.is_connected():
        conn.close()


@contextmanager
def get_connection() -> Generator[mysql.connector.MySQLConnection, None, None]:
    """Context manager for MySQL connections."""
    conn = None
    try:
        conn = _connect()
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        _close(conn)


# ---------------------------------------------------------------------------
# DETECTORS (Locations table)
# ---------------------------------------------------------------------------

def save_detectors(detectors: Iterable[Dict[str, Any]]) -> List[str]:
    """Insert or update detector rows in the Locations table.

    Expects items with keys: Detector, Latitude, Longitude.
    Returns a list of error messages (empty on success).
    """
    detectors = list(detectors or [])
    if not detectors:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()

        sql = """
            INSERT INTO Locations (Detector_name, latitude, longitude)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                latitude = VALUES(latitude),
                longitude = VALUES(longitude)
        """
        rows = [
            (d["Detector"], float(d["Latitude"]), float(d["Longitude"]))
            for d in detectors
        ]
        cursor.executemany(sql, rows)
        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


def load_detectors_from_db() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load detectors from Locations table.

    Returns (detectors, errors) where detectors is a list of
    { "Detector ID", "Latitude", "Longitude" }.
    """
    detectors: List[Dict[str, Any]] = []
    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT Detector_name, latitude, longitude FROM Locations "
            "ORDER BY Detector_name"
        )
        for row in cursor.fetchall():
            detectors.append(
                {
                    "Detector ID": row["Detector_name"],
                    "Latitude": str(row["latitude"]),
                    "Longitude": str(row["longitude"]),
                }
            )
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return detectors, errors


def delete_detectors(detector_ids: Iterable[str]) -> List[str]:
    """Delete detector rows from Locations by Detector_name."""
    detector_ids = [str(d).strip() for d in (detector_ids or []) if str(d).strip()]
    if not detector_ids:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.executemany(
            "DELETE FROM Locations WHERE Detector_name = %s",
            [(d,) for d in detector_ids],
        )
        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


# ---------------------------------------------------------------------------
# SPECIES (Bats table)
# ---------------------------------------------------------------------------

def save_species(species: Iterable[Dict[str, Any]]) -> List[str]:
    """Insert or update species rows in the Bats table.

    Expects items with keys: Abbreviation, LatinName, CommonName.
    Returns a list of error messages (empty on success).
    """
    species = list(species or [])
    if not species:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()
        sql = """
            INSERT INTO Bats (abbreviation, latin_name, common_name)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                latin_name = VALUES(latin_name),
                common_name = VALUES(common_name)
        """
        # Use NULL for empty common_name so multiple species can have no common name
        # (Bats has UNIQUE on common_name; multiple NULLs are allowed, duplicate '' is not)
        rows = [
            (
                s["Abbreviation"],
                s["LatinName"],
                (s.get("CommonName") or "").strip() or None,
            )
            for s in species
        ]
        cursor.executemany(sql, rows)
        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors

config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'test', # Placeholder password for testing. In order to establish a connection, replace with new password
    'database': 'batlab_schema', # Placeholder database name for testing. In order to establish a connection, replace with new database name
    'auth_plugin': 'mysql_native_password'
}

def load_species_from_db() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load species from Bats table.

    Returns (species, errors) where species is a list of
    { "Abbreviation", "Latin Name", "Common Name" }.
    """
    species_list: List[Dict[str, Any]] = []
    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT abbreviation, latin_name, common_name "
            "FROM Bats ORDER BY abbreviation"
        )
        for row in cursor.fetchall():
            species_list.append(
                {
                    "Abbreviation": row["abbreviation"],
                    "Latin Name": row["latin_name"],
                    "Common Name": row.get("common_name") or "",
                }
            )
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return species_list, errors


def delete_species(abbreviations: Iterable[str]) -> List[str]:
    """Delete species rows from Bats by abbreviation."""
    abbreviations = [str(a).strip() for a in (abbreviations or []) if str(a).strip()]
    if not abbreviations:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.executemany(
            "DELETE FROM Bats WHERE abbreviation = %s",
            [(a,) for a in abbreviations],
        )
        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


# ---------------------------------------------------------------------------
# TRAINING METADATA
# ---------------------------------------------------------------------------

def load_training_records_df() -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate training records from Call_Library + Bats.

    Returns (DataFrame, errors) where DataFrame has columns:
    [species_abbreviation, location_name, file_count].
    """
    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        query = """
            SELECT
                b.abbreviation AS species_abbreviation,
                c.Detector_name AS location_name,
                COUNT(*) AS file_count
            FROM Call_Library AS c
            JOIN Bats AS b
              ON c.latin_name = b.latin_name
            GROUP BY b.abbreviation, c.Detector_name
            ORDER BY c.Detector_name, b.abbreviation
        """
        df = pd.read_sql(query, conn)
        return df, errors
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    empty = pd.DataFrame(
        columns=["species_abbreviation", "location_name", "file_count"]
    )
    return empty, errors


# ---------------------------------------------------------------------------
# CALL LIBRARY (Call_Library table)
# ---------------------------------------------------------------------------


def save_training_data(
    training_entries: Iterable[Dict[str, Any]],
    file_bytes: Dict[str, bytes],
) -> List[str]:
    """Persist training data into Call_Library so load_training_records_df works.

    Writes one row per WAV file into Call_Library (files, latin_name, Detector_name).
    latin_name is resolved from Bats by species abbreviation; Detector_name must
    exist in Locations. After this, load_training_records_df will return aggregates
    (species_abbreviation, location_name, file_count) from these rows.

    Accepts training_entries as a list of dicts with:
        - Species  – species abbreviation (must exist in Bats.abbreviation)
        - Location or Detector – detector ID (must exist in Locations.Detector_name)
        - FileNames – list of filenames (keys into file_bytes), or
        - file      – single filename (alternative to FileNames)
    """
    training_entries = list(training_entries or [])
    if not training_entries:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()
        latin_cache: Dict[str, str] = {}

        def _latin_for_abbr(abbr: str) -> str | None:
            if abbr in latin_cache:
                return latin_cache[abbr]
            cursor.execute(
                "SELECT latin_name FROM Bats WHERE abbreviation = %s", (abbr,)
            )
            row = cursor.fetchone()
            if not row:
                errors.append(
                    f"No matching species in Bats for abbreviation '{abbr}'."
                )
                latin_cache[abbr] = None  # type: ignore[assignment]
                return None
            latin = row[0]
            latin_cache[abbr] = latin
            return latin

        sql = """
            INSERT INTO Call_Library (files, latin_name, Detector_name)
            VALUES (%s, %s, %s)
        """

        for t in training_entries:
            abbr = t.get("Species")
            location = t.get("Location") or t.get("Detector")
            if not abbr or not location:
                errors.append(
                    "Each training entry must have Species and Location or Detector."
                )
                continue
            filenames = t.get("FileNames")
            if filenames is None and "file" in t:
                filenames = [t["file"]]
            filenames = list(filenames or [])

            latin = _latin_for_abbr(abbr)
            if not latin:
                continue

            for fname in filenames:
                blob = file_bytes.get(fname)
                if not blob:
                    errors.append(
                        f"No bytes for file '{fname}' (species {abbr}, detector {location})."
                    )
                    continue
                try:
                    cursor.execute(sql, (blob, latin, location))
                except Error as e:
                    errors.append(str(e))

        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


def update_call_library_data(
    updates: Iterable[Dict[str, Any]],
) -> List[str]:
    """Update species/detector assignments for existing Call_Library rows.

    Each update item expects:
        - old_species: abbreviation currently assigned in Call_Library
        - new_species: abbreviation to assign
        - old_detector: current Detector_name
        - new_detector: new Detector_name
        - file_count (optional): number of rows represented by the edited row
    """
    updates = list(updates or [])
    if not updates:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()
        latin_cache: Dict[str, str | None] = {}

        def _latin_for_abbr(abbr: str) -> str | None:
            if abbr in latin_cache:
                return latin_cache[abbr]
            cursor.execute(
                "SELECT latin_name FROM Bats WHERE abbreviation = %s",
                (abbr,),
            )
            row = cursor.fetchone()
            if not row:
                errors.append(f"No matching species in Bats for abbreviation '{abbr}'.")
                latin_cache[abbr] = None
                return None
            latin_cache[abbr] = row[0]
            return row[0]

        select_ids_sql = """
            SELECT c.idTraining
            FROM Call_Library AS c
            JOIN Bats AS b
              ON c.latin_name = b.latin_name
            WHERE b.abbreviation = %s
              AND c.Detector_name = %s
            ORDER BY c.idTraining
            LIMIT %s
        """
        update_sql = """
            UPDATE Call_Library
            SET latin_name = %s, Detector_name = %s
            WHERE idTraining = %s
        """

        for item in updates:
            old_species = (item.get("old_species") or "").strip()
            new_species = (item.get("new_species") or "").strip()
            old_detector = (item.get("old_detector") or "").strip()
            new_detector = (item.get("new_detector") or "").strip()
            try:
                file_count = int(item.get("file_count", 1))
            except (TypeError, ValueError):
                file_count = 1

            if not old_species or not new_species or not old_detector or not new_detector:
                errors.append(
                    "Each update must include old_species, new_species, old_detector, and new_detector."
                )
                continue

            if file_count < 1:
                continue
            if old_species == new_species and old_detector == new_detector:
                continue

            new_latin = _latin_for_abbr(new_species)
            if not new_latin:
                continue

            cursor.execute(select_ids_sql, (old_species, old_detector, file_count))
            ids_to_update = [row[0] for row in cursor.fetchall()]
            if len(ids_to_update) < file_count:
                errors.append(
                    f"Only found {len(ids_to_update)} of {file_count} rows for "
                    f"species '{old_species}' at detector '{old_detector}'."
                )
            for row_id in ids_to_update:
                cursor.execute(update_sql, (new_latin, new_detector, row_id))

        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


def delete_call_library_data(
    deletions: Iterable[Dict[str, Any]],
) -> List[str]:
    """Delete existing Call_Library rows represented by grouped training rows.

    Each item expects:
        - species: abbreviation currently assigned
        - detector: Detector_name currently assigned
        - file_count (optional): number of rows to remove for this group
    """
    deletions = list(deletions or [])
    if not deletions:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()

        select_ids_sql = """
            SELECT c.idTraining
            FROM Call_Library AS c
            JOIN Bats AS b
              ON c.latin_name = b.latin_name
            WHERE b.abbreviation = %s
              AND c.Detector_name = %s
            ORDER BY c.idTraining
            LIMIT %s
        """
        delete_sql = "DELETE FROM Call_Library WHERE idTraining = %s"

        for item in deletions:
            species = (item.get("species") or "").strip()
            detector = (item.get("detector") or "").strip()
            try:
                file_count = int(item.get("file_count", 0))
            except (TypeError, ValueError):
                file_count = 0

            if not species or not detector or file_count < 1:
                continue

            cursor.execute(select_ids_sql, (species, detector, file_count))
            ids_to_delete = [row[0] for row in cursor.fetchall()]
            if len(ids_to_delete) < file_count:
                errors.append(
                    f"Only found {len(ids_to_delete)} of {file_count} rows for "
                    f"species '{species}' at detector '{detector}'."
                )
            for row_id in ids_to_delete:
                cursor.execute(delete_sql, (row_id,))

        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


def save_call_library_entries(
    entries: Iterable[Dict[str, Any]],
    file_bytes: Dict[str, bytes],
) -> List[str]:
    """Insert WAV training calls into Call_Library.

    ``entries``: list of dicts with keys:
        Species  – abbreviation (e.g. 'MYLU')
        Location – detector name (Detector_name)
        FileNames – list of filenames (keys into file_bytes)

    ``file_bytes``: mapping from filename -> raw audio bytes.
    """
    entries = list(entries or [])
    if not entries:
        return []

    errors: List[str] = []
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()

        latin_cache: Dict[str, str] = {}

        def _latin_for_abbr(abbr: str) -> str | None:
            if abbr in latin_cache:
                return latin_cache[abbr]
            cursor.execute(
                "SELECT latin_name FROM Bats WHERE abbreviation = %s", (abbr,)
            )
            row = cursor.fetchone()
            if not row:
                errors.append(
                    f"No matching species in Bats for abbreviation '{abbr}'."
                )
                latin_cache[abbr] = None  # type: ignore[assignment]
                return None
            latin = row[0]
            latin_cache[abbr] = latin
            return latin

        sql = """
            INSERT INTO Call_Library (files, latin_name, Detector_name)
            VALUES (%s, %s, %s)
        """

        for entry in entries:
            abbr = entry["Species"]
            location = entry["Location"]
            latin = _latin_for_abbr(abbr)
            if not latin:
                # Error already recorded above.
                continue

            for fname in entry.get("FileNames", []):
                blob = file_bytes.get(fname)
                if not blob:
                    errors.append(
                        f"No bytes found for training file '{fname}' (species {abbr}, detector {location})."
                    )
                    continue
                cursor.execute(sql, (blob, latin, location))

        conn.commit()
    except Error as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))
    finally:
        _close(conn)

    return errors


def get_call_library_data(location: str | None = None) -> pd.DataFrame:
    """Return training calls from Call_Library as a DataFrame.

    The returned DataFrame has columns: [file, bat, location],
    where:
        * file     – absolute path to a temporary .wav file on disk
        * bat      – species abbreviation (from Bats.abbreviation)
        * location – detector name (Detector_name)
    """
    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()

        base_sql = """
            SELECT
                c.idTraining,
                c.files,
                c.Detector_name,
                b.abbreviation AS bat
            FROM Call_Library AS c
            JOIN Bats AS b
              ON c.latin_name = b.latin_name
        """
        params: Tuple[Any, ...] = ()
        if location:
            base_sql += " WHERE c.Detector_name = %s"
            params = (location,)

        cursor.execute(base_sql, params)
        rows = cursor.fetchall()

        # Materialise blobs to disk so the ML pipeline can work with file paths.
        project_root = Path(__file__).resolve().parents[2]
        out_dir = project_root / "data" / "db_call_library"
        out_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for row in rows:
            id_training, blob, det_name, bat_abbr = row
            if blob is None:
                continue
            file_path = out_dir / f"{id_training}.wav"
            if not file_path.exists():
                try:
                    with file_path.open("wb") as f:
                        f.write(blob)
                except Exception as e:
                    # Skip this record but continue with others.
                    print(f"Failed to write training file {file_path}: {e}")
                    continue

            records.append(
                {
                    "file": str(file_path),
                    "bat": bat_abbr,
                    "location": det_name,
                }
            )

        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error reading Call_Library: {e}")
        return pd.DataFrame(columns=["file", "bat", "location"])
    finally:
        _close(conn)