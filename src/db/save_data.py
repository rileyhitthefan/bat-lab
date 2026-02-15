"""Save user input data from BatLab app tabs 2, 3, and 4 to MySQL."""

import json
from typing import Any

from src.db.mysql_connection import get_connection


def save_detectors(detectors: list[dict[str, Any]]) -> list[str]:
    """
    Save detectors (Tab 2) to Locations table.
    Each detector: {"Detector": str, "Latitude": str, "Longitude": str}

    Returns list of error messages (empty on success).
    """
    errors = []
    try:
        with get_connection() as conn:
            if conn is None:
                return ["MySQL connector not installed. Run: pip install mysql-connector-python"]

            cursor = conn.cursor()
            for d in detectors:
                try:
                    cursor.execute(
                        """
                        INSERT INTO Locations (area_name, latitude, longitude)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE latitude = VALUES(latitude), longitude = VALUES(longitude)
                        """,
                        (
                            d["Detector"].strip(),
                            float(d["Latitude"]),
                            float(d["Longitude"]),
                        ),
                    )
                except Exception as e:
                    errors.append(f"Detector {d.get('Detector', '?')}: {e}")
            cursor.close()
    except Exception as e:
        errors.append(str(e))
    return errors


def save_species(species: list[dict[str, Any]]) -> list[str]:
    """
    Save species (Tab 3) to Bats table.
    Each species: {"Abbreviation": str, "LatinName": str, "CommonName": str}

    Returns list of error messages (empty on success).
    """
    errors = []
    try:
        with get_connection() as conn:
            if conn is None:
                return ["MySQL connector not installed. Run: pip install mysql-connector-python"]

            cursor = conn.cursor()
            for s in species:
                try:
                    cursor.execute(
                        """
                        INSERT INTO Bats (abbreviation, latin_name, common_name)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE latin_name = VALUES(latin_name), common_name = VALUES(common_name)
                        """,
                        (
                            s["Abbreviation"].strip(),
                            (s.get("LatinName") or "").strip() or None,
                            (s.get("CommonName") or "").strip() or None,
                        ),
                    )
                except Exception as e:
                    errors.append(f"Species {s.get('Abbreviation', '?')}: {e}")
            cursor.close()
    except Exception as e:
        errors.append(str(e))
    return errors


def save_training_entries(training_entries: list[dict[str, Any]]) -> list[str]:
    """
    Save training data (Tab 4) to Training_Records table.
    Each entry: {"Species": str, "Location": str, "FileCount": int, "FileNames": list[str]}

    Creates Training_Records table if it does not exist.
    Returns list of error messages (empty on success).
    """
    errors = []
    try:
        with get_connection() as conn:
            if conn is None:
                return ["MySQL connector not installed. Run: pip install mysql-connector-python"]

            cursor = conn.cursor()

            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Training_Records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    species_abbreviation VARCHAR(45) NOT NULL,
                    location_name VARCHAR(255) NOT NULL,
                    file_count INT NOT NULL,
                    file_names JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            for t in training_entries:
                try:
                    file_names_json = json.dumps(t.get("FileNames", []))
                    cursor.execute(
                        """
                        INSERT INTO Training_Records (species_abbreviation, location_name, file_count, file_names)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            t["Species"].strip(),
                            t["Location"].strip(),
                            int(t.get("FileCount", 0)),
                            file_names_json,
                        ),
                    )
                except Exception as e:
                    errors.append(f"Training entry {t.get('Species', '?')} @ {t.get('Location', '?')}: {e}")
            cursor.close()
    except Exception as e:
        errors.append(str(e))
    return errors
