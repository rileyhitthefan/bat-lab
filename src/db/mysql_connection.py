"""MySQL connection for BatLab database."""

import os
from contextlib import contextmanager
from typing import Generator

try:
    import mysql.connector
    from mysql.connector import MySQLConnection
except ImportError:
    MySQLConnection = None
    mysql = None


def get_connection_params() -> dict:
    """Read MySQL connection params from environment or defaults."""
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "batlab_schema"),
    }


@contextmanager
def get_connection() -> Generator["MySQLConnection | None", None, None]:
    """
    Context manager for MySQL connections.
    Yields a connection or None if mysql-connector-python is not installed.
    """
    if mysql is None:
        yield None
        return

    conn = None
    try:
        params = get_connection_params()
        conn = mysql.connector.connect(**params)
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()
