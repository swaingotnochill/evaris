"""Database module for evaris-server.

Handles PostgreSQL connections with RLS context management.
"""

from evaris_server.db.client import (
    Database,
    get_database,
)

__all__ = [
    "Database",
    "get_database",
]
