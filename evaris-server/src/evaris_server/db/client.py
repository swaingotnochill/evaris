"""Database client with Prisma and RLS context management.

This module provides a database client that:
1. Uses Prisma for type-safe database access
2. Sets session variables for Row Level Security
3. Provides async context manager for scoped operations
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from prisma import Prisma
from prisma.engine.errors import EngineConnectionError

from evaris_server.config import Settings, get_settings


class Database:
    """Prisma database client with RLS context support.

    This client manages Prisma connections and provides methods to set
    PostgreSQL session variables for Row Level Security enforcement.

    The RLS context is set per-transaction using raw SQL.

    Usage:
        db = Database(settings)
        await db.connect()

        async with db.with_org_context("org_xxx") as client:
            # All queries respect RLS for org_xxx
            evals = await client.eval.find_many()

        await db.disconnect()
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize database client.

        Args:
            settings: Server settings. If None, loads from environment.
        """
        self.settings = settings or get_settings()
        self._client: Prisma | None = None

    async def connect(self) -> None:
        """Establish connection to PostgreSQL via Prisma."""
        if self._client is not None:
            return

        self._client = Prisma()
        await self._client.connect()

    async def disconnect(self) -> None:
        """Close the Prisma connection."""
        if self._client:
            await self._client.disconnect()
            self._client = None

    @property
    def client(self) -> Prisma:
        """Get the Prisma client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client

    async def _set_rls_context(
        self,
        organization_id: str,
        is_admin: bool = False,
    ) -> None:
        """Set RLS context variables for the current session.

        Args:
            organization_id: The organization ID to set as context
            is_admin: If True, sets admin bypass flag
        """
        await self._client.execute_raw(
            f"SET app.current_organization_id = '{organization_id}'"
        )
        if is_admin:
            await self._client.execute_raw("SET app.is_admin = 'true'")

    async def _reset_rls_context(self) -> None:
        """Reset RLS context variables."""
        await self._client.execute_raw("RESET app.current_organization_id")
        await self._client.execute_raw("RESET app.is_admin")

    @asynccontextmanager
    async def with_org_context(
        self,
        organization_id: str,
        is_admin: bool = False,
    ) -> AsyncGenerator[Prisma, None]:
        """Get the Prisma client with RLS context set for the given organization.

        This sets the PostgreSQL session variable that RLS policies check.
        All queries executed within this context will be scoped to the organization.

        Args:
            organization_id: The organization ID to set as context
            is_admin: If True, enables admin bypass for RLS

        Yields:
            Prisma client with RLS context set

        Example:
            async with db.with_org_context("org_123") as client:
                # RLS enforced - only sees org_123 data
                evals = await client.eval.find_many()
        """
        await self._set_rls_context(organization_id, is_admin)
        try:
            yield self.client
        finally:
            await self._reset_rls_context()

    async def health_check(self) -> bool:
        """Check database connectivity.

        Returns:
            True if connected and healthy, False otherwise
        """
        try:
            await self._client.execute_raw("SELECT 1")
            return True
        except (EngineConnectionError, Exception):
            return False


# Global database instance
_database: Database | None = None
_lock = asyncio.Lock()


async def get_database() -> Database:
    """Get the global database instance.

    Creates and connects the database on first call.
    Thread-safe via asyncio lock.

    Returns:
        Connected Database instance
    """
    global _database

    if _database is None:
        async with _lock:
            if _database is None:
                _database = Database()
                await _database.connect()

    return _database


async def close_database() -> None:
    """Close the global database instance."""
    global _database

    if _database is not None:
        await _database.disconnect()
        _database = None
