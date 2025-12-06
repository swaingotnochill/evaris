"""Unit tests for evaris_server.db.client module.

Tests the Database class and RLS context management, including:
- Database initialization
- Connection lifecycle
- RLS context setting and resetting
- Health check behavior
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris_server.config import Settings
from evaris_server.db.client import Database, close_database, get_database


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        host="127.0.0.1",
        port=8080,
        environment="test",
        debug=True,
        database_url="postgresql://localhost:5432/test",
        internal_jwt_secret="test-secret",
        internal_jwt_algorithm="HS256",
        judge_model="openai/gpt-4o-mini",
        judge_provider="openrouter",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
    )


class TestDatabaseInit:
    """Tests for Database initialization."""

    def test_init_with_settings(self, settings: Settings) -> None:
        """Test initialization with explicit settings."""
        db = Database(settings=settings)

        assert db.settings == settings
        assert db._client is None

    def test_init_without_settings(self) -> None:
        """Test initialization loads settings from environment."""
        with patch("evaris_server.db.client.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_get.return_value = mock_settings

            db = Database()

            assert db.settings == mock_settings


class TestDatabaseConnection:
    """Tests for Database connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, settings: Settings) -> None:
        """Test that connect creates Prisma client."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            MockPrisma.assert_called_once()
            mock_client.connect.assert_called_once()
            assert db._client is not None

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, settings: Settings) -> None:
        """Test that multiple connects don't create multiple clients."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()
            await db.connect()  # Second call should be no-op

            # Should only be called once
            assert MockPrisma.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self, settings: Settings) -> None:
        """Test that disconnect closes and clears client."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()
            await db.disconnect()

            mock_client.disconnect.assert_called_once()
            assert db._client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, settings: Settings) -> None:
        """Test disconnect when not connected is safe."""
        db = Database(settings=settings)

        # Should not raise
        await db.disconnect()


class TestDatabaseClient:
    """Tests for Database.client property."""

    def test_client_raises_when_not_connected(self, settings: Settings) -> None:
        """Test that accessing client before connect raises."""
        db = Database(settings=settings)

        with pytest.raises(RuntimeError) as exc_info:
            _ = db.client

        assert "not connected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_client_returns_prisma(self, settings: Settings) -> None:
        """Test that client returns Prisma instance after connect."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            assert db.client == mock_client


class TestRLSContext:
    """Tests for RLS context management."""

    @pytest.mark.asyncio
    async def test_with_org_context_sets_variable(self, settings: Settings) -> None:
        """Test that context manager sets RLS variable."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            async with db.with_org_context("org_123"):
                # Verify SET was called
                mock_client.execute_raw.assert_any_call(
                    "SET app.current_organization_id = 'org_123'"
                )

    @pytest.mark.asyncio
    async def test_with_org_context_resets_on_exit(self, settings: Settings) -> None:
        """Test that context manager resets RLS on exit."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            async with db.with_org_context("org_123"):
                pass

            # Verify RESET was called
            calls = [str(c) for c in mock_client.execute_raw.call_args_list]
            assert any("RESET app.current_organization_id" in c for c in calls)

    @pytest.mark.asyncio
    async def test_with_org_context_yields_client(self, settings: Settings) -> None:
        """Test that context manager yields Prisma client."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            async with db.with_org_context("org_123") as client:
                assert client == mock_client

    @pytest.mark.asyncio
    async def test_with_org_context_admin_mode(self, settings: Settings) -> None:
        """Test that admin mode sets additional variable."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            async with db.with_org_context("org_123", is_admin=True):
                pass

            # Verify admin flag was set
            calls = [str(c) for c in mock_client.execute_raw.call_args_list]
            assert any("app.is_admin" in c for c in calls)

    @pytest.mark.asyncio
    async def test_with_org_context_resets_on_exception(self, settings: Settings) -> None:
        """Test that RLS is reset even if exception occurs."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            MockPrisma.return_value = mock_client

            await db.connect()

            with pytest.raises(ValueError):
                async with db.with_org_context("org_123"):
                    raise ValueError("Test error")

            # RESET should still be called
            calls = [str(c) for c in mock_client.execute_raw.call_args_list]
            assert any("RESET" in c for c in calls)


class TestHealthCheck:
    """Tests for Database.health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, settings: Settings) -> None:
        """Test health check returns True when DB is responsive."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            mock_client.execute_raw = AsyncMock(return_value=None)
            MockPrisma.return_value = mock_client

            await db.connect()

            result = await db.health_check()

            assert result is True
            mock_client.execute_raw.assert_called_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self, settings: Settings) -> None:
        """Test health check returns False when DB is unreachable."""
        db = Database(settings=settings)

        with patch("evaris_server.db.client.Prisma") as MockPrisma:
            mock_client = AsyncMock()
            mock_client.execute_raw = AsyncMock(side_effect=Exception("Connection failed"))
            MockPrisma.return_value = mock_client

            await db.connect()

            result = await db.health_check()

            assert result is False


class TestGlobalDatabase:
    """Tests for global database instance functions."""

    @pytest.mark.asyncio
    async def test_get_database_creates_instance(self) -> None:
        """Test get_database creates and connects database."""
        # Reset global state
        import evaris_server.db.client as db_module
        db_module._database = None

        with patch("evaris_server.db.client.Database") as MockDB:
            mock_instance = AsyncMock()
            MockDB.return_value = mock_instance

            result = await get_database()

            MockDB.assert_called_once()
            mock_instance.connect.assert_called_once()
            assert result == mock_instance

    @pytest.mark.asyncio
    async def test_get_database_returns_same_instance(self) -> None:
        """Test get_database returns same instance on subsequent calls."""
        import evaris_server.db.client as db_module
        db_module._database = None

        with patch("evaris_server.db.client.Database") as MockDB:
            mock_instance = AsyncMock()
            MockDB.return_value = mock_instance

            result1 = await get_database()
            result2 = await get_database()

            # Should only create once
            assert MockDB.call_count == 1
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_close_database(self) -> None:
        """Test close_database disconnects and clears global."""
        import evaris_server.db.client as db_module

        mock_db = AsyncMock()
        db_module._database = mock_db

        await close_database()

        mock_db.disconnect.assert_called_once()
        assert db_module._database is None

    @pytest.mark.asyncio
    async def test_close_database_when_none(self) -> None:
        """Test close_database when no database exists."""
        import evaris_server.db.client as db_module
        db_module._database = None

        # Should not raise
        await close_database()
