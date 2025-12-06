"""Test fixtures for evaris-server integration tests.

This module provides fixtures for testing evaris-server directly,
bypassing evaris-web by generating internal JWT tokens.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from evaris_server.app import create_app
from evaris_server.config import Settings
from evaris_server.db import Database
from evaris_server.middleware.auth import create_internal_token


# ==============================================================================
# Settings Fixtures
# ==============================================================================


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with known values."""
    return Settings(
        host="127.0.0.1",
        port=8080,
        environment="test",
        debug=True,
        database_url="postgresql://localhost:5432/evaris_test",
        internal_jwt_secret="test-secret-for-testing-only",
        internal_jwt_algorithm="HS256",
        judge_model="openai/gpt-4o-mini",
        judge_provider="openrouter",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
    )


# ==============================================================================
# JWT Token Fixtures
# ==============================================================================


@pytest.fixture
def test_organization_id() -> str:
    """Test organization ID for RLS context (tenant root)."""
    return "org_test_123"


@pytest.fixture
def other_organization_id() -> str:
    """Different organization ID for isolation tests."""
    return "org_other_456"


@pytest.fixture
def test_project_id() -> str:
    """Test project ID for scoping within organization."""
    return "proj_test_123"


@pytest.fixture
def other_project_id() -> str:
    """Different project ID for isolation tests."""
    return "proj_other_456"


@pytest.fixture
def test_user_id() -> str:
    """Test user ID."""
    return "user_test_789"


@pytest.fixture
def valid_token(
    test_settings: Settings,
    test_organization_id: str,
    test_project_id: str,
    test_user_id: str,
) -> str:
    """Generate a valid JWT token for testing."""
    return create_internal_token(
        organization_id=test_organization_id,
        project_id=test_project_id,
        user_id=test_user_id,
        settings=test_settings,
        expires_in_seconds=3600,
    )


@pytest.fixture
def other_org_token(
    test_settings: Settings,
    other_organization_id: str,
    other_project_id: str,
    test_user_id: str,
) -> str:
    """Generate a JWT token for a different organization (isolation tests)."""
    return create_internal_token(
        organization_id=other_organization_id,
        project_id=other_project_id,
        user_id=test_user_id,
        settings=test_settings,
        expires_in_seconds=3600,
    )


@pytest.fixture
def expired_token(
    test_settings: Settings,
    test_organization_id: str,
    test_project_id: str,
    test_user_id: str,
) -> str:
    """Generate an expired JWT token for testing auth failures."""
    return create_internal_token(
        organization_id=test_organization_id,
        project_id=test_project_id,
        user_id=test_user_id,
        settings=test_settings,
        expires_in_seconds=-10,  # Already expired
    )


# ==============================================================================
# Mock Database Fixtures
# ==============================================================================


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self) -> None:
        self.executed_queries: list[tuple[str, tuple]] = []
        self.fetch_results: list[Any] = []
        self._fetch_index = 0

    async def execute(self, query: str, *args: Any) -> str:
        """Record executed query."""
        self.executed_queries.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Return mock fetch results."""
        self.executed_queries.append((query, args))
        if self._fetch_index < len(self.fetch_results):
            result = self.fetch_results[self._fetch_index]
            self._fetch_index += 1
            return result
        return []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        """Return mock fetchrow result."""
        results = await self.fetch(query, *args)
        return results[0] if results else None


class MockDatabase:
    """Mock database for testing without real PostgreSQL."""

    def __init__(self) -> None:
        self._pool_connected = True
        self._connections: list[MockConnection] = []

    async def connect(self) -> None:
        """Mock connect."""
        self._pool_connected = True

    async def disconnect(self) -> None:
        """Mock disconnect."""
        self._pool_connected = False

    @property
    def pool(self) -> Any:
        """Mock pool property."""
        if not self._pool_connected:
            raise RuntimeError("Database not connected")
        return MagicMock()

    async def with_org_context(
        self, organization_id: str, is_admin: bool = False
    ) -> AsyncGenerator[MockConnection, None]:
        """Mock context manager that yields a mock connection with RLS context."""
        conn = MockConnection()
        self._connections.append(conn)
        # Simulate setting RLS context
        await conn.execute("SET app.current_organization_id = $1", organization_id)
        if is_admin:
            await conn.execute("SET app.is_admin = 'true'")
        try:
            yield conn
        finally:
            await conn.execute("RESET app.current_organization_id")
            await conn.execute("RESET app.is_admin")

    async def fetch(self, query: str, *args: Any, project_id: str | None = None) -> list[Any]:
        """Mock fetch method."""
        if query == "SELECT 1":
            return [{"?column?": 1}]
        return []


@pytest.fixture
def mock_db() -> MockDatabase:
    """Create a mock database for testing."""
    return MockDatabase()


# ==============================================================================
# FastAPI TestClient Fixtures
# ==============================================================================


@pytest.fixture
def app(test_settings: Settings, mock_db: MockDatabase, monkeypatch: pytest.MonkeyPatch):
    """Create FastAPI app with mocked dependencies."""
    # Mock the settings
    monkeypatch.setattr("evaris_server.config.get_settings", lambda: test_settings)

    # Mock the database
    async def mock_get_database() -> MockDatabase:
        return mock_db

    monkeypatch.setattr("evaris_server.db.get_database", mock_get_database)
    monkeypatch.setattr("evaris_server.api.routes.get_database", mock_get_database)

    return create_app()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ==============================================================================
# Direct Test Client Helper
# ==============================================================================


class DirectTestClient:
    """Helper for making authenticated requests directly to evaris-server.

    This client bypasses evaris-web by generating internal JWT tokens
    and sending them in the X-Context-Token header.

    Usage:
        direct_client = DirectTestClient(client, valid_token)
        response = direct_client.post_assess(
            name="test-run",
            test_cases=[...],
            metrics=["faithfulness"],
        )
    """

    def __init__(self, client: TestClient, token: str):
        self.client = client
        self.token = token

    @property
    def headers(self) -> dict[str, str]:
        """Get headers with JWT token."""
        return {
            "X-Context-Token": self.token,
            "Content-Type": "application/json",
        }

    def post_assess(
        self,
        name: str,
        test_cases: list[dict[str, Any]],
        metrics: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Send assessment request to /internal/evaluate."""
        return self.client.post(
            "/internal/evaluate",
            headers=self.headers,
            json={
                "name": name,
                "test_cases": test_cases,
                "metrics": metrics,
                "metadata": metadata or {},
            },
        )

    def post_trace(
        self,
        name: str,
        spans: list[dict[str, Any]],
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Send trace request to /internal/trace."""
        return self.client.post(
            "/internal/trace",
            headers=self.headers,
            json={
                "name": name,
                "spans": spans,
                "duration_ms": duration_ms,
                "metadata": metadata or {},
            },
        )

    def post_log(
        self,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> Any:
        """Send log request to /internal/log."""
        return self.client.post(
            "/internal/log",
            headers=self.headers,
            json={
                "message": message,
                "level": level,
                "metadata": metadata or {},
                "timestamp": timestamp,
            },
        )

    def get_health(self) -> Any:
        """Check health endpoint (no auth required)."""
        return self.client.get("/health")


@pytest.fixture
def direct_client(client: TestClient, valid_token: str) -> DirectTestClient:
    """Create a direct test client with valid authentication."""
    return DirectTestClient(client, valid_token)


@pytest.fixture
def unauthenticated_client(client: TestClient) -> DirectTestClient:
    """Create a direct test client without authentication."""
    return DirectTestClient(client, "")


# ==============================================================================
# Sample Test Data Fixtures
# ==============================================================================


@pytest.fixture
def sample_test_case() -> dict[str, Any]:
    """Sample test case for assessment tests."""
    return {
        "input": "What is the capital of France?",
        "expected": "Paris",
        "actual_output": "The capital of France is Paris.",
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_test_cases(sample_test_case: dict[str, Any]) -> list[dict[str, Any]]:
    """Multiple sample test cases."""
    return [
        sample_test_case,
        {
            "input": "What is 2 + 2?",
            "expected": "4",
            "actual_output": "The answer is 4.",
            "metadata": {"source": "test"},
        },
    ]


@pytest.fixture
def sample_span() -> dict[str, Any]:
    """Sample span for trace tests."""
    return {
        "name": "llm_call",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_ms": 150.5,
        "input": {"prompt": "Hello"},
        "output": {"response": "Hi there!"},
        "metadata": {"model": "gpt-4"},
        "children": [],
    }


@pytest.fixture
def nested_spans(sample_span: dict[str, Any]) -> list[dict[str, Any]]:
    """Nested spans for trace tests."""
    return [
        {
            "name": "agent_run",
            "duration_ms": 500.0,
            "metadata": {},
            "children": [
                sample_span,
                {
                    "name": "tool_call",
                    "duration_ms": 50.0,
                    "input": {"tool": "search"},
                    "output": {"results": []},
                    "metadata": {},
                    "children": [],
                },
            ],
        }
    ]
