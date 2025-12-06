"""E2E test fixtures and configuration.

This module provides fixtures for end-to-end testing of the full Evaris
stack: Python SDK -> evaris-server -> PostgreSQL (Supabase).

Fixtures handle:
- Service availability checks (skip tests if services unavailable)
- JWT token generation for internal auth
- Test data setup and cleanup
- Authenticated client creation
"""

import os
import socket
from datetime import datetime, timezone
from typing import Any, Generator

import pytest

# Check if services are available before importing SDK
E2E_ENABLED = os.environ.get("EVARIS_E2E_TESTS", "0") == "1"
EVARIS_API_URL = os.environ.get("EVARIS_API_URL", "http://localhost:8081")
EVARIS_TEST_API_KEY = os.environ.get("EVARIS_TEST_API_KEY", "test_e2e_api_key")

# JWT configuration for internal auth (matches evaris-server/.env.test)
INTERNAL_JWT_SECRET = os.environ.get("INTERNAL_JWT_SECRET", "test-secret-for-testing-only")
INTERNAL_JWT_ALGORITHM = os.environ.get("INTERNAL_JWT_ALGORITHM", "HS256")

# Test tenant identifiers - read from environment (set in evaris-server/.env)
TEST_ORGANIZATION_ID = os.environ.get("TEST_ORGANIZATION_ID", "cmiql8v0e0001erfq78jce2yk")
TEST_PROJECT_ID = os.environ.get("TEST_PROJECT_ID", "cmiqlbo7e0005erfq7htyf6t1")
TEST_USER_ID = os.environ.get("TEST_USER_ID", "wuTuX0pdrF3ICICxiP7lIZhadfhUjYek")


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open on a host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _parse_url_for_port(url: str) -> tuple[str, int]:
    """Parse URL to extract host and port."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end (requires running services)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip e2e tests unless services are available or --e2e flag is set."""
    run_e2e = config.getoption("--e2e", default=False)

    if run_e2e or E2E_ENABLED:
        # Check if evaris-server is reachable
        host, port = _parse_url_for_port(EVARIS_API_URL)
        if not _is_port_open(host, port):
            skip_marker = pytest.mark.skip(
                reason=f"evaris-server not reachable at {EVARIS_API_URL}"
            )
            for item in items:
                if "e2e" in item.keywords:
                    item.add_marker(skip_marker)
    else:
        # Skip all e2e tests
        skip_marker = pytest.mark.skip(
            reason="E2E tests disabled. Run with --e2e or set EVARIS_E2E_TESTS=1"
        )
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_marker)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --e2e option to pytest."""
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="run end-to-end tests (requires running services)",
    )


# ==============================================================================
# Database Test Data Setup
# ==============================================================================

# Use hardcoded test entity IDs - these must exist in the database
# Create them manually in Supabase if they don't exist:
#   INSERT INTO organization (id, name, slug, "isPersonal") VALUES ('org_e2e_test', 'E2E Test Org', 'e2e-test-org', false);
#   INSERT INTO "Project" (id, name, "organizationId") VALUES ('proj_e2e_test', 'E2E Test Project', 'org_e2e_test');
_test_org_id: str = TEST_ORGANIZATION_ID
_test_project_id: str = TEST_PROJECT_ID


# ==============================================================================
# JWT Token Generation
# ==============================================================================


def create_internal_token(
    organization_id: str | None = None,
    project_id: str | None = None,
    user_id: str = TEST_USER_ID,
    permissions: list[str] | None = None,
    expires_in_seconds: int = 3600,
) -> str:
    """Create an internal JWT token for E2E testing.

    This generates a token compatible with evaris-server's auth middleware.
    The token contains organization_id, project_id, user_id claims required
    for Row Level Security (RLS) enforcement.

    Args:
        organization_id: Organization ID for RLS scoping (uses test org if None)
        project_id: Project ID for data isolation (uses test project if None)
        user_id: User ID for audit logging
        permissions: List of permissions (default: ['read', 'write'])
        expires_in_seconds: Token validity duration (default: 1 hour)

    Returns:
        Signed JWT token string
    """
    from jose import jwt

    # Use global test IDs if not provided (set by setup_test_data fixture)
    org_id = organization_id or _test_org_id or TEST_ORGANIZATION_ID
    proj_id = project_id or _test_project_id or TEST_PROJECT_ID

    now = datetime.now(timezone.utc)
    payload = {
        "organization_id": org_id,
        "project_id": proj_id,
        "user_id": user_id,
        "permissions": permissions or ["read", "write"],
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + expires_in_seconds,
    }

    return jwt.encode(
        payload,
        INTERNAL_JWT_SECRET,
        algorithm=INTERNAL_JWT_ALGORITHM,
    )


# ==============================================================================
# Client Fixtures
# ==============================================================================


@pytest.fixture
def e2e_api_url() -> str:
    """Get the evaris API URL for E2E tests."""
    return EVARIS_API_URL


@pytest.fixture
def e2e_api_key() -> str:
    """Get the test API key for E2E tests."""
    return EVARIS_TEST_API_KEY


@pytest.fixture
def internal_token() -> str:
    """Generate a fresh JWT token for E2E tests.

    Uses hardcoded test organization and project IDs.
    These must exist in Supabase for tests to pass.
    """
    return create_internal_token(
        organization_id=TEST_ORGANIZATION_ID,
        project_id=TEST_PROJECT_ID,
    )


@pytest.fixture
def evaris_client(internal_token: str):
    """Create an EvarisClient for E2E tests.

    Uses internal JWT token for direct server authentication,
    bypassing the API gateway.
    """
    from evaris import EvarisClient, RetryConfig

    # Use fast retry settings for tests
    config = RetryConfig(
        max_retries=2,
        base_delay_ms=100,
        jitter=False,
    )

    with EvarisClient(
        api_key=EVARIS_TEST_API_KEY,  # Still required but not used with internal_token
        base_url=EVARIS_API_URL,
        timeout=30.0,
        retry_config=config,
        internal_token=internal_token,
    ) as client:
        yield client


@pytest.fixture
async def async_evaris_client(internal_token: str):
    """Create an async EvarisClient for E2E tests."""
    from evaris import EvarisClient, RetryConfig

    config = RetryConfig(
        max_retries=2,
        base_delay_ms=100,
        jitter=False,
    )

    async with EvarisClient(
        api_key=EVARIS_TEST_API_KEY,
        base_url=EVARIS_API_URL,
        timeout=30.0,
        retry_config=config,
        internal_token=internal_token,
    ) as client:
        yield client


# ==============================================================================
# Test Data Fixtures
# ==============================================================================


@pytest.fixture
def sample_test_cases() -> list[dict[str, Any]]:
    """Sample test cases for E2E assessment tests."""
    from evaris import TestCase

    return [
        TestCase(
            input="What is the capital of France?",
            expected="Paris",
            actual_output="The capital of France is Paris.",
        ),
        TestCase(
            input="What is 2 + 2?",
            expected="4",
            actual_output="The answer is 4.",
        ),
        TestCase(
            input="Who wrote Romeo and Juliet?",
            expected="William Shakespeare",
            actual_output="Shakespeare wrote Romeo and Juliet.",
        ),
    ]


@pytest.fixture
def sample_spans() -> list:
    """Sample spans for E2E trace tests."""
    from datetime import datetime, timezone
    from evaris import Span

    now = datetime.now(timezone.utc)

    return [
        Span(
            name="llm_call",
            start_time=now,
            duration_ms=150.5,
            input={"prompt": "Hello, how are you?"},
            output={"response": "I'm doing well, thanks!"},
            metadata={"model": "gpt-4o-mini"},
        ),
        Span(
            name="tool_call",
            start_time=now,
            duration_ms=50.0,
            input={"tool": "search", "query": "weather"},
            output={"results": ["Sunny, 72F"]},
            metadata={"source": "weather_api"},
        ),
    ]


# ==============================================================================
# Cleanup Fixtures
# ==============================================================================


@pytest.fixture
def cleanup_assessments(evaris_client) -> Generator[list[str], None, None]:
    """Track and cleanup assessments created during tests.

    Usage:
        def test_something(cleanup_assessments, evaris_client):
            result = evaris_client.assess_sync(...)
            cleanup_assessments.append(result.assessment_id)
            # Assessment will be cleaned up after test
    """
    assessment_ids: list[str] = []
    yield assessment_ids

    # Cleanup logic would go here if the API supports deletion
    # For now, we just track what was created
    if assessment_ids:
        print(f"\nCreated {len(assessment_ids)} assessments during test: {assessment_ids}")
