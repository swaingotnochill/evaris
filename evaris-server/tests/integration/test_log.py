"""Integration tests for the /internal/log endpoint."""

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient


class TestLogEndpointAuth:
    """Tests for authentication on /internal/log."""

    def test_missing_token_returns_401(self, client: TestClient) -> None:
        """Request without X-Context-Token should return 401."""
        response = client.post(
            "/internal/log",
            json={
                "message": "test log message",
                "level": "info",
            },
        )

        assert response.status_code == 401

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        """Request with invalid JWT should return 401."""
        response = client.post(
            "/internal/log",
            headers={"X-Context-Token": "invalid-token"},
            json={
                "message": "test log message",
                "level": "info",
            },
        )

        assert response.status_code == 401


class TestLogEndpointValidation:
    """Tests for request validation on /internal/log."""

    def test_missing_message_returns_422(self, direct_client) -> None:
        """Request without message should return 422."""
        response = direct_client.client.post(
            "/internal/log",
            headers=direct_client.headers,
            json={
                "level": "info",
            },
        )

        assert response.status_code == 422

    def test_invalid_level_returns_422(self, direct_client) -> None:
        """Request with invalid log level should return 422."""
        response = direct_client.client.post(
            "/internal/log",
            headers=direct_client.headers,
            json={
                "message": "test message",
                "level": "invalid_level",
            },
        )

        assert response.status_code == 422


class TestLogEndpointSuccess:
    """Tests for successful log requests."""

    def test_info_log_returns_201(
        self,
        direct_client,
        test_project_id: str,
    ) -> None:
        """Info level log should succeed."""
        response = direct_client.post_log(
            message="This is an info message",
            level="info",
        )

        assert response.status_code == 201
        data = response.json()
        assert data["level"] == "info"
        assert data["project_id"] == test_project_id
        assert "log_id" in data
        assert data["log_id"].startswith("log_")

    def test_debug_log_returns_201(
        self,
        direct_client,
        test_project_id: str,
    ) -> None:
        """Debug level log should succeed."""
        response = direct_client.post_log(
            message="Debug message with details",
            level="debug",
        )

        assert response.status_code == 201
        data = response.json()
        assert data["level"] == "debug"

    def test_warning_log_returns_201(
        self,
        direct_client,
        test_project_id: str,
    ) -> None:
        """Warning level log should succeed."""
        response = direct_client.post_log(
            message="Warning: something might be wrong",
            level="warning",
        )

        assert response.status_code == 201
        data = response.json()
        assert data["level"] == "warning"

    def test_error_log_returns_201(
        self,
        direct_client,
        test_project_id: str,
    ) -> None:
        """Error level log should succeed."""
        response = direct_client.post_log(
            message="Error: something went wrong",
            level="error",
        )

        assert response.status_code == 201
        data = response.json()
        assert data["level"] == "error"

    def test_default_level_is_info(
        self,
        direct_client,
    ) -> None:
        """Default log level should be info."""
        response = direct_client.client.post(
            "/internal/log",
            headers=direct_client.headers,
            json={
                "message": "Message without level",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["level"] == "info"

    def test_custom_timestamp_is_used(
        self,
        direct_client,
    ) -> None:
        """Custom timestamp should be preserved."""
        custom_time = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        response = direct_client.post_log(
            message="Message with custom timestamp",
            level="info",
            timestamp=custom_time.isoformat(),
        )

        assert response.status_code == 201
        data = response.json()
        # The created_at should match our custom timestamp
        assert "2024-01-15" in data["created_at"]

    def test_metadata_is_accepted(
        self,
        direct_client,
    ) -> None:
        """Metadata should be stored with log."""
        custom_metadata = {
            "user_id": "user_123",
            "action": "login",
            "ip_address": "192.168.1.1",
        }

        response = direct_client.post_log(
            message="User logged in",
            level="info",
            metadata=custom_metadata,
        )

        assert response.status_code == 201
        # Metadata is stored but not returned in response

    def test_long_message_is_accepted(
        self,
        direct_client,
    ) -> None:
        """Long log messages should be accepted."""
        long_message = "A" * 10000  # 10KB message

        response = direct_client.post_log(
            message=long_message,
            level="info",
        )

        assert response.status_code == 201


class TestLogEndpointTimestamp:
    """Tests for timestamp handling."""

    def test_default_timestamp_is_now(
        self,
        direct_client,
    ) -> None:
        """Without timestamp, should use current time."""
        before = datetime.now(timezone.utc)

        response = direct_client.post_log(
            message="Message without timestamp",
            level="info",
        )

        after = datetime.now(timezone.utc)

        assert response.status_code == 201
        data = response.json()

        # Parse the returned timestamp
        created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        # Should be between before and after
        assert before <= created_at <= after
