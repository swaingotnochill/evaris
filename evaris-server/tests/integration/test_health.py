"""Integration tests for the health check endpoint."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint should return OK status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["database"] == "connected"

    def test_health_no_auth_required(self, client: TestClient) -> None:
        """Health endpoint should not require authentication."""
        # No X-Context-Token header
        response = client.get("/health")

        assert response.status_code == 200
        # Should still work without auth

    def test_health_returns_version(self, client: TestClient) -> None:
        """Health endpoint should return server version."""
        response = client.get("/health")

        data = response.json()
        assert "version" in data
        # Version should be a valid semver string
        assert data["version"].count(".") >= 1
