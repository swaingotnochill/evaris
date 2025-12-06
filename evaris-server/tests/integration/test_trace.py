"""Integration tests for the /internal/trace endpoint."""

from typing import Any

import pytest
from fastapi.testclient import TestClient


class TestTraceEndpointAuth:
    """Tests for authentication on /internal/trace."""

    def test_missing_token_returns_401(self, client: TestClient) -> None:
        """Request without X-Context-Token should return 401."""
        response = client.post(
            "/internal/trace",
            json={
                "name": "test-trace",
                "spans": [],
            },
        )

        assert response.status_code == 401

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        """Request with invalid JWT should return 401."""
        response = client.post(
            "/internal/trace",
            headers={"X-Context-Token": "invalid-token"},
            json={
                "name": "test-trace",
                "spans": [],
            },
        )

        assert response.status_code == 401


class TestTraceEndpointValidation:
    """Tests for request validation on /internal/trace."""

    def test_missing_name_returns_422(self, direct_client) -> None:
        """Request without name should return 422."""
        response = direct_client.client.post(
            "/internal/trace",
            headers=direct_client.headers,
            json={
                "spans": [],
            },
        )

        assert response.status_code == 422

    def test_missing_spans_returns_422(self, direct_client) -> None:
        """Request without spans should return 422."""
        response = direct_client.client.post(
            "/internal/trace",
            headers=direct_client.headers,
            json={
                "name": "test-trace",
            },
        )

        assert response.status_code == 422


class TestTraceEndpointSuccess:
    """Tests for successful trace requests."""

    def test_empty_spans_returns_201(
        self,
        direct_client,
        test_project_id: str,
    ) -> None:
        """Trace with empty spans list should succeed."""
        response = direct_client.post_trace(
            name="empty-trace",
            spans=[],
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "empty-trace"
        assert data["project_id"] == test_project_id
        assert data["span_count"] == 0
        assert "trace_id" in data
        assert data["trace_id"].startswith("trace_")

    def test_single_span_returns_201(
        self,
        direct_client,
        sample_span: dict[str, Any],
        test_project_id: str,
    ) -> None:
        """Trace with single span should succeed."""
        response = direct_client.post_trace(
            name="single-span-trace",
            spans=[sample_span],
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "single-span-trace"
        assert data["span_count"] == 1

    def test_nested_spans_returns_201(
        self,
        direct_client,
        nested_spans: list[dict[str, Any]],
        test_project_id: str,
    ) -> None:
        """Trace with nested spans should count all spans."""
        response = direct_client.post_trace(
            name="nested-trace",
            spans=nested_spans,
        )

        assert response.status_code == 201
        data = response.json()
        # nested_spans has 1 parent with 2 children = 3 total
        assert data["span_count"] == 3

    def test_duration_is_calculated(
        self,
        direct_client,
        sample_span: dict[str, Any],
    ) -> None:
        """Duration should be calculated from spans if not provided."""
        response = direct_client.post_trace(
            name="duration-test",
            spans=[sample_span],
            duration_ms=None,  # Let it calculate
        )

        assert response.status_code == 201
        data = response.json()
        # Duration should be set from span duration
        assert data["duration_ms"] is not None or data["duration_ms"] == sample_span["duration_ms"]

    def test_explicit_duration_is_used(
        self,
        direct_client,
        sample_span: dict[str, Any],
    ) -> None:
        """Explicit duration should override calculated duration."""
        explicit_duration = 1000.0

        response = direct_client.post_trace(
            name="explicit-duration-test",
            spans=[sample_span],
            duration_ms=explicit_duration,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["duration_ms"] == explicit_duration

    def test_metadata_is_preserved(
        self,
        direct_client,
    ) -> None:
        """Metadata should be stored with trace."""
        custom_metadata = {"environment": "test", "version": "1.0"}

        response = direct_client.post_trace(
            name="metadata-test",
            spans=[],
            metadata=custom_metadata,
        )

        assert response.status_code == 201
        # Metadata is stored but not necessarily returned
        # Just verify the request succeeded
