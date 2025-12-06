"""Unit tests for evaris_server.middleware.auth module.

Tests JWT token handling, including:
- Token encoding and decoding
- Required claims validation
- Token expiration
- Error handling
"""

from datetime import datetime, timezone

import pytest
from jose import jwt

from evaris_server.config import Settings
from evaris_server.middleware.auth import (
    InternalAuthContext,
    InvalidTokenError,
    MissingTokenError,
    create_internal_token,
    decode_internal_token,
)


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        host="127.0.0.1",
        port=8080,
        environment="test",
        debug=True,
        database_url="postgresql://localhost:5432/test",
        internal_jwt_secret="test-secret-key-for-unit-tests",
        internal_jwt_algorithm="HS256",
        judge_model="openai/gpt-4o-mini",
        judge_provider="openrouter",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
    )


class TestInternalAuthContext:
    """Tests for InternalAuthContext model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating context with required fields only."""
        ctx = InternalAuthContext(
            organization_id="org_123",
            project_id="proj_456",
            user_id="user_789",
        )

        assert ctx.organization_id == "org_123"
        assert ctx.project_id == "proj_456"
        assert ctx.user_id == "user_789"
        assert ctx.permissions == []
        assert ctx.issued_at is None
        assert ctx.expires_at is None

    def test_create_with_all_fields(self) -> None:
        """Test creating context with all fields."""
        now = datetime.now(timezone.utc)

        ctx = InternalAuthContext(
            organization_id="org_123",
            project_id="proj_456",
            user_id="user_789",
            permissions=["read", "write"],
            issued_at=now,
            expires_at=now,
        )

        assert ctx.permissions == ["read", "write"]
        assert ctx.issued_at == now


class TestCreateInternalToken:
    """Tests for create_internal_token function."""

    def test_create_basic_token(self, settings: Settings) -> None:
        """Test creating a basic token."""
        token = create_internal_token(
            organization_id="org_test",
            project_id="proj_test",
            user_id="user_test",
            settings=settings,
        )

        # Verify it's a valid JWT
        assert isinstance(token, str)
        parts = token.split(".")
        assert len(parts) == 3  # header.payload.signature

    def test_token_contains_required_claims(self, settings: Settings) -> None:
        """Test that token contains all required claims."""
        token = create_internal_token(
            organization_id="org_abc",
            project_id="proj_xyz",
            user_id="user_123",
            settings=settings,
        )

        # Decode without verification to check claims
        payload = jwt.decode(
            token,
            settings.internal_jwt_secret,
            algorithms=[settings.internal_jwt_algorithm],
        )

        assert payload["organization_id"] == "org_abc"
        assert payload["project_id"] == "proj_xyz"
        assert payload["user_id"] == "user_123"
        assert "iat" in payload
        assert "exp" in payload

    def test_token_with_custom_permissions(self, settings: Settings) -> None:
        """Test token with custom permissions."""
        token = create_internal_token(
            organization_id="org_test",
            project_id="proj_test",
            user_id="user_test",
            settings=settings,
            permissions=["admin", "delete"],
        )

        payload = jwt.decode(
            token,
            settings.internal_jwt_secret,
            algorithms=[settings.internal_jwt_algorithm],
        )

        assert payload["permissions"] == ["admin", "delete"]

    def test_token_default_permissions(self, settings: Settings) -> None:
        """Test token has default read/write permissions."""
        token = create_internal_token(
            organization_id="org_test",
            project_id="proj_test",
            user_id="user_test",
            settings=settings,
        )

        payload = jwt.decode(
            token,
            settings.internal_jwt_secret,
            algorithms=[settings.internal_jwt_algorithm],
        )

        assert payload["permissions"] == ["read", "write"]

    def test_token_expiration(self, settings: Settings) -> None:
        """Test token expiration is set correctly."""
        token = create_internal_token(
            organization_id="org_test",
            project_id="proj_test",
            user_id="user_test",
            settings=settings,
            expires_in_seconds=600,  # 10 minutes
        )

        payload = jwt.decode(
            token,
            settings.internal_jwt_secret,
            algorithms=[settings.internal_jwt_algorithm],
        )

        # Expiration should be ~600 seconds from issued time
        assert payload["exp"] - payload["iat"] == 600


class TestDecodeInternalToken:
    """Tests for decode_internal_token function."""

    def test_decode_valid_token(self, settings: Settings) -> None:
        """Test decoding a valid token."""
        token = create_internal_token(
            organization_id="org_decode",
            project_id="proj_decode",
            user_id="user_decode",
            settings=settings,
        )

        ctx = decode_internal_token(token, settings)

        assert ctx.organization_id == "org_decode"
        assert ctx.project_id == "proj_decode"
        assert ctx.user_id == "user_decode"
        assert ctx.permissions == ["read", "write"]
        assert ctx.issued_at is not None
        assert ctx.expires_at is not None

    def test_decode_invalid_signature(self, settings: Settings) -> None:
        """Test that invalid signature raises error."""
        token = create_internal_token(
            organization_id="org_test",
            project_id="proj_test",
            user_id="user_test",
            settings=settings,
        )

        # Modify the token to invalidate signature
        tampered_token = token[:-5] + "XXXXX"

        with pytest.raises(InvalidTokenError) as exc_info:
            decode_internal_token(tampered_token, settings)

        assert "decode failed" in str(exc_info.value.detail).lower()

    def test_decode_expired_token(self, settings: Settings) -> None:
        """Test that expired token raises error."""
        token = create_internal_token(
            organization_id="org_test",
            project_id="proj_test",
            user_id="user_test",
            settings=settings,
            expires_in_seconds=-10,  # Already expired
        )

        with pytest.raises(InvalidTokenError) as exc_info:
            decode_internal_token(token, settings)

        assert "expired" in str(exc_info.value.detail).lower()

    def test_decode_missing_organization_id(self, settings: Settings) -> None:
        """Test that missing organization_id raises error."""
        # Create token manually without organization_id
        payload = {
            "project_id": "proj_test",
            "user_id": "user_test",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
        }

        token = jwt.encode(
            payload,
            settings.internal_jwt_secret,
            algorithm=settings.internal_jwt_algorithm,
        )

        with pytest.raises(InvalidTokenError) as exc_info:
            decode_internal_token(token, settings)

        assert "organization_id" in str(exc_info.value.detail)

    def test_decode_missing_project_id(self, settings: Settings) -> None:
        """Test that missing project_id raises error."""
        payload = {
            "organization_id": "org_test",
            "user_id": "user_test",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
        }

        token = jwt.encode(
            payload,
            settings.internal_jwt_secret,
            algorithm=settings.internal_jwt_algorithm,
        )

        with pytest.raises(InvalidTokenError) as exc_info:
            decode_internal_token(token, settings)

        assert "project_id" in str(exc_info.value.detail)

    def test_decode_missing_user_id(self, settings: Settings) -> None:
        """Test that missing user_id raises error."""
        payload = {
            "organization_id": "org_test",
            "project_id": "proj_test",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
        }

        token = jwt.encode(
            payload,
            settings.internal_jwt_secret,
            algorithm=settings.internal_jwt_algorithm,
        )

        with pytest.raises(InvalidTokenError) as exc_info:
            decode_internal_token(token, settings)

        assert "user_id" in str(exc_info.value.detail)

    def test_decode_wrong_algorithm(self, settings: Settings) -> None:
        """Test that wrong algorithm raises error."""
        payload = {
            "organization_id": "org_test",
            "project_id": "proj_test",
            "user_id": "user_test",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
        }

        # Encode with different algorithm
        token = jwt.encode(payload, "different-secret", algorithm="HS384")

        with pytest.raises(InvalidTokenError):
            decode_internal_token(token, settings)

    def test_decode_malformed_token(self, settings: Settings) -> None:
        """Test that malformed token raises error."""
        with pytest.raises(InvalidTokenError):
            decode_internal_token("not.a.valid.token", settings)

        with pytest.raises(InvalidTokenError):
            decode_internal_token("", settings)

        with pytest.raises(InvalidTokenError):
            decode_internal_token("gibberish", settings)


class TestErrorClasses:
    """Tests for error classes."""

    def test_invalid_token_error(self) -> None:
        """Test InvalidTokenError attributes."""
        error = InvalidTokenError("Custom message")

        assert error.status_code == 401
        assert error.detail == "Custom message"
        assert error.headers == {"WWW-Authenticate": "Bearer"}

    def test_invalid_token_error_default_message(self) -> None:
        """Test InvalidTokenError default message."""
        error = InvalidTokenError()

        assert "Invalid or expired" in error.detail

    def test_missing_token_error(self) -> None:
        """Test MissingTokenError attributes."""
        error = MissingTokenError()

        assert error.status_code == 401
        assert "Missing" in error.detail
        assert error.headers == {"WWW-Authenticate": "Bearer"}
