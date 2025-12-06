"""Internal authentication middleware.

Verifies JWT tokens from evaris-web and sets PostgreSQL session context for RLS.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel

from evaris_server.config import Settings, get_settings


class InternalAuthContext(BaseModel):
    """Authenticated context from evaris-web.

    Contains the organization_id, project_id, and other metadata extracted from the JWT.
    Used to set PostgreSQL session variables for RLS enforcement.
    """

    organization_id: str
    project_id: str
    user_id: str
    permissions: list[str] = []
    issued_at: datetime | None = None
    expires_at: datetime | None = None


class InvalidTokenError(HTTPException):
    """Raised when the internal JWT token is invalid."""

    def __init__(self, detail: str = "Invalid or expired internal token"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class MissingTokenError(HTTPException):
    """Raised when the internal JWT token is missing."""

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing internal authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def decode_internal_token(token: str, settings: Settings) -> InternalAuthContext:
    """Decode and validate the internal JWT token from evaris-web.

    Args:
        token: The JWT token from X-Context-Token header
        settings: Server settings containing the JWT secret

    Returns:
        InternalAuthContext with organization_id, project_id, and metadata

    Raises:
        InvalidTokenError: If token is invalid, expired, or missing required claims
    """
    try:
        payload = jwt.decode(
            token,
            settings.internal_jwt_secret,
            algorithms=[settings.internal_jwt_algorithm],
        )
    except JWTError as e:
        raise InvalidTokenError(f"Token decode failed: {str(e)}")

    # Extract required claims
    organization_id = payload.get("organization_id")
    if not organization_id:
        raise InvalidTokenError("Token missing required 'organization_id' claim")

    project_id = payload.get("project_id")
    if not project_id:
        raise InvalidTokenError("Token missing required 'project_id' claim")

    user_id = payload.get("user_id")
    if not user_id:
        raise InvalidTokenError("Token missing required 'user_id' claim")

    # Check expiration (jose handles exp claim, but we double-check)
    exp = payload.get("exp")
    if exp:
        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
        if exp_dt < datetime.now(timezone.utc):
            raise InvalidTokenError("Token has expired")

    # Build context
    iat = payload.get("iat")
    return InternalAuthContext(
        organization_id=organization_id,
        project_id=project_id,
        user_id=user_id,
        permissions=payload.get("permissions", []),
        issued_at=datetime.fromtimestamp(iat, tz=timezone.utc) if iat else None,
        expires_at=datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None,
    )


async def verify_internal_request(
    x_context_token: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> InternalAuthContext:
    """FastAPI dependency to verify internal requests from evaris-web.

    This dependency:
    1. Extracts the JWT from X-Context-Token header
    2. Verifies the signature and expiration
    3. Returns the authenticated context with organization_id and project_id

    The returned context should be used to set PostgreSQL session variables
    for Row Level Security (RLS) enforcement.

    Usage:
        @app.post("/internal/evaluate")
        async def evaluate(
            auth: InternalAuthContext = Depends(verify_internal_request),
            db: Database = Depends(get_db),
        ):
            async with db.with_org_context(auth.organization_id) as client:
                ...

    Args:
        x_context_token: JWT token from evaris-web in header

    Returns:
        InternalAuthContext with organization_id, project_id, and metadata

    Raises:
        MissingTokenError: If no token provided
        InvalidTokenError: If token is invalid or expired
    """
    if not x_context_token:
        raise MissingTokenError()

    return decode_internal_token(x_context_token, settings)


def create_internal_token(
    organization_id: str,
    project_id: str,
    user_id: str,
    settings: Settings,
    permissions: list[str] | None = None,
    expires_in_seconds: int = 300,  # 5 minutes default
) -> str:
    """Create an internal JWT token (for testing or evaris-web reference).

    This function shows how evaris-web should create tokens.

    Args:
        organization_id: The organization ID (required for RLS)
        project_id: The project ID (required for scoping)
        user_id: The user ID (required for audit logging)
        settings: Server settings with JWT secret
        permissions: List of permissions (e.g., ['read', 'write'])
        expires_in_seconds: Token validity duration

    Returns:
        Signed JWT token string
    """
    now = datetime.now(timezone.utc)
    payload = {
        "organization_id": organization_id,
        "project_id": project_id,
        "user_id": user_id,
        "permissions": permissions or ["read", "write"],
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + expires_in_seconds,
    }

    return jwt.encode(
        payload,
        settings.internal_jwt_secret,
        algorithm=settings.internal_jwt_algorithm,
    )
