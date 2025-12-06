"""Main FastAPI application for evaris-server.

This is the internal evaluation server that handles heavy compute tasks:
- Running LLM judge evaluations
- Storing traces and logs
- Computing metrics

It is NOT exposed directly to the internet. Instead, evaris-web acts as
the API gateway and forwards authenticated requests here.

Architecture:
    Client -> evaris-web (auth, rate limiting) -> evaris-server (compute)
                                               -> PostgreSQL (shared, RLS)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evaris_server.api import router
from evaris_server.config import get_settings
from evaris_server.db import get_database


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize database connection pool
    - Shutdown: Close database connections gracefully
    """
    settings = get_settings()

    # Startup
    print(f"Starting evaris-server...")
    print(f"  Environment: {settings.environment}")
    print(f"  Judge model: {settings.judge_model}")

    # Initialize database
    db = await get_database()
    print(f"  Database: connected")

    yield

    # Shutdown
    print("Shutting down evaris-server...")
    await db.disconnect()
    print("  Database: disconnected")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title="evaris-server",
        description="Internal evaluation server for Evaris platform",
        version="0.1.0",
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
        openapi_url="/openapi.json" if settings.environment == "development" else None,
        lifespan=lifespan,
    )

    # CORS middleware (only needed for development)
    if settings.environment == "development":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include API routes
    app.include_router(router)

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "evaris_server.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )
