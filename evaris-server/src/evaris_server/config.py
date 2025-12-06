"""Configuration for evaris-server."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars not defined in this model
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    environment: str = "development"
    debug: bool = False

    # Database (PostgreSQL)
    database_url: str = "postgresql://localhost:5432/evaris"

    # Internal authentication (from evaris-web)
    internal_jwt_secret: str = "change-me-in-production"
    internal_jwt_algorithm: str = "HS256"

    # LLM Provider for Judge (Evaris's API keys)
    openrouter_api_key: Optional[str] = None
    judge_model: str = "openai/gpt-4o-mini"
    judge_provider: str = "openrouter"

    # Redis (for async job queue)
    redis_url: str = "redis://localhost:6379/0"

    # Observability
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
