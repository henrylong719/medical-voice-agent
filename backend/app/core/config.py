"""
Application configuration loaded from environment variables.

Pydantic BaseSettings automatically reads from .env files and validates types.
If SUPABASE_URL is missing or not a valid URL, the app won't even start —
you’ll get a clear error instead of a mysterious runtime crash later.
"""

import secrets

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BeforeValidator, computed_field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)

class Settings(BaseSettings):
    """All app configuration in one place."""
    
    model_config = SettingsConfigDict(
        # Use top level .env file (one level above ./backend/)
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
    
    
    API_V1_STR: str = "/api/v1"
    
    # App environment
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    FRONTEND_HOST: str = "http://localhost:5173"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []
    

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    
    # Anthropic (Claude) — used by the LangChain agent
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"

    # OpenAI — used only for embeddings (text-embedding-3-small)
    OPENAI_API_KEY: str = ""
    
     # Supabase connection
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # Supabase direct Postgres connection — used by PostgresSaver for
    # persistent conversation memory. Find this in Supabase Dashboard →
    # Settings → Database → Connection string (URI).
    SUPABASE_DB_URI: str = ""

    # AssemblyAI — streaming speech-to-text (Phase 6)
    ASSEMBLYAI_API_KEY: str = ""

    # Cartesia — streaming text-to-speech (Phase 6)
    CARTESIA_API_KEY: str = ""

    # LangSmith observability — traces every LLM call and tool invocation
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_PROJECT: str = "medical-voice-agent"

    # Client scheduling config
    TIMEZONE: str = "America/Chicago"
    SCHEDULING_HORIZON_DAYS: int = 30  # how far ahead patients can book
    DEFAULT_SLOT_DURATION_MIN: int = 30  # fallback if not specified per doctor

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Prefer backend/.env during local development so stale exported shell vars
        do not silently override the project configuration.
        """
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )


# Singleton instance — import this everywhere
# Created once when the module is first imported
settings = Settings()
