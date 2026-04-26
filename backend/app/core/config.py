"""
Application configuration loaded from environment variables.

Pydantic BaseSettings automatically reads from .env files and validates types.
If SUPABASE_URL is missing or not a valid URL, the app won't even start —
you’ll get a clear error instead of a mysterious runtime crash later.
"""

import json

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import AnyUrl, BeforeValidator, PostgresDsn, TypeAdapter, computed_field
from pydantic import field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


BACKEND_DIR = Path(__file__).resolve().parents[2]
LOCAL_DEV_JWT_SECRET = "local-development-jwt-secret-change-me"

_ANY_URL_ADAPTER = TypeAdapter(AnyUrl)
_POSTGRES_DSN_ADAPTER = TypeAdapter(PostgresDsn)


def parse_cors(v: Any) -> list[Any]:
    if isinstance(v, str):
        value = v.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("BACKEND_CORS_ORIGINS must be valid JSON") from exc
            if isinstance(parsed, list):
                return parsed
            raise ValueError("BACKEND_CORS_ORIGINS JSON value must be a list")
        return [i.strip() for i in value.split(",") if i.strip()]
    elif isinstance(v, list):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    """All app configuration in one place."""

    model_config = SettingsConfigDict(
        # Use backend/.env regardless of the shell's current working directory.
        env_file=BACKEND_DIR / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"

    # App environment
    JWT_SECRET_KEY: str = LOCAL_DEV_JWT_SECRET
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    REFRESH_COOKIE_NAME: str = "refresh_token"
    COOKIE_SECURE: bool = False
    FRONTEND_HOST: str = "http://localhost:5173"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        cors_origins = parse_cors(self.BACKEND_CORS_ORIGINS)
        return [str(origin).rstrip("/") for origin in cors_origins] + [
            self.FRONTEND_HOST.rstrip("/")
        ]

    # Anthropic (Claude) — used by the LangChain agent
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"

    # OpenAI — used only for embeddings (text-embedding-3-small)
    OPENAI_API_KEY: str = ""

    # Supabase connection
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # Supabase direct Postgres connection — used by PostgresSaver for
    # persistent conversation memory. Find this in Supabase Dashboard →
    # Settings → Database → Connection string (URI).
    SUPABASE_DB_URI: str

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

    @field_validator("FRONTEND_HOST")
    @classmethod
    def normalize_frontend_host(cls, value: str) -> str:
        return str(_ANY_URL_ADAPTER.validate_python(value.strip())).rstrip("/")

    @field_validator("SUPABASE_URL")
    @classmethod
    def validate_supabase_url(cls, value: str) -> str:
        return str(_ANY_URL_ADAPTER.validate_python(value.strip())).rstrip("/")

    @field_validator("SUPABASE_DB_URI")
    @classmethod
    def validate_supabase_db_uri(cls, value: str) -> str:
        return str(_POSTGRES_DSN_ADAPTER.validate_python(value.strip()))

    @field_validator("SUPABASE_SERVICE_KEY")
    @classmethod
    def validate_supabase_service_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("SUPABASE_SERVICE_KEY is required")
        return value

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret_key(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return value

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Self:
        if self.ENVIRONMENT != "local" and self.JWT_SECRET_KEY == LOCAL_DEV_JWT_SECRET:
            raise ValueError("JWT_SECRET_KEY must be set outside local development")
        return self

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
settings = Settings()  # pyright: ignore[reportCallIssue] - BaseSettings loads required values from env/.env.
