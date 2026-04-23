"""
Application configuration loaded from environment variables.

Pydantic BaseSettings automatically reads from .env files and validates types.
If SUPABASE_URL is missing or not a valid URL, the app won't even start —
you’ll get a clear error instead of a mysterious runtime crash later.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """All app configuration in one place."""

    # Supabase connection
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Supabase direct Postgres connection — used by PostgresSaver for
    # persistent conversation memory. Find this in Supabase Dashboard →
    # Settings → Database → Connection string (URI).
    supabase_db_uri: str = ""

    # Anthropic (Claude) — used by the LangChain agent
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # OpenAI — used only for embeddings (text-embedding-3-small)
    openai_api_key: str = ""
    
    # AssemblyAI — streaming speech-to-text (Phase 6)
    assemblyai_api_key: str = ""

    # Cartesia — streaming text-to-speech (Phase 6)
    cartesia_api_key: str = ""

    # LangSmith observability — traces every LLM call and tool invocation
    langsmith_api_key: str = ""
    langsmith_tracing: str = "true"
    langsmith_project: str = "medical-voice-agent"

    # Client scheduling config
    timezone: str = "America/Chicago"
    scheduling_horizon_days: int = 30  # how far ahead patients can book
    default_slot_duration_min: int = 30  # fallback if not specified per doctor

    model_config = {
        "env_file": ENV_FILE,  # load backend/.env regardless of cwd
        "env_file_encoding": "utf-8",
        "case_sensitive": False,  # SUPABASE_URL and supabase_url both work
    }

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
