"""
Application configuration loaded from environment variables.
 
Pydantic BaseSettings automatically reads from .env files and validates types.
If SUPABASE_URL is missing or not a valid URL, the app won't even start —
you'll get a clear error instead of a mysterious runtime crash later.
"""

from pydantic_settings import BaseSettings


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

     # LangSmith observability — traces every LLM call and tool invocation
     langsmith_api_key: str = ""
     langsmith_tracing: str = "true"
     langsmith_project: str = "medical-voice-agent"

     # Client scheduling config
     timezone: str = 'America/Chicago'
     scheduling_horizon_days: int = 30          # how far ahead patients can book
     default_slot_duration_min: int = 30        # fallback if not specified per doctor
 
     model_config = {
        "env_file": ".env",                    # load from .env in working directory
        "env_file_encoding": "utf-8",
        "case_sensitive": False,               # SUPABASE_URL and supabase_url both work
    }
     
     
# Singleton instance — import this everywhere
# Created once when the module is first imported
settings = Settings()
     
     
