import os


os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "service-role-key")
os.environ.setdefault(
    "SUPABASE_DB_URI",
    "postgresql://user:password@localhost:5432/postgres",
)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")
