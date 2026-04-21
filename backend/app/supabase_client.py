"""
Supabase client singleton.

We create one Supabase client when the app starts and reuse it everywhere.
This is efficient — the client manages its own HTTP connection pool internally.

We use the service_role key (not the anon key) because this is a backend service
that needs full access. The anon key is for browser clients with Row Level Security.
"""

from httpx import Client as HttpxClient, Timeout
from postgrest.constants import DEFAULT_POSTGREST_CLIENT_TIMEOUT
from supabase import create_client, Client
from supabase.lib.client_options import SyncClientOptions

from app.config import settings


def _build_http_client() -> HttpxClient:
    """Provide a shared HTTP client so Supabase doesn't use deprecated kwargs."""
    return HttpxClient(
        timeout=Timeout(DEFAULT_POSTGREST_CLIENT_TIMEOUT),
        follow_redirects=True,
        http2=True,
    )


def _create_supabase_client() -> Client:
    """Create and return a Supabase client instance."""
    options = SyncClientOptions(httpx_client=_build_http_client())
    return create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_key,
        options=options,
    )


# Singleton - import this wherever you need database access
supabase: Client = _create_supabase_client()
