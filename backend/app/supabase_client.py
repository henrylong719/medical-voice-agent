"""
Supabase client singleton.
 
We create one Supabase client when the app starts and reuse it everywhere.
This is efficient — the client manages its own HTTP connection pool internally.
 
We use the service_role key (not the anon key) because this is a backend service
that needs full access. The anon key is for browser clients with Row Level Security.
"""

from supabase import create_client, Client

from app.config import settings


def _create_supabase_client() -> Client:
    """Create and return a Supabase client instance."""
    return create_client(
		supabase_url=settings.supabase_url,
  		supabase_key=settings.supabase_service_key,
	)
    
    
# Singleton - import this wherever you need database access
supabase: Client = _create_supabase_client()