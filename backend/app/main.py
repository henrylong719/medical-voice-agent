"""
FastAPI application entry point.

Run with: uvicorn app.main:app --reload
The --reload flag watches for file changes during development.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

import os
from contextlib import asynccontextmanager


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.main import api_router

from app.core.config import settings

from app.agent.graph import cleanup_checkpointer

# ── LangSmith environment setup ──────────────────────────────
# LangSmith's SDK reads config from environment variables, not
# from our Pydantic settings. We push them into os.environ here
# at import time — before any LangChain modules initialize —
# so tracing is picked up automatically.
#
# PII redaction: we pre-initialize the LangSmith cached client
# singleton with our anonymizer BEFORE any LangChain code runs.
# LangChain's auto-tracing calls get_cached_client() internally,
# and since it's already initialized with our anonymizer, all
# traces will have PII masked automatically.
if settings.LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_TRACING", settings.LANGSMITH_TRACING)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)

    from langsmith.run_trees import get_cached_client
    from app.agent.pii_redactor import redact_pii

    # Pre-initialize the singleton with our anonymizer. This MUST
    # happen before any LangChain import that triggers tracing,
    # because get_cached_client() only accepts kwargs on first call.
    get_cached_client(anonymizer=redact_pii)
    logging.getLogger(__name__).info(
        "LangSmith PII redaction enabled — patient data will be masked in traces"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle: startup and shutdown.

    FastAPI's lifespan context manager runs code before the app starts
    accepting requests (above yield) and after it stops (below yield).
    We use it to cleanly close the PostgresSaver connection pool on shutdown.
    """
    yield
    # Shutdown: close the Postgres connection pool used by the checkpointer
    await cleanup_checkpointer()


app = FastAPI(
    title="Medical Voice Agent API",
    description="Backend API for the medical voice scheduling agent",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — allows the frontend (browser) to call our API.
# In production, you'd restrict origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.all_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "healthy", "version": "0.1.0"}
