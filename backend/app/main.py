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

from app.config import settings

# ── LangSmith environment setup ──────────────────────────────
# LangSmith's SDK reads config from environment variables, not
# from our Pydantic settings. We push them into os.environ here
# at import time — before any LangChain modules initialize —
# so tracing is picked up automatically.
if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_TRACING", settings.langsmith_tracing)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)

from app.api.admin.specialty_routes import router as specialty_router
from app.api.admin.doctor_routes import router as doctor_router
from app.api.admin.patient_routes import router as patient_router
from app.api.admin.appointment_routes import router as appointment_router
from app.api.admin.block_routes import router as block_router
from app.api.admin.slot_routes import router as slot_router
from app.api.chat.routes import router as chat_router
from app.agent.graph import cleanup_checkpointer


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
    allow_origins=["*"],           # allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount admin route groups — each domain gets its own prefix
API_PREFIX = "/api/v1/admin"
app.include_router(specialty_router, prefix=API_PREFIX)
app.include_router(doctor_router, prefix=API_PREFIX)
app.include_router(patient_router, prefix=API_PREFIX)
app.include_router(appointment_router, prefix=API_PREFIX)
app.include_router(block_router, prefix=API_PREFIX)
app.include_router(slot_router, prefix=API_PREFIX)

# Mount chat routes — the agent-facing API
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "healthy", "version": "0.1.0"}