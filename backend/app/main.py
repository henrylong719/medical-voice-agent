"""
FastAPI application entry point.
 
Run with: uvicorn app.main:app --reload
The --reload flag watches for file changes during development.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin.routes import router as admin_router

app = FastAPI(
	title="Medical Voice Agent API",
	description="Backend API for the medical voice scheduling agent",
 	version="0.1.0"
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

# Mount route groups
app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "healthy", "version": "0.1.0"}