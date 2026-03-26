"""
FastAPI application entry point.
 
Run with: uvicorn app.main:app --reload
The --reload flag watches for file changes during development.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin.specialty_routes import router as specialty_router
from app.api.admin.doctor_routes import router as doctor_router
from app.api.admin.patient_routes import router as patient_router
from app.api.admin.appointment_routes import router as appointment_router
from app.api.admin.block_routes import router as block_router

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


# Mount admin route groups — each domain gets its own prefix
API_PREFIX = "/api/v1/admin"
app.include_router(specialty_router, prefix=API_PREFIX)
app.include_router(doctor_router, prefix=API_PREFIX)
app.include_router(patient_router, prefix=API_PREFIX)
app.include_router(appointment_router, prefix=API_PREFIX)
app.include_router(block_router, prefix=API_PREFIX)


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "healthy", "version": "0.1.0"}