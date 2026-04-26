from fastapi import APIRouter

from app.core.config import settings
from app.api.routes.admin import appointment_routes, block_routes, doctor_routes, patient_routes, slot_routes, specialty_routes
from app.api.routes.chat import chat_routes

api_router = APIRouter()

api_router.include_router(specialty_routes.router, prefix='/admin')
api_router.include_router(doctor_routes.router, prefix='/admin')
api_router.include_router(patient_routes.router, prefix='/admin')
api_router.include_router(appointment_routes.router, prefix='/admin')
api_router.include_router(block_routes.router, prefix='/admin')
api_router.include_router(slot_routes.router, prefix='/admin')

# Mount chat routes — the agent-facing API
api_router.include_router(chat_routes.router)


# if settings.ENVIRONMENT == "local":
    # api_router.include_router(private.router)
