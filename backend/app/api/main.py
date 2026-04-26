from fastapi import APIRouter

from app.core.config import settings
from app.api.routes.admin import specialty
from app.api.routes.chat import chat
from app.api.routes.admin import appointment, block, doctor, patient, slot

api_router = APIRouter()

api_router.include_router(specialty.router, prefix="/admin")
api_router.include_router(doctor.router, prefix="/admin")
api_router.include_router(patient.router, prefix="/admin")
api_router.include_router(appointment.router, prefix="/admin")
api_router.include_router(block.router, prefix="/admin")
api_router.include_router(slot.router, prefix="/admin")

# Mount chat routes — the agent-facing API
api_router.include_router(chat.router)


# if settings.ENVIRONMENT == "local":
# api_router.include_router(private.router)
