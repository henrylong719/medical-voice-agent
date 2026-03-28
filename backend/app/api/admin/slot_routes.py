"""Admin routes for finding available appointment slots."""

from fastapi import APIRouter

from app.services.slot_engine import find_slots_for_specialty, find_slots_for_doctor


router = APIRouter(prefix="/slots", tags=["slots"])


@router.get("/by-specialty")
def get_slots_by_specialty(
    specialty_id: str,
    preferred_day: str | None = None,
    preferred_time: str | None = None,
    max_results: int = 10,
):
    """
    Find available slots across all doctors in a specialty.

    Used during triage: "When can I see a cardiologist?"

    Examples:
        GET /slots/by-specialty?specialty_id=...
        GET /slots/by-specialty?specialty_id=...&preferred_day=next tuesday
        GET /slots/by-specialty?specialty_id=...&preferred_day=tomorrow&preferred_time=morning
        GET /slots/by-specialty?specialty_id=...&preferred_day=earliest
    """
    return find_slots_for_specialty(
        specialty_id=specialty_id,
        preferred_day=preferred_day,
        preferred_time=preferred_time,
        max_results=max_results,
    )


@router.get("/by-doctor")
def get_slots_by_doctor(
    doctor_id: str,
    specialty_id: str,
    preferred_day: str | None = None,
    preferred_time: str | None = None,
    max_results: int = 10,
):
    """
    Find available slots for a specific doctor.

    Used when rescheduling: "When can I see Dr. Chen next?"

    Examples:
        GET /slots/by-doctor?doctor_id=...&specialty_id=...
        GET /slots/by-doctor?doctor_id=...&specialty_id=...&preferred_day=this week
    """
    return find_slots_for_doctor(
        doctor_id=doctor_id,
        specialty_id=specialty_id,
        preferred_day=preferred_day,
        preferred_time=preferred_time,
        max_results=max_results,
    )