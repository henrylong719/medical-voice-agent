"""Admin routes for managing appointments."""

from fastapi import APIRouter, Depends, HTTPException

from app.supabase_client import supabase

from app.api.deps import require_superuser


router = APIRouter(
    prefix="/appointments",
    tags=["appointments"],
    dependencies=[Depends(require_superuser)],
)

@router.get("")
def list_appointment(
    patient_id: str | None = None,
    doctor_id: str | None = None,
    status: str | None = None,
):
    """List appointments with optional filters."""
    query = supabase.table("appointments").select(
        "*, patients(full_name, date_of_birth, phone), doctors(full_name), specialties(name)"
    )

    if patient_id:
        query = query.eq("patient_id", patient_id)
    if doctor_id:
        query = query.eq("doctor_id", doctor_id)
    if status:
        query = query.eq("status", status)

    result = query.order("start_at", desc=True).execute()
    return result.data


@router.get("/{appointment_id}")
def get_appointment(appointment_id: str):
    """Get a single appointment with full details."""
    result = (
        supabase.table("appointments")
        .select(
            "*, patients(full_name, date_of_birth, phone), doctors(full_name), specialties(name)"
        )
        .eq("id", appointment_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return result.data[0]
