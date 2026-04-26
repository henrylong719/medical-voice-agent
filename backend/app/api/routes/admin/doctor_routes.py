"""Admin routes for managing doctors."""

from turtle import reset

from fastapi import APIRouter, HTTPException

from app.supabase_client import supabase
from app.models.doctor import DoctorCreateIn


router = APIRouter(prefix="/doctors", tags=["doctors"])


@router.get("")
def list_doctors(specialty_id: str | None = None):
    """
    List all doctors, optionally filtered by specialty.
    Returns doctors with their specialties joined.
    """
    query = (
        supabase.table("doctors")
        .select("*, doctor_specialties(specialty_id, specialties(name))")
        .eq("is_active", True)
    )

    if specialty_id:
        # Filter to only doctors who have this specialty
        # We need to query through the junction table
        doctor_ids_result = (
            supabase.table("doctor_specialties")
            .select("doctor_id")
            .eq("specialty_id", specialty_id)
            .execute()
        )

        doctor_ids = [row["doctor_id"] for row in doctor_ids_result.data]  # type: ignore[index]

        if not doctor_ids:
            return []

        query = query.in_("id", doctor_ids)

    result = query.order("full_name").execute()
    return result.data


@router.get("/{doctor_id}")
def get_doctor(doctor_id: str):
    """Get a single doctor with their specialties and availability."""

    result = (
        supabase.table("doctors")
        .select(
            "*, "
            "doctor_specialties(specialty_id, specialties(name)), "
            "doctor_availability(*)"
        )
        .eq("id", doctor_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Doctor not found")

    return result.data[0]


@router.post("", status_code=201)
def create_doctor(payload: DoctorCreateIn):
    """
    Create a new doctor with their specialties and availability.

    Uses a Postgres RPC function to run all three steps (insert doctor,
    link specialties, add availability) in a single database transaction.
    If any step fails, the entire operation is rolled back — no orphaned data.
    """
    # Convert availability models to dicts for JSONB parameter
    availability_data = [avail.model_dump() for avail in payload.availability]

    result = supabase.rpc(
        "create_doctor_with_details",
        {
            "p_full_name": payload.doctor.full_name,
            "p_email": payload.doctor.email,
            "p_phone": payload.doctor.phone,
            "p_image_url": payload.doctor.image_url,
            "p_specialty_ids": payload.specialty_ids,
            "p_availability": availability_data,
        },
    ).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create doctor")

    return result.data
