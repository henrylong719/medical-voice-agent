
"""Admin routes for managing patients."""

from fastapi import APIRouter, HTTPException

from app.models.patient import PatientIdentifierIn, PatientIn, PatientSearchIn
from app.supabase_client import supabase


router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("")
def list_patients():
    """List all patients."""
    result = supabase.table("patients").select("*").order("full_name").execute()
    return result.data


@router.post("/search")
def search_patients(payload: PatientSearchIn):
    """Search for patients by demographics."""
    query = (
        supabase.table("patients")
        .select("*")
        .eq("full_name", payload.full_name)
        .eq("date_of_birth", payload.date_of_birth)
    )
    if payload.phone:
        query = query.eq("phone", payload.phone)

    result = query.execute()
    return result.data


@router.get("/{patient_id}")
def get_patient(patient_id: str):
    """Get a patient by internal UUID."""
    result = (
        supabase.table("patients")
        .select("*")
        .eq("id", patient_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Patient not found")
    return result.data[0]


@router.post("", status_code=201)
def create_patient(payload: PatientIn):
    """Register a new patient."""
    result = (
        supabase.table("patients")
        .insert(payload.model_dump())
        .execute()
    )
    return result.data[0]


@router.post("/{patient_id}/identifiers", status_code=201)
def add_patient_identifier(patient_id: str, payload: PatientIdentifierIn):
    """Attach an identifier such as an MRN or passport number."""
    patient = (
        supabase.table("patients")
        .select("id")
        .eq("id", patient_id)
        .execute()
    )
    if not patient.data:
        raise HTTPException(status_code=404, detail="Patient not found")

    data = payload.model_dump()
    data["patient_id"] = patient_id

    result = (
        supabase.table("patient_identifiers")
        .insert(data)
        .execute()
    )
    return result.data[0]
 
