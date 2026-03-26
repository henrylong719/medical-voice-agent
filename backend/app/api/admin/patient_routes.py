
"""Admin routes for managing patients."""
 
from fastapi import APIRouter, HTTPException
 
from app.supabase_client import supabase
from app.models.patient import PatientIn
 
 
router = APIRouter(prefix="/patients", tags=["patients"])
 


@router.get("")
def list_patients():
    """List all patients."""
    result = supabase.table("patients").select("*").order("full_name").execute()
    return result.data


@router.get("/uin/{uin}")
def get_patient_by_uin(uin: str):
    """
    Look up a patient by their 9-digit UIN.
    This is how the voice agent identifies returning patients.
    """
    result = (
        supabase.table("patients")
        .select("*")
        .eq("uin", uin)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Patient not found")
    return result.data[0]


@router.post("", status_code=201)
def create_patient(payload: PatientIn):
    """Register a new patient."""
    # Check if UIN already exists
    existing = (
        supabase.table("patients")
        .select("id")
        .eq("uin", payload.uin)
        .execute()
    )
    if existing.data:
        raise HTTPException(status_code=409, detail="Patient with this UIN already exists")
 
    result = (
        supabase.table("patients")
        .insert(payload.model_dump())
        .execute()
    )
    return result.data[0]
 