"""Admin routes for managing specialties."""

from fastapi import APIRouter, HTTPException

from app.supabase_client import supabase
from app.models.specialty import SpecialtyOut


router = APIRouter(prefix="/specialties", tags=["specialties"])


@router.get("", response_model=list[SpecialtyOut])
def list_specialties():
    """List all available specialties."""
    result = supabase.table("specialties").select("*").order("name").execute()
    return result.data


@router.get("/{specialty_id}", response_model=SpecialtyOut)
def get_specialty(specialty_id: str):
    """Get a single specialty by ID."""
    result = supabase.table("specialties").select("*").eq("id", specialty_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Specialty not found")

    return result.data[0]
