"""Admin routes for managing doctor blocks (time off)."""
 
from fastapi import APIRouter
 
from app.supabase_client import supabase
from app.models.block import BlockIn
 
 
router = APIRouter(prefix="/blocks", tags=["blocks"])


@router.get("")
def list_blocks(doctor_id: str | None = None):
    """List doctor blocks (time off), optionally filtered by doctor."""
    query = supabase.table("doctor_blocks").select("*, doctors(full_name)")
 
    if doctor_id:
        query = query.eq("doctor_id", doctor_id)
 
    result = query.order("start_at").execute()
    return result.data


@router.post("", status_code=201)
def create_block(payload: BlockIn):
    """Add a new time-off block for a doctor."""
    result = (
        supabase.table("doctor_blocks")
        .insert(payload.model_dump(exclude_none=True))
        .execute()
    )
    return result.data[0]