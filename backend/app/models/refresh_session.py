from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RefreshSession(BaseModel):
    """Pydantic model for a persisted refresh-token session."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID

    token_hash: str = Field(min_length=1)
    expires_at: datetime
    revoked_at: datetime | None = None

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
