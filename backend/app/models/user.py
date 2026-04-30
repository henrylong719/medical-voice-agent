from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(BaseModel):
    """Pydantic model for an application user."""

    id: UUID = Field(default_factory=uuid4)

    email: str = Field(min_length=1, max_length=255)
    full_name: str | None = Field(default=None, max_length=255)
    password_hash: str = Field(min_length=1)

    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
