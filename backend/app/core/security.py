import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings

password_hash = PasswordHash.recommended()
DUMMY_PASSWORD_HASH = password_hash.hash("not-a-real-password")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


def verify_password_or_dummy(
    plain_password: str,
    hashed_password: str | None,
) -> bool:
    """
    Run a hash verification even when the user does not exist.
    This reduces username/email enumeration through timing differences.
    """
    return verify_password(plain_password, hashed_password or DUMMY_PASSWORD_HASH)


def create_access_token(
    *,
    user_id: UUID | str,
    email: str,
    is_superuser: bool,
    expires_delta: timedelta | None = None,
) -> str:
    now = utc_now()
    expires_at = now + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "is_superuser": is_superuser,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": expires_at,
    }
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    

def decode_access_token(token: str) -> dict[str, Any]:
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )
    if payload.get("type") != "access":
        raise InvalidTokenError("Invalid token type")
    return payload


def new_refresh_token() -> str:
    return secrets.token_urlsafe(64)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
