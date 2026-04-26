from datetime import timedelta
from typing import Any, cast
from uuid import UUID

from fastapi import HTTPException, Request, status
from postgrest.exceptions import APIError

from app.core.config import settings
from app.core.security import (
    create_access_token,
    get_password_hash,
    hash_refresh_token,
    new_refresh_token,
    utc_now,
    verify_password_or_dummy,
)
from app.models.auth import LoginIn, RegisterIn
from app.supabase_client import supabase


USER_TABLE = "users"
REFRESH_TABLE = "refresh_sessions"


def _first_row(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None

    return cast(dict[str, Any], data[0])


def normalize_email(email: str) -> str:
    return email.strip().lower()

def get_user_by_email(email: str) -> dict[str, Any] | None:
    result = (
        supabase.table(USER_TABLE)
        .select("*")
        .eq("email", normalize_email(email))
        .limit(1)
        .execute()
    )
    
    return _first_row(result.data)


def get_user_by_id(user_id: str | UUID) -> dict[str, Any] | None:
    result = (
        supabase.table(USER_TABLE)
        .select("*")
        .eq("id", str(user_id))
        .limit(1)
        .execute()
    )
    return _first_row(result.data)


def register_user(payload: RegisterIn) -> dict[str, Any]:
    email = normalize_email(str(payload.email))
    if get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    try:
        result = (
            supabase.table(USER_TABLE)
            .insert(
                {
                    "email": email,
                    "full_name": payload.full_name,
                    "password_hash": get_password_hash(payload.password),
                }
            )
            .execute()
        )
    except APIError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        ) from None

    created_user = _first_row(result.data)
    if not created_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed",
        )

    return created_user

def authenticate_user(payload: LoginIn) -> dict[str, Any]:
    user = get_user_by_email(str(payload.email))
    password_hash = user["password_hash"] if user else None

    if not verify_password_or_dummy(payload.password, password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    return user


def create_token_pair(
    *,
    user: dict[str, Any],
    request: Request,
) -> tuple[str, str]:
    access_token = create_access_token(
        user_id=user["id"],
        email=user["email"],
        is_superuser=user["is_superuser"],
    )
    refresh_token = new_refresh_token()
    expires_at = utc_now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    client_host = request.client.host if request.client else None

    supabase.table(REFRESH_TABLE).insert(
        {
            "user_id": user["id"],
            "token_hash": hash_refresh_token(refresh_token),
            "expires_at": expires_at.isoformat(),
            "user_agent": request.headers.get("user-agent"),
            "ip_address": client_host,
        }
    ).execute()

    return access_token, refresh_token



def rotate_refresh_token(
    *,
    refresh_token: str,
    request: Request,
) -> tuple[dict[str, Any], str, str]:
    now = utc_now()
    token_hash = hash_refresh_token(refresh_token)

    session_result = (
        supabase.table(REFRESH_TABLE)
        .select("*")
        .eq("token_hash", token_hash)
        .is_("revoked_at", "null")
        .gt("expires_at", now.isoformat())
        .limit(1)
        .execute()
    )
    session = _first_row(session_result.data)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh session",
        )

    user = get_user_by_id(session["user_id"])
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh session",
        )

    supabase.table(REFRESH_TABLE).update(
        {"revoked_at": now.isoformat()}
    ).eq("id", session["id"]).execute()

    access_token, next_refresh_token = create_token_pair(
        user=user,
        request=request,
    )
    return user, access_token, next_refresh_token


def revoke_refresh_token(refresh_token: str | None) -> None:
    if not refresh_token:
        return

    supabase.table(REFRESH_TABLE).update(
        {"revoked_at": utc_now().isoformat()}
    ).eq("token_hash", hash_refresh_token(refresh_token)).execute()
