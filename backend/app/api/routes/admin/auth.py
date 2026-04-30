"""Admin routes for auth."""

from fastapi import APIRouter, Cookie, Depends, Request, Response, status

from app.api.deps import CurrentUser
from app.core.config import settings
from app.models.auth import LoginIn, RegisterIn, TokenOut, UserOut
from app.services.auth_service import (
    authenticate_user,
    create_token_pair,
    register_user,
    revoke_refresh_token,
    rotate_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_COOKIE_PATH = f"{settings.API_V1_STR}/admin/auth"

def access_token_expires_in_seconds() -> int:
    return settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


def set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        path=AUTH_COOKIE_PATH,
    )


def clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path=AUTH_COOKIE_PATH,
    )


def token_response(user: dict, access_token: str) -> TokenOut:
    return TokenOut(
        access_token=access_token,
        expires_in=access_token_expires_in_seconds(),
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=TokenOut, status_code=201)
def register(
    payload: RegisterIn,
    request: Request,
    response: Response,
) -> TokenOut:
    user = register_user(payload)
    access_token, refresh_token = create_token_pair(user=user, request=request)
    set_refresh_cookie(response, refresh_token)
    return token_response(user, access_token)


@router.post("/login", response_model=TokenOut)
def login(
    payload: LoginIn,
    request: Request,
    response: Response,
) -> TokenOut:
    user = authenticate_user(payload)
    access_token, refresh_token = create_token_pair(user=user, request=request)
    set_refresh_cookie(response, refresh_token)
    return token_response(user, access_token)


@router.post("/refresh", response_model=TokenOut)
def refresh(
    request: Request,
    response: Response,
    refresh_token: str | None = Cookie(
        default=None,
        alias=settings.REFRESH_COOKIE_NAME,
    ),
) -> TokenOut:
    user, access_token, next_refresh_token = rotate_refresh_token(
        refresh_token=refresh_token or "",
        request=request,
    )
    set_refresh_cookie(response, next_refresh_token)
    return token_response(user, access_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    response: Response,
    refresh_token: str | None = Cookie(
        default=None,
        alias=settings.REFRESH_COOKIE_NAME,
    ),
) -> Response:
    revoke_refresh_token(refresh_token)
    clear_refresh_cookie(response)
    return response


@router.get("/me", response_model=UserOut)
def me(current_user: CurrentUser) -> UserOut:
    return current_user
