from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError

from app.core.security import decode_access_token
from app.models.auth import UserOut
from app.services.auth_service import get_user_by_id


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/admin/auth/login",
)


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserOut:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception from None

    user = get_user_by_id(user_id)
    if not user or not user["is_active"]:
        raise credentials_exception

    return UserOut.model_validate(user)


CurrentUser = Annotated[UserOut, Depends(get_current_user)]


def require_superuser(current_user: CurrentUser) -> UserOut:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


SuperUser = Annotated[UserOut, Depends(require_superuser)]
