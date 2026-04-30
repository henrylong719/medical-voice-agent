# Production JWT Auth With React, FastAPI, and Supabase Postgres

This is a production-oriented auth example tailored to the current backend structure in this repository.

It does not use Supabase Auth. Supabase is used as the Postgres database through your existing backend-only `app.supabase_client.supabase` singleton. FastAPI owns password hashing, JWT signing, refresh-token rotation, logout, and authorization. React talks to FastAPI only.

## Why this shape fits this backend

Your backend already has these pieces:

- `backend/app/supabase_client.py`: a backend Supabase client using `SUPABASE_SERVICE_KEY`.
- `backend/app/core/security.py`: currently started for JWT/password helpers.
- `backend/app/models/user.py`: Pydantic user model.
- `backend/app/models/refresh_session.py`: Pydantic refresh-session model.
- `backend/app/api/routes/admin/auth.py`: auth route stub.
- `backend/app/api/deps.py`: empty dependency module, ideal for `get_current_user`.
- `backend/app/db/session.py`: empty, which is fine because this project uses Supabase/PostgREST rather than SQLAlchemy sessions.

The main design choice is:

- Access token: short-lived JWT returned to React and kept in memory.
- Refresh token: long-lived opaque random secret stored only in an HttpOnly cookie.
- Database stores only `sha256(refresh_token)`, never the raw refresh token.
- Logout revokes the refresh session and clears the cookie.
- React never receives the Supabase service key.
- React does not use `supabase.auth.*`.

## References

Primary docs used for this architecture:

- FastAPI OAuth2/JWT guidance: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
- Supabase Python client initialization: https://supabase.com/docs/reference/python/initializing
- Supabase data security guidance: https://supabase.com/docs/guides/database/secure-data
- Supabase Row Level Security and service-key notes: https://supabase.com/docs/guides/database/postgres/row-level-security

## Database migration

Create a new migration, for example:

`backend/app/db/sql/006_auth.sql`

```sql
-- ============================================================
-- Custom application auth
-- Supabase Auth is intentionally not used.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "citext";

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           CITEXT NOT NULL UNIQUE,
    full_name       TEXT,
    password_hash   TEXT NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active);

CREATE TABLE IF NOT EXISTS refresh_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      TEXT NOT NULL UNIQUE,
    expires_at      TIMESTAMPTZ NOT NULL,
    revoked_at      TIMESTAMPTZ,
    user_agent      TEXT,
    ip_address      INET,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_refresh_sessions_user_id
    ON refresh_sessions (user_id);

CREATE INDEX IF NOT EXISTS idx_refresh_sessions_valid
    ON refresh_sessions (token_hash, expires_at)
    WHERE revoked_at IS NULL;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_users_updated_at ON users;
CREATE TRIGGER set_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS set_refresh_sessions_updated_at ON refresh_sessions;
CREATE TRIGGER set_refresh_sessions_updated_at
    BEFORE UPDATE ON refresh_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

GRANT ALL PRIVILEGES ON TABLE users TO service_role;
GRANT ALL PRIVILEGES ON TABLE refresh_sessions TO service_role;
```

This creates `public.users`. Supabase Auth has a separate `auth.users` table, so use schema-qualified names like `public.users` in raw SQL whenever there is any chance of ambiguity.

## Backend configuration

Your `.env.example` already contains auth variables, but `backend/app/core/config.py` currently does not define all of them. Add these settings to the `Settings` class.

```python
# Auth
JWT_SECRET_KEY: str = secrets.token_urlsafe(64)
JWT_ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
REFRESH_TOKEN_EXPIRE_DAYS: int = 7
REFRESH_COOKIE_NAME: str = "refresh_token"
COOKIE_SECURE: bool = False
```

For production:

```env
JWT_SECRET_KEY=generate-a-long-random-secret-64-plus-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
COOKIE_SECURE=true
BACKEND_CORS_ORIGINS=https://your-frontend.example.com
FRONTEND_HOST=https://your-frontend.example.com
```

If your frontend and backend are on different sites in production, use `SameSite=None` and `Secure=true` for the refresh cookie. If they are same-site subdomains, `SameSite=Lax` is usually simpler.

## Backend dependencies

Your backend already includes:

- `fastapi`
- `supabase`
- `pyjwt`
- `pwdlib`

For strict email validation, add Pydantic email support:

```bash
uv add "pydantic[email]"
```

If you do not want that dependency, change `EmailStr` below to `str` and normalize/validate email yourself.

## `app/core/security.py`

Replace the current skeleton with:

```python
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
```

## `app/models/auth.py`

Create:

```python
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str | None = None
    is_active: bool
    is_superuser: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut
```

## `app/services/auth_service.py`

Create:

```python
from datetime import timedelta
from typing import Any
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
    return result.data[0] if result.data else None


def get_user_by_id(user_id: str | UUID) -> dict[str, Any] | None:
    result = (
        supabase.table(USER_TABLE)
        .select("*")
        .eq("id", str(user_id))
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


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

    return result.data[0]


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
    if not session_result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh session",
        )

    session = session_result.data[0]
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
```

## `app/api/deps.py`

Create:

```python
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
```

## `app/api/routes/admin/auth.py`

Replace the route stub with:

```python
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
```

## Wire the auth router

Update `backend/app/api/main.py`:

```python
from app.api.routes.admin import appointment, auth, block, doctor, patient, slot

api_router.include_router(auth.router, prefix="/admin")
```

This makes the routes:

- `POST /api/v1/admin/auth/register`
- `POST /api/v1/admin/auth/login`
- `POST /api/v1/admin/auth/refresh`
- `POST /api/v1/admin/auth/logout`
- `GET /api/v1/admin/auth/me`

## Protect existing admin routes

For admin routes like doctors, patients, appointments, and slots, add the dependency at router level:

```python
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user


router = APIRouter(
    prefix="/patients",
    tags=["patients"],
    dependencies=[Depends(get_current_user)],
)
```

For stricter admin-only access, add:

```python
from fastapi import HTTPException, status

from app.api.deps import CurrentUser


def require_superuser(current_user: CurrentUser) -> CurrentUser:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user
```

Then:

```python
router = APIRouter(
    prefix="/doctors",
    tags=["doctors"],
    dependencies=[Depends(require_superuser)],
)
```

## CORS requirements

Your `app/main.py` already has:

```python
allow_credentials=True
```

That is required because React needs to send the HttpOnly refresh cookie on `/refresh` and `/logout`.

In production, do not use wildcard origins with credentials. Configure exact origins:

```env
BACKEND_CORS_ORIGINS=https://app.example.com
FRONTEND_HOST=https://app.example.com
```

## React auth client

This assumes a Vite React app.

Install:

```bash
npm install react-router-dom
```

Create:

`src/auth/api.ts`

```ts
export type User = {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

let accessToken: string | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (response.status === 204) {
    return undefined as T;
  }

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const message =
      data?.detail ?? data?.message ?? `Request failed with ${response.status}`;
    throw new Error(message);
  }

  return data as T;
}

export async function refreshAccessToken(): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/admin/auth/refresh`, {
    method: "POST",
    credentials: "include",
  });

  const data = await parseJson<TokenResponse>(response);
  setAccessToken(data.access_token);
  return data;
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");

  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  let response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });

  if (response.status === 401 && accessToken) {
    await refreshAccessToken();

    const retryHeaders = new Headers(options.headers);
    retryHeaders.set("Content-Type", "application/json");
    if (accessToken) {
      retryHeaders.set("Authorization", `Bearer ${accessToken}`);
    }

    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: retryHeaders,
      credentials: "include",
    });
  }

  return parseJson<T>(response);
}

export async function register(input: {
  email: string;
  password: string;
  full_name?: string;
}) {
  const data = await apiFetch<TokenResponse>("/admin/auth/register", {
    method: "POST",
    body: JSON.stringify(input),
  });
  setAccessToken(data.access_token);
  return data;
}

export async function login(input: { email: string; password: string }) {
  const data = await apiFetch<TokenResponse>("/admin/auth/login", {
    method: "POST",
    body: JSON.stringify(input),
  });
  setAccessToken(data.access_token);
  return data;
}

export async function logout() {
  await apiFetch<void>("/admin/auth/logout", { method: "POST" });
  setAccessToken(null);
}

export async function getMe() {
  return apiFetch<User>("/admin/auth/me");
}
```

## React auth provider

Create:

`src/auth/AuthProvider.tsx`

```tsx
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  User,
  login as loginRequest,
  logout as logoutRequest,
  refreshAccessToken,
  register as registerRequest,
} from "./api";

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  login: (input: { email: string; password: string }) => Promise<void>;
  register: (input: {
    email: string;
    password: string;
    full_name?: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    refreshAccessToken()
      .then((data) => {
        if (!cancelled) {
          setUser(data.user);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setUser(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (input: { email: string; password: string }) => {
    const data = await loginRequest(input);
    setUser(data.user);
  }, []);

  const register = useCallback(
    async (input: { email: string; password: string; full_name?: string }) => {
      const data = await registerRequest(input);
      setUser(data.user);
    },
    [],
  );

  const logout = useCallback(async () => {
    await logoutRequest();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return value;
}
```

## Protected routes in React

Create:

`src/auth/RequireAuth.tsx`

```tsx
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "./AuthProvider";

export function RequireAuth() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return <Outlet />;
}
```

Example router:

```tsx
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./auth/AuthProvider";
import { RequireAuth } from "./auth/RequireAuth";
import { Dashboard } from "./pages/Dashboard";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route element={<RequireAuth />}>
            <Route path="/" element={<Dashboard />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
```

## Login page example

```tsx
import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const from = (location.state as { from?: Location })?.from?.pathname ?? "/";

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      await login({ email, password });
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main>
      <form onSubmit={onSubmit}>
        <h1>Sign in</h1>

        <label>
          Email
          <input
            autoComplete="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </label>

        <label>
          Password
          <input
            autoComplete="current-password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>

        {error ? <p role="alert">{error}</p> : null}

        <button disabled={submitting} type="submit">
          {submitting ? "Signing in..." : "Sign in"}
        </button>

        <Link to="/register">Create an account</Link>
      </form>
    </main>
  );
}
```

## Register page example

```tsx
import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";

export function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      await register({
        email,
        password,
        full_name: fullName || undefined,
      });
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main>
      <form onSubmit={onSubmit}>
        <h1>Create account</h1>

        <label>
          Full name
          <input
            autoComplete="name"
            value={fullName}
            onChange={(event) => setFullName(event.target.value)}
          />
        </label>

        <label>
          Email
          <input
            autoComplete="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </label>

        <label>
          Password
          <input
            autoComplete="new-password"
            minLength={12}
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>

        {error ? <p role="alert">{error}</p> : null}

        <button disabled={submitting} type="submit">
          {submitting ? "Creating..." : "Create account"}
        </button>

        <Link to="/login">Sign in instead</Link>
      </form>
    </main>
  );
}
```

## Logout button

```tsx
import { useAuth } from "../auth/AuthProvider";

export function LogoutButton() {
  const { logout } = useAuth();

  return (
    <button type="button" onClick={() => void logout()}>
      Sign out
    </button>
  );
}
```

## Calling protected backend APIs from React

Use `apiFetch` so the access token is attached and refreshed automatically.

```ts
import { apiFetch } from "../auth/api";

export async function listPatients() {
  return apiFetch("/admin/patients");
}
```

## Important Supabase notes

In this architecture, Supabase is your database, not your auth provider.

Do:

- Use `SUPABASE_SERVICE_KEY` only in FastAPI.
- Keep all protected reads/writes behind FastAPI endpoints.
- Enforce authorization in FastAPI because the service role can bypass RLS.
- Store password hashes with `pwdlib.PasswordHash.recommended()`.
- Store only hashed refresh tokens.
- Rotate refresh tokens after every refresh.
- Revoke refresh sessions on logout.

Do not:

- Put `SUPABASE_SERVICE_KEY` in React.
- Call `supabase.auth.signInWithPassword`, `signUp`, or `signOut`.
- Store refresh tokens in `localStorage`.
- Store raw refresh tokens in Postgres.
- Trust decoded JWT claims on the frontend for real authorization.

You can still use Supabase RLS as defense in depth if you query through non-bypass database roles. But with the service-role key used by the backend client, authorization must primarily happen in FastAPI.

## Production hardening checklist

- Use a 64-plus character `JWT_SECRET_KEY`.
- Keep access tokens short-lived, usually 5 to 15 minutes.
- Use HttpOnly, Secure refresh cookies in production.
- Use exact CORS origins with `allow_credentials=True`.
- Add rate limiting to `/login`, `/register`, and `/refresh`.
- Add audit logging for login success, login failure, refresh, and logout.
- Add email verification before enabling sensitive account actions.
- Add password reset with one-time, hashed reset tokens.
- Add password breach checks or minimum complexity policy.
- Add account lockout or step-up verification after repeated failures.
- Revoke all user refresh sessions on password change.
- Use HTTPS everywhere.
- Do not log passwords, refresh tokens, JWTs, or authorization headers.
- Consider an asymmetric JWT algorithm such as RS256 when multiple services must verify tokens.

## Minimal backend test cases

Create tests around the public auth contract:

```python
def test_register_login_refresh_logout(client):
    register_response = client.post(
        "/api/v1/admin/auth/register",
        json={
            "email": "admin@example.com",
            "password": "very-secure-password",
            "full_name": "Admin User",
        },
    )
    assert register_response.status_code == 201
    assert "access_token" in register_response.json()
    assert "refresh_token" in register_response.cookies

    me_response = client.get(
        "/api/v1/admin/auth/me",
        headers={
            "Authorization": (
                f"Bearer {register_response.json()['access_token']}"
            )
        },
    )
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "admin@example.com"

    refresh_response = client.post("/api/v1/admin/auth/refresh")
    assert refresh_response.status_code == 200
    assert "access_token" in refresh_response.json()

    logout_response = client.post("/api/v1/admin/auth/logout")
    assert logout_response.status_code == 204
```

Depending on how your test Supabase client is mocked, you may prefer service-level tests for `register_user`, `authenticate_user`, `rotate_refresh_token`, and `revoke_refresh_token`.

## Endpoint behavior summary

| Endpoint | Purpose | Body | Auth |
| --- | --- | --- | --- |
| `POST /api/v1/admin/auth/register` | Create user and start session | `email`, `password`, `full_name` | Public |
| `POST /api/v1/admin/auth/login` | Verify password and start session | `email`, `password` | Public |
| `POST /api/v1/admin/auth/refresh` | Rotate refresh token and return new access token | None | Refresh cookie |
| `POST /api/v1/admin/auth/logout` | Revoke refresh session and clear cookie | None | Refresh cookie |
| `GET /api/v1/admin/auth/me` | Return current user | None | Bearer access token |

## The key tradeoff

This approach is more work than Supabase Auth, but it gives your FastAPI backend full control over the identity model, token contents, refresh-session storage, audit logs, and authorization rules. That is often the right tradeoff for a medical scheduling/admin backend, especially when protected operations already flow through FastAPI.
