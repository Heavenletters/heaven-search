"""Simple API-key authentication for the search API.

Supports two modes:
1. Static API key (set in env: HEAVEN_API_KEY)
2. User/password → JWT (for web UI login)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Configuration ───────────────────────────────────────────────────

SECRET_KEY = os.environ.get("HEAVEN_JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Static API key from environment (simplest auth)
API_KEY = os.environ.get("HEAVEN_API_KEY", "")

# Admin user for web UI login
ADMIN_USERNAME = os.environ.get("HEAVEN_ADMIN_USER", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("HEAVEN_ADMIN_PASSWORD_HASH", "")
ADMIN_PASSWORD_FILE = os.environ.get("HEAVEN_ADMIN_PASSWORD_FILE", "")

# Prefer file-based password hash (avoids docker-compose $ escaping)
if ADMIN_PASSWORD_FILE and os.path.isfile(ADMIN_PASSWORD_FILE):
    with open(ADMIN_PASSWORD_FILE) as f:
        ADMIN_PASSWORD_HASH = f.read().strip()

security_scheme = HTTPBearer(auto_error=False)


# ── Models ──────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Token utilities ─────────────────────────────────────────────────

def create_access_token(username: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode = {"sub": username, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str | None:
    """Verify JWT, return username if valid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ── FastAPI dependency ──────────────────────────────────────────────

async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> str:
    """FastAPI dependency: authenticate via Bearer token or API key query param."""

    token: str | None = None

    if credentials:
        token = credentials.credentials

    # Static API key
    if token and API_KEY and token == API_KEY:
        return "api_key"

    # JWT token
    if token:
        username = verify_token(token)
        if username:
            return username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def login(request: LoginRequest) -> TokenResponse:
    """Authenticate user/password, return JWT."""
    if request.username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not ADMIN_PASSWORD_HASH:
        raise HTTPException(status_code=500, detail="Admin password not configured")

    try:
        valid = bcrypt.checkpw(
            request.password.encode("utf-8"),
            ADMIN_PASSWORD_HASH.encode("utf-8"),
        )
    except ValueError:
        raise HTTPException(status_code=500, detail="Invalid password hash")

    if not valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(request.username)
    return TokenResponse(access_token=token)


def hash_password(password: str) -> str:
    """Generate bcrypt hash for a password (for setup only)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
