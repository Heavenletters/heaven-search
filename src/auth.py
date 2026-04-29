"""Simple API-key authentication for the search API.

Supports two modes:
1. Static API key (set in env: HEAVEN_API_KEY)
2. User/password → JWT (for web UI login)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
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

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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

    # Option 1: Authorization header
    if credentials:
        token = credentials.credentials

    # Option 2: Static API key (check if token matches API_KEY directly)
    if token and API_KEY and token == API_KEY:
        return "api_key"

    # Option 3: JWT token
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

    if not pwd_context.verify(request.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(request.username)
    return TokenResponse(access_token=token)


def hash_password(password: str) -> str:
    """Generate bcrypt hash for a password (for setup only)."""
    return pwd_context.hash(password)
