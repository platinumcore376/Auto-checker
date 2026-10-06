import hmac
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import bcrypt
import jwt
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel

import config

logger = logging.getLogger("autochecker.auth")

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

# In-memory rate limiting: IP -> list of failed attempt timestamps
_failed_attempts: Dict[str, List[float]] = {}
MAX_FAILED_ATTEMPTS = 5
RATE_LIMIT_WINDOW_SECONDS = 300  # 5 minutes

class LoginRequest(BaseModel):
    username: str
    password: str

def _check_rate_limit(client_ip: str):
    """Enforces max 5 failed login attempts per 5 minutes per client IP."""
    now = time.time()
    attempts = _failed_attempts.get(client_ip, [])
    # Keep only attempts within the window
    attempts = [t for t in attempts if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _failed_attempts[client_ip] = attempts

    if len(attempts) >= MAX_FAILED_ATTEMPTS:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - attempts[0]))
        logger.warning(f"Rate limit exceeded for IP {client_ip}. Retry after {retry_after}s")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Please try again in {max(1, retry_after)} seconds.",
            headers={"Retry-After": str(max(1, retry_after))}
        )

def _record_failed_attempt(client_ip: str):
    now = time.time()
    if client_ip not in _failed_attempts:
        _failed_attempts[client_ip] = []
    _failed_attempts[client_ip].append(now)

def _clear_failed_attempts(client_ip: str):
    if client_ip in _failed_attempts:
        del _failed_attempts[client_ip]

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=config.JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, config.JWT_SECRET, algorithm="HS256")

def decode_access_token(token: str) -> dict:
    """Validates and decodes a signed JWT access token."""
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please log in again."
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )

def require_user(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> dict:
    """
    FastAPI dependency that enforces authentication.
    Checks HttpOnly cookie first, then Authorization Bearer header.
    """
    token = request.cookies.get(config.COOKIE_NAME)
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in."
        )

    return decode_access_token(token)

@auth_router.post("/login")
def login(request: Request, response: Response, body: LoginRequest):
    """
    Authenticates username and password against environment secrets.
    Issues an HttpOnly, SameSite=Lax JWT cookie.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    # Constant-time username comparison to avoid timing attacks
    username_match = hmac.compare_digest(body.username.strip(), config.AUTH_USERNAME)

    # Bcrypt password verification
    password_match = False
    if config.AUTH_PASSWORD_HASH:
        try:
            password_match = bcrypt.checkpw(
                body.password.encode("utf-8"),
                config.AUTH_PASSWORD_HASH.encode("utf-8")
            )
        except Exception as e:
            logger.error(f"Error checking password hash: {e}")
            password_match = False

    if not (username_match and password_match):
        _record_failed_attempt(client_ip)
        logger.warning(f"Failed login attempt for username '{body.username}' from IP {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    # Successful login
    _clear_failed_attempts(client_ip)
    token = create_access_token({"sub": body.username})

    # Set secure HttpOnly cookie (SameSite=Lax)
    is_localhost = client_ip in ["127.0.0.1", "localhost", "::1"]
    response.set_cookie(
        key=config.COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=config.JWT_EXPIRE_MINUTES * 60,
        expires=config.JWT_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=not is_localhost,
    )
    logger.info(f"User '{body.username}' logged in successfully from IP {client_ip}")
    return {"status": "ok", "user": {"username": body.username}}

@auth_router.post("/logout")
def logout(response: Response):
    """Logs out by clearing the HttpOnly auth cookie."""
    response.delete_cookie(
        key=config.COOKIE_NAME,
        httponly=True,
        samesite="lax",
    )
    return {"status": "logged_out"}

@auth_router.get("/me")
def get_current_user(user: dict = Depends(require_user)):
    """Returns the currently authenticated user's profile."""
    return {"user": {"username": user.get("sub", "")}}
