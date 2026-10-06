import pytest
from fastapi import HTTPException
import config
from auth import (
    _failed_attempts,
    create_access_token,
    decode_access_token,
)
from config import validate_auth_config

def test_startup_refusal_missing_password_hash(monkeypatch):
    """App must refuse to start if AUTH_PASSWORD_HASH is missing."""
    monkeypatch.setattr(config, "AUTH_PASSWORD_HASH", None)
    with pytest.raises(RuntimeError) as exc:
        validate_auth_config()
    assert "AUTH_PASSWORD_HASH is not set" in str(exc.value)

def test_startup_refusal_missing_or_short_jwt_secret(monkeypatch):
    """App must refuse to start if JWT_SECRET is missing or < 32 chars."""
    monkeypatch.setattr(config, "JWT_SECRET", None)
    with pytest.raises(RuntimeError) as exc:
        validate_auth_config()
    assert "JWT_SECRET is not set" in str(exc.value)

    monkeypatch.setattr(config, "JWT_SECRET", "short_secret_key")
    with pytest.raises(RuntimeError) as exc:
        validate_auth_config()
    assert "at least 32 characters" in str(exc.value)

def test_login_success(test_client):
    """Valid credentials return 200 and set HttpOnly auth cookie."""
    response = test_client.post(
        "/auth/login",
        json={"username": "admin@rait.ac.in", "password": "admin"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert config.COOKIE_NAME in response.cookies

def test_login_invalid_password(test_client):
    """Invalid password returns 401."""
    response = test_client.post(
        "/auth/login",
        json={"username": "admin@rait.ac.in", "password": "wrong_password"}
    )
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["detail"]

def test_similarity_requires_auth(test_client):
    """Unauthenticated request to /similarity returns 401."""
    test_client.cookies.clear()
    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "sample key"},
        files={"answer_key_diagram": ("key.png", b"fake", "image/png")}
    )
    assert response.status_code == 401
    assert "Authentication required" in response.json()["detail"]

def test_similarity_succeeds_with_auth_cookie(test_client, monkeypatch):
    """Authenticated request with valid JWT token succeeds."""
    token = create_access_token({"sub": "admin@rait.ac.in"})
    test_client.cookies.set(config.COOKIE_NAME, token)

    from PIL import Image
    dummy_page = Image.new("RGB", (200, 200), color="white")
    monkeypatch.setattr("server.convert_from_bytes", lambda *args, **kwargs: [dummy_page])

    from tests.test_api import _create_minimal_pdf_bytes, _create_png_bytes

    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "Machine learning"},
        files={
            "answer_key_diagram": ("key.png", _create_png_bytes(), "image/png"),
            "answer_sheets": ("student.pdf", _create_minimal_pdf_bytes(), "application/pdf"),
        }
    )
    assert response.status_code == 200

def test_auth_me_endpoint(test_client):
    """GET /auth/me returns the current user profile."""
    token = create_access_token({"sub": "admin@rait.ac.in"})
    test_client.cookies.set(config.COOKIE_NAME, token)
    response = test_client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "admin@rait.ac.in"

def test_auth_logout(test_client):
    """POST /auth/logout clears the session cookie."""
    token = create_access_token({"sub": "admin@rait.ac.in"})
    test_client.cookies.set(config.COOKIE_NAME, token)
    response = test_client.post("/auth/logout")
    assert response.status_code == 200
    assert response.json()["status"] == "logged_out"

def test_login_rate_limiting(test_client):
    """5 consecutive failed attempts trigger 429 Too Many Requests."""
    _failed_attempts.clear()
    for _ in range(5):
        test_client.post(
            "/auth/login",
            json={"username": "admin@rait.ac.in", "password": "wrong_password"}
        )
    # 6th attempt must be rate-limited
    response = test_client.post(
        "/auth/login",
        json={"username": "admin@rait.ac.in", "password": "wrong_password"}
    )
    assert response.status_code == 429
    assert "Too many failed login attempts" in response.json()["detail"]
    _failed_attempts.clear()
