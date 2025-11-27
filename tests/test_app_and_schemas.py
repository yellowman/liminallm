import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from liminallm import app as app_module
from liminallm.api import schemas


@pytest.fixture
def fresh_app(monkeypatch):
    """Reload the app module to respect env overrides for CORS tests."""

    # ensure environment-driven defaults are picked up when reloading
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    reloaded = importlib.reload(app_module)
    try:
        yield reloaded.app
    finally:
        importlib.reload(app_module)


def test_security_headers_and_health(fresh_app):
    client = TestClient(fresh_app)
    response = client.get("/healthz", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")
    assert response.headers["Content-Security-Policy"].startswith("default-src")
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_frontend_routes_serve_bundled_assets():
    client = TestClient(app_module.app)

    chat_response = client.get("/")
    assert chat_response.status_code == 200
    assert chat_response.text.lower().lstrip().startswith("<!doctype html")

    admin_response = client.get("/admin", headers={"Authorization": "Bearer stub"})
    assert admin_response.status_code != 404


def test_allowed_origins_default(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    origins = app_module._allowed_origins()
    assert "http://localhost" in origins
    assert "http://127.0.0.1:5173" in origins


def test_allowed_origins_override(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://example.com, https://demo.local")
    origins = app_module._allowed_origins()
    assert origins == ["https://example.com", "https://demo.local"]


def test_envelope_status_validation():
    with pytest.raises(ValidationError):
        schemas.Envelope(status="pending")


def test_signup_request_validates_email_and_password():
    with pytest.raises(ValidationError):
        schemas.SignupRequest(email="invalid", password="Short1")

    req = schemas.SignupRequest(email=" User@example.com ", password="Password1")
    assert req.email == "User@example.com"


def test_schema_payload_aliasing_and_defaults():
    payload = {"schema": {"title": "Example"}, "name": "artifact"}
    req = schemas.ArtifactRequest(**payload)

    dumped = req.model_dump(by_alias=True)
    assert dumped["schema"] == {"title": "Example"}
    assert "schema_" not in dumped

    response = schemas.ChatResponse(message_id="m", conversation_id="c", content="hi")
    response.adapters.append("a1")
    second = schemas.ChatResponse(message_id="m2", conversation_id="c2", content="hi")
    assert second.adapters == []


def test_frontend_files_exist_on_disk():
    frontend_dir = Path(app_module.STATIC_DIR)
    assert (frontend_dir / "index.html").exists()
    assert (frontend_dir / "admin.html").exists()
