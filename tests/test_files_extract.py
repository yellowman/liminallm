"""API-level tests for archive upload + safe extraction (SPEC §18)."""
import io
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def user(client):
    email = f"extract_{uuid.uuid4().hex[:8]}@test.local"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPass123!"}
    )
    assert resp.status_code == 201, resp.json()
    token = resp.json()["data"]["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def upload(client, headers, filename: str, payload: bytes, content_type="application/zip"):
    return client.post(
        "/v1/files/upload",
        headers=headers,
        files={"file": (filename, io.BytesIO(payload), content_type)},
    )


def test_upload_and_extract_zip(user):
    client, headers = user
    payload = zip_bytes({"notes.txt": b"hello", "docs/readme.md": b"# md"})
    resp = upload(client, headers, "bundle.zip", payload)
    assert resp.status_code in (200, 201), resp.json()

    resp = client.post("/v1/files/bundle.zip/extract", headers=headers)
    assert resp.status_code == 200, resp.json()
    data = resp.json()["data"]
    assert data["extracted_to"] == "bundle"
    assert sorted(data["files"]) == ["docs/readme.md", "notes.txt"]

    # Extracted files appear in the recursive listing with relative paths
    listing = client.get("/v1/files", headers=headers).json()["data"]
    names = {f["name"] for f in listing["files"]}
    assert {"bundle.zip", "bundle/notes.txt", "bundle/docs/readme.md"} <= names

    # Signed URL + download works for a subpath file
    resp = client.get("/v1/files/bundle/notes.txt/url", headers=headers)
    assert resp.status_code == 200, resp.json()
    resp = client.get(resp.json()["data"]["download_url"], headers=headers)
    assert resp.status_code == 200
    assert resp.content == b"hello"


def test_extract_conflict_and_folder_delete(user):
    client, headers = user
    upload(client, headers, "twice.zip", zip_bytes({"a.txt": b"a"}))
    assert client.post("/v1/files/twice.zip/extract", headers=headers).status_code == 200
    # Second extraction refuses to clobber
    assert client.post("/v1/files/twice.zip/extract", headers=headers).status_code == 409
    # Deleting the folder clears the way
    assert client.delete("/v1/files/twice", headers=headers).status_code == 200
    assert client.post("/v1/files/twice.zip/extract", headers=headers).status_code == 200


def test_extract_rejects_non_archive(user):
    client, headers = user
    upload(client, headers, "plain.txt", b"not an archive", "text/plain")
    resp = client.post("/v1/files/plain.txt/extract", headers=headers)
    assert resp.status_code == 400


def test_extract_missing_file_404(user):
    client, headers = user
    assert client.post("/v1/files/ghost.zip/extract", headers=headers).status_code == 404


def test_zip_slip_never_escapes(user):
    client, headers = user
    payload = zip_bytes({"../../../../evil.txt": b"pwn", "safe.txt": b"ok"})
    upload(client, headers, "slip.zip", payload)
    resp = client.post("/v1/files/slip.zip/extract", headers=headers)
    assert resp.status_code == 200, resp.json()
    data = resp.json()["data"]
    assert data["files"] == ["safe.txt"]
    assert data["skipped"][0]["reason"] == "unsafe path"
    # The traversal name never landed anywhere reachable
    names = {f["name"] for f in client.get("/v1/files", headers=headers).json()["data"]["files"]}
    assert not any("evil" in n for n in names)


def test_disallowed_payload_types_skipped(user):
    client, headers = user
    payload = zip_bytes({"tool.exe": b"MZ", "ok.md": b"x"})
    upload(client, headers, "mixed.zip", payload)
    data = client.post("/v1/files/mixed.zip/extract", headers=headers).json()["data"]
    assert data["files"] == ["ok.md"]
    assert "not allowed" in data["skipped"][0]["reason"]


def test_archive_upload_with_context_rejected(user):
    client, headers = user
    resp = client.post(
        "/v1/files/upload",
        headers=headers,
        files={"file": ("a.zip", io.BytesIO(zip_bytes({"x.txt": b"1"})), "application/zip")},
        data={"context_id": "ctx-123"},
    )
    assert resp.status_code == 400
    assert "extract" in resp.json()["error"]["message"]


def test_traversal_in_url_rejected(user):
    client, headers = user
    resp = client.get("/v1/files/../../../etc/passwd/url", headers=headers)
    assert resp.status_code in (400, 404)
    resp = client.delete("/v1/files/..%2f..%2fescape", headers=headers)
    assert resp.status_code in (400, 404)
