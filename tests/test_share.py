"""Conversation sharing: private by default, owner-only toggle, noindex."""
import uuid

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


def make_user(client):
    email = f"share_{uuid.uuid4().hex[:8]}@test.local"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPass123!"}
    )
    assert resp.status_code == 201, resp.json()
    token = resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def make_conversation(client, headers, title="Share test"):
    resp = client.post("/v1/conversations", headers=headers, json={"title": title})
    assert resp.status_code == 201, resp.json()
    return resp.json()["data"]["id"]


def test_private_by_default(client):
    headers = make_user(client)
    conv_id = make_conversation(client, headers)
    detail = client.get(f"/v1/conversations/{conv_id}", headers=headers).json()["data"]
    assert detail["public"] is False
    # Unauthenticated public read is refused
    assert client.get(f"/v1/public/conversations/{conv_id}").status_code == 404


def test_share_and_unshare(client):
    headers = make_user(client)
    conv_id = make_conversation(client, headers, title="Now public")

    resp = client.post(
        f"/v1/conversations/{conv_id}/share", headers=headers, json={"public": True}
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()["data"]
    assert data["public"] is True
    assert data["share_path"] == f"/share/{conv_id}"

    # Public, unauthenticated read now works and carries noindex
    pub = client.get(f"/v1/public/conversations/{conv_id}")
    assert pub.status_code == 200
    assert pub.headers.get("x-robots-tag") == "noindex, nofollow"
    assert pub.json()["data"]["title"] == "Now public"

    # Owner sees the flag on the detail endpoint
    detail = client.get(f"/v1/conversations/{conv_id}", headers=headers).json()["data"]
    assert detail["public"] is True

    # Unshare returns it to private
    client.post(
        f"/v1/conversations/{conv_id}/share", headers=headers, json={"public": False}
    )
    assert client.get(f"/v1/public/conversations/{conv_id}").status_code == 404


def test_only_owner_can_share(client):
    owner = make_user(client)
    stranger = make_user(client)
    conv_id = make_conversation(client, owner)
    resp = client.post(
        f"/v1/conversations/{conv_id}/share", headers=stranger, json={"public": True}
    )
    assert resp.status_code == 404
    assert client.get(f"/v1/public/conversations/{conv_id}").status_code == 404


def test_public_messages_expose_only_safe_fields(client):
    headers = make_user(client)
    conv_id = make_conversation(client, headers)
    client.post(
        f"/v1/conversations/{conv_id}/share", headers=headers, json={"public": True}
    )
    data = client.get(f"/v1/public/conversations/{conv_id}").json()["data"]
    for msg in data["messages"]:
        assert set(msg.keys()) == {"role", "content", "created_at"}


def test_public_directory_lists_only_shared(client):
    headers = make_user(client)
    private_id = make_conversation(client, headers, title="Stays private")
    public_id = make_conversation(client, headers, title="Goes public")
    client.post(
        f"/v1/conversations/{public_id}/share", headers=headers, json={"public": True}
    )

    resp = client.get("/v1/public/conversations")
    assert resp.status_code == 200
    assert resp.headers.get("x-robots-tag") == "noindex, nofollow"
    ids = {c["id"] for c in resp.json()["data"]["items"]}
    assert public_id in ids
    assert private_id not in ids


def test_share_pages_are_noindex(client):
    for path in ("/share", f"/share/{uuid.uuid4()}"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers.get("x-robots-tag") == "noindex, nofollow"
        assert 'name="robots" content="noindex, nofollow"' in resp.text


def test_robots_txt_disallows_share(client):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    assert "Disallow: /share" in resp.text
