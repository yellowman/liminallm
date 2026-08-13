"""API keys: minted by a session, honored only by the served Responses API.

The containment property under test is structural, not policy: get_user never
reads keys, so a leaked key can drive /v1/responses and nothing else — it
cannot list conversations, cannot mint another key, cannot revoke anything.
"""

from liminallm.service.auth import API_KEY_PREFIX


def _mint(client, headers, name="test key"):
    resp = client.post("/v1/auth/api-keys", headers=headers, json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


class TestApiKeyLifecycle:
    def test_mint_returns_plaintext_once_and_lists_prefix_only(
        self, client, auth_headers
    ):
        created = _mint(client, auth_headers, name="agent runner")

        assert created["api_key"].startswith(API_KEY_PREFIX)
        assert created["name"] == "agent runner"
        assert created["api_key"].startswith(created["prefix"])
        # The prefix identifies, never authenticates.
        assert len(created["prefix"]) < len(created["api_key"])

        listing = client.get("/v1/auth/api-keys", headers=auth_headers)
        assert listing.status_code == 200, listing.text
        (item,) = listing.json()["data"]["items"]
        assert item["id"] == created["id"]
        assert item["prefix"] == created["prefix"]
        assert "api_key" not in item
        assert item["revoked_at"] is None

    def test_key_authenticates_the_responses_surface(self, client, auth_headers):
        created = _mint(client, auth_headers)
        key_headers = {"Authorization": f"Bearer {created['api_key']}"}

        resp = client.post(
            "/v1/responses", headers=key_headers, json={"input": "hello from an agent"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["object"] == "response"

        # The turn landed in the minting user's account, visible to their session.
        convs = client.get("/v1/conversations", headers=auth_headers).json()["data"][
            "items"
        ]
        assert len(convs) == 1
        assert convs[0]["source"] == "responses"

    def test_key_is_refused_everywhere_else(self, client, auth_headers):
        created = _mint(client, auth_headers)
        key_headers = {"Authorization": f"Bearer {created['api_key']}"}

        assert client.get("/v1/conversations", headers=key_headers).status_code == 401
        # A key cannot mint keys...
        assert (
            client.post(
                "/v1/auth/api-keys", headers=key_headers, json={"name": "escalation"}
            ).status_code
            == 401
        )
        # ...and cannot revoke them.
        assert (
            client.delete(
                f"/v1/auth/api-keys/{created['id']}", headers=key_headers
            ).status_code
            == 401
        )

    def test_revoked_key_stops_authenticating(self, client, auth_headers):
        created = _mint(client, auth_headers)
        key_headers = {"Authorization": f"Bearer {created['api_key']}"}
        assert (
            client.post(
                "/v1/responses", headers=key_headers, json={"input": "before"}
            ).status_code
            == 200
        )

        revoke = client.delete(
            f"/v1/auth/api-keys/{created['id']}", headers=auth_headers
        )
        assert revoke.status_code == 200, revoke.text
        assert revoke.json()["data"]["revoked"] is True

        after = client.post(
            "/v1/responses", headers=key_headers, json={"input": "after"}
        )
        assert after.status_code == 401

        (item,) = client.get("/v1/auth/api-keys", headers=auth_headers).json()["data"][
            "items"
        ]
        assert item["revoked_at"] is not None

    def test_revoking_someone_elses_key_is_a_miss(self, client, auth_headers):
        import uuid

        other = client.post(
            "/v1/auth/signup",
            json={
                "email": f"other_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
        )
        other_headers = {
            "Authorization": f"Bearer {other.json()['data']['access_token']}"
        }
        theirs = _mint(client, other_headers)

        resp = client.delete(
            f"/v1/auth/api-keys/{theirs['id']}", headers=auth_headers
        )
        assert resp.status_code == 404

        # Still alive for its owner.
        key_headers = {"Authorization": f"Bearer {theirs['api_key']}"}
        assert (
            client.post(
                "/v1/responses", headers=key_headers, json={"input": "still here"}
            ).status_code
            == 200
        )

    def test_active_key_cap(self, client):
        import uuid

        from liminallm.api.routes import MAX_ACTIVE_API_KEYS
        from liminallm.service.runtime import get_runtime

        signup = client.post(
            "/v1/auth/signup",
            json={
                "email": f"cap_{uuid.uuid4().hex[:8]}@example.com",
                "password": "TestPassword123!",
            },
        )
        assert signup.status_code == 201, signup.text
        headers = {
            "Authorization": f"Bearer {signup.json()['data']['access_token']}"
        }
        user_id = signup.json()["data"]["user_id"]

        runtime = get_runtime()
        # Seed to the cap through the service, not the route: an HTTP loop
        # would trip the write rate limit first and test the wrong thing.
        while runtime.store.count_active_api_keys(user_id) < MAX_ACTIVE_API_KEYS:
            runtime.auth.mint_api_key(user_id, name="filler")

        resp = client.post(
            "/v1/auth/api-keys", headers=headers, json={"name": "one too many"}
        )
        assert resp.status_code == 400
        assert "revoke" in resp.text

        # Revoking one frees a slot.
        victim = runtime.store.list_api_keys(user_id)[0]
        assert (
            client.delete(
                f"/v1/auth/api-keys/{victim.id}", headers=headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/auth/api-keys", headers=headers, json={"name": "replacement"}
            ).status_code
            == 201
        )

    def test_garbage_bearer_is_still_401(self, client):
        headers = {"Authorization": f"Bearer {API_KEY_PREFIX}not-a-real-key"}
        resp = client.post("/v1/responses", headers=headers, json={"input": "hi"})
        assert resp.status_code == 401
