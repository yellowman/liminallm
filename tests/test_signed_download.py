"""A signed download URL is a capability, and these are its edges.

SPEC §18: "downloads use signed URLs with 10m expiry and content-disposition
set to prevent inline execution". A signature alone satisfies none of that —
what matters is which object the token names, whether the name can be changed
after signing, whether expiry is checked when the token is *used* rather than
when it was made, and whether the bytes come back in a form a browser will save
instead of execute.

One structural fact shapes all of it, and is asserted below rather than assumed:
redemption depends on `get_user`, so this is not a bearer token. It cannot be
handed to a browser without the session, and it cannot be replayed by a second
account. That makes several of the classic signed-URL attacks impossible by
construction — and the tests say which ones, so a later change that drops the
dependency fails here instead of silently turning the URL into a bearer grant.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from liminallm.service.fs import generate_signed_url


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


def _account(client, runtime):
    email = f"{_unique('dl')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return (
        runtime.store.get_user(data["user_id"]),
        {"Authorization": f"Bearer {data['access_token']}"},
    )


def _upload(client, headers, name="report.md", body=b"quarterly figures\n"):
    resp = client.post(
        "/v1/files/upload",
        headers=headers,
        files={"file": (name, body, "text/markdown")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]["fs_path"]


def _mint(client, headers, name):
    resp = client.get(f"/v1/files/{name}/url", headers=headers)
    assert resp.status_code == 200, resp.text
    url = resp.json()["data"]["download_url"]
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def _redeem(client, headers, params):
    return client.get("/v1/files/download", headers=headers, params=params)


class TestTheTokenNamesOneObject:
    def test_a_minted_token_redeems_its_own_file(self, client, runtime):
        _user, headers = _account(client, runtime)
        name = _upload(client, headers)
        resp = _redeem(client, headers, _mint(client, headers, name))
        assert resp.status_code == 200, resp.text
        assert resp.content == b"quarterly figures\n"

    def test_changing_the_path_invalidates_the_signature(self, client, runtime):
        _user, headers = _account(client, runtime)
        _upload(client, headers, "report.md")
        _upload(client, headers, "salaries.md", b"secret\n")
        params = _mint(client, headers, "report.md")
        params["path"] = "salaries.md"

        resp = _redeem(client, headers, params)
        assert resp.status_code == 401, resp.text

    def test_extending_the_expiry_invalidates_the_signature(self, client, runtime):
        _user, headers = _account(client, runtime)
        name = _upload(client, headers)
        params = _mint(client, headers, name)
        params["expires"] = str(int(params["expires"]) + 86_400)

        resp = _redeem(client, headers, params)
        assert resp.status_code == 401, resp.text

    def test_a_forged_signature_is_refused(self, client, runtime):
        _user, headers = _account(client, runtime)
        name = _upload(client, headers)
        params = _mint(client, headers, name)
        params["sig"] = "0" * 64

        resp = _redeem(client, headers, params)
        assert resp.status_code == 401, resp.text


class TestExpiryIsCheckedWhenTheTokenIsUsed:
    def test_an_expired_token_is_refused_at_redemption(self, client, runtime):
        """Minted valid, redeemed late. Checking expiry only at issue time
        would make the window unbounded, which is the whole of the 10m rule."""
        user, headers = _account(client, runtime)
        name = _upload(client, headers)
        stale = generate_signed_url(
            file_path=name,
            user_id=user.id,
            secret_key=runtime.settings.jwt_secret,
            expiry_seconds=-1,
        )
        params = {k: v[0] for k, v in parse_qs(urlparse(stale).query).items()}
        assert int(params["expires"]) < time.time()

        resp = _redeem(client, headers, params)
        assert resp.status_code == 401, resp.text

    def test_the_window_is_the_ten_minutes_spec_asks_for(self, client, runtime):
        _user, headers = _account(client, runtime)
        name = _upload(client, headers)
        params = _mint(client, headers, name)
        remaining = int(params["expires"]) - int(time.time())
        assert 0 < remaining <= 600, remaining


class TestTheTokenIsNotABearerGrant:
    """Redemption re-resolves identity, and the signature carries it too.

    Two independent reasons a second account cannot use someone else's token,
    tested separately: remove either and one of these fails.
    """

    def test_another_account_cannot_redeem_it(self, client, runtime):
        _owner, owner_headers = _account(client, runtime)
        name = _upload(client, owner_headers, "report.md", b"owner only\n")
        params = _mint(client, owner_headers, name)

        _stranger, stranger_headers = _account(client, runtime)
        resp = _redeem(client, stranger_headers, params)
        assert resp.status_code in (401, 404), resp.text
        assert b"owner only" not in resp.content

    def test_an_unauthenticated_request_cannot_redeem_it(self, client, runtime):
        _owner, owner_headers = _account(client, runtime)
        name = _upload(client, owner_headers)
        params = _mint(client, owner_headers, name)

        resp = client.get("/v1/files/download", params=params)
        assert resp.status_code in (401, 403), resp.text

    def test_the_signature_binds_the_user_not_just_the_path(self, runtime):
        """Even given the same path and expiry, two users get two tokens."""
        first = generate_signed_url(
            file_path="report.md",
            user_id="user-a",
            secret_key=runtime.settings.jwt_secret,
        )
        second = generate_signed_url(
            file_path="report.md",
            user_id="user-b",
            secret_key=runtime.settings.jwt_secret,
        )
        sig_of = lambda url: parse_qs(urlparse(url).query)["sig"][0]  # noqa: E731
        assert sig_of(first) != sig_of(second)


class TestRedemptionResolvesOwnershipAgain:
    def test_a_traversal_path_cannot_be_signed_into_validity(
        self, client, runtime
    ):
        """A token is not a licence to skip `safe_join`. Signed by the server,
        so the signature is genuine — the path still has to be the caller's."""
        owner, owner_headers = _account(client, runtime)
        victim, victim_headers = _account(client, runtime)
        _upload(client, victim_headers, "private.md", b"not yours\n")
        escape = f"../../{victim.id}/files/private.md"
        forged = generate_signed_url(
            file_path=escape,
            user_id=owner.id,
            secret_key=runtime.settings.jwt_secret,
        )
        params = {k: v[0] for k, v in parse_qs(urlparse(forged).query).items()}

        resp = _redeem(client, owner_headers, params)
        assert resp.status_code in (400, 404), resp.text
        assert b"not yours" not in resp.content


def _place(runtime, user, name: str, body: bytes = b"payload\n") -> None:
    """Put a file in a user's area directly.

    Upload sanitizes its filenames, so a hostile name cannot arrive that way —
    but `interpreter.publish_artifacts` refuses only `/` and a leading dot, and
    `.txt` is an allowed extension, so model-written code can produce one. The
    model's choices are attacker-influenced the moment it has read a page.
    """
    files_dir = Path(runtime.settings.shared_fs_root) / "users" / user.id / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / name).write_bytes(body)


def _disposition_filename(header: str):
    """What a client will actually call the file, decoded the way one does."""
    from email.message import Message

    message = Message()
    message["content-disposition"] = header
    return message.get_filename()


class TestTheResponseIsSavedNotExecuted:
    def test_the_disposition_is_attachment(self, client, runtime):
        """An html file is the case that matters, and uploads refuse that
        extension — so it is placed directly, as published code could."""
        user, headers = _account(client, runtime)
        _place(runtime, user, "page.html", b"<script>alert(1)</script>")
        resp = _redeem(client, headers, _mint(client, headers, "page.html"))
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-disposition"].startswith("attachment"), (
            resp.headers["content-disposition"]
        )
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_a_quote_in_the_filename_cannot_forge_header_parameters(
        self, client, runtime
    ):
        """The header used to be `f'attachment; filename="{path}"'`, so a name
        containing a quote closed the string and added a second `filename=`
        parameter: `attachment; filename="evil";filename="innocent.txt"`. A
        client taking the last one saves the file under a name and extension
        the injected page chose.
        """
        user, headers = _account(client, runtime)
        hostile = 'evil";filename="innocent.txt'
        _place(runtime, user, hostile)

        resp = _redeem(client, headers, _mint(client, headers, hostile))
        assert resp.status_code == 200, resp.text
        disposition = resp.headers["content-disposition"]
        # Asserted on the *decoded* value, not on the raw string: the encoded
        # form legitimately contains the letters "filename" inside its own
        # percent-encoded payload, so counting substrings measures nothing.
        assert _disposition_filename(disposition) == hostile, disposition
        assert '"' not in disposition, disposition

    def test_an_ordinary_filename_is_still_readable(self, client, runtime):
        """The encoding must not turn every download into an escaped blob."""
        _user, headers = _account(client, runtime)
        name = _upload(client, headers, "report.md")
        resp = _redeem(client, headers, _mint(client, headers, name))
        assert _disposition_filename(resp.headers["content-disposition"]) == "report.md"
