"""Deleting an account is one lifetime boundary, not a list of deletions.

`DELETE /v1/admin/users/{id}` removes an account, and `delete_user` says it
removes all of that account's data. It removed rows. Everything the account
owned on the filesystem stayed: `/users/<id>` holds their uploaded files and
their content-addressed attachment generations, `.archive-staging/<id>` holds
whole-tree archive work.

The clock is the harder half. Three collectors already walk that namespace on
their own schedules, and each one measures age from something on disk. Once
the account's rows are gone, `sweep_generations` finds no conversations, so
its referenced-checksum set is empty and every generation the account ever
made becomes collectable at once — judged by the blob's own mtime, which is
weeks old. A turn that resolved one of those blobs a moment before the
deletion then reads a filesystem where it is gone.

So the account's retirement has to outrank the subordinate lifetimes: while it
is pending, none of them may touch that user, and when it comes due the whole
identity-derived namespace goes at once.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client, *, admin=False, email=None):
    email = email or f"{_unique('era')}@example.com"
    password = "TestPassword123!"
    resp = client.post("/v1/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    user_id = data["user_id"]
    if admin:
        get_runtime().store.update_user_role(user_id, role="admin")
        resp = client.post(
            "/v1/auth/login", json={"email": email, "password": password}
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
    return user_id, email, {"Authorization": f"Bearer {data['access_token']}"}


def _count(sql, params) -> int:
    with get_runtime().store._connect() as conn:
        return int(conn.execute(sql, params).fetchone()["n"])


def _namespace(user_id: str) -> Path:
    return Path(get_runtime().settings.shared_fs_root) / "users" / user_id


def _staging(user_id: str) -> Path:
    return Path(get_runtime().settings.shared_fs_root) / ".archive-staging" / user_id


def _age(root: Path, *, days: int = 7) -> None:
    """Backdate a tree past every subordinate sweep's threshold.

    Each of those sweeps measures age from something on disk, so an account
    whose files are all minutes old is protected by its own freshness and
    proves nothing about the exclusion under test.
    """
    import os
    import time

    when = time.time() - days * 86400
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        os.utime(path, (when, when))
    os.utime(root, (when, when))


def _populate(client, headers, user_id) -> tuple[Path, Path]:
    """Give the account something on disk that every sweep would collect.

    Uploaded file and attachment generation, scratch tmp, archive staging: one
    for each collector that walks this namespace, all backdated a week so the
    only thing keeping them is the account's own retirement.
    """
    from liminallm.service.attachments import INLINE_MAX_BYTES

    conversation = client.post(
        "/v1/conversations", headers=headers, json={"title": "chat"}
    )
    assert conversation.status_code in (200, 201), conversation.text
    body = ("a line worth indexing.\n" * 600).encode()
    assert len(body) > INLINE_MAX_BYTES
    upload = client.post(
        "/v1/files/upload",
        headers={**headers, "Idempotency-Key": _unique("k")},
        files={"file": ("kept.md", body, "text/markdown")},
        data={"conversation_id": conversation.json()["data"]["id"]},
    )
    assert upload.status_code == 200, upload.text

    namespace = _namespace(user_id)
    (namespace / "tmp").mkdir(parents=True, exist_ok=True)
    (namespace / "tmp" / "scratch.txt").write_text("interpreter scratch")

    staging = _staging(user_id)
    (staging / "in-progress").mkdir(parents=True, exist_ok=True)
    (staging / "in-progress" / "part.bin").write_bytes(b"archive work")

    _age(namespace)
    _age(staging)
    return namespace, staging


class TestAResetTokenNamesAnAccount:
    """A token that names an email transfers to whoever holds it next.

    Issuance stored the address and completion looked up whichever account
    owned it at the time. Delete the requester, register the same address, and
    the old token changes the new account's password — a credential transfer
    between two unrelated users, using nothing but the ordinary reset flow.
    """

    @pytest.mark.asyncio
    async def test_a_token_does_not_follow_the_email_to_a_new_account(self, client):
        runtime = get_runtime()
        victim_id, email, _ = _account(client)
        _, _, admin_headers = _account(client, admin=True)

        token = await runtime.auth.initiate_password_reset(
            runtime.store.get_user(victim_id)
        )
        assert token

        assert client.delete(
            f"/v1/admin/users/{victim_id}", headers=admin_headers
        ).status_code in (200, 204)

        # The address is free again, and somebody else takes it.
        successor_id, _, _ = _account(client, email=email)
        assert successor_id != victim_id
        before = runtime.store.get_password_record(successor_id)

        completed = await runtime.auth.complete_password_reset(token, "Attacker123!")
        assert completed is False, (
            "a reset token issued for a deleted account changed the password "
            "of the account that later took its email address"
        )
        after = runtime.store.get_password_record(successor_id)
        assert after == before, "the successor's credentials were rewritten"

    @pytest.mark.asyncio
    async def test_the_in_process_fallback_binds_the_same_way(self, client):
        """Redis is optional here. The rule that makes the token safe is not.

        A deployment without Redis keeps reset tokens in a dictionary on the
        service, and that dictionary held the address for the same reason the
        Redis key did. A real `AuthService` with no cache, so the branch under
        test is the branch that runs.
        """
        from liminallm.service.auth import AuthService

        runtime = get_runtime()
        auth = AuthService(runtime.store, None, runtime.settings)
        victim_id, email, _ = _account(client)
        _, _, admin_headers = _account(client, admin=True)

        # Both halves, because refusing everything would satisfy the second
        # one on its own. A fallback that resolves nothing is not secure, it
        # is broken, and this is the branch no Redis-backed test reaches.
        ordinary_id, _, _ = _account(client)
        rotated = await auth.initiate_password_reset(
            runtime.store.get_user(ordinary_id)
        )
        assert auth._password_reset_tokens, "the fallback branch did not run"
        was = runtime.store.get_password_record(ordinary_id)
        assert await auth.complete_password_reset(rotated, "NewPassword123!")
        assert runtime.store.get_password_record(ordinary_id) != was

        token = await auth.initiate_password_reset(runtime.store.get_user(victim_id))
        assert client.delete(
            f"/v1/admin/users/{victim_id}", headers=admin_headers
        ).status_code in (200, 204)
        successor_id, _, _ = _account(client, email=email)
        before = runtime.store.get_password_record(successor_id)

        assert await auth.complete_password_reset(token, "Attacker123!") is False, (
            "the in-process fallback let a deleted account's token change the "
            "password of whoever took its email address"
        )
        assert runtime.store.get_password_record(successor_id) == before

    @pytest.mark.asyncio
    async def test_an_ordinary_reset_still_works(self, client):
        runtime = get_runtime()
        user_id, email, _ = _account(client)
        before = runtime.store.get_password_record(user_id)

        token = await runtime.auth.initiate_password_reset(
            runtime.store.get_user(user_id)
        )
        assert await runtime.auth.complete_password_reset(token, "NewPassword123!")
        assert runtime.store.get_password_record(user_id) != before


class TestTheAccountOwnsItsWholeNamespace:
    def test_deleting_an_account_records_a_namespace_retirement(self, client):
        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        _populate(client, headers, user_id)

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (user_id,),
        ) == 1, (
            "the account went and nothing recorded that its filesystem "
            "namespace is now waiting to be reclaimed"
        )

    @pytest.mark.asyncio
    async def test_a_due_retirement_takes_both_identity_derived_trees(self, client):
        from liminallm.app import _run_cleanup_pass

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        namespace, staging = _populate(client, headers, user_id)
        assert namespace.is_dir() and staging.is_dir()

        runtime = get_runtime()
        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE user_namespace_retirement "
                "SET retired_at = now() - interval '2 days' WHERE user_id = %s",
                (user_id,),
            )

        await _run_cleanup_pass(
            runtime, Path(runtime.settings.shared_fs_root), max_age_hours=24
        )
        assert not namespace.exists(), "the account's files survived its erasure"
        assert not staging.exists(), "the account's archive staging survived"
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (user_id,),
        ) == 0

    @pytest.mark.asyncio
    async def test_a_pending_retirement_is_not_collected_early(self, client):
        from liminallm.app import _run_cleanup_pass

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        namespace, staging = _populate(client, headers, user_id)
        runtime = get_runtime()

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        await _run_cleanup_pass(
            runtime, Path(runtime.settings.shared_fs_root), max_age_hours=24
        )
        assert namespace.is_dir(), (
            "the namespace was reclaimed inside the grace period a still "
            "in-flight request needs"
        )
        assert staging.is_dir()
        # Every collector that walks this namespace, one assertion each. All of
        # it is a week old, so nothing but the pending retirement is keeping
        # any of it; drop the exclusion from any one sweep and its line fails.
        assert (namespace / "tmp" / "scratch.txt").exists(), (
            "the scratch sweep emptied a namespace mid-erasure"
        )
        assert (staging / "in-progress").is_dir(), (
            "the archive-staging sweep took a tree mid-erasure"
        )
        assert list(
            (namespace / "attachment-generations" / "sha256").glob("*/*")
        ), "the generation sweep took the account's blobs mid-erasure"


class TestSubordinateSweepsDoNotUndercutTheGrace:
    """The account's clock outranks every collector inside its namespace.

    `sweep_generations` marks from what conversations reference. Delete the
    account and there are no conversations, so the mark set is empty and every
    generation the user ever made looks unreferenced — judged by the blob's own
    mtime, which is old. That is the right clock for its normal race and the
    wrong one for this event.
    """

    @pytest.mark.asyncio
    async def test_an_old_generation_survives_the_pass_that_follows_deletion(
        self, client
    ):
        import os
        import time

        from liminallm.app import _run_cleanup_pass

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        _populate(client, headers, user_id)
        runtime = get_runtime()

        generations = (
            _namespace(user_id) / "attachment-generations" / "sha256"
        )
        blobs = list(generations.glob("*/*")) if generations.is_dir() else []
        assert blobs, "the account has no attachment generation to protect"
        blob = blobs[0]
        week_ago = time.time() - 7 * 86400
        os.utime(blob, (week_ago, week_ago))

        # A turn resolved that blob a moment ago; then the account is deleted.
        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        await _run_cleanup_pass(
            runtime, Path(runtime.settings.shared_fs_root), max_age_hours=24
        )

        assert blob.exists(), (
            "an attachment generation was reclaimed the moment its account's "
            "rows disappeared, because the mark set emptied and the blob's own "
            "age was already past the sweep's threshold"
        )

    def test_the_generation_sweep_skips_a_pending_user(self, client):
        """Directly, so the exclusion is not only tested through the loop."""
        import os
        import time

        from liminallm.service.attachments import sweep_generations

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        _populate(client, headers, user_id)
        runtime = get_runtime()
        generations = _namespace(user_id) / "attachment-generations" / "sha256"
        # This account's own objects, not a total: the shared filesystem root
        # outlives each test's database, so earlier tests leave namespaces
        # whose users no longer exist, and the sweep is right to take those.
        before = set(generations.glob("*/*"))
        assert before
        week_ago = time.time() - 7 * 86400
        for blob in before:
            os.utime(blob, (week_ago, week_ago))

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        sweep_generations(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert set(generations.glob("*/*")) == before, (
            "the sweep took a pending user's generations"
        )


class TestRetirementIsDurableAndRetryable:
    def test_a_failed_removal_keeps_the_record(self, client):
        from liminallm.service import users as users_module
        from liminallm.service.users import sweep_user_namespaces

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        namespace, _ = _populate(client, headers, user_id)
        runtime = get_runtime()
        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)

        def refuse(*a, **kw):
            raise OSError("device or resource busy")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(users_module.shutil, "rmtree", refuse)
        try:
            removed = sweep_user_namespaces(
                runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
            )
        finally:
            monkeypatch.undo()

        assert removed == 0
        assert namespace.is_dir()
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (user_id,),
        ) == 1, "the retirement was forgotten after a failed removal"

        assert sweep_user_namespaces(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        ) >= 1
        assert not namespace.exists()

    def test_debris_from_before_this_existed_is_enrolled_not_removed(self, client):
        """First observed, then collected — never removed on sight."""
        from liminallm.service.users import sweep_user_namespaces

        runtime = get_runtime()
        stray = str(uuid.uuid4())
        tree = _namespace(stray)
        (tree / "files").mkdir(parents=True, exist_ok=True)
        (tree / "files" / "old.md").write_text("left behind")

        removed = sweep_user_namespaces(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=3600
        )
        assert removed == 0
        assert tree.is_dir()
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (stray,),
        ) == 1, "debris was neither enrolled nor removed, so it is immortal"

        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE user_namespace_retirement "
                "SET retired_at = now() - interval '2 hours' WHERE user_id = %s",
                (stray,),
            )
        assert sweep_user_namespaces(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=3600
        ) >= 1
        assert not tree.exists()

    def test_a_directory_that_is_not_an_identity_is_left_alone(self, client):
        """The destructive target is derived from a user id or from nothing.

        Two layers, and the test asks for both. Enrolment refuses a name that
        is not a user id, so nothing that is not one enters the queue; and the
        join refuses to leave the shared root, so a name that somehow did
        could still not choose a path.
        """
        from liminallm.service.fs import PathTraversalError
        from liminallm.service.users import namespace_dirs, sweep_user_namespaces

        runtime = get_runtime()
        root = Path(runtime.settings.shared_fs_root)
        stray = root / "users" / "not-a-uuid"
        stray.mkdir(parents=True, exist_ok=True)
        (stray / "keep.txt").write_text("operator's own directory")

        assert runtime.store.enrol_user_namespace_retirement("not-a-uuid") is False
        sweep_user_namespaces(
            runtime.store, str(root), grace_seconds=0
        )
        assert stray.is_dir() and (stray / "keep.txt").exists(), (
            "a directory the server never named was reclaimed as an account"
        )

        with pytest.raises(PathTraversalError):
            namespace_dirs(root, "../../etc")

    def test_a_live_users_namespace_is_never_enrolled(self, client):
        from liminallm.service.users import sweep_user_namespaces

        user_id, _, headers = _account(client)
        _populate(client, headers, user_id)
        runtime = get_runtime()

        sweep_user_namespaces(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (user_id,),
        ) == 0
        assert _namespace(user_id).is_dir()


class TestTheNamespaceRetirementIsVerifiedAtStartup:
    def test_startup_refuses_a_database_without_the_table(self, client):
        import os

        from liminallm.storage.postgres import PostgresStore
        from tests.harness import apply_schema

        runtime = get_runtime()
        url = os.environ["DATABASE_URL"]
        with runtime.store._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS user_namespace_retirement CASCADE")
        try:
            with pytest.raises(RuntimeError) as caught:
                PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))
            assert "migrate" in str(caught.value).lower(), str(caught.value)
        finally:
            apply_schema(url, embedding_dim=64)
        PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))

    @pytest.mark.parametrize(
        "sabotage",
        [
            "ALTER TABLE app_user DISABLE TRIGGER app_user_retire_namespace",
            "ALTER TABLE app_user ENABLE REPLICA TRIGGER app_user_retire_namespace",
        ],
    )
    def test_startup_refuses_a_trigger_that_will_not_fire(self, client, sabotage):
        import os

        from liminallm.storage.postgres import PostgresStore
        from tests.harness import apply_schema

        runtime = get_runtime()
        url = os.environ["DATABASE_URL"]
        with runtime.store._connect() as conn:
            conn.execute(sabotage)
        try:
            with pytest.raises(RuntimeError) as caught:
                PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))
            assert "migrate" in str(caught.value).lower(), str(caught.value)
        finally:
            with runtime.store._connect() as conn:
                conn.execute(
                    "ALTER TABLE app_user ENABLE TRIGGER app_user_retire_namespace"
                )
            apply_schema(url, embedding_dim=64)


class TestHotStateGoesWithTheAccount:
    @pytest.mark.asyncio
    async def test_deleting_an_account_retires_its_cached_conversations(
        self, client
    ):
        """Individual chat DELETE retires this; bulk erasure bypassed it."""
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        conversation = client.post(
            "/v1/conversations", headers=headers, json={"title": "chat"}
        ).json()["data"]["id"]
        secret = f"SECRET-{uuid.uuid4().hex[:10]}"
        await runtime.cache.set_conversation_summary(
            conversation, {"recent_messages": [{"content": secret}]}
        )
        assert await runtime.cache.get_conversation_summary(conversation)

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        assert await runtime.cache.get_conversation_summary(conversation) is None, (
            "the erased account's messages are still readable in the cache"
        )

    @pytest.mark.asyncio
    async def test_deleting_an_account_revokes_its_cached_sessions(self, client):
        """The row is gone; the cached answer to "whose session is this" is not."""
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")

        email = f"{_unique('era')}@example.com"
        password = "TestPassword123!"
        resp = client.post(
            "/v1/auth/signup", json={"email": email, "password": password}
        )
        assert resp.status_code == 201, resp.text
        user_id = resp.json()["data"]["user_id"]
        _, _, admin_headers = _account(client, admin=True)

        # The real login path, because that is what populates the cache.
        _, session, _ = await runtime.auth.login(email, password)
        assert await runtime.cache.get_session_user(session.id) == (True, user_id)

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        present, cached_user = await runtime.cache.get_session_user(session.id)
        assert not present and cached_user is None, (
            "an erased account's session still resolves from the cache"
        )

    @pytest.mark.asyncio
    async def test_a_redis_outage_does_not_roll_back_the_erasure(self, client):
        """Postgres is canonical. The purge is best-effort and comes after."""
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        _populate(client, headers, user_id)

        async def unreachable(*a, **kw):
            raise ConnectionError("redis is down")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(runtime.cache, "revoke_user_sessions", unreachable)
        try:
            assert client.delete(
                f"/v1/admin/users/{user_id}", headers=admin_headers
            ).status_code in (200, 204)
        finally:
            monkeypatch.undo()

        assert runtime.store.get_user(user_id) is None, (
            "the account survived because a cache could not be reached"
        )
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (user_id,),
        ) == 1


def test_the_audit_log_does_not_keep_the_erased_email():
    """An endpoint that promises erasure must not re-record the address.

    Correlation is what the log is for, and the user id serves that. Writing
    the raw email back out puts the identifier the deletion just removed into
    a store with its own retention.
    """
    import inspect

    from liminallm.api import routes

    source = inspect.getsource(routes.admin_delete_user)
    assert "deleted_email" not in source, (
        "the deletion audit entry still records the erased email address"
    )
