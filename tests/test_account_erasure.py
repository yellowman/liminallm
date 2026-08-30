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
made becomes collectable at once - judged by the blob's own mtime, which is
weeks old. A turn that resolved one of those blobs a moment before the
deletion then reads a filesystem where it is gone.

So the account's retirement has to outrank the subordinate lifetimes: while it
is pending, none of them may touch that user, and when it comes due the whole
identity-derived namespace goes at once.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    the old token changes the new account's password - a credential transfer
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

    @pytest.mark.asyncio
    async def test_the_same_pass_collects_when_nothing_is_pending(self, client):
        """The other half of the pair above, and the reason it means anything.

        Every assertion there is that something still exists, which is also
        what a pass that ran no sweeps produces. Measured: unwire any of the
        three from `_run_cleanup_pass` and that test still passes - the
        exclusion was never what kept those files. So the same fixture runs
        the same pass against a live account, where all three collectors must
        take their own kind of debris.
        """
        from liminallm.app import _run_cleanup_pass

        user_id, _, headers = _account(client)
        namespace, staging = _populate(client, headers, user_id)
        runtime = get_runtime()
        # A generation no attachment names. The one `_populate` leaves is
        # referenced by a live conversation, so a correct sweep keeps it and
        # it cannot stand in for the sweep having run.
        orphan = (
            namespace
            / "attachment-generations"
            / "sha256"
            / "ab"
            / f"ab{uuid.uuid4().hex}"
        )
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"a generation nothing references")
        _age(orphan.parent)

        await _run_cleanup_pass(
            runtime, Path(runtime.settings.shared_fs_root), max_age_hours=24
        )
        assert not (namespace / "tmp" / "scratch.txt").exists(), (
            "the cleanup pass never ran the scratch sweep"
        )
        assert not (staging / "in-progress").exists(), (
            "the cleanup pass never ran the archive-staging sweep"
        )
        assert not orphan.exists(), (
            "the cleanup pass never ran the attachment-generation sweep"
        )


class TestSubordinateSweepsDoNotUndercutTheGrace:
    """The account's clock outranks every collector inside its namespace.

    `sweep_generations` marks from what conversations reference. Delete the
    account and there are no conversations, so the mark set is empty and every
    generation the user ever made looks unreferenced - judged by the blob's own
    mtime, which is old. That is the right clock for its normal race and the
    wrong one for this event.
    """

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

    def test_a_sweep_already_running_cannot_be_overtaken_by_the_deletion(
        self, client
    ):
        """The order the skip-list could not describe.

        Every other red here deletes first and sweeps second, which a snapshot
        taken at the top of the pass answers correctly. This is the other
        order: the sweep reaches the account, and the deletion lands while it
        is working. A question asked once is stale from the moment after it is
        asked, so the only thing that can hold here is a lock.

        Both legal outcomes keep the blob, which is what makes the assertion
        one sentence. If the sweep holds the account first it runs against
        pre-deletion state, where the conversation still names the generation.
        If the deletion holds it first, the sweep waits and then finds a
        retirement. There is no interleaving in between.
        """
        import threading

        from liminallm.service.attachments import sweep_generations

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        _populate(client, headers, user_id)
        runtime = get_runtime()
        generations = _namespace(user_id) / "attachment-generations" / "sha256"
        before = set(generations.glob("*/*"))
        assert before

        reached = threading.Event()
        release = threading.Event()
        real = runtime.store.referenced_attachment_checksums

        def pause_at(target_user_id):
            if target_user_id == user_id:
                reached.set()
                assert release.wait(timeout=30), "the sweep was never released"
            return real(target_user_id)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime.store, "referenced_attachment_checksums", pause_at
        )
        deletion: dict = {}

        def delete_the_account():
            resp = client.delete(
                f"/v1/admin/users/{user_id}", headers=admin_headers
            )
            deletion["status"] = resp.status_code

        sweeper = threading.Thread(
            target=sweep_generations,
            args=(runtime.store, runtime.settings.shared_fs_root),
            kwargs={"grace_seconds": 0},
            daemon=True,
        )
        try:
            sweeper.start()
            assert reached.wait(timeout=30), "the sweep never reached the account"
            deleter = threading.Thread(target=delete_the_account, daemon=True)
            deleter.start()
            # An unserialized deletion commits in tens of milliseconds, so a
            # second is a wide margin for the schedule under test. A
            # serialized one is still blocked on the lock.
            deleter.join(timeout=1)
            release.set()
            sweeper.join(timeout=30)
            deleter.join(timeout=30)
        finally:
            release.set()
            monkeypatch.undo()

        assert not sweeper.is_alive() and not deleter.is_alive()
        assert deletion.get("status") in (200, 204), deletion
        assert set(generations.glob("*/*")) == before, (
            "an account deletion landed inside a running generation sweep, and "
            "the sweep finished on its stale answer - reclaiming a generation "
            "the erasure had just promised an hour of grace"
        )

    @pytest.mark.parametrize("sweep_name", ["tmp", "archive_staging"])
    def test_a_path_sweep_holds_the_account_it_is_working_on(
        self, client, sweep_name
    ):
        """These two are serialized, not protected.

        Their contents are not what the grace period is for. A week-old
        scratch file is legitimately collectable while the account is alive,
        and an abandoned staging tree is collectable by definition, so
        asserting they survive would be asserting that a correct sweep did
        nothing. What has to hold is the ordering: the deletion cannot land in
        the middle of one of these accounts, because the namespace retirement
        removes the whole tree and a sweep pruning directories inside it is
        the one other writer.

        So the assertion is about the schedule directly, and it is taken at the
        destructive step rather than at the guard: while this account's files
        are being removed, the deletion is still waiting. Pausing at the guard
        instead would only prove the guard was entered - measured, a body
        moved outside the `with` survived that version of this test. A set
        read once at the top of the pass makes neither true.

        The seam is the per-account helper, which exists only because the work
        had to be separable from the decision, so this red cannot run on a
        tree that has no guard. Its mutations stand in for that.
        """
        import threading

        from liminallm import app as app_module

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        _populate(client, headers, user_id)
        runtime = get_runtime()
        root = Path(runtime.settings.shared_fs_root)
        sweep, helper = (
            (app_module._sweep_tmp_dirs, "_sweep_one_tmp_dir")
            if sweep_name == "tmp"
            else (app_module._sweep_archive_staging, "_sweep_one_staging_dir")
        )

        reached = threading.Event()
        release = threading.Event()
        real = getattr(app_module, helper)

        def pause_at_the_removal(target, *args, **kwargs):
            if user_id in str(target):
                reached.set()
                assert release.wait(timeout=30), "the sweep was never released"
            return real(target, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(app_module, helper, pause_at_the_removal)
        deletion: dict = {}

        def delete_the_account():
            deletion["status"] = client.delete(
                f"/v1/admin/users/{user_id}", headers=admin_headers
            ).status_code

        sweeper = threading.Thread(
            target=sweep, args=(runtime.store, root, 24), daemon=True
        )
        deleter = threading.Thread(target=delete_the_account, daemon=True)
        try:
            sweeper.start()
            assert reached.wait(timeout=30), "the sweep never reached the account"
            deleter.start()
            # A second: an unserialized deletion commits in tens of
            # milliseconds. A serialized one is still blocked on the lock.
            deleter.join(timeout=1)
            assert deleter.is_alive() and "status" not in deletion, (
                "an account deletion committed while a sweep was working "
                "inside that account's namespace"
            )
            release.set()
            sweeper.join(timeout=30)
            deleter.join(timeout=30)
        finally:
            release.set()
            monkeypatch.undo()

        assert not sweeper.is_alive() and not deleter.is_alive()
        assert deletion.get("status") in (200, 204), (
            f"the deletion never completed after the {sweep_name} sweep "
            f"released the account: {deletion}"
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
        """First observed, then collected - never removed on sight."""
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
    async def test_every_family_the_purge_names_is_actually_purged(self, client):
        """One row per family in `purge_user_state`, because seven had none.

        Measured before this existed: disable the session-index, session
        activity, session rotation, MFA, router-cache, concurrency or
        email-verification family and the whole suite still passed. A family
        purged only by code nothing exercises stops being purged the next time
        its key shape changes, and says nothing when it does.

        The table is the list the purge iterates, so a family added to
        production without a witness leaves a row to write here.
        """
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
        headers = {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}
        _, _, admin_headers = _account(client, admin=True)
        # A real session and a real conversation: the per-session and
        # per-conversation families are addressed from the deleting
        # transaction's own rows, not from anything Redis holds.
        _, session, _ = await runtime.auth.login(email, password)
        conversation = client.post(
            "/v1/conversations", headers=headers, json={"title": "chat"}
        ).json()["data"]["id"]

        keys = {
            "sessions": f"auth:session:{session.id}",
            "session_index": f"auth:user_sessions:{user_id}",
            "session_activity": f"session:activity:{session.id}",
            "session_rotation": f"session:rotation:{session.id}",
            "conversation_summaries": f"chat:summary:{conversation}",
            "mfa_attempts": f"mfa:attempts:{user_id}",
            "mfa_lockout": f"mfa:lockout:{user_id}",
            "idempotency": f"idemp:chat:{user_id}:{_unique('k')}",
            "router_cache": f"router:last:model:{user_id}:0",
            "concurrency": f"concurrency:workflow:{user_id}",
        }
        for key in keys.values():
            await runtime.cache.client.set(key, "the account's own content")
        # These two are found by value: the token itself is opaque, so the
        # purge scans the prefix and keeps only the keys naming this account.
        for family, prefix in (("reset_tokens", "reset"), ("verify_tokens", "verify")):
            keys[family] = f"{prefix}:{uuid.uuid4().hex}"
            await runtime.cache.client.set(keys[family], user_id)
        for family, key in keys.items():
            assert await runtime.cache.client.exists(key), f"unwritten: {family}"

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        left = []
        for family, key in keys.items():
            if await runtime.cache.client.exists(key):
                left.append(family)
        assert not left, (
            f"the erased account's own content is still cached: {sorted(left)}"
        )

    @pytest.mark.asyncio
    async def test_the_session_index_is_not_the_authority_on_sessions(self, client):
        """`auth:user_sessions` is a convenience index with its own TTL.

        Deriving what to purge from it means purging nothing exactly when it
        has expired and the session keys it should have named have not. The
        ids are read from Postgres inside the deleting transaction instead,
        which is why this can be forced: drop the index and the session must
        still go.

        The plain case - delete an account, its cached session stops
        resolving - is this test with the index left in place, so it is this
        one and not two.
        """
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

        _, session, _ = await runtime.auth.login(email, password)
        assert await runtime.cache.get_session_user(session.id) == (True, user_id)

        # The index expires; the session key it named does not.
        await runtime.cache.client.delete(f"auth:user_sessions:{user_id}")
        assert await runtime.cache.get_session_user(session.id) == (True, user_id)

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        present, cached_user = await runtime.cache.get_session_user(session.id)
        assert not present and cached_user is None, (
            "the purge found nothing to do because Redis's own session index "
            "had expired, and left the erased account's session resolvable"
        )

    @pytest.mark.asyncio
    async def test_an_identity_token_does_not_outlive_its_account(self, client):
        """A reset token names an account. The account is gone."""
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")

        user_id, _, _ = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        token = await runtime.auth.initiate_password_reset(
            runtime.store.get_user(user_id)
        )
        assert await runtime.cache.client.get(f"reset:{token}") == user_id

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)
        assert await runtime.cache.client.get(f"reset:{token}") is None, (
            "a password reset token for an erased account is still stored"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refused", ["auth:", "idemp:"], ids=["listed", "scanned"])
    async def test_one_unreachable_family_does_not_cancel_the_others(
        self, client, refused
    ):
        """Every category is its own attempt.

        The first version ran all of them inside one `try`, so a failure
        revoking sessions meant no conversation summary was even attempted -
        one unreachable key pattern leaving an account's messages readable.

        The purge has two loops - the families it can address by name and the
        ones it has to scan for - and each keeps its own `try`. So each is
        refused a family it attempts early, and the assertion is on a family it
        attempts after that: refusing a later one would prove nothing, because
        the earlier one is gone whether or not the categories are independent.
        """
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")

        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        conversation = client.post(
            "/v1/conversations", headers=headers, json={"title": "chat"}
        ).json()["data"]["id"]
        await runtime.cache.set_conversation_summary(
            conversation, {"recent_messages": [{"content": "still here"}]}
        )
        # One key in each refused family, or the purge finds nothing there to
        # fail on and the refusal never happens.
        await runtime.cache.set_idempotency_record(
            "chat", user_id, _unique("key"), {"status": "completed"}
        )
        token = await runtime.auth.initiate_password_reset(
            runtime.store.get_user(user_id)
        )
        real_delete = type(runtime.cache.client).delete

        async def refuse_one_family(self, *keys, **kw):
            if any(str(k).startswith(refused) for k in keys):
                raise ConnectionError("this key family is unavailable")
            return await real_delete(self, *keys, **kw)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(type(runtime.cache.client), "delete", refuse_one_family)
        try:
            assert client.delete(
                f"/v1/admin/users/{user_id}", headers=admin_headers
            ).status_code in (200, 204)
        finally:
            monkeypatch.undo()

        # Sessions are the first family the purge lists; the idempotency scan
        # is the first it scans for. The summary comes after the one, the reset
        # token after the other.
        later = (
            runtime.cache.get_conversation_summary(conversation)
            if refused == "auth:"
            else runtime.cache.client.get(f"reset:{token}")
        )
        assert await later is None, (
            "one failing key family stopped the rest of the purge, so the "
            "erased account's own content stayed readable"
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
        monkeypatch.setattr(runtime.cache, "purge_user_state", unreachable)
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


class TestAnInFlightRequestCannotUndoTheErasure:
    """The purge is complete at an instant. That is not the same as gone.

    A request authorized before the deletion is allowed to finish, and it
    finishes by writing. Every cached write those requests make lands after
    the purge has already run, so the erased account's own content comes back
    under a key naming the account, for as long as that key's TTL:

        CHAT                          ADMIN DELETE
        ----                          ------------
        authorized as U
        turn finishes
                                      delete U
                                      purge every cached key of U
                                      200
        store the idempotency record
          -> the completed response,
             back for 24 hours

    A liveness check before the write does not close it - that is the same
    check-then-act the collectors had, one participant further along. Only a
    lock held across the decision and the write does.

    Each of these closes by asserting the key is gone, so each is also the
    plain "the purge removed this family" witness, under the harder schedule
    and through the production write path rather than the store's own setter.
    """

    async def _forced_schedule(
        self, client, user_id, admin_headers, write, key_exists,
        reached=None, release=None,
    ):
        """Pause a cache write mid-flight, then erase the account under it.

        Returns nothing; asserts both directions of the one property. While
        the writer holds the account the deletion must still be waiting, and
        once everything settles the key must not exist.

        With no `reached`/`release` the pause is installed just inside the
        guard, which shows the guard is entered and held. A caller that passes
        its own pair has put the pause at the statement that actually writes,
        which is the stronger placement and the one that catches a guard
        released too early - measured, a claim written after the `with` block
        survived the weaker version of this.
        """
        import threading

        runtime = get_runtime()
        monkeypatch = pytest.MonkeyPatch()
        if reached is None or release is None:
            reached = threading.Event()
            release = threading.Event()
            real = runtime.store.hold_live_user

            @contextlib.contextmanager
            def pause_inside(target_user_id):
                with real(target_user_id) as live:
                    if target_user_id == user_id:
                        reached.set()
                        assert release.wait(timeout=30), "the writer was not released"
                    yield live

            monkeypatch.setattr(runtime.store, "hold_live_user", pause_inside)
        deletion: dict = {}

        def delete_the_account():
            deletion["status"] = client.delete(
                f"/v1/admin/users/{user_id}", headers=admin_headers
            ).status_code

        def run_the_write():
            asyncio.run(write())

        writer = threading.Thread(target=run_the_write, daemon=True)
        deleter = threading.Thread(target=delete_the_account, daemon=True)
        try:
            writer.start()
            assert reached.wait(timeout=30), "the write never reached the guard"
            deleter.start()
            # A second: an unguarded deletion commits and purges in tens of
            # milliseconds, which is the schedule under test.
            deleter.join(timeout=1)
            assert deleter.is_alive() and "status" not in deletion, (
                "the account was erased and purged while a write on its "
                "behalf was already in flight"
            )
            release.set()
            writer.join(timeout=30)
            deleter.join(timeout=30)
        finally:
            release.set()
            monkeypatch.undo()

        assert not writer.is_alive() and not deleter.is_alive()
        assert deletion.get("status") in (200, 204), deletion
        assert not await key_exists(), (
            "an in-flight request wrote the erased account's content back "
            "into the cache after the purge had already run"
        )

    @pytest.mark.asyncio
    async def test_an_in_flight_idempotency_record_does_not_resurrect(self, client):
        from liminallm.service.runtime import _set_cached_idempotency_record

        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        key = _unique("key")
        secret = f"SECRET-{uuid.uuid4().hex[:10]}"

        async def write():
            await _set_cached_idempotency_record(
                runtime,
                "chat",
                user_id,
                key,
                {"status": "completed", "response": {"message": secret}},
            )

        async def key_exists():
            return [
                k
                async for k in runtime.cache.client.scan_iter(
                    match=f"idemp:*{user_id}*", count=500
                )
            ]

        await self._forced_schedule(
            client, user_id, admin_headers, write, key_exists
        )

    @pytest.mark.asyncio
    async def test_an_in_flight_conversation_summary_does_not_resurrect(self, client):
        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        conversation = client.post(
            "/v1/conversations", headers=headers, json={"title": "chat"}
        ).json()["data"]["id"]
        history = runtime.store.list_messages(conversation, user_id=user_id)

        async def write():
            await runtime.workflow.cache_conversation_state(
                conversation, history, user_id
            )

        async def key_exists():
            return await runtime.cache.get_conversation_summary(conversation)

        await self._forced_schedule(
            client, user_id, admin_headers, write, key_exists
        )

    @pytest.mark.asyncio
    async def test_an_in_flight_idempotency_claim_does_not_resurrect(self, client):
        """The claim is a key too, and it is written by a different call.

        Deleting first and then entering the guard proves the liveness
        predicate and nothing about where the lock is held. This pauses at the
        cache acquisition itself - the statement that actually creates
        `idemp:...:<user>:...` - so a guard that answered and released before
        it lets the deletion through during the pause.
        """
        import threading

        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        key = _unique("key")

        reached = threading.Event()
        release = threading.Event()
        real_acquire = runtime.cache.acquire_idempotency_slot

        async def pause_at_the_claim(route, target_user_id, *args, **kwargs):
            if target_user_id == user_id:
                reached.set()
                assert release.wait(timeout=30), "the claim was never released"
            return await real_acquire(route, target_user_id, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime.cache, "acquire_idempotency_slot", pause_at_the_claim
        )

        async def write():
            from liminallm.service.runtime import _acquire_idempotency_slot

            await _acquire_idempotency_slot(
                runtime, "chat", user_id, key, {"status": "in_progress"}
            )

        async def key_exists():
            return [
                k
                async for k in runtime.cache.client.scan_iter(
                    match=f"idemp:*{user_id}*", count=500
                )
            ]

        try:
            await self._forced_schedule(
                client, user_id, admin_headers, write, key_exists, reached, release
            )
        finally:
            monkeypatch.undo()

    @pytest.mark.asyncio
    async def test_a_write_for_an_account_that_is_already_gone_does_nothing(
        self, client
    ):
        """The other history: the deletion took the lock first."""
        from liminallm.service.runtime import _set_cached_idempotency_record

        runtime = get_runtime()
        if runtime.cache is None:
            pytest.skip("no Redis in this environment")
        user_id, _, headers = _account(client)
        _, _, admin_headers = _account(client, admin=True)
        conversation = client.post(
            "/v1/conversations", headers=headers, json={"title": "chat"}
        ).json()["data"]["id"]
        history = runtime.store.list_messages(conversation, user_id=user_id)

        assert client.delete(
            f"/v1/admin/users/{user_id}", headers=admin_headers
        ).status_code in (200, 204)

        # Exactly what the in-flight request would have done, a moment late.
        await _set_cached_idempotency_record(
            runtime,
            "chat",
            user_id,
            _unique("key"),
            {"status": "completed", "response": {"message": "too late"}},
        )
        await runtime.workflow.cache_conversation_state(
            conversation, history, user_id
        )

        # And through the real guard, which claims a slot before it has a
        # result to store. That claim is a key naming the account too, and the
        # purge it would have to survive has already run.
        from liminallm.api.idempotency import IdempotencyGuard
        from liminallm.api.schemas import Envelope

        async with IdempotencyGuard(
            "chat", user_id, _unique("key"), require=True
        ) as guard:
            assert guard.cached is None
            await guard.store_result(
                Envelope(status="ok", data={"late": True}, request_id=guard.request_id)
            )

        left = [
            k
            async for k in runtime.cache.client.scan_iter(
                match=f"idemp:*{user_id}*", count=500
            )
        ]
        assert left == [], f"a write for an erased account created keys: {left}"
        assert await runtime.cache.get_conversation_summary(conversation) is None

    def test_a_sweep_does_not_make_a_deletion_wait_on_an_upload(self, client):
        """The account's lifetime is held; the blob's lock is not waited on.

        Holding the account across a blocking per-blob wait means a deletion
        inherits that wait, once per contended blob. The sweep takes each
        blob's lock without waiting instead, because a blob it skips is one
        the next pass collects - the upload is the side that must publish.
        """
        import hashlib
        import os
        import threading
        import time as time_

        from liminallm.service.attachments import (
            generation_lock,
            store_generation,
            sweep_generations,
        )

        user_id, _, headers = _account(client)
        runtime = get_runtime()
        root = runtime.settings.shared_fs_root

        # Unreferenced and old, so the sweep actually reaches its lock. A blob
        # an attachment still names is skipped before that, which is why the
        # first version of this test passed with the blocking wait in place.
        body = f"orphan {uuid.uuid4().hex}".encode()
        checksum = hashlib.sha256(body).hexdigest()
        blob = store_generation(root, user_id, body, checksum)
        assert blob is not None and blob.is_file()
        week_ago = time_.time() - 7 * 86400
        os.utime(blob, (week_ago, week_ago))
        assert checksum not in runtime.store.referenced_attachment_checksums(user_id)

        held = threading.Event()
        release = threading.Event()

        def hold_the_blob():
            with generation_lock(root, user_id, checksum):
                held.set()
                release.wait(timeout=30)

        holder = threading.Thread(target=hold_the_blob, daemon=True)
        holder.start()
        try:
            assert held.wait(timeout=10), "the fixture never took the blob lock"
            started = time_.monotonic()
            sweep_generations(runtime.store, root, grace_seconds=0)
            elapsed = time_.monotonic() - started
        finally:
            release.set()
            holder.join(timeout=30)

        # The blocking wait is 30s per contended blob; anything near that means
        # the sweep queued behind the upload while holding the account.
        assert elapsed < 10, (
            f"the sweep waited {elapsed:.1f}s on a contended generation lock "
            "while holding the account's lifetime, which is a wait the "
            "account's own deletion would have inherited"
        )
        assert blob.exists(), "the contended blob was taken anyway"

    def test_the_two_guards_answer_different_questions(self, client):
        """Debris to a collector is not a principal to a writer.

        A user id with no account row and no retirement - the namespace of an
        account erased long enough ago that its record was cleared - is
        something a collector may act on and something no write may happen on
        behalf of. Reusing the collector's boolean on the write side is how a
        caller ends up writing for an account that is not there.
        """
        runtime = get_runtime()
        gone = str(uuid.uuid4())
        assert runtime.store.get_user(gone) is None
        assert _count(
            "SELECT COUNT(*) AS n FROM user_namespace_retirement WHERE user_id = %s",
            (gone,),
        ) == 0

        with runtime.store.hold_user_lifetime(gone) as collectable:
            assert collectable is True
        with runtime.store.hold_live_user(gone) as live:
            assert live is False, (
                "the write guard treated an id with no account as a principal"
            )

        # And they agree on a live account, so the difference is about
        # existence rather than about being generally stricter.
        live_id, _, _ = _account(client)
        with runtime.store.hold_user_lifetime(live_id) as collectable:
            assert collectable is True
        with runtime.store.hold_live_user(live_id) as live:
            assert live is True


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
