"""A user can retire a knowledge context they own, and only that.

SPEC §12.3 gives users CRUD over their contexts. The API had create, list,
chunks and source add/list - no direct read, no edit, no delete.

Two boundaries make this more than adding three routes. A conversation's
implicit attachment index is a `knowledge_context` too, and it must not be
reachable here: it belongs to the chat's lifetime (SPEC §19.5), not the
user's context collection. And a context is referenced from two directions -
`context_source` and `knowledge_chunk` hang off it, while `conversation`
points at it through `active_context_id` - so deleting one is only correct
while both of those relationships are what the schema claims they are.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('ctx')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}


def _context(client, headers, name=None) -> str:
    resp = client.post(
        "/v1/contexts",
        headers=headers,
        json={"name": name or _unique("ctx"), "description": "a corpus"},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _conversation(client, headers) -> str:
    resp = client.post("/v1/conversations", headers=headers, json={"title": "chat"})
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _implicit_context(client, headers, user_id) -> tuple[str, str]:
    """A conversation and the implicit index its attachment builds."""
    from liminallm.service.attachments import INLINE_MAX_BYTES

    conversation_id = _conversation(client, headers)
    body = ("indexed line worth keeping.\n" * 600).encode()
    assert len(body) > INLINE_MAX_BYTES
    resp = client.post(
        "/v1/files/upload",
        headers={**headers, "Idempotency-Key": _unique("k")},
        files={"file": ("attached.md", body, "text/markdown")},
        data={"conversation_id": conversation_id},
    )
    assert resp.status_code == 200, resp.text
    context = get_runtime().store.get_conversation_attachment_context(
        user_id, conversation_id
    )
    assert context is not None, "no implicit index was built"
    return conversation_id, context.id


def _rows(sql, params):
    with get_runtime().store._connect() as conn:
        return conn.execute(sql, params).fetchall()


def _count(sql, params) -> int:
    return int(_rows(sql, params)[0]["n"])


class TestReadingAndEditingAContext:
    def test_get_answers_by_identity_not_by_page(self, client):
        """A direct read must not depend on where the row sorts.

        The same mistake as the implicit-index lookup one tranche ago: the
        only way to see a context was a listing that pages in SQL, so an
        account with more contexts than a page held could not reach its
        older ones at all.
        """
        user_id, headers = _account(client)
        wanted = _context(client, headers, name="the-one-i-want")

        # Bury it under more contexts than a page holds. Inserted directly:
        # this is about the read path, and 150 HTTP round trips are not.
        store = get_runtime().store
        for _ in range(150):
            store.upsert_context(user_id, _unique("filler"), "filler")

        resp = client.get(f"/v1/contexts/{wanted}", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["id"] == wanted
        assert data["name"] == "the-one-i-want"

    def test_patch_changes_name_and_description(self, client):
        _, headers = _account(client)
        context_id = _context(client, headers)
        resp = client.patch(
            f"/v1/contexts/{context_id}",
            headers=headers,
            json={"name": "renamed", "description": "described"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["name"] == "renamed"
        assert data["description"] == "described"

    def test_patch_one_field_leaves_the_other_alone(self, client):
        _, headers = _account(client)
        context_id = _context(client, headers, name="keep me")
        assert client.patch(
            f"/v1/contexts/{context_id}",
            headers=headers,
            json={"description": "only this"},
        ).status_code == 200
        got = client.get(f"/v1/contexts/{context_id}", headers=headers).json()["data"]
        assert got["name"] == "keep me", "a description change blanked the name"
        assert got["description"] == "only this"

    @pytest.mark.parametrize(
        "payload, field",
        [
            ({"meta": {"auto": True}}, "meta"),
            ({"conversation_id": str(uuid.uuid4())}, "conversation_id"),
            ({"fs_path": "/etc"}, "fs_path"),
            ({"text": "smuggled corpus"}, "text"),
        ],
    )
    def test_patch_refuses_anything_else(self, client, payload, field):
        """Refused, not ignored.

        `meta` and `conversation_id` are how a context would claim to be a
        conversation's implicit index; `fs_path` and `text` are ingestion,
        which is a different mutation with its own path authority.
        """
        _, headers = _account(client)
        context_id = _context(client, headers, name="untouched")
        resp = client.patch(
            f"/v1/contexts/{context_id}",
            headers=headers,
            json={"name": "renamed", **payload},
        )
        assert resp.status_code == 422, resp.text
        assert field in resp.text
        got = client.get(f"/v1/contexts/{context_id}", headers=headers).json()["data"]
        assert got["name"] == "untouched"


class TestDeletingAContext:
    def test_delete_takes_the_sources_and_chunks_with_it(self, client):
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        # Give it something to lose.
        upload = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("corpus.md", b"# Corpus\n\n" + b"a line.\n" * 400,
                            "text/markdown")},
            data={"context_id": context_id},
        )
        assert upload.status_code == 200, upload.text
        assert _count(
            "SELECT COUNT(*) AS n FROM knowledge_chunk WHERE context_id = %s",
            (context_id,),
        ) > 0

        resp = client.delete(f"/v1/contexts/{context_id}", headers=headers)
        assert resp.status_code == 200, resp.text

        assert _count(
            "SELECT COUNT(*) AS n FROM knowledge_context WHERE id = %s", (context_id,)
        ) == 0
        assert _count(
            "SELECT COUNT(*) AS n FROM context_source WHERE context_id = %s",
            (context_id,),
        ) == 0
        assert _count(
            "SELECT COUNT(*) AS n FROM knowledge_chunk WHERE context_id = %s",
            (context_id,),
        ) == 0
        assert client.get(
            f"/v1/contexts/{context_id}", headers=headers
        ).status_code == 404

    def test_delete_leaves_the_source_files_alone(self, client):
        """A context references files; it does not own them."""
        from pathlib import Path

        user_id, headers = _account(client)
        context_id = _context(client, headers)
        client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("keepme.md", b"# Keep\n\n" + b"a line.\n" * 400,
                            "text/markdown")},
            data={"context_id": context_id},
        )
        runtime = get_runtime()
        on_disk = (
            Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"
            / "keepme.md"
        )
        assert on_disk.exists()

        assert client.delete(
            f"/v1/contexts/{context_id}", headers=headers
        ).status_code == 200
        assert on_disk.exists(), "deleting a context deleted the user's file"

    def test_delete_releases_every_conversation_bound_to_it(self, client):
        """`active_context_id` must be nulled, not left dangling."""
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = _conversation(client, headers)
        second = _conversation(client, headers)
        store = get_runtime().store
        with store._connect() as conn:
            conn.execute(
                "UPDATE conversation SET active_context_id = %s WHERE id = ANY(%s::uuid[])",
                (context_id, [first, second]),
            )

        assert client.delete(
            f"/v1/contexts/{context_id}", headers=headers
        ).status_code == 200

        for conversation_id in (first, second):
            conversation = store.get_conversation(conversation_id, user_id=user_id)
            assert conversation is not None, "the conversation was deleted too"
            assert conversation.active_context_id is None, (
                "the conversation still points at a context that is gone"
            )


class TestTheseRoutesAreOwnerOnlyAndOrdinaryOnly:
    def test_another_user_gets_nothing(self, client):
        _, owner_headers = _account(client)
        _, other_headers = _account(client)
        context_id = _context(client, owner_headers, name="private-corpus")

        for method, kwargs in (
            ("get", {}),
            ("patch", {"json": {"name": "taken"}}),
            ("delete", {}),
        ):
            resp = getattr(client, method)(
                f"/v1/contexts/{context_id}", headers=other_headers, **kwargs
            )
            assert resp.status_code in (403, 404), f"{method}: {resp.status_code}"
            assert "private-corpus" not in resp.text

        got = client.get(f"/v1/contexts/{context_id}", headers=owner_headers)
        assert got.status_code == 200
        assert got.json()["data"]["name"] == "private-corpus"

    def test_a_conversations_implicit_index_is_not_reachable(self, client):
        """Its lifetime is the chat's, and it is not part of this collection."""
        user_id, headers = _account(client)
        conversation_id, context_id = _implicit_context(client, headers, user_id)

        for method, kwargs in (
            ("get", {}),
            ("patch", {"json": {"name": "escaped"}}),
            ("delete", {}),
        ):
            resp = getattr(client, method)(
                f"/v1/contexts/{context_id}", headers=headers, **kwargs
            )
            assert resp.status_code == 404, f"{method}: {resp.status_code} {resp.text}"

        # Still there, still the conversation's.
        assert get_runtime().store.get_conversation_attachment_context(
            user_id, conversation_id
        ) is not None

    def test_the_store_itself_refuses_an_implicit_context(self, client):
        """The predicate lives in the mutation, not only in the route helper.

        A route-only guard stays green while someone weakens the store, and
        the store is what a future caller reaches directly. So the SQL for
        both mutations carries `conversation_id IS NULL` itself.
        """
        user_id, headers = _account(client)
        conversation_id, context_id = _implicit_context(client, headers, user_id)
        store = get_runtime().store

        assert store.update_context(
            context_id, owner_user_id=user_id, name="I escaped"
        ) is None
        assert store.delete_context(context_id, owner_user_id=user_id) is False

        survivor = store.get_conversation_attachment_context(user_id, conversation_id)
        assert survivor is not None and survivor.id == context_id
        assert survivor.name != "I escaped"

    def test_the_store_refuses_another_users_context(self, client):
        owner_id, owner_headers = _account(client)
        other_id, _ = _account(client)
        context_id = _context(client, owner_headers, name="mine")
        store = get_runtime().store

        assert store.update_context(
            context_id, owner_user_id=other_id, name="theirs"
        ) is None
        assert store.delete_context(context_id, owner_user_id=other_id) is False
        assert store.get_context(context_id).name == "mine"


class TestTheBindingIsVerifiedAtStartup:
    """`ON DELETE SET NULL` is what makes context deletion safe.

    The schema creates that foreign key conditionally, and the condition is a
    name lookup in `information_schema.table_constraints` - which lists every
    constraint type. A constraint of the same name that is not a foreign key
    satisfies the guard, the FK is never created, and deleting a context then
    leaves conversations pointing at a row that is gone.
    """

    def test_a_same_named_check_constraint_does_not_satisfy_the_check(self, client):
        import os

        from liminallm.storage.postgres import PostgresStore
        from tests.harness import apply_schema

        runtime = get_runtime()
        url = os.environ["DATABASE_URL"]
        with runtime.store._connect() as conn:
            conn.execute(
                "ALTER TABLE conversation "
                "DROP CONSTRAINT conversation_active_context_id_fkey"
            )
            conn.execute(
                "ALTER TABLE conversation "
                "ADD CONSTRAINT conversation_active_context_id_fkey "
                "CHECK (active_context_id IS NULL OR active_context_id IS NOT NULL)"
            )
        try:
            with pytest.raises(RuntimeError) as caught:
                PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))
            assert "migrate" in str(caught.value).lower(), str(caught.value)
        finally:
            with runtime.store._connect() as conn:
                conn.execute(
                    "ALTER TABLE conversation "
                    "DROP CONSTRAINT conversation_active_context_id_fkey"
                )
            apply_schema(url, embedding_dim=64)

        PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))

    def test_the_schema_replaces_a_constraint_wearing_the_right_name(self, client):
        """The other half: the schema has to be able to repair this.

        Refusing to start is only useful if `scripts/migrate.sh` then fixes
        it. A guard that asks information_schema for the *name* finds the
        CHECK constraint, concludes its work is done, and leaves the database
        in exactly the state startup refuses - so the operator is told to run
        a command that changes nothing.
        """
        import os

        from liminallm.storage.postgres import PostgresStore
        from tests.harness import apply_schema

        runtime = get_runtime()
        url = os.environ["DATABASE_URL"]
        with runtime.store._connect() as conn:
            conn.execute(
                "ALTER TABLE conversation "
                "DROP CONSTRAINT conversation_active_context_id_fkey"
            )
            conn.execute(
                "ALTER TABLE conversation "
                "ADD CONSTRAINT conversation_active_context_id_fkey "
                "CHECK (active_context_id IS NULL OR active_context_id IS NOT NULL)"
            )

        # The repair, without touching the constraint by hand first.
        apply_schema(url, embedding_dim=64)

        PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))
        shape = _rows(
            """
            SELECT c.contype, c.confdeltype
            FROM pg_constraint c
            WHERE c.conrelid = 'conversation'::regclass
              AND c.conname = 'conversation_active_context_id_fkey'
            """,
            (),
        )
        assert len(shape) == 1, shape
        assert shape[0]["contype"] == "f", "still not a foreign key"
        assert shape[0]["confdeltype"] == "n", "not ON DELETE SET NULL"

    def test_a_binding_that_cascades_instead_of_nulling_is_refused(self, client):
        """`ON DELETE CASCADE` here would delete the user's conversations.

        Same column, same tables, still a foreign key - and retiring a
        context would take every chat bound to it.
        """
        import os

        from liminallm.storage.postgres import PostgresStore
        from tests.harness import apply_schema

        runtime = get_runtime()
        url = os.environ["DATABASE_URL"]
        with runtime.store._connect() as conn:
            conn.execute(
                "ALTER TABLE conversation "
                "DROP CONSTRAINT conversation_active_context_id_fkey"
            )
            conn.execute(
                "ALTER TABLE conversation "
                "ADD CONSTRAINT conversation_active_context_id_fkey "
                "FOREIGN KEY (active_context_id) REFERENCES knowledge_context(id) "
                "ON DELETE CASCADE"
            )
        try:
            with pytest.raises(RuntimeError) as caught:
                PostgresStore(url, fs_root=str(runtime.settings.shared_fs_root))
            assert "migrate" in str(caught.value).lower(), str(caught.value)
        finally:
            with runtime.store._connect() as conn:
                conn.execute(
                    "ALTER TABLE conversation "
                    "DROP CONSTRAINT conversation_active_context_id_fkey"
                )
            apply_schema(url, embedding_dim=64)


class TestDeleteAgainstIngestion:
    def test_ingestion_that_loses_the_race_leaves_nothing_behind(self, client):
        """Source ingestion is insert-then-work, so deletion can land inside.

        `add_context_source` records the source and the reading, chunking and
        embedding happen afterwards. A delete in that window must not be
        undone by the work still in flight: whichever request wins, the
        durable state has to be all of the context or none of it.
        """
        from liminallm.api import routes as routes_module

        user_id, headers = _account(client)
        context_id = _context(client, headers)
        # Something for the source to find.
        client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("sourced.md", b"# Sourced\n\n" + b"a line.\n" * 400,
                            "text/markdown")},
        )

        from pathlib import Path

        runtime = get_runtime()
        sourced = (
            Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"
            / "sourced.md"
        )
        assert sourced.exists(), "the file the source will read is missing"

        real_add = routes_module.get_runtime().store.add_context_source
        deleted: dict = {}

        def add_then_delete(*args, **kwargs):
            source = real_add(*args, **kwargs)
            if not deleted:
                resp = client.delete(f"/v1/contexts/{context_id}", headers=headers)
                deleted["status"] = resp.status_code
            return source

        store = get_runtime().store
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(store, "add_context_source", add_then_delete)
        try:
            resp = client.post(
                f"/v1/contexts/{context_id}/sources",
                headers=headers,
                json={"fs_path": str(sourced), "recursive": False},
            )
        finally:
            monkeypatch.undo()

        assert deleted.get("status") == 200, "the deletion under test did not happen"
        # The property is the durable state, not which request won.
        assert _count(
            "SELECT COUNT(*) AS n FROM knowledge_context WHERE id = %s", (context_id,)
        ) == 0, "the context came back"
        assert _count(
            "SELECT COUNT(*) AS n FROM context_source WHERE context_id = %s",
            (context_id,),
        ) == 0, "a source row survived the deletion"
        assert _count(
            "SELECT COUNT(*) AS n FROM knowledge_chunk WHERE context_id = %s",
            (context_id,),
        ) == 0, "chunks were written into a context that was deleted"
        assert resp.status_code >= 400, (
            "the source was reported as added to a context that no longer "
            f"exists: {resp.status_code} {resp.text[:300]}"
        )
