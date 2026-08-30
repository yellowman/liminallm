"""A conversation owns the state that exists only for that conversation.

SPEC §12.3 gives users CRUD over their own conversations; SPEC §19.5 scopes a
conversation attachment to "that chat only". Those two sentences meet at
deletion: if a chat can be deleted, everything scoped to it has to go with it,
and nothing may be able to recreate that state afterwards.

The implicit attachment index is the hard case. It is a separate
`knowledge_context` row, and its only tie to the chat used to live in
`meta.conversation_id` - a JSON string with no relational meaning. Deleting
the conversation left the index and its chunks behind, still holding the text
of files the user attached to a chat they deleted.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('life')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}


def _conversation(client, headers, title="chat") -> str:
    resp = client.post("/v1/conversations", headers=headers, json={"title": title})
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


#: Inline and searchable are exclusive, and the split is on size: a text file
#: at or under `INLINE_MAX_BYTES` is injected into the prompt, and only a
#: larger one is chunked into the conversation's implicit index. A small file
#: therefore builds no index at all, which is not the case under test here.
def _searchable_body(marker: str) -> bytes:
    from liminallm.service.attachments import INLINE_MAX_BYTES

    filler = f"{marker} line of text worth indexing.\n" * 600
    body = f"# {marker}\n\n{filler}".encode()
    assert len(body) > INLINE_MAX_BYTES, len(body)
    return body


def _attach(client, headers, conversation_id, name, body):
    """Upload a searchable file into a conversation's implicit index."""
    return client.post(
        "/v1/files/upload",
        headers={**headers, "Idempotency-Key": _unique("k")},
        files={"file": (name, body, "text/markdown")},
        data={"conversation_id": conversation_id},
    )


def _implicit_context(user_id, conversation_id):
    return get_runtime().store.get_conversation_attachment_context(
        user_id, conversation_id
    )


def _auto_contexts(user_id) -> list:
    """Every implicit context this user has, read straight from the table."""
    store = get_runtime().store
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT id, meta FROM knowledge_context WHERE owner_user_id = %s",
            (user_id,),
        ).fetchall()
    out = []
    for row in rows:
        meta = row["meta"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        if (meta or {}).get("auto"):
            out.append(str(row["id"]))
    return out


def _chunk_count_for_owner(user_id) -> int:
    """Every chunk under any of this user's contexts."""
    store = get_runtime().store
    with store._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge_chunk kc "
            "JOIN knowledge_context ctx ON ctx.id = kc.context_id "
            "WHERE ctx.owner_user_id = %s",
            (user_id,),
        ).fetchone()
    return int(row["n"])


def _chunk_count(context_id) -> int:
    store = get_runtime().store
    with store._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge_chunk WHERE context_id = %s",
            (context_id,),
        ).fetchone()
    return int(row["n"])


class TestDeletingAChatTakesItsChatOnlyState:
    """Deletion is not complete while the attachment index survives it."""

    def test_deleting_a_chat_removes_its_index_and_the_chunks_under_it(self, client):
        """The chat, its implicit context and its chunks go together.

        `delete_conversation` removed the conversation row and its messages.
        The implicit context is a different table with no relational tie to
        the conversation, so it stayed - and its chunks stayed with it,
        holding the text of a file attached to a chat that no longer exists.
        """
        user_id, headers = _account(client)
        conversation_id = _conversation(client, headers)
        resp = _attach(
            client, headers, conversation_id, "secret.md", _searchable_body("secret")
        )
        assert resp.status_code == 200, resp.text

        context = _implicit_context(user_id, conversation_id)
        assert context is not None, "the attachment did not build an implicit index"
        assert _chunk_count(context.id) > 0, "nothing was indexed to delete"

        deleted = client.delete(
            f"/v1/conversations/{conversation_id}", headers=headers
        )
        assert deleted.status_code == 200, deleted.text

        assert (
            client.get(f"/v1/conversations/{conversation_id}", headers=headers)
        ).status_code == 404
        assert _implicit_context(user_id, conversation_id) is None
        assert _auto_contexts(user_id) == [], (
            "the implicit context outlived the conversation it belongs to"
        )
        assert _chunk_count(context.id) == 0, (
            "the deleted chat's file text is still indexed"
        )

    def test_a_generation_another_chat_still_names_is_not_reclaimed(self, client):
        """Deleting one chat must not take bytes another chat still uses.

        Two chats attaching identical bytes share one content-addressed
        object. The sweep is driven by what conversations still name, so
        deleting the first chat must leave the object in place for the
        second.
        """
        from liminallm.service.attachments import generation_path, sweep_generations

        user_id, headers = _account(client)
        first = _conversation(client, headers, "first")
        second = _conversation(client, headers, "second")
        body = _searchable_body("shared")
        assert _attach(client, headers, first, "shared.md", body).status_code == 200
        assert _attach(client, headers, second, "shared.md", body).status_code == 200

        runtime = get_runtime()
        conversation = runtime.store.get_conversation(second, user_id=user_id)
        attachments = (conversation.meta or {}).get("attachments") or []
        checksum = attachments[0]["checksum"]
        blob = generation_path(runtime.settings.shared_fs_root, user_id, checksum)
        assert blob.exists()

        assert client.delete(
            f"/v1/conversations/{first}", headers=headers
        ).status_code == 200

        sweep_generations(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert blob.exists(), (
            "the object was reclaimed while a second conversation still "
            "names that checksum"
        )
        # And the surviving chat can still resolve it.
        assert runtime.store.attachment_checksum_referenced(user_id, checksum)

    def test_the_last_chat_naming_a_generation_releases_it(self, client):
        """Once no conversation names the bytes, the sweep may take them.

        The other half of the previous test: the object is held by
        references, not held forever.
        """
        from liminallm.service.attachments import generation_path, sweep_generations

        user_id, headers = _account(client)
        first = _conversation(client, headers, "first")
        second = _conversation(client, headers, "second")
        body = _searchable_body("shared")
        assert _attach(client, headers, first, "shared.md", body).status_code == 200
        assert _attach(client, headers, second, "shared.md", body).status_code == 200

        runtime = get_runtime()
        conversation = runtime.store.get_conversation(second, user_id=user_id)
        checksum = ((conversation.meta or {}).get("attachments") or [])[0]["checksum"]
        blob = generation_path(runtime.settings.shared_fs_root, user_id, checksum)

        for conversation_id in (first, second):
            assert client.delete(
                f"/v1/conversations/{conversation_id}", headers=headers
            ).status_code == 200

        assert not runtime.store.attachment_checksum_referenced(user_id, checksum)
        sweep_generations(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=0
        )
        assert not blob.exists(), (
            "no conversation names this checksum any more and the object "
            "was still not reclaimed"
        )


class TestAnUploadCannotOutliveItsConversation:
    """The lifetime boundary, forced rather than hoped for."""

    def test_a_chat_deleted_mid_upload_gets_no_index(self, client, monkeypatch):
        """Validate, delete, resume: the upload must fail and leave nothing.

        The upload validates the conversation early and does the expensive
        work afterwards, so a deletion in between is a real schedule, not a
        contrived one. Recording the attachment already noticed - the row
        lock finds no conversation and returns None - but that None was
        turned into an empty list and the route still answered 200, after
        creating an implicit context and indexing chunks under a
        conversation that no longer existed.

        The pause is taken at `classify_attachment`, which the route calls
        immediately after validating the conversation and before it touches
        the index.
        """
        from liminallm.api import routes as routes_module

        user_id, headers = _account(client)
        conversation_id = _conversation(client, headers)
        real_classify = routes_module.classify_attachment
        deleted: dict = {}

        def classify_then_delete(name, size):
            caps = real_classify(name, size)
            if not deleted:
                resp = client.delete(
                    f"/v1/conversations/{conversation_id}", headers=headers
                )
                deleted["status"] = resp.status_code
            return caps

        monkeypatch.setattr(
            routes_module, "classify_attachment", classify_then_delete
        )
        resp = _attach(
            client, headers, conversation_id, "late.md", _searchable_body("late")
        )

        assert deleted.get("status") == 200, "the deletion under test did not happen"
        assert resp.status_code == 409, (
            "the upload reported success for a conversation that was deleted "
            f"before it indexed anything: {resp.status_code} {resp.text[:400]}"
        )
        assert "conversation" in resp.text.lower(), resp.text
        assert _auto_contexts(user_id) == [], (
            "an implicit context exists for a conversation that does not"
        )
        assert _chunk_count_for_owner(user_id) == 0, (
            "chunks were indexed for a conversation that no longer exists"
        )

    def test_an_inline_attachment_racing_deletion_also_fails(self, client):
        """The same race one path over, where no foreign key is involved.

        A small text file is inlined rather than indexed, so the upload never
        creates an implicit context and the foreign key never gets a chance
        to refuse anything. The only thing that notices is the attachment
        record: the store takes the conversation's row lock, finds nothing,
        and returns None. That None used to become an empty list and the
        route answered 200 for a chat that no longer existed.
        """
        from liminallm.api import routes as routes_module

        _, headers = _account(client)
        conversation_id = _conversation(client, headers)
        real_classify = routes_module.classify_attachment
        deleted: dict = {}

        def classify_then_delete(name, size):
            caps = real_classify(name, size)
            assert caps["inline"] and not caps["searchable"], caps
            if not deleted:
                resp = client.delete(
                    f"/v1/conversations/{conversation_id}", headers=headers
                )
                deleted["status"] = resp.status_code
            return caps

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(routes_module, "classify_attachment", classify_then_delete)
        try:
            resp = _attach(client, headers, conversation_id, "note.md", b"# tiny\n")
        finally:
            monkeypatch.undo()

        assert deleted.get("status") == 200, "the deletion under test did not happen"
        assert resp.status_code == 409, (
            "the upload reported success for a conversation deleted while it "
            f"ran: {resp.status_code} {resp.text[:400]}"
        )


class TestOnlyTheOwnerChangesAConversation:
    """CRUD is owner-only, and PATCH is narrow."""

    def test_another_user_cannot_patch_or_delete_the_conversation(self, client):
        """A stranger gets the same answer whether or not the chat exists."""
        owner_id, owner_headers = _account(client)
        _, other_headers = _account(client)
        conversation_id = _conversation(client, owner_headers, "private")

        patched = client.patch(
            f"/v1/conversations/{conversation_id}",
            headers=other_headers,
            json={"title": "taken over"},
        )
        assert patched.status_code == 404, patched.text
        deleted = client.delete(
            f"/v1/conversations/{conversation_id}", headers=other_headers
        )
        assert deleted.status_code == 404, deleted.text

        # Neither the title nor the existence of the chat leaked, and nothing
        # about it changed.
        assert "private" not in patched.text
        still_there = client.get(
            f"/v1/conversations/{conversation_id}", headers=owner_headers
        )
        assert still_there.status_code == 200
        assert still_there.json()["data"]["title"] == "private"

    def test_patch_changes_title_and_status(self, client):
        """The two editable fields move, and nothing else does."""
        user_id, headers = _account(client)
        conversation_id = _conversation(client, headers, "before")

        resp = client.patch(
            f"/v1/conversations/{conversation_id}",
            headers=headers,
            json={"title": "after", "status": "archived"},
        )
        assert resp.status_code == 200, resp.text

        conversation = get_runtime().store.get_conversation(
            conversation_id, user_id=user_id
        )
        assert conversation.title == "after"
        assert conversation.status == "archived"
        assert not (conversation.meta or {}).get("public")
        assert conversation.active_context_id is None

    def test_patch_only_one_field_leaves_the_other_alone(self, client):
        """A PATCH is not a replace: an omitted field keeps its value."""
        user_id, headers = _account(client)
        conversation_id = _conversation(client, headers, "keep me")
        assert client.patch(
            f"/v1/conversations/{conversation_id}",
            headers=headers,
            json={"status": "archived"},
        ).status_code == 200

        conversation = get_runtime().store.get_conversation(
            conversation_id, user_id=user_id
        )
        assert conversation.title == "keep me", "a status change blanked the title"
        assert conversation.status == "archived"

    @pytest.mark.parametrize(
        "payload, field",
        [
            ({"meta": {"public": True}}, "meta"),
            ({"meta": {"attachments": [{"name": "injected"}]}}, "meta"),
            ({"active_context_id": str(uuid.uuid4())}, "active_context_id"),
        ],
    )
    def test_patch_refuses_fields_that_are_not_its_business(
        self, client, payload, field
    ):
        """Refused, not ignored.

        `meta` carries the public-share flag and the attachment records, and
        `active_context_id` names a context whose ownership is checked where
        contexts are chosen. Silently dropping them would answer 200 to a
        request that did not happen; the caller is told instead.
        """
        user_id, headers = _account(client)
        conversation_id = _conversation(client, headers, "untouched")

        resp = client.patch(
            f"/v1/conversations/{conversation_id}",
            headers=headers,
            json={"title": "renamed", **payload},
        )
        assert resp.status_code == 422, resp.text
        assert field in resp.text

        # And the request was refused whole: the title did not change either.
        conversation = get_runtime().store.get_conversation(
            conversation_id, user_id=user_id
        )
        assert conversation.title == "untouched"
        assert not (conversation.meta or {}).get("public")
        assert not (conversation.meta or {}).get("attachments")
        assert conversation.active_context_id is None

    @pytest.mark.parametrize("status", ["deleted", "", "not-a-status"])
    def test_patch_refuses_a_status_outside_the_allowed_set(self, client, status):
        """Status is an enumeration, not free text."""
        _, headers = _account(client)
        conversation_id = _conversation(client, headers)
        resp = client.patch(
            f"/v1/conversations/{conversation_id}",
            headers=headers,
            json={"status": status},
        )
        assert resp.status_code == 422, resp.text


class TestTheGuardKeysOnTheKeyNotTheDescription:
    """`meta.auto` describes an implicit index; the foreign key defines it."""

    def test_a_context_tied_to_a_chat_is_not_nameable_without_the_json_marker(
        self, client
    ):
        """A row can carry the relationship and not the description.

        The capability guard refuses conversation indexes so one chat cannot
        read another's attachment by naming its id (SPEC §19.5). It used to
        ask `meta.auto`, which is a description the row may simply not have -
        anything inserting the relationship without also writing the JSON got
        a context that authorization treated as an ordinary one.
        """
        user_id, headers = _account(client)
        conversation_id = _conversation(client, headers)
        store = get_runtime().store
        with store._connect() as conn:
            row = conn.execute(
                "INSERT INTO knowledge_context "
                "(id, owner_user_id, name, description, conversation_id) "
                "VALUES (gen_random_uuid(), %s, %s, %s, %s) RETURNING id",
                (user_id, "no marker", "", conversation_id),
            ).fetchone()
        context_id = str(row["id"])

        # Reading the context's chunks is the escape this guard exists for:
        # it is how one chat would read another chat's attachment text.
        for path in (
            f"/v1/contexts/{context_id}/chunks",
            f"/v1/contexts/{context_id}/sources",
        ):
            resp = client.get(path, headers=headers)
            assert resp.status_code == 404, (
                f"{path} was reachable for a context tied to a conversation, "
                f"because the row carried no meta.auto: {resp.status_code}"
            )

        # Naming it as an upload target is refused for the same reason.
        upload = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("x.md", _searchable_body("x"), "text/markdown")},
            data={"context_id": context_id},
        )
        assert upload.status_code == 404, upload.text

        # And it is not offered in the listing either.
        listed = client.get("/v1/contexts", headers=headers)
        assert listed.status_code == 200, listed.text
        assert context_id not in listed.text


class TestDeletingAChatRetiresItsCachedState:
    """Postgres is not the only place a conversation's text lives.

    The relational lifetime is exact now, and it covers exactly the tables.
    Redis holds the same content on its own schedule: recent messages are
    cached under `chat:summary:<id>` with an hour's TTL so a follow-up turn
    does not re-read them. Deleting the chat left that behind, so the text of
    a deleted conversation stayed readable for up to an hour by anything
    holding its id.
    """

    def _cache(self):
        cache = get_runtime().cache
        if cache is None:
            pytest.skip("no Redis in this environment; nothing to retire")
        return cache

    def test_deleting_a_chat_drops_its_cached_history(self, client):
        import asyncio

        _, headers = _account(client)
        conversation_id = _conversation(client, headers)
        cache = self._cache()

        secret = f"SECRET-{uuid.uuid4().hex[:10]}"
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            cache.set_conversation_summary(
                conversation_id, {"recent_messages": [{"content": secret}]}
            )
        )

        def _summary():
            loop = asyncio.get_event_loop_policy().new_event_loop()
            try:
                return loop.run_until_complete(
                    cache.get_conversation_summary(conversation_id)
                )
            finally:
                loop.close()

        cached = _summary()
        assert cached and secret in json.dumps(cached), "nothing was cached to retire"

        assert client.delete(
            f"/v1/conversations/{conversation_id}", headers=headers
        ).status_code == 200

        assert _summary() is None, (
            "the deleted conversation's messages are still in the cache, "
            "readable for as long as the TTL lasts"
        )

    def test_a_finished_workflow_leaves_no_state_behind(self, client):
        """Terminal workflow state is not retained at all.

        The engine wrote `completed`, `failed` and `timeout` states carrying
        result content, traces, context snippets and vars - and nothing ever
        read them back. Retaining them meant a second copy of a chat's
        content with its own lifetime, so deletion would have needed
        machinery to enumerate and remove it. Not writing it is smaller and
        leaves nothing to enumerate.
        """
        import asyncio

        _, headers = _account(client)
        conversation_id = _conversation(client, headers)
        cache = self._cache()

        resp = client.post(
            "/v1/chat",
            headers=headers,
            json={
                "conversation_id": conversation_id,
                "message": {"content": "a question worth answering"},
            },
        )
        assert resp.status_code == 200, resp.text

        loop = asyncio.get_event_loop_policy().new_event_loop()
        try:
            keys = loop.run_until_complete(
                cache.client.keys(f"workflow:state:*{conversation_id}*")
            )
        finally:
            loop.close()
        assert keys == [], (
            f"the finished workflow left state behind: {keys}"
        )
