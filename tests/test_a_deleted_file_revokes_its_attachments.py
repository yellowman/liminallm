"""Deleting a file revokes the attachments backed by it.

An attachment record is not a label on a filename. It carries a checksum, and
`resolve_attachment` turns that checksum into an object in the write-once
generation store - a different place from `/users/{u}/files/{name}`, and one
the pathname's removal does not touch. So the record, not the file, is what
holds a conversation's capabilities open, and it holds all three at once:
inline injection reads the generation, interpreter staging resolves it, and
`file_search` retrieves chunks indexed under its generation key.

Measured on `main`, deleting a file left every one of them working. The bytes
left the disk, the request returned 200, and the same conversation still
inlined the deleted file's text into later turns, still resolved it for
staging, and still returned its contents from the workflow's own retrieval,
rendered under the original filename.

So the fix revokes the record first and prunes the index as a consequence.
The prune is context-local and computed from the records that *remain*,
because `generation_key` is a checksum and a format rather than a name: two
files holding identical bytes share one reading, and deleting that key
outright would take a surviving record's chunks with it.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

from liminallm.service import attachments as attachments_service
from liminallm.service.fs import namespace_key, path_lock
from liminallm.service.runtime import get_runtime

MARKER = f"ZQX{uuid.uuid4().hex[:10].upper()}"
#: Past INLINE_MAX_BYTES (12_000), so the classifier calls it searchable.
SEARCHABLE_BODY = (
    "Notes about the quarterly migration plan. " * 300
    + f" The agreed rollback codeword is {MARKER}. "
    + "Further notes about the migration plan. " * 300
).encode()
INLINE_MARKER = f"INL{uuid.uuid4().hex[:8].upper()}"
INLINE_BODY = f"A short attached note. The inline codeword is {INLINE_MARKER}.".encode()
QUESTION = "What is the agreed rollback codeword in the migration plan?"


@pytest.fixture
def auth(client):
    email = f"att_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/v1/auth/signup",
                       json={"email": email, "password": "TestPassword123!"})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    user_id = client.get("/v1/me", headers=headers).json()["data"]["id"]
    return {"headers": headers, "user_id": user_id}


def _conversation(client, auth, title="attached"):
    return client.post("/v1/conversations", headers=auth["headers"],
                       json={"title": title}).json()["data"]["id"]


def _attach(client, auth, conversation_id, name, body):
    resp = client.post(
        "/v1/files/upload", headers=auth["headers"],
        files={"file": (name, body, "text/plain")},
        data={"conversation_id": conversation_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp


def _records(conversation_id, user_id):
    convo = get_runtime().store.get_conversation(conversation_id, user_id=user_id)
    if convo is None:
        return []
    return [a for a in ((convo.meta or {}).get("attachments") or [])
            if isinstance(a, dict)]


def _names(conversation_id, user_id):
    return {str(r.get("name")) for r in _records(conversation_id, user_id)}


def _search(conversation_id, user_id):
    """The workflow's own retrieval, with the inputs it builds for itself."""
    text, snippets, _chunks, _hints = get_runtime().workflow._run_file_search(
        QUESTION, 8,
        conversation_id=conversation_id, context_id=None,
        user_id=user_id, tenant_id=None,
    )
    return str(text or "") + str(snippets or "")


def _chunks_in_context(conversation_id, user_id):
    store = get_runtime().store
    context = store.get_conversation_attachment_context(user_id, conversation_id)
    if context is None:
        return 0
    with store._connect() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM knowledge_chunk WHERE context_id = %s",
            (context.id,),
        ).fetchone()["n"]


def _files_dir(user_id):
    from pathlib import Path

    return (
        Path(get_runtime().settings.shared_fs_root) / "users" / user_id / "files"
    )


class TestTheThreeCapabilitiesAreRevoked:
    """One record, three ways to reach the bytes. All of them must close."""

    def test_a_searchable_attachment_is_no_longer_retrievable(self, client, auth):
        convo = _conversation(client, auth)
        name = f"plan{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, name, SEARCHABLE_BODY)

        assert MARKER in _search(convo, auth["user_id"]), (
            "the control failed: retrieval never found the marker, so a "
            "negative result after the delete would prove nothing"
        )
        assert _chunks_in_context(convo, auth["user_id"]) > 0

        assert client.delete(
            f"/v1/files/{name}", headers=auth["headers"]
        ).status_code == 200

        assert name not in _names(convo, auth["user_id"]), (
            "the record survived, so the generation is still authorized"
        )
        assert MARKER not in _search(convo, auth["user_id"])
        assert _chunks_in_context(convo, auth["user_id"]) == 0

    def test_an_inline_attachment_is_no_longer_injected(self, client, auth):
        """Inline needs no chunks at all - it reads the generation directly."""
        convo = _conversation(client, auth)
        name = f"small{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, name, INLINE_BODY)
        root = get_runtime().settings.shared_fs_root

        before = attachments_service.read_inline_contents(
            _records(convo, auth["user_id"]), fs_root=root, user_id=auth["user_id"]
        )
        assert any(INLINE_MARKER in item["content"] for item in before), (
            "the control failed: nothing was inlined before the delete"
        )

        assert client.delete(
            f"/v1/files/{name}", headers=auth["headers"]
        ).status_code == 200

        after = attachments_service.read_inline_contents(
            _records(convo, auth["user_id"]), fs_root=root, user_id=auth["user_id"]
        )
        assert not any(INLINE_MARKER in item["content"] for item in after), (
            "a deleted file's text is still being inlined into later turns"
        )

    def test_an_analyzable_attachment_no_longer_resolves(self, client, auth):
        """What staging and the interpreter are handed."""
        convo = _conversation(client, auth)
        name = f"rows{uuid.uuid4().hex[:6]}.csv"
        _attach(client, auth, convo, name, b"col_a,col_b\n1,2\n")
        root = get_runtime().settings.shared_fs_root

        before = attachments_service.resolved_sources(
            _records(convo, auth["user_id"]), fs_root=root, user_id=auth["user_id"]
        )
        assert name in {n for n, _ in before}, "the control failed"

        assert client.delete(
            f"/v1/files/{name}", headers=auth["headers"]
        ).status_code == 200

        after = attachments_service.resolved_sources(
            _records(convo, auth["user_id"]), fs_root=root, user_id=auth["user_id"]
        )
        assert name not in {n for n, _ in after}, (
            "the deleted file still resolves for staging"
        )


class TestASharedReadingIsNotCollateral:
    """`generation_key` is a checksum and a format, never a name."""

    def test_identical_bytes_under_another_name_survive(self, client, auth):
        """The case that makes a blind generation-key delete wrong.

        `foo` and `bar` holding the same bytes in the same format authorize
        one reading. Revoking `foo` must leave `bar`'s, because a record still
        names it.
        """
        convo = _conversation(client, auth)
        foo = f"foo{uuid.uuid4().hex[:6]}.txt"
        bar = f"bar{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, foo, SEARCHABLE_BODY)
        _attach(client, auth, convo, bar, SEARCHABLE_BODY)

        key_foo = attachments_service.generation_key(
            next(r["checksum"] for r in _records(convo, auth["user_id"])
                 if r["name"] == foo), foo
        )
        key_bar = attachments_service.generation_key(
            next(r["checksum"] for r in _records(convo, auth["user_id"])
                 if r["name"] == bar), bar
        )
        assert key_foo == key_bar, (
            "this test only means something while the two names share a "
            "reading; they no longer do, so it has stopped being the case "
            "it was written for"
        )

        assert client.delete(
            f"/v1/files/{foo}", headers=auth["headers"]
        ).status_code == 200

        names = _names(convo, auth["user_id"])
        assert foo not in names and bar in names
        assert MARKER in _search(convo, auth["user_id"]), (
            "deleting one name took the other name's still-authorized reading"
        )

    def test_another_conversation_keeps_its_own_reading(self, client, auth):
        """A second chat holding the same bytes under a surviving name."""
        keeper = _conversation(client, auth, "keeps it")
        doomed = _conversation(client, auth, "loses it")
        kept = f"kept{uuid.uuid4().hex[:6]}.txt"
        gone = f"gone{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, keeper, kept, SEARCHABLE_BODY)
        _attach(client, auth, doomed, gone, SEARCHABLE_BODY)

        assert MARKER in _search(keeper, auth["user_id"]), "the control failed"

        assert client.delete(
            f"/v1/files/{gone}", headers=auth["headers"]
        ).status_code == 200

        assert kept in _names(keeper, auth["user_id"])
        assert MARKER in _search(keeper, auth["user_id"]), (
            "another conversation's reading of the same bytes was revoked"
        )
        assert gone not in _names(doomed, auth["user_id"])


class TestRevocationReachesEveryConversation:
    def test_one_filename_attached_twice_is_revoked_from_both(self, client, auth):
        """The file is the user's; every chat that took it loses it."""
        first = _conversation(client, auth, "first")
        second = _conversation(client, auth, "second")
        name = f"shared{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, first, name, SEARCHABLE_BODY)
        _attach(client, auth, second, name, SEARCHABLE_BODY)

        assert name in _names(first, auth["user_id"])
        assert name in _names(second, auth["user_id"])

        assert client.delete(
            f"/v1/files/{name}", headers=auth["headers"]
        ).status_code == 200

        assert name not in _names(first, auth["user_id"])
        assert name not in _names(second, auth["user_id"]), (
            "only one conversation was revoked"
        )

    def test_a_sibling_name_is_not_caught_by_the_prefix(self, client, auth):
        """Path boundary, not raw string prefix.

        The pair matters. `planX.txt` against `planX2.txt` cannot tell the two
        rules apart, because the sibling does not begin with the whole deleted
        name - an earlier version of this test used exactly that pair and the
        raw-prefix mutant walked straight through it.

        `planX.txtmore.txt` does begin with it, and the uploader's sanitizer
        keeps word characters and dots, so it is a name a user can really
        create.
        """
        convo = _conversation(client, auth)
        target = f"plan{uuid.uuid4().hex[:5]}.txt"
        sibling = f"{target}more.txt"
        assert sibling.startswith(target), "the pair cannot discriminate"
        _attach(client, auth, convo, target, SEARCHABLE_BODY)
        _attach(client, auth, convo, sibling, INLINE_BODY)

        assert client.delete(
            f"/v1/files/{target}", headers=auth["headers"]
        ).status_code == 200

        names = _names(convo, auth["user_id"])
        assert target not in names
        assert sibling in names, (
            f"deleting {target!r} also revoked {sibling!r}, so the match is a "
            "raw prefix rather than a path boundary"
        )


class TestTheFailurePolicyIsUnchanged:
    def test_a_failed_revocation_leaves_the_file_in_place(self, client, auth):
        """Durable work first, pathname last - the route's existing rule.

        "nothing was deleted and the request failed" is a state the user can
        act on. "the file is gone and the request failed" is not.
        """
        convo = _conversation(client, auth)
        name = f"keep{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, name, SEARCHABLE_BODY)

        store = get_runtime().store
        real = store.retire_file_attachments

        def _boom(*args, **kwargs):
            raise RuntimeError("the store fell over")

        store.retire_file_attachments = _boom
        try:
            # The route does not catch this, so it becomes a 500 in
            # production and the test client re-raises it. Either way the
            # request failed; what this is about is what it left behind.
            with pytest.raises(RuntimeError):
                client.delete(f"/v1/files/{name}", headers=auth["headers"])
        finally:
            store.retire_file_attachments = real

        assert (_files_dir(auth["user_id"]) / name).exists(), (
            "the bytes were removed while the request reported failure"
        )
        assert name in _names(convo, auth["user_id"])

    def test_the_delete_still_waits_for_the_publication_lock(self, client, auth):
        """A lock, not an existence check.

        Holding the name's publication lock must make the delete conflict.
        Sequential ordering alone cannot tell those two apart.
        """
        convo = _conversation(client, auth)
        name = f"lock{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, name, INLINE_BODY)

        files_dir = _files_dir(auth["user_id"])
        key = namespace_key(files_dir, name)
        root = get_runtime().settings.shared_fs_root
        outcome: dict = {}

        def _delete():
            resp = client.delete(f"/v1/files/{name}", headers=auth["headers"])
            outcome["status"] = resp.status_code

        with path_lock(root, key):
            worker = threading.Thread(target=_delete, daemon=True)
            worker.start()
            time.sleep(2.0)
            outcome["still_running"] = worker.is_alive()
            worker.join(timeout=90)

        assert outcome.get("still_running"), (
            "the delete did not wait for the lock at all"
        )
        assert outcome.get("status") == 409, outcome
        assert (files_dir / name).exists(), "a conflicted delete removed the file"


class TestWhatDeliberatelySurvives:
    def test_the_generation_object_outlives_the_record(self, client, auth):
        """Reclamation is the sweep's job, not this route's.

        The object is write-once and shared, and an execution that resolved it
        before the delete may still be reading it. What must not survive is a
        way to reach it: with no record naming it, nothing authorizes it.
        """
        convo = _conversation(client, auth)
        name = f"gen{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, name, SEARCHABLE_BODY)
        root = get_runtime().settings.shared_fs_root
        record = next(r for r in _records(convo, auth["user_id"])
                      if r["name"] == name)
        generation = attachments_service.resolve_attachment(
            root, auth["user_id"], record
        )
        assert generation is not None and generation.exists()

        assert client.delete(
            f"/v1/files/{name}", headers=auth["headers"]
        ).status_code == 200

        assert os.path.exists(generation), (
            "the generation object was reclaimed inline; that belongs to the "
            "sweep, which is what gives a live execution its grace period"
        )
        assert _records(convo, auth["user_id"]) == [], (
            "nothing may still name it"
        )

    def test_deleting_the_conversation_still_clears_its_context(self, client, auth):
        """The neighbouring path this fix must not disturb."""
        convo = _conversation(client, auth)
        name = f"whole{uuid.uuid4().hex[:6]}.txt"
        _attach(client, auth, convo, name, SEARCHABLE_BODY)
        assert _chunks_in_context(convo, auth["user_id"]) > 0

        store = get_runtime().store
        context = store.get_conversation_attachment_context(
            auth["user_id"], convo
        )
        assert client.delete(
            f"/v1/conversations/{convo}", headers=auth["headers"]
        ).status_code == 200

        with store._connect() as conn:
            left = conn.execute(
                "SELECT count(*) AS n FROM knowledge_chunk WHERE context_id = %s",
                (context.id,),
            ).fetchone()["n"]
        assert left == 0
