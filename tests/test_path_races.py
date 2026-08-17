"""Check and use are two moments, and something can happen in between.

Every test here interleaves two requests that each succeed on their own, and
asks what the filesystem says afterwards. The interleavings are forced rather
than hoped for: a race that reproduces one run in fifty is a race that passes
CI, so each one gates a real request at the exact point the window opens.

The invariant is not "no two requests may touch one name" — that would be a
policy the SPEC does not state. It is that whatever survives describes *one*
generation: the bytes on disk, the chunks in the index, and the checksum in
the manifest are three records of the same upload, or they are lying about
each other. SPEC §22 puts `shared_fs_root` in common across replicas, so a
process-local lock cannot be the answer either.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('race')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["user_id"], {"Authorization": f"Bearer {data['access_token']}"}


def _context(client, headers) -> str:
    resp = client.post(
        "/v1/contexts",
        headers=headers,
        json={"name": _unique("ctx"), "description": "race"},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _files_dir(runtime, user_id: str) -> Path:
    return Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"


def _manifest(runtime, user_id: str) -> dict:
    path = _files_dir(runtime, user_id) / ".checksums.json"
    return json.loads(path.read_text()) if path.exists() else {}


class TestOneNameLeavesOneGeneration:
    """Two uploads of the same filename with different bytes.

    Different idempotency keys, so the request-idempotency system correctly
    treats them as two requests rather than a duplicate. Each phase succeeds;
    the damage is in the order they land:

        A: write bytes A
        B: write bytes B
        A: ingest the path  -> reads B
        A: write manifest   -> records checksum A

    Afterwards the disk holds B, the index holds B, and the manifest swears
    the file is A. Nothing errored, and the deduplication check on the next
    upload of that name now compares against a checksum no file ever had.
    """

    def test_disk_index_and_manifest_agree_after_a_concurrent_overwrite(
        self, client
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        name = "notes.md"
        alpha = b"# alpha\n" + b"the first upload's distinctive body\n" * 40
        beta = b"# beta\n" + b"the second upload's distinctive body\n" * 40

        reached_ingest = threading.Event()
        may_continue = threading.Event()
        real_ingest = runtime.rag.ingest_file
        gated = {"armed": True}

        def ingest(ctx, path, **kwargs):
            # The window the first upload sits in: its bytes are on disk and
            # it is about to reopen the path to read them back.
            if gated["armed"]:
                gated["armed"] = False
                reached_ingest.set()
                may_continue.wait(20)
            return real_ingest(ctx, path, **kwargs)

        def upload(body: bytes, key: str):
            return client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": key},
                files={"file": (name, body, "text/markdown")},
                data={"context_id": context_id},
            )

        # Both uploads run in threads and the window is opened for the second
        # one whether or not it can use it: serialised, it waits there and the
        # test still finishes; unserialised, it completes inside the window and
        # the assertions below describe what it left behind.
        results: dict = {}
        first = threading.Thread(
            target=lambda: results.update(a=upload(alpha, _unique("k"))), daemon=True
        )
        second = threading.Thread(
            target=lambda: results.update(b=upload(beta, _unique("k"))), daemon=True
        )
        runtime.rag.ingest_file = ingest
        try:
            first.start()
            assert reached_ingest.wait(30), "the first upload never reached ingestion"
            second.start()
            time.sleep(1.0)  # long enough for an unserialised second to finish
            may_continue.set()
            first.join(60)
            second.join(60)
        finally:
            may_continue.set()
            runtime.rag.ingest_file = real_ingest
        assert not first.is_alive() and not second.is_alive(), "an upload hung"

        assert results["a"].status_code == 200, results["a"].text
        assert results["b"].status_code == 200, results["b"].text

        on_disk = (_files_dir(runtime, user_id) / name).read_bytes()
        manifest = _manifest(runtime, user_id)
        recorded = (manifest.get(name) or {}).get("checksum")

        assert recorded == hashlib.sha256(on_disk).hexdigest(), (
            "the manifest describes a generation that is not on disk: "
            f"disk={hashlib.sha256(on_disk).hexdigest()[:12]} "
            f"manifest={str(recorded)[:12]}"
        )

        # And the bytes that survived are the bytes that got indexed — the
        # failure this catches is an upload indexing *someone else's* generation
        # under its own provenance. Read straight out of the store rather than
        # through `retrieve`: what is indexed is the question, and what a
        # similarity search returns is a ranking question that would answer a
        # different one.
        marker = "alpha" if on_disk == alpha else "beta"
        stale = "beta" if marker == "alpha" else "alpha"
        chunks = runtime.store.list_chunks(context_id, limit=200)
        texts = " ".join(c.content or "" for c in chunks)
        assert marker in texts, f"the surviving file was never indexed: {texts[:200]}"
        assert stale not in texts, (
            "the index still describes a generation the file no longer holds"
        )


class TestOneManifestHoldsEveryName:
    """The checksum manifest is one file for the whole directory.

    So a lock keyed on the file being uploaded does not protect it: two
    uploads of *different* names take different locks, run concurrently, and
    each does a read-modify-write of the same JSON. The later write is built
    from a copy taken before the earlier one landed, so the earlier entry
    disappears — and the next upload of that name finds no prior checksum,
    fails to deduplicate, and re-ingests a file that never changed.

    Found by mutation: moving the manifest read back outside the lock did not
    fail the same-name test, which is how it surfaced that the same-name test
    was not the one that could see this.
    """

    def test_a_concurrent_upload_of_another_name_keeps_both_entries(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)

        reached_ingest = threading.Event()
        may_continue = threading.Event()
        real_ingest = runtime.rag.ingest_file
        gated = {"armed": True}

        def ingest(ctx, path, **kwargs):
            # The first upload pauses with its bytes written and its manifest
            # entry not yet stored.
            if gated["armed"]:
                gated["armed"] = False
                reached_ingest.set()
                may_continue.wait(20)
            return real_ingest(ctx, path, **kwargs)

        def upload(name: str, body: bytes):
            return client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": _unique("k")},
                files={"file": (name, body, "text/markdown")},
                data={"context_id": context_id},
            )

        results: dict = {}
        first = threading.Thread(
            target=lambda: results.update(
                a=upload("first.md", b"# first\nbody of the first file\n" * 20)
            ),
            daemon=True,
        )
        second = threading.Thread(
            target=lambda: results.update(
                b=upload("second.md", b"# second\nbody of the second file\n" * 20)
            ),
            daemon=True,
        )
        runtime.rag.ingest_file = ingest
        try:
            first.start()
            assert reached_ingest.wait(30), "the first upload never reached ingestion"
            second.start()
            time.sleep(1.0)
            may_continue.set()
            first.join(60)
            second.join(60)
        finally:
            may_continue.set()
            runtime.rag.ingest_file = real_ingest

        assert results["a"].status_code == 200, results["a"].text
        assert results["b"].status_code == 200, results["b"].text
        manifest = _manifest(runtime, user_id)
        assert "first.md" in manifest and "second.md" in manifest, (
            f"an entry was lost to a concurrent upload of another name: "
            f"{sorted(manifest)}"
        )


class TestReingestingAPathReplacesItsChunks:
    """Found while building the race above, and it is not a race.

    Ingestion appended, so after two uploads of one name the index held both
    generations and a search could return, as the contents of `notes.md`, text
    that file had not held since the first upload. No interleaving reaches it —
    two sequential uploads are enough — which is why it sits here rather than
    among the races.

    `replace_chunks_for_path` is the narrow answer: within one context, a
    path's chunks are made to *be* the new generation rather than to join the
    old one, deleting and inserting in a single transaction so a reader never
    sees the path with no chunks at all. §2.5 dedupes by checksum and path and
    refreshes a changed path by ingesting it, which describes one generation.
    Its deletion half is what `DELETE /files/{name}` will want when that route
    gets its own consistency pass; that is not this tranche.
    """

    def _upload(self, client, headers, context_id, name, body):
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, body, "text/markdown")},
            data={"context_id": context_id},
        )
        assert resp.status_code == 200, resp.text
        return resp

    def test_an_empty_generation_replaces_the_last_one(self, client):
        """The `ingest_text` branch, driven at the service rather than the route.

        Measured: a whitespace-only upload does not reach it — `extract_text`
        strips and refuses, so the route arrives by the refusal path below.
        This branch is reachable through the ingestion API itself, and the two
        have to agree, or "no text this time" means one thing when the
        extractor says it and another when normalization does.
        """
        runtime = get_runtime()
        _user_id, headers = _account(client)
        context_id = _context(client, headers)
        path = "/srv/does-not-need-to-exist/notes.md"

        assert runtime.rag.ingest_text(
            context_id, "the first generation body\n" * 20, source_path=path
        ) > 0
        assert "first generation" in " ".join(
            c.content or "" for c in runtime.store.list_chunks(context_id, limit=200)
        ), "the first generation was never indexed; the test proves nothing"

        assert runtime.rag.ingest_text(context_id, "   \n\t\n  ", source_path=path) == 0
        texts = " ".join(
            c.content or "" for c in runtime.store.list_chunks(context_id, limit=200)
        )
        assert "first generation" not in texts, (
            "an empty generation left the previous one standing as current"
        )

    def test_a_generation_the_extractor_refuses_still_replaces_the_last_one(
        self, client
    ):
        """Same rule by the other route: `ingest_file` returns zero before it
        ever reaches `ingest_text`, so the replacement has to happen there too.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        name = "notes.md"

        self._upload(
            client, headers, context_id, name,
            b"# alpha\nthe first generation body\n" * 20,
        )
        assert "alpha" in " ".join(
            c.content or "" for c in runtime.store.list_chunks(context_id, limit=200)
        ), "the first generation was never indexed; the test proves nothing"

        # Accepted by the upload endpoint (extension and MIME are fine) and
        # refused by the shared extractor, which is what makes this the other
        # zero-chunk path rather than a rejected upload.
        self._upload(
            client, headers, context_id, name,
            b"\x00\x01\x02\x00\xff\xfe" * 400,
        )

        on_disk = (_files_dir(runtime, user_id) / name).read_bytes()
        assert on_disk.startswith(b"\x00\x01"), "the binary generation is not on disk"
        texts = " ".join(
            c.content or "" for c in runtime.store.list_chunks(context_id, limit=200)
        )
        assert "alpha" not in texts, (
            "an unreadable generation left the readable one indexed as current"
        )

    def test_the_index_holds_only_the_current_generation(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        name = "notes.md"

        for body in (
            b"# alpha\nthe first generation body\n" * 20,
            b"# beta\nthe second generation body\n" * 20,
        ):
            resp = client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": _unique("k")},
                files={"file": (name, body, "text/markdown")},
                data={"context_id": context_id},
            )
            assert resp.status_code == 200, resp.text

        on_disk = (_files_dir(runtime, user_id) / name).read_bytes()
        assert b"beta" in on_disk, "the second upload is not the one on disk"
        chunks = runtime.store.list_chunks(context_id, limit=200)
        texts = " ".join(c.content or "" for c in chunks)
        assert "alpha" not in texts, (
            "the index still describes a generation the file no longer has"
        )


class TestAConversationDescribesTheFileThatSurvived:
    """Attachment metadata is model-visible state, and it was outside the lock.

    §19.5 makes inline/searchable/analyzable part of how a conversation uses a
    file, and the classification comes from the size. So the record is not
    bookkeeping that can settle in any order: a 5KB `.md` is `inline` and the
    same name at 20KB is `searchable`, and `read_inline_contents` opens the
    file on disk using whichever classification the conversation ended up
    holding.
    """

    def _conversation(self, client, headers) -> str:
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("conv")}
        )
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["data"]["id"]

    def _attachments(self, runtime, conversation_id, user_id) -> list:
        conv = runtime.store.get_conversation(conversation_id, user_id=user_id)
        return list((conv.meta or {}).get("attachments") or [])

    def _race(self, runtime, monkeypatch, jobs, *, gate: str):
        """Run `jobs` with the first one paused at `gate`.

        Two windows, so two gate points. `record` pauses before the record is
        written at all, which is the gap between releasing the publication
        lock and describing what was published. `merge` pauses *inside*
        `record_attachment`, after it has read the attachment list and before
        it writes the edited copy back — a different bug in the same line of
        code, and one no file lock can reach because the state is in Postgres.

        Neither is reachable by luck: the first request wins the sprint from
        one to the next almost every time, which is what makes these races
        that pass CI rather than races that do not exist.
        """
        from liminallm.api import routes

        reached = threading.Event()
        may_continue = threading.Event()
        armed = {"on": True}

        def pause():
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)

        if gate == "record":
            real_record = routes.record_attachment

            def wrapper(*args, **kwargs):
                pause()
                return real_record(*args, **kwargs)

            monkeypatch.setattr(routes, "record_attachment", wrapper)
        else:
            real_merge = runtime.store.merge_conversation_meta

            def wrapper(*args, **kwargs):  # noqa: F811 - one name, two gates
                pause()
                return real_merge(*args, **kwargs)

            monkeypatch.setattr(
                runtime.store, "merge_conversation_meta", wrapper, raising=False
            )

        results: dict = {}
        threads = [
            threading.Thread(
                target=lambda i=i, j=j: results.update({i: j()}), daemon=True
            )
            for i, j in enumerate(jobs)
        ]
        threads[0].start()
        assert reached.wait(30), f"the first request never reached the {gate} gate"
        for t in threads[1:]:
            t.start()
        time.sleep(1.0)
        may_continue.set()
        for t in threads:
            t.join(60)
        assert all(not t.is_alive() for t in threads), "a request hung"
        return results

    def test_the_recorded_size_is_the_size_of_the_file_on_disk(
        self, client, monkeypatch
    ):
        """Small then large, concurrently. Whichever generation survives on
        disk, the conversation must describe *that* one — a record written
        after the lock was released can be the loser's."""
        from liminallm.service.attachments import INLINE_MAX_BYTES

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        name = "notes.md"
        small = b"s" * (INLINE_MAX_BYTES // 2)
        large = b"l" * (INLINE_MAX_BYTES * 2)

        def upload(body):
            return lambda: client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": _unique("k")},
                files={"file": (name, body, "text/markdown")},
                data={"conversation_id": conversation_id},
            )

        # Small first: it pauses at its record, the large one then
        # publishes fully, and the small one's record lands last.
        results = self._race(
            runtime, monkeypatch, [upload(small), upload(large)], gate="record"
        )
        for resp in results.values():
            assert resp.status_code == 200, resp.text

        on_disk = (_files_dir(runtime, user_id) / name).read_bytes()
        records = [
            a for a in self._attachments(runtime, conversation_id, user_id)
            if a.get("name") == name
        ]
        assert len(records) == 1, records
        record = records[0]
        assert record["size"] == len(on_disk), (
            f"the conversation says {record['size']} bytes; disk holds "
            f"{len(on_disk)}"
        )
        assert record["inline"] is (len(on_disk) <= INLINE_MAX_BYTES), (
            "the classification describes the generation that lost"
        )

    def test_concurrent_attachment_records_all_survive(self, client):
        """The list is one JSON value holding every attachment.

        Editing it means read, change one entry, write the whole thing back —
        and two writers that both read before either wrote each store their
        own copy, so one addition disappears. Driven straight at
        `record_attachment` with a barrier rather than through the route,
        because after the fix the read and the write are one transaction and
        there is no longer a seam between them to pause at; what is left to
        test is the property, under real contention.
        """
        from liminallm.service.attachments import record_attachment

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)

        names = [f"file{i}.md" for i in range(8)]
        start = threading.Barrier(len(names))
        errors: list = []

        def add(name: str):
            try:
                start.wait(30)
                record_attachment(
                    runtime.store,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    name=name,
                    size=100,
                    capabilities={"inline": True, "searchable": False,
                                  "analyzable": False},
                    chunk_count=None,
                )
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=add, args=(n,), daemon=True) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors, errors
        assert all(not t.is_alive() for t in threads), "a writer hung"

        recorded = {
            a.get("name")
            for a in self._attachments(runtime, conversation_id, user_id)
        }
        assert recorded == set(names), (
            f"records were lost to concurrent writers: missing "
            f"{sorted(set(names) - recorded)}"
        )
