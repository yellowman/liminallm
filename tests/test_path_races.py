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

import asyncio
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import httpx
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


def _zip_bytes(name: str, body: bytes) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, body)
    return buf.getvalue()


def _targz_bytes(name: str, body: bytes) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
    return buf.getvalue()


class TestOneDestinationHasOnePublisher:
    """Two archives, one destination.

    `bundle.zip` and `bundle.tar.gz` both extract to `bundle/`, so the
    conflict is not two requests for one archive — it is two *different*
    requests whose only shared state is where they land. The route checked
    `dest_path.exists()` in the API process and started the sandbox much
    later; inside, extraction does `mkdir(exist_ok=True)` and, on failure,
    `rmtree` of that same directory. So both requests could pass the check,
    both could write into one tree, and either could delete the other's.

    Locking has to be keyed on the resolved destination for the same reason:
    the archive names are deliberately different.
    """

    def _publish_archives(self, client, headers):
        for name, data in (
            ("bundle.zip", _zip_bytes("zip.txt", b"from the zip\n" * 40)),
            ("bundle.tar.gz", _targz_bytes("tar.txt", b"from the tarball\n" * 40)),
        ):
            resp = client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": _unique("k")},
                files={"file": (name, data, "application/zip")},
            )
            assert resp.status_code == 200, resp.text

    def _race_extract(self, client, runtime, monkeypatch, headers, names):
        """Both extractions, with the first paused inside the sandbox call."""
        from liminallm.api import routes

        reached = threading.Event()
        may_continue = threading.Event()
        armed = {"on": True}
        real_extract = routes.extract_archive_sandboxed

        def gated(*args, **kwargs):
            # The window: the destination check has passed and the tree has
            # not been written yet.
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)
            return real_extract(*args, **kwargs)

        monkeypatch.setattr(routes, "extract_archive_sandboxed", gated)
        results: dict = {}

        def run(index, name):
            results[index] = client.post(
                f"/v1/files/{name}/extract", headers=headers
            )

        threads = [
            threading.Thread(target=run, args=(i, n), daemon=True)
            for i, n in enumerate(names)
        ]
        threads[0].start()
        assert reached.wait(30), "the first extraction never started"
        for t in threads[1:]:
            t.start()
        time.sleep(1.0)
        may_continue.set()
        for t in threads:
            t.join(90)
        assert all(not t.is_alive() for t in threads), "an extraction hung"
        return results

    def test_only_one_archive_lands_in_the_shared_destination(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        self._publish_archives(client, headers)

        results = self._race_extract(
            client, runtime, monkeypatch, headers, ["bundle.zip", "bundle.tar.gz"]
        )
        codes = sorted(r.status_code for r in results.values())
        assert codes == [200, 409], [
            (r.status_code, r.text[:200]) for r in results.values()
        ]

        tree = sorted(p.name for p in (_files_dir(runtime, user_id) / "bundle").iterdir())
        assert tree in (["zip.txt"], ["tar.txt"]), (
            f"two archives were published into one destination: {tree}"
        )

    def test_a_failing_extraction_never_deletes_a_published_tree(
        self, client, monkeypatch
    ):
        """The failure path removes the destination directory. If the two
        requests share it, the loser's cleanup takes the winner's files."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        good = _zip_bytes("zip.txt", b"from the zip\n" * 40)
        # A tarball header with a truncated body: accepted as an upload,
        # refused by the extractor, and its refusal removes the destination.
        corrupt = _targz_bytes("tar.txt", b"x" * 40)[:60]
        for name, data in (("bundle.zip", good), ("bundle.tar.gz", corrupt)):
            resp = client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": _unique("k")},
                files={"file": (name, data, "application/zip")},
            )
            assert resp.status_code == 200, resp.text

        # The corrupt one goes first and pauses, so the good one publishes
        # inside its window and the failure's cleanup lands afterwards. The
        # other order proves nothing: the destination does not exist yet when
        # the failure tidies up.
        results = self._race_extract(
            client, runtime, monkeypatch, headers, ["bundle.tar.gz", "bundle.zip"]
        )
        assert results[1].status_code == 200, results[1].text

        dest = _files_dir(runtime, user_id) / "bundle"
        assert dest.is_dir(), "the failing extraction deleted the published tree"
        assert sorted(p.name for p in dest.iterdir()) == ["zip.txt"], (
            "the published tree did not survive intact"
        )


class TestTheParentOpensOnlyWhatTheChildProduced:
    """§21.2 confines model-written code. The process that publishes its
    output does not run under that confinement.

    A pathname is not a capability the child has to hold. `run_python` pivots
    the child's root away, so it cannot open `/etc/passwd` — but creating a
    *link* with that target costs it nothing and needs no target to exist on
    its side. The parent then resolves that link in its own namespace and
    copies what it finds into the caller's file area. The authorized object
    was a file the child produced; the object read was never that one.

    This is the confused-deputy form of a check/use gap: the check ("is this a
    regular file I may publish?") and the use ("read it") were two operations
    against a name rather than one against an object.
    """

    def _publish(self, workdir, dest, names, allowed):
        from liminallm.service.interpreter import publish_artifacts
        from liminallm.service.invocation import Invocation, current_invocation

        invocation = Invocation("publish", tool="code.python_v1")
        invocation.begin_attempt()
        try:
            with current_invocation(invocation):
                return publish_artifacts(
                    str(workdir),
                    str(dest),
                    [{"name": n} for n in names],
                    allowed_extensions=allowed,
                )
        finally:
            invocation.close()

    def test_a_link_to_a_host_file_publishes_nothing(self, client, tmp_path):
        import os

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        os.symlink("/etc/passwd", workdir / "result.txt")

        published = self._publish(workdir, dest, ["result.txt"], {".txt"})
        out = dest / "result.txt"
        assert published == [], published
        assert not out.exists(), (
            f"a host file was published into the user's area: "
            f"{out.read_bytes()[:60]!r}"
        )

    def test_a_link_to_another_users_file_publishes_nothing(self, client, tmp_path):
        import os

        runtime = get_runtime()
        victim, victim_headers = _account(client)
        secret = b"THE VICTIM'S PRIVATE PLAN\n"
        resp = client.post(
            "/v1/files/upload",
            headers={**victim_headers, "Idempotency-Key": _unique("k")},
            files={"file": ("private.md", secret, "text/markdown")},
        )
        assert resp.status_code == 200, resp.text
        victim_file = _files_dir(runtime, victim) / "private.md"
        assert victim_file.is_file()

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        os.symlink(str(victim_file), workdir / "stolen.md")

        published = self._publish(workdir, dest, ["stolen.md"], {".md"})
        out = dest / "stolen.md"
        assert published == [], published
        assert not out.exists() or secret not in out.read_bytes(), (
            "another user's file was published into this caller's area"
        )

    def test_a_file_the_code_actually_wrote_still_publishes(self, client, tmp_path):
        """The refusals above must be about the link, not about publishing."""
        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        (workdir / "result.txt").write_bytes(b"computed by the model's code\n")

        published = self._publish(workdir, dest, ["result.txt"], {".txt"})
        assert published == ["result.txt"], published
        assert (dest / "result.txt").read_bytes() == b"computed by the model's code\n"

    def test_the_publication_identity_does_not_read_through_a_link(self, tmp_path):
        """The identity hash opens the same child-named paths, so it is the
        same defect one function earlier — and a hash of `/etc/passwd` is a
        read of `/etc/passwd` whether or not anything is published."""
        import hashlib
        import os

        from liminallm.service.agent_tools import _durable_identity

        workdir = tmp_path / "w"
        workdir.mkdir()
        os.symlink("/etc/passwd", workdir / "result.txt")

        identity = _durable_identity(str(workdir), [{"name": "result.txt"}])
        host = hashlib.sha256(Path("/etc/passwd").read_bytes()).hexdigest()
        assert identity[0]["sha256"] != host, (
            "the identity hash is a hash of a host file the child named"
        )

    def test_a_fabricated_absolute_name_is_never_opened(self, client, tmp_path):
        """The whole sandbox result is attacker-controlled, names included.

        `os.path.join(workdir, "/etc/passwd")` is `/etc/passwd` — an absolute
        second argument discards the first. Publication rejects names holding
        a separator, but the identity hash runs before publication, so the
        read has already happened by the time that check is reached.
        """
        import hashlib

        from liminallm.service.agent_tools import _durable_identity
        from liminallm.service.interpreter import open_produced_file

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()

        assert open_produced_file(str(workdir), "/etc/passwd") is None
        identity = _durable_identity(str(workdir), [{"name": "/etc/passwd"}])
        host = hashlib.sha256(Path("/etc/passwd").read_bytes()).hexdigest()
        assert identity[0]["sha256"] != host, (
            "the parent hashed a host file the child merely named"
        )
        assert self._publish(workdir, dest, ["/etc/passwd"], {".txt"}) == []

    def test_the_identity_hash_stops_at_the_publishable_size(self, tmp_path):
        """A file too large to publish is not worth reading whole to decide
        it is the same one, and the child chooses how large it is."""
        import hashlib

        from liminallm.service.agent_tools import _durable_identity
        from liminallm.service.interpreter import MAX_ARTIFACT_BYTES

        workdir = tmp_path / "w"
        workdir.mkdir()
        body = b"a" * (MAX_ARTIFACT_BYTES + 4096)
        (workdir / "big.txt").write_bytes(body)

        identity = _durable_identity(str(workdir), [{"name": "big.txt"}])
        assert identity[0]["sha256"] == hashlib.sha256(
            body[:MAX_ARTIFACT_BYTES]
        ).hexdigest()
        assert identity[0]["sha256"] != hashlib.sha256(body).hexdigest()

    def test_a_traversal_name_is_never_opened(self, client, tmp_path):
        """Same defect spelled relatively."""
        import hashlib

        from liminallm.service.agent_tools import _durable_identity
        from liminallm.service.interpreter import open_produced_file

        workdir = tmp_path / "w" / "inner"
        workdir.mkdir(parents=True)
        outside = tmp_path / "w" / "outside.txt"
        outside.write_bytes(b"not the child's to name\n")
        name = "../outside.txt"

        assert open_produced_file(str(workdir), name) is None
        identity = _durable_identity(str(workdir), [{"name": name}])
        assert identity[0]["sha256"] != hashlib.sha256(
            outside.read_bytes()
        ).hexdigest(), "the parent hashed a file above the workdir"

    def test_real_sandboxed_code_cannot_name_a_file_the_parent_opens(
        self, client, tmp_path
    ):
        """The parent must distrust the child, not a helper's arguments.

        `execute_python` builds `created_files` from process-local state
        *after* running the code, so the code can change what that state
        reports. Measured, this returned
        `[{'name': '/etc/passwd', 'size': 1}]` through the real sandbox and
        the real wire.
        """
        import hashlib

        from liminallm.service.agent_tools import _durable_identity
        from liminallm.service.interpreter import run_python_sandboxed

        workdir = tmp_path / "wd"
        workdir.mkdir()
        result = run_python_sandboxed(
            "open('out.txt','w').write('x')\n"
            "import pathlib\n"
            "pathlib.PurePath.name = property(lambda self: '/etc/passwd')\n",
            workdir=str(workdir),
            confine_root=str(tmp_path / "root"),
            timeout=15,
        )
        created = result.get("created_files") or []
        assert created and created[0]["name"] == "/etc/passwd", (
            f"the child did not fabricate a name, so nothing is proved: {result}"
        )

        identity = _durable_identity(str(workdir), created)
        host = hashlib.sha256(Path("/etc/passwd").read_bytes()).hexdigest()
        assert identity[0]["sha256"] != host, (
            "the parent hashed the host file the sandboxed code named"
        )

    def test_a_fifo_named_as_an_artifact_is_refused_without_blocking(
        self, client, tmp_path
    ):
        """`O_NOFOLLOW` refuses a link and says nothing about a fifo, and
        opening one for reading waits for a writer. Measured, `os.open` on a
        fifo never returned — which parks a thread of the API process for as
        long as the child leaves it there. The test has its own clock because
        the failure mode is a hang, not a wrong answer."""
        import os
        import threading

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        os.mkfifo(workdir / "result.txt")

        done = threading.Event()
        outcome: dict = {}

        def publish():
            outcome["published"] = self._publish(
                workdir, dest, ["result.txt"], {".txt"}
            )
            done.set()

        threading.Thread(target=publish, daemon=True).start()
        assert done.wait(20), "publication blocked on a fifo the child created"
        assert outcome["published"] == [], outcome
        assert not (dest / "result.txt").exists()

    def test_the_destination_is_not_followed_either(self, client, tmp_path):
        """Defence in depth, stated as such: no writer under `files/` can
        plant a link today, which is why this plants one by hand. The write
        side deserves the same treatment as the read side because it is the
        same mistake — trusting a name to still mean the object it meant."""
        import os

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        (workdir / "result.txt").write_bytes(b"the model's output\n")
        elsewhere = tmp_path / "elsewhere.txt"
        elsewhere.write_bytes(b"someone else's file\n")
        os.symlink(str(elsewhere), dest / "result.txt")

        self._publish(workdir, dest, ["result.txt"], {".txt"})
        assert elsewhere.read_bytes() == b"someone else's file\n", (
            "the publication wrote through a link at the destination"
        )


class TestInterpreterPublicationDoesNotClobber:
    """Two producers write to the user's file area, and only one keeps books.

    `/files/upload` serialises a name, records its checksum in the manifest,
    and replaces that path's indexed generation. `publish_artifacts` writes
    into the same directory with `O_CREAT|O_TRUNC`, takes no lock, and updates
    neither. So an interpreter artifact could replace an uploaded file while
    the manifest and the index went on describing the file it replaced — and
    the next upload of those same bytes then saw a dedupe hit and returned
    success without restoring them.

    SPEC does not say whether model-produced artifacts may overwrite an
    existing user filename, so this does not decide that they may. It takes
    the narrow contract: publication never replaces a name that is already
    there. The artifact keeps a distinct name instead of being dropped, which
    matches how `notes/from-file` already disambiguates a title.
    """

    def test_an_artifact_never_replaces_an_uploaded_file(self, client, tmp_path):
        from liminallm.service.interpreter import publish_artifacts
        from liminallm.service.invocation import Invocation, current_invocation

        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        name = "report.txt"
        uploaded = b"the user's own uploaded report\n" * 20

        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, uploaded, "text/plain")},
            data={"context_id": context_id},
        )
        assert resp.status_code == 200, resp.text
        files_dir = _files_dir(runtime, user_id)
        before = _manifest(runtime, user_id).get(name, {}).get("checksum")
        assert before == hashlib.sha256(uploaded).hexdigest()

        workdir = tmp_path / "w"
        workdir.mkdir()
        (workdir / name).write_bytes(b"written by the model's code\n")
        invocation = Invocation("clobber", tool="code.python_v1")
        invocation.begin_attempt()
        try:
            with current_invocation(invocation):
                published = publish_artifacts(
                    str(workdir), str(files_dir), [{"name": name}], {".txt"}
                )
        finally:
            invocation.close()

        assert (files_dir / name).read_bytes() == uploaded, (
            "an interpreter artifact replaced the file the user uploaded"
        )
        assert _manifest(runtime, user_id).get(name, {}).get("checksum") == before
        # The artifact is kept, under a name that was free.
        assert published and published != [name], published
        assert (files_dir / published[0]).read_bytes() == b"written by the model's code\n"


class TestAnAuthorizedSourceBoundsItsDescendants:
    """§18: authority is the caller's own `/users/{user_id}` area, or an
    artifact covering a particular path. Membership somewhere under
    `shared_fs_root` is not authority.

    `add_context_source` authorizes the source correctly and then hands
    `ingest_path` the *shared root* as its allowed base, which throws that
    narrower authority away. `ingest_path` validates only the starting path,
    then globs descendants and calls `is_file()` on each — and `is_file()`
    follows a link. So a link inside an authorized directory reads whatever it
    points at, including another user's files and paths outside the shared
    root entirely.

    No supported writer can plant such a link under `files/` today: uploads
    write bytes, the archive extractor skips links, and interpreter
    publication refuses non-regular sources. The link here is planted
    directly, because the authority check is wrong whether or not the API
    currently offers a way to exploit it, and externally provisioned source
    trees are not bound by the API's write set.
    """

    def test_a_link_inside_the_source_does_not_reach_the_index(self, client):
        import os

        runtime = get_runtime()
        victim, victim_headers = _account(client)
        secret = b"THE VICTIM'S PRIVATE CORPUS ENTRY\n"
        resp = client.post(
            "/v1/files/upload",
            headers={**victim_headers, "Idempotency-Key": _unique("k")},
            files={"file": ("private.md", secret, "text/markdown")},
        )
        assert resp.status_code == 200, resp.text
        victim_file = _files_dir(runtime, victim) / "private.md"
        assert victim_file.is_file()

        user_id, headers = _account(client)
        corpus = _files_dir(runtime, user_id) / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "innocent.txt").write_bytes(b"material belonging to the caller\n" * 20)
        os.symlink(str(victim_file), corpus / "secret.txt")

        context_id = _context(client, headers)
        added = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": str(corpus), "recursive": True},
        )
        assert added.status_code in (200, 201), added.text

        texts = " ".join(
            c.content or ""
            for c in runtime.store.list_chunks(context_id, limit=500)
        )
        assert "belonging to the caller" in texts, (
            "the source was never indexed, so nothing is proved"
        )
        assert "PRIVATE CORPUS ENTRY" not in texts, (
            "a link inside the source indexed another user's file"
        )

    def test_a_link_pointing_outside_the_shared_root_is_refused_too(
        self, client, tmp_path
    ):
        """The allowed base has already been satisfied by the directory, so
        the target is not confined to `shared_fs_root` either."""
        import os

        runtime = get_runtime()
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"NOT UNDER THE SHARED ROOT AT ALL\n")

        user_id, headers = _account(client)
        corpus = _files_dir(runtime, user_id) / "corpus2"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "innocent.txt").write_bytes(b"material belonging to the caller\n" * 20)
        os.symlink(str(outside), corpus / "escape.txt")

        context_id = _context(client, headers)
        added = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": str(corpus), "recursive": True},
        )
        assert added.status_code in (200, 201), added.text

        texts = " ".join(
            c.content or ""
            for c in runtime.store.list_chunks(context_id, limit=500)
        )
        assert "belonging to the caller" in texts
        assert "NOT UNDER THE SHARED ROOT" not in texts, (
            "a link read a path outside the shared root entirely"
        )

    def test_a_hardlink_is_refused_though_no_path_reveals_it(self, client):
        """A hardlink *is* the file it points at. Nothing in the path says so,
        so neither the link test nor resolved containment can refuse it —
        measured, both accepted one. The archive extractor already skips
        hardlinked members for the same reason."""
        import os

        runtime = get_runtime()
        victim, victim_headers = _account(client)
        secret = b"THE VICTIM HARDLINKED ENTRY\n"
        resp = client.post(
            "/v1/files/upload",
            headers={**victim_headers, "Idempotency-Key": _unique("k")},
            files={"file": ("private.md", secret, "text/markdown")},
        )
        assert resp.status_code == 200, resp.text
        victim_file = _files_dir(runtime, victim) / "private.md"

        user_id, headers = _account(client)
        corpus = _files_dir(runtime, user_id) / "corpus3"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "innocent.txt").write_bytes(b"material belonging to the caller\n" * 20)
        os.link(str(victim_file), corpus / "hard.md")

        context_id = _context(client, headers)
        added = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": str(corpus), "recursive": True},
        )
        assert added.status_code in (200, 201), added.text
        texts = " ".join(
            c.content or ""
            for c in runtime.store.list_chunks(context_id, limit=500)
        )
        assert "belonging to the caller" in texts
        assert "VICTIM HARDLINKED ENTRY" not in texts, (
            "a hardlink inside the source indexed another user's file"
        )

    def test_each_descendant_test_refuses_something_the_others_accept(
        self, tmp_path
    ):
        """Kept separate because mutation showed the three overlap on the
        route-level cases, and code no test distinguishes is code nobody is
        checking."""
        import os

        from liminallm.service.rag import _within_source

        root = tmp_path / "src"
        root.mkdir()
        (root / "real.txt").write_bytes(b"inside")
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"outside")

        # Refused by the link test alone: it resolves inside the root, so
        # containment accepts it, and it is not hardlinked.
        os.symlink(str(root / "real.txt"), root / "inside-link.txt")
        assert not _within_source(root / "inside-link.txt", root.resolve())

        # Refused by containment alone: the final component is a real file and
        # is not hardlinked, but its parent is a link out of the root.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "file.txt").write_bytes(b"beyond")
        os.symlink(str(elsewhere), root / "subdir")
        assert not _within_source(root / "subdir" / "file.txt", root.resolve())

        # Refused by the hardlink test alone: no link in the path, and it
        # resolves inside the root.
        os.link(str(outside), root / "hard.txt")
        assert not _within_source(root / "hard.txt", root.resolve())

        # And an ordinary file is still accepted.
        assert _within_source(root / "real.txt", root.resolve())


class TestDeleteJoinsTheSameProtocol:
    """Upload and extraction each treat a name as one critical section.
    Deletion took no lock at all.

    Upload holds `path_lock(dest_path)` across disk, index, and manifest, and
    its own comment says the manifest is meaningful only under that lock.
    Extraction holds `path_lock(dest_path)` across the whole destination.
    `DELETE` resolved a path, checked it, removed it, and then did an
    unlocked read-modify-write of the shared manifest.
    """

    def _conversation_free_upload(self, client, headers, name, body, context_id):
        return client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, body, "text/markdown")},
            data={"context_id": context_id},
        )

    def test_a_delete_inside_an_upload_leaves_no_impossible_state(self, client):
        """Neither serialization order produces "file absent, manifest and
        index describe it, both requests succeeded"."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        name = "report.md"
        body = b"# report\nthe uploaded body\n" * 20

        reached = threading.Event()
        may_continue = threading.Event()
        real_ingest = runtime.rag.ingest_file
        armed = {"on": True}

        def ingest(ctx, path, **kwargs):
            out = real_ingest(ctx, path, **kwargs)
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)
            return out

        results: dict = {}
        upload = threading.Thread(
            target=lambda: results.update(
                up=self._conversation_free_upload(
                    client, headers, name, body, context_id
                )
            ),
            daemon=True,
        )
        deleter = threading.Thread(
            target=lambda: results.update(
                rm=client.delete(f"/v1/files/{name}", headers=headers)
            ),
            daemon=True,
        )
        runtime.rag.ingest_file = ingest
        try:
            upload.start()
            assert reached.wait(30), "the upload never reached ingestion"
            deleter.start()
            time.sleep(1.0)
            may_continue.set()
            upload.join(60)
            deleter.join(60)
        finally:
            may_continue.set()
            runtime.rag.ingest_file = real_ingest

        on_disk = (_files_dir(runtime, user_id) / name).exists()
        recorded = _manifest(runtime, user_id).get(name)
        chunks = " ".join(
            c.content or "" for c in runtime.store.list_chunks(context_id, limit=200)
        )
        indexed = "the uploaded body" in chunks
        # Either the upload won and everything describes it, or the delete won
        # and nothing does. A file that is absent while the manifest still
        # names it is neither.
        assert (on_disk, bool(recorded)) in ((True, True), (False, False)), (
            f"disk={on_disk} manifest={bool(recorded)} indexed={indexed}; "
            "no ordering of these two requests produces that"
        )
        # `indexed` is reported and not asserted on. Deletion has never
        # removed a path's chunks, in any ordering, so a deleted file leaving
        # its chunks behind is not this race — it is the recorded consistency
        # pass that `DELETE /files/{name}` still needs, and the deletion half
        # of `replace_chunks_for_path` is what it will use.

    def test_deleting_inside_a_tree_conflicts_with_the_tree_being_published(
        self, client, monkeypatch
    ):
        """The lock key is the namespace, not the target.

        Extraction publishes `bundle/` under a lock on `bundle`. Deleting
        `bundle/subdir` has to conflict with that, or the delete removes part
        of a tree the extraction goes on to finish and report as whole.
        """
        import io
        import zipfile

        from liminallm.api import routes

        runtime = get_runtime()
        user_id, headers = _account(client)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("subdir/one.txt", b"first member\n" * 20)
            zf.writestr("subdir/two.txt", b"second member\n" * 20)
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("bundle.zip", buf.getvalue(), "application/zip")},
        )
        assert resp.status_code == 200, resp.text

        reached = threading.Event()
        may_continue = threading.Event()
        armed = {"on": True}
        real_extract = routes.extract_archive_sandboxed

        def gated(*args, **kwargs):
            out = real_extract(*args, **kwargs)
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)
            return out

        monkeypatch.setattr(routes, "extract_archive_sandboxed", gated)
        results: dict = {}
        released = threading.Event()

        def delete_and_record():
            resp = client.delete("/v1/files/bundle/subdir", headers=headers)
            # Recorded at the moment the delete returns: whether the
            # extraction had let go of `bundle` by then.
            results["rm"] = resp
            results["waited_for_release"] = released.is_set()

        extractor = threading.Thread(
            target=lambda: results.update(
                ex=client.post("/v1/files/bundle.zip/extract", headers=headers)
            ),
            daemon=True,
        )
        deleter = threading.Thread(target=delete_and_record, daemon=True)
        extractor.start()
        assert reached.wait(30), "the extraction never ran"
        deleter.start()
        time.sleep(1.0)
        released.set()
        may_continue.set()
        extractor.join(60)
        deleter.join(60)

        assert results["ex"].status_code == 200, results["ex"].text
        # The outcome asserted on is the contention, not the final tree: a
        # delete that runs *after* a completed extraction is a correct
        # ordering and removes what it was asked to. What must not happen is
        # the delete finishing while the extraction still owns `bundle`.
        assert results["waited_for_release"], (
            "the delete completed while the extraction still held the "
            "destination lock, so it was not keyed on the same namespace"
        )

    def test_deleting_an_ancestor_conflicts_with_a_nested_extraction(
        self, client, monkeypatch
    ):
        """The trap in an exact-path delete lock.

        A nested archive is reachable: extraction leaves nested archives
        opaque, and the API lets the user extract one afterwards. So
        `outer/dir/inner.zip` publishes into `outer/dir/inner` while a
        recursive `DELETE outer` targets an ancestor. Locking the exact
        destination on one side and the exact target on the other gives two
        different keys, and the delete walks straight through — removing
        files the child already wrote, while later members recreate the
        ancestry with `mkdir(parents=True, exist_ok=True)`, so both requests
        report success over a partial tree.

        A `path_lock(str(file_path))` in the delete route closes the flat case
        and leaves this one alive, which is why it has its own test.
        """
        import io
        import zipfile

        from liminallm.api import routes

        runtime = get_runtime()
        user_id, headers = _account(client)

        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("one.txt", b"first member\n" * 20)
            zf.writestr("two.txt", b"second member\n" * 20)
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dir/inner.zip", inner.getvalue())
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("outer.zip", outer.getvalue(), "application/zip")},
        )
        assert resp.status_code == 200, resp.text
        first = client.post("/v1/files/outer.zip/extract", headers=headers)
        assert first.status_code == 200, first.text
        nested = _files_dir(runtime, user_id) / "outer" / "dir" / "inner.zip"
        assert nested.is_file(), "the nested archive was not published"

        reached = threading.Event()
        may_continue = threading.Event()
        armed = {"on": True}
        real_extract = routes.extract_archive_sandboxed

        def gated(*args, **kwargs):
            out = real_extract(*args, **kwargs)
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)
            return out

        monkeypatch.setattr(routes, "extract_archive_sandboxed", gated)
        results: dict = {}
        released = threading.Event()

        def delete_and_record():
            results["rm"] = client.delete("/v1/files/outer", headers=headers)
            results["waited_for_release"] = released.is_set()

        extractor = threading.Thread(
            target=lambda: results.update(
                ex=client.post(
                    "/v1/files/outer/dir/inner.zip/extract", headers=headers
                )
            ),
            daemon=True,
        )
        deleter = threading.Thread(target=delete_and_record, daemon=True)
        extractor.start()
        assert reached.wait(30), "the nested extraction never ran"
        deleter.start()
        time.sleep(1.0)
        released.set()
        may_continue.set()
        extractor.join(60)
        deleter.join(60)

        assert results["ex"].status_code == 200, results["ex"].text
        assert results["waited_for_release"], (
            "the delete of an ancestor completed while the nested extraction "
            "still owned its destination"
        )

    def test_a_delete_does_not_erase_another_names_manifest_entry(self, client):
        """The manifest is one object for the whole directory, so deletion's
        read-modify-write can drop an entry it never touched."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        keep, drop = "keep.md", "drop.md"
        for name in (keep, drop):
            resp = self._conversation_free_upload(
                client, headers, name, b"# first\nbody\n" * 20, context_id
            )
            assert resp.status_code == 200, resp.text

        reached = threading.Event()
        may_continue = threading.Event()
        armed = {"on": True}
        real_loads = json.loads

        def gated_loads(*args, **kwargs):
            out = real_loads(*args, **kwargs)
            # After the delete has read the manifest and before it writes.
            if armed["on"] and isinstance(out, dict) and keep in out:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)
            return out

        results: dict = {}
        deleter = threading.Thread(
            target=lambda: results.update(
                rm=client.delete(f"/v1/files/{drop}", headers=headers)
            ),
            daemon=True,
        )
        import liminallm.api.routes as routes

        monkey = routes.json.loads
        routes.json.loads = gated_loads
        try:
            deleter.start()
            if reached.wait(20):
                second = self._conversation_free_upload(
                    client, headers, keep, b"# second\nreplaced body\n" * 20, context_id
                )
                assert second.status_code == 200, second.text
                may_continue.set()
            deleter.join(60)
        finally:
            may_continue.set()
            routes.json.loads = monkey

        manifest = _manifest(runtime, user_id)
        on_disk = (_files_dir(runtime, user_id) / keep).read_bytes()
        assert keep in manifest, (
            f"the delete of {drop} erased {keep}'s manifest entry: "
            f"{sorted(manifest)}"
        )
        assert manifest[keep]["checksum"] == hashlib.sha256(on_disk).hexdigest(), (
            "the surviving manifest entry describes bytes that are not on disk"
        )


class TestExtractionOwnsItsDestinationUntilItIsIndexed:
    """Extracting into a context is one operation, not two.

    The destination lock was released as soon as the sandbox returned, and
    ingestion ran after it. `ingest_path` walks the tree it is given and
    catches per-file errors, returning the count it managed rather than
    failing, so a delete landing in that window removed the folder, the walk
    found nothing, and the request still reported 200 with every extracted
    file listed in its body.
    """

    def _bundle(self) -> bytes:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("one.txt", b"the first member says alpha\n" * 20)
            zf.writestr("two.txt", b"the second member says beta\n" * 20)
        return buf.getvalue()

    def test_a_delete_cannot_land_between_extraction_and_ingestion(self, client):
        runtime = get_runtime()
        _user_id, headers = _account(client)
        context_id = _context(client, headers)
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("bundle.zip", self._bundle(), "application/zip")},
        )
        assert resp.status_code == 200, resp.text

        reached = threading.Event()
        may_continue = threading.Event()
        released = threading.Event()
        armed = {"on": True}
        real_ingest = runtime.rag.ingest_path

        def gated(*args, **kwargs):
            # Entered, with nothing walked yet: the whole ingestion is still
            # ahead, which is the widest form of the window.
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(20)
            return real_ingest(*args, **kwargs)

        results: dict = {}

        def delete_and_record():
            results["rm"] = client.delete("/v1/files/bundle", headers=headers)
            # Recorded at the moment the delete returns: whether the
            # extraction had finished indexing by then.
            results["waited_for_release"] = released.is_set()

        extractor = threading.Thread(
            target=lambda: results.update(
                ex=client.post(
                    f"/v1/files/bundle.zip/extract?context_id={context_id}",
                    headers=headers,
                )
            ),
            daemon=True,
        )
        deleter = threading.Thread(target=delete_and_record, daemon=True)
        runtime.rag.ingest_path = gated
        try:
            extractor.start()
            assert reached.wait(30), "the extraction never reached ingestion"
            deleter.start()
            time.sleep(1.0)
            released.set()
            may_continue.set()
            extractor.join(60)
            deleter.join(60)
        finally:
            may_continue.set()
            runtime.rag.ingest_path = real_ingest

        assert results["ex"].status_code == 200, results["ex"].text
        assert results["waited_for_release"], (
            "the delete removed the destination while the extraction was "
            "still indexing it"
        )
        # A delete that runs after a finished extraction is a correct
        # ordering, so the tree is gone by now. What it must not have taken
        # with it is the indexing the extraction reported doing.
        indexed = " ".join(
            c.content or "" for c in runtime.store.list_chunks(context_id, limit=500)
        )
        for marker in ("the first member says alpha", "the second member says beta"):
            assert marker in indexed, (
                f"the extraction reported success without indexing {marker!r}; "
                f"body: {results['ex'].json()['data']}"
            )


async def _asgi_request(request, *, at_first_block=None):
    """Run one request through the app, suspended between two body blocks.

    starlette's `TestClient` runs the app to completion and only then hands
    back a response, so nothing it returns is still being produced and no
    interleaving can be placed inside one — measured, `iter_bytes()` yielded
    the whole 512 KiB in a single block. Driving the ASGI app directly gives
    back the real blocks, and `at_first_block` is awaited between two of
    them, so a second real request runs while the body is half sent.

    Only the body hook exists. A hook on `http.response.start` looks like it
    would name the moment after the headers and before the file is opened,
    but the app wraps five `BaseHTTPMiddleware` layers and each relays
    messages through a memory stream, so the inner response is already past
    that point by the time the outermost `send` is called. A window that
    narrow has to be held inside the route, not observed from outside it.
    """
    url = request.url
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": request.method,
        "scheme": "http",
        "path": url.path,
        "raw_path": url.raw_path.split(b"?")[0],
        "query_string": url.query,
        "root_path": "",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in request.headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    body = request.read()
    state = {"sent": False, "fired": False}
    out = {"status": None, "headers": [], "body": b""}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        # Never `http.disconnect`: a StreamingResponse races the body against
        # a disconnect watcher and cancels itself when one arrives, which
        # silently produced an empty 200.
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = message["headers"]
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            out["body"] += chunk
            if chunk and at_first_block is not None and not state["fired"]:
                state["fired"] = True
                await at_first_block()

    from liminallm import app as app_module

    await app_module.app(scope, receive, send)
    return out


class TestADownloadReadsOneGeneration:
    """`FileResponse` takes a pathname and opens it later.

    Between the route's existence check and the first byte read there is a
    window, and two ordinary requests reach into it. An upload of the same
    name rewrote the file in place, so a download already in progress read
    the head of one generation and the tail of another — measured, 512 KiB
    of bytes that were never a file. A delete in the same window left the
    response headers already sent and then failed to open anything.

    A signed URL names a path, not a generation, so it may resolve to either
    one. It may not resolve to half of one, and it may not resolve to a
    header with no body behind it.
    """

    NAME = "payload.txt"
    BODY_A = b"A" * (512 * 1024)
    BODY_B = b"B" * (512 * 1024)

    def _upload_request(self, headers, body):
        return httpx.Request(
            "POST",
            "http://testserver/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (self.NAME, body, "text/plain")},
        )

    def _upload(self, client, headers, body):
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (self.NAME, body, "text/plain")},
        )
        assert resp.status_code == 200, resp.text

    def _download_request(self, client, headers):
        """A minted signed URL, as an unsent request."""
        from urllib.parse import parse_qs, urlparse

        resp = client.get(f"/v1/files/{self.NAME}/url", headers=headers)
        assert resp.status_code == 200, resp.text
        query = urlparse(resp.json()["data"]["download_url"]).query
        return httpx.Request(
            "GET",
            "http://testserver/v1/files/download",
            headers=headers,
            params={k: v[0] for k, v in parse_qs(query).items()},
        )

    def test_an_overwrite_during_a_download_never_tears_the_body(self, client):
        user_id, headers = _account(client)
        self._upload(client, headers, self.BODY_A)
        download = self._download_request(client, headers)
        overwrite = self._upload_request(headers, self.BODY_B)
        landed: dict = {}

        async def second_generation():
            landed["upload"] = await _asgi_request(overwrite)

        out = asyncio.run(
            _asgi_request(download, at_first_block=second_generation)
        )

        assert landed["upload"]["status"] == 200, landed["upload"]["body"][:200]
        assert out["status"] == 200, out["status"]
        body = out["body"]
        assert body in (self.BODY_A, self.BODY_B), (
            f"the download returned {len(body)} of an expected "
            f"{len(self.BODY_A)} bytes, made of {sorted(set(body))}: neither "
            "generation, so the body was torn across both"
        )
        assert (_files_dir(get_runtime(), user_id) / self.NAME).read_bytes() == (
            self.BODY_B
        ), "the overwrite never landed, so nothing was raced"

    def test_a_delete_during_a_download_still_delivers_the_file(
        self, client, monkeypatch
    ):
        """The check and the open have to be one act.

        `FileResponse` is given a pathname: it stats the file in the API
        process, sends the headers, and opens the name later. Between the
        route's existence check and that open, a delete leaves the response
        already started with nothing behind it.

        The window is held inside the route, where it is a single point
        rather than something to aim at. The delete is a real request, run
        from another thread on its own event loop — the download's loop is
        blocked while it is paused, which is what a second worker or a second
        replica looks like from here.
        """
        from liminallm.api import routes

        runtime = get_runtime()
        user_id, headers = _account(client)
        self._upload(client, headers, self.BODY_A)
        download = self._download_request(client, headers)

        reached = threading.Event()
        done = threading.Event()
        armed = {"on": True}
        real_guess = routes.mimetypes.guess_type
        results: dict = {}

        def gated(name, *args, **kwargs):
            # The route has decided the file is there. Whatever it holds at
            # this point is what it will read the body from.
            if armed["on"] and str(name).endswith(self.NAME):
                armed["on"] = False
                reached.set()
                done.wait(30)
            return real_guess(name, *args, **kwargs)

        def delete_when_open():
            if not reached.wait(30):
                return
            results["rm"] = client.delete(f"/v1/files/{self.NAME}", headers=headers)
            done.set()

        monkeypatch.setattr(routes.mimetypes, "guess_type", gated)
        deleter = threading.Thread(target=delete_when_open, daemon=True)
        deleter.start()
        try:
            out = asyncio.run(_asgi_request(download))
        finally:
            done.set()
            deleter.join(60)

        assert not armed["on"], "the window never opened, so nothing was raced"
        assert results["rm"].status_code == 200, results["rm"].text
        assert not (_files_dir(runtime, user_id) / self.NAME).exists(), (
            "the delete returned 200 without removing the file"
        )
        assert out["status"] == 200, out["status"]
        assert out["body"] == self.BODY_A, (
            f"the download sent a 200 and then {len(out['body'])} of "
            f"{len(self.BODY_A)} bytes"
        )


class TestAListingToleratesADisappearance:
    """`GET /files` is observational, and a name it saw may already be gone.

    The route asked `is_file()` and then `stat()` — two questions about one
    name — and caught only `PermissionError`. A delete between them raised
    `FileNotFoundError` out of the route, so an unrelated user listing their
    files got an internal failure because someone deleted a file.
    """

    def _upload(self, client, headers, name):
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, b"# body\ncontent\n" * 20, "text/markdown")},
        )
        assert resp.status_code == 200, resp.text

    def _list_with_vanishing(self, client, monkeypatch, headers, doomed, when):
        """List while `doomed` disappears, before or after it is measured."""
        armed = {"on": True}
        real_stat = Path.stat

        def gated(path, *args, **kwargs):
            if not (armed["on"] and path == doomed):
                return real_stat(path, *args, **kwargs)
            armed["on"] = False
            if when == "before":
                doomed.unlink()
                return real_stat(path, *args, **kwargs)
            out = real_stat(path, *args, **kwargs)
            doomed.unlink()
            return out

        monkeypatch.setattr(Path, "stat", gated)
        resp = client.get("/v1/files", headers=headers)
        assert not armed["on"], "the listing never asked about the doomed name"
        return resp

    def test_a_delete_between_two_questions_about_one_name_is_not_an_error(
        self, client, monkeypatch
    ):
        """The name answers the first question and is gone by the second.

        This is the window the old code had. A route that asks once cannot be
        caught by it, which is the point: it stays as the guard against
        anything reintroducing a second question.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        for name in ("keep.md", "doomed.md"):
            self._upload(client, headers, name)

        resp = self._list_with_vanishing(
            client,
            monkeypatch,
            headers,
            _files_dir(runtime, user_id) / "doomed.md",
            when="after",
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"
        names = {f["name"] for f in resp.json()["data"]["files"]}
        assert "keep.md" in names, names
        # Reported from the one measurement the route took, which is the
        # whole point of taking one: the entry was there when it was asked
        # about, and a listing describes the moment it looked.
        assert "doomed.md" in names, names

    def test_a_name_that_vanishes_before_it_is_measured_is_simply_omitted(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        for name in ("keep.md", "doomed.md"):
            self._upload(client, headers, name)

        resp = self._list_with_vanishing(
            client,
            monkeypatch,
            headers,
            _files_dir(runtime, user_id) / "doomed.md",
            when="before",
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"
        data = resp.json()["data"]
        names = {f["name"] for f in data["files"]}
        assert "keep.md" in names, names
        assert "doomed.md" not in names, (
            "a name that no longer exists was reported with a size and dates "
            "that came from somewhere"
        )
        assert data["total"] == len(data["files"]), (
            "the count and the list disagree about how many files there are"
        )


class TestTheFileResponsesSayWhatTheSpecSays:
    """§13.3 states two response bodies this route did not produce."""

    def test_a_delete_reports_only_that_it_deleted(self, client):
        _user_id, headers = _account(client)
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("gone.md", b"# body\n" * 20, "text/markdown")},
        )
        assert resp.status_code == 200, resp.text

        resp = client.delete("/v1/files/gone.md", headers=headers)
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"] == {"deleted": True}, resp.json()["data"]

    def test_a_signed_url_says_when_it_expires(self, client):
        import datetime as dt

        _user_id, headers = _account(client)
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("report.md", b"# body\n" * 20, "text/markdown")},
        )
        assert resp.status_code == 200, resp.text

        before = dt.datetime.now(dt.timezone.utc)
        resp = client.get("/v1/files/report.md/url", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert "expires_at" in data, sorted(data)
        expires_at = dt.datetime.fromisoformat(data["expires_at"])
        assert expires_at.tzinfo is not None, data["expires_at"]
        # §18 fixes the window at ten minutes; the field has to describe the
        # same window `expires_in` reports, not a second, different one.
        window = (expires_at - before).total_seconds()
        assert 590 <= window <= 610, f"{window}s from {data['expires_at']}"
        assert data["expires_in"] == 600, data["expires_in"]
