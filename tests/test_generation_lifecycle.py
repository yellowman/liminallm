"""A chunk whose `fs_path` is P claims to be the contents of P.

That claim has a lifetime. It is true from the moment the bytes at P are
indexed until the moment P holds different bytes, or holds nothing at all.
Nothing in the schema records which generation of P a chunk came from, so
the claim is about P *now* — and the index has no way to say "this is a
snapshot" instead.

Everything here follows from that one reading. A path that is deleted must
stop being described. A path that is replaced must not leave an older
generation standing somewhere else. And a record that says a generation
landed must not survive the write that was supposed to land it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
import zipfile
import zlib
from pathlib import Path

import pytest

from liminallm.service.runtime import get_runtime


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _account(client):
    email = f"{_unique('gen')}@example.com"
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
        json={"name": _unique("ctx"), "description": "generation"},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["data"]["id"]


def _files_dir(runtime, user_id: str) -> Path:
    return Path(runtime.settings.shared_fs_root) / "users" / user_id / "files"


def _manifest(runtime, user_id: str) -> dict:
    path = _files_dir(runtime, user_id) / ".checksums.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _upload(client, headers, name, body, context_id=None, media="text/markdown"):
    data = {"context_id": context_id} if context_id else None
    return client.post(
        "/v1/files/upload",
        headers={**headers, "Idempotency-Key": _unique("k")},
        files={"file": (name, body, media)},
        data=data,
    )


def _chunks(runtime, context_id):
    return runtime.store.list_chunks(context_id, limit=500)


def _described_paths(runtime, context_id) -> set[str]:
    return {c.fs_path for c in _chunks(runtime, context_id) if c.fs_path}


def _text(runtime, context_id) -> str:
    return " ".join(c.content or "" for c in _chunks(runtime, context_id))


class TestDeletingAFileStopsItBeingDescribed:
    """The recorded consistency pass, and the reason it is not cosmetic.

    Deletion removed the bytes, the manifest entry and nothing else. The
    chunks stayed, so a grounded conversation still answered with the
    contents of a file the user had deleted — the deletion the user asked
    for did not happen, it only became invisible in the file listing.
    """

    def test_a_deleted_file_is_described_by_no_context(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        first, second = _context(client, headers), _context(client, headers)
        secret = b"# evidence\nTHE DELETED PARAGRAPH IS STILL HERE\n" * 12

        # The same bytes into two contexts. The second upload dedupes on
        # checksum and still ingests, because the context is new to the
        # manifest entry — so one pathname is described by two contexts,
        # which is what makes a single-context cleanup wrong.
        for context_id in (first, second):
            resp = _upload(client, headers, "evidence.md", secret, context_id)
            assert resp.status_code == 200, resp.text
        target = _files_dir(runtime, user_id) / "evidence.md"
        for context_id in (first, second):
            assert "THE DELETED PARAGRAPH IS STILL HERE" in _text(runtime, context_id)

        resp = client.delete("/v1/files/evidence.md", headers=headers)
        assert resp.status_code == 200, resp.text

        assert not target.exists(), "the file survived its own deletion"
        assert "evidence.md" not in _manifest(runtime, user_id)
        for name, context_id in (("first", first), ("second", second)):
            assert str(target) not in _described_paths(runtime, context_id), (
                f"the {name} context still describes the deleted path"
            )
            assert "THE DELETED PARAGRAPH IS STILL HERE" not in _text(
                runtime, context_id
            ), f"the deleted contents are still retrievable from the {name} context"

    def test_deleting_a_tree_stops_every_path_under_it_being_described(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.txt", b"THE TOP MEMBER OF THE TREE\n" * 12)
            zf.writestr("sub/b.md", b"THE NESTED MEMBER OF THE TREE\n" * 12)
        resp = _upload(
            client, headers, "bundle.zip", buf.getvalue(), media="application/zip"
        )
        assert resp.status_code == 200, resp.text
        resp = client.post(
            f"/v1/files/bundle.zip/extract?context_id={context_id}", headers=headers
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["chunk_count"], resp.json()["data"]

        tree = _files_dir(runtime, user_id) / "bundle"
        for marker in ("THE TOP MEMBER OF THE TREE", "THE NESTED MEMBER OF THE TREE"):
            assert marker in _text(runtime, context_id), marker

        resp = client.delete("/v1/files/bundle", headers=headers)
        assert resp.status_code == 200, resp.text

        assert not tree.exists(), "the tree survived its own deletion"
        described = _described_paths(runtime, context_id)
        assert not [p for p in described if p.startswith(str(tree))], (
            f"paths under the deleted tree are still described: {sorted(described)}"
        )
        for marker in ("THE TOP MEMBER OF THE TREE", "THE NESTED MEMBER OF THE TREE"):
            assert marker not in _text(runtime, context_id), (
                f"the deleted tree's contents are still retrievable: {marker}"
            )

    def test_a_sibling_whose_name_shares_the_prefix_is_left_alone(self, client):
        """`bundle` and `bundle2` are different names.

        A prefix match written as "starts with the deleted path" takes both.
        The boundary is the separator, and this is the test that says so.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.txt", b"THE MEMBER OF THE DELETED TREE\n" * 12)
        assert (
            _upload(
                client, headers, "bundle.zip", buf.getvalue(), media="application/zip"
            ).status_code
            == 200
        )
        assert (
            _upload(
                client, headers, "bundle2.md", b"THE SIBLING SURVIVES\n" * 12, context_id
            ).status_code
            == 200
        )
        resp = client.post(
            f"/v1/files/bundle.zip/extract?context_id={context_id}", headers=headers
        )
        assert resp.status_code == 200, resp.text

        resp = client.delete("/v1/files/bundle", headers=headers)
        assert resp.status_code == 200, resp.text

        assert (_files_dir(runtime, user_id) / "bundle2.md").exists()
        assert "THE MEMBER OF THE DELETED TREE" not in _text(runtime, context_id)
        assert "THE SIBLING SURVIVES" in _text(runtime, context_id), (
            "deleting `bundle` took `bundle2.md` with it"
        )

    def test_another_users_identical_filename_is_untouched(self, client):
        """Two accounts, one filename, and one of them deletes it.

        This passes because every account's files live under its own
        directory, so the two uploads have different absolute paths and the
        owner predicate never decides anything here. The predicate is tested
        where it does decide, in the test below.
        """
        runtime = get_runtime()
        _mine, my_headers = _account(client)
        _theirs, their_headers = _account(client)
        my_context = _context(client, my_headers)
        their_context = _context(client, their_headers)
        body = b"# shared name\nEACH ACCOUNT HAS ITS OWN COPY\n" * 12
        for headers, context_id in (
            (my_headers, my_context),
            (their_headers, their_context),
        ):
            assert (
                _upload(client, headers, "report.md", body, context_id).status_code
                == 200
            )

        resp = client.delete("/v1/files/report.md", headers=my_headers)
        assert resp.status_code == 200, resp.text

        assert "EACH ACCOUNT HAS ITS OWN COPY" not in _text(runtime, my_context)
        assert "EACH ACCOUNT HAS ITS OWN COPY" in _text(runtime, their_context), (
            "one account's delete removed another account's chunks"
        )

    def test_a_failed_index_cleanup_leaves_everything_in_place(self, client, monkeypatch):
        """No transaction spans Postgres and the filesystem, so the order
        inside the lock decides which half can be left behind.

        The bookkeeping runs first and the pathname goes last, which makes a
        failure look like "nothing was deleted" rather than like "the file is
        gone and its contents are still retrievable".
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        body = b"# receipt\nTHE CONTENTS OF THE RECEIPT\n" * 12
        assert _upload(client, headers, "receipt.md", body, context_id).status_code == 200

        def unreachable(*args, **kwargs):
            raise OSError("the index is unreachable")

        monkeypatch.setattr(runtime.store, "delete_chunks_under_path", unreachable)
        with pytest.raises(OSError):
            client.delete("/v1/files/receipt.md", headers=headers)

        assert (_files_dir(runtime, user_id) / "receipt.md").exists(), (
            "the file was removed although the request failed, so its "
            "contents stayed retrievable while the user was told the "
            "deletion did not happen"
        )
        assert "THE CONTENTS OF THE RECEIPT" in _text(runtime, context_id)
        assert "receipt.md" in _manifest(runtime, user_id)

    def test_a_failed_manifest_write_leaves_the_file_in_place(self, client, monkeypatch):
        """The manifest is one of the three records, not a note about them."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        body = b"# invoice\nTHE CONTENTS OF THE INVOICE\n" * 12
        assert _upload(client, headers, "invoice.md", body, context_id).status_code == 200

        manifest_path = _files_dir(runtime, user_id) / ".checksums.json"
        real_write = Path.write_text

        def gated(path, *args, **kwargs):
            if path == manifest_path:
                raise OSError("the manifest is unwritable")
            return real_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", gated)
        with pytest.raises(OSError):
            client.delete("/v1/files/invoice.md", headers=headers)
        monkeypatch.undo()

        assert (_files_dir(runtime, user_id) / "invoice.md").exists(), (
            "the file was removed although its manifest entry could not be"
        )
        assert "invoice.md" in _manifest(runtime, user_id)

    def test_the_cleanup_never_reaches_another_owners_context(self, client):
        """The owner predicate, tested where it decides something.

        Driven against the store rather than the routes, because the routes
        cannot produce this state: `safe_join` keeps every account inside its
        own directory, so no two accounts' uploads ever share an absolute
        path. A context source pointing into a shared corpus is the shape
        that would, and the predicate is what stops one account's delete
        reaching the other's rows when it does.
        """
        from liminallm.storage.models import KnowledgeChunk

        runtime = get_runtime()
        mine, my_headers = _account(client)
        _theirs, their_headers = _account(client)
        my_context = _context(client, my_headers)
        their_context = _context(client, their_headers)
        body = b"# corpus\nONE PATH DESCRIBED BY TWO ACCOUNTS\n" * 12
        assert _upload(client, my_headers, "corpus.md", body, my_context).status_code == 200
        shared_path = str(_files_dir(runtime, mine) / "corpus.md")

        # The other account's context, describing the same absolute path.
        runtime.store.add_chunks(
            their_context,
            [
                KnowledgeChunk(
                    context_id=their_context,
                    fs_path=shared_path,
                    content="ONE PATH DESCRIBED BY TWO ACCOUNTS",
                    embedding=runtime.rag.embed("ONE PATH DESCRIBED BY TWO ACCOUNTS"),
                    chunk_index=0,
                )
            ],
        )
        assert shared_path in _described_paths(runtime, their_context)

        removed = runtime.store.delete_chunks_under_path(mine, shared_path)

        assert removed, "the owner's own rows were not removed"
        assert shared_path not in _described_paths(runtime, my_context)
        assert shared_path in _described_paths(runtime, their_context), (
            "the cleanup removed a row from a context the caller does not own"
        )


class TestTheManifestIsARecordNotANote:
    """The manifest is one of the three records that describe a generation.

    Upload caught every exception around its manifest write, logged a
    warning and returned 200. That reopens the false-dedupe history 2E.1
    closed, from the other end: the manifest keeps naming the previous
    checksum and the previous context set, so the *next* upload of those
    bytes matches a record no file has, skips the write, skips the ingest,
    and reports success while the disk keeps the bytes it already had.
    """

    def _arm_manifest_failure(self, monkeypatch, manifest_path):
        """Fail the next write of the manifest, and only that one."""
        armed = {"on": True}
        real_write = Path.write_text

        def gated(path, *args, **kwargs):
            if armed["on"] and path == manifest_path:
                armed["on"] = False
                raise OSError("no space left on device")
            return real_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", gated)
        return armed

    def test_an_unrecorded_generation_is_not_a_successful_upload(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE FIRST GENERATION\n" * 12
        second = b"# report\nTHE SECOND GENERATION\n" * 12
        target = _files_dir(runtime, user_id) / "report.md"
        manifest_path = _files_dir(runtime, user_id) / ".checksums.json"

        assert _upload(client, headers, "report.md", first, context_id).status_code == 200
        assert _manifest(runtime, user_id)["report.md"]["checksum"] == (
            hashlib.sha256(first).hexdigest()
        )

        armed = self._arm_manifest_failure(monkeypatch, manifest_path)
        key = _unique("retry")
        with pytest.raises(OSError):
            client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": key},
                files={"file": ("report.md", second, "text/markdown")},
                data={"context_id": context_id},
            )
        assert not armed["on"], "the manifest write never happened"

        # The retry repairs it. The same key, because the request failed and
        # a failed slot is reclaimable — this is the client doing what a 5xx
        # tells it to do.
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": key},
            files={"file": ("report.md", second, "text/markdown")},
            data={"context_id": context_id},
        )
        assert resp.status_code == 200, resp.text
        assert target.read_bytes() == second
        assert _manifest(runtime, user_id)["report.md"]["checksum"] == (
            hashlib.sha256(second).hexdigest()
        ), "the retry did not repair the record"
        assert "THE SECOND GENERATION" in _text(runtime, context_id)

        # The payoff. Re-uploading the first generation must actually write
        # it. Against a manifest still describing the first checksum, this
        # is a dedupe hit: no write, no ingest, and a 200 over bytes that
        # never changed.
        resp = _upload(client, headers, "report.md", first, context_id)
        assert resp.status_code == 200, resp.text
        assert target.read_bytes() == first, (
            "the upload reported success and left the previous generation on "
            "disk, because the manifest was describing bytes no file had"
        )
        assert "THE FIRST GENERATION" in _text(runtime, context_id)
        assert "THE SECOND GENERATION" not in _text(runtime, context_id)

    def test_an_unreadable_manifest_does_not_erase_the_names_it_holds(
        self, client, monkeypatch
    ):
        """A read failure says nothing about what the manifest holds.

        Read failures were swallowed and the manifest treated as empty. The
        write that follows rebuilds the whole object from that empty copy,
        so one transient read error dropped every other name's entry.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        for name in ("kept.md", "other.md"):
            assert (
                _upload(client, headers, name, b"# body\nCONTENT\n" * 12, context_id)
            ).status_code == 200
        manifest_path = _files_dir(runtime, user_id) / ".checksums.json"
        before = _manifest(runtime, user_id)
        assert set(before) == {"kept.md", "other.md"}

        armed = {"on": True}
        real_read = Path.read_text

        def gated(path, *args, **kwargs):
            if armed["on"] and path == manifest_path:
                armed["on"] = False
                raise OSError("the manifest is unreadable")
            return real_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", gated)
        with pytest.raises(OSError):
            _upload(client, headers, "fresh.md", b"# fresh\nNEW\n" * 12, context_id)
        monkeypatch.undo()

        assert set(_manifest(runtime, user_id)) == {"kept.md", "other.md"}, (
            "a failed read of the manifest erased the entries it could not read"
        )

    def test_a_corrupt_manifest_is_rebuilt_rather_than_fatal(self, client):
        """Corruption is different: it is what the file holds, not a failure
        to find out. Reading it as empty is the recovery."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        assert (
            _upload(client, headers, "first.md", b"# body\nCONTENT\n" * 12, context_id)
        ).status_code == 200
        manifest_path = _files_dir(runtime, user_id) / ".checksums.json"
        manifest_path.write_text("{ this is not json")

        resp = _upload(client, headers, "second.md", b"# body\nMORE\n" * 12, context_id)
        assert resp.status_code == 200, resp.text
        assert "second.md" in _manifest(runtime, user_id)


class TestAContextSourceCannotCommitAStaleGeneration:
    """`POST /contexts/{id}/sources` reads a path and commits what it read.

    Between those two moments the path can change, and the route took part
    in none of the serialization the other writers of that pathname now use.
    Upload holds the namespace lock across disk, index and manifest;
    extraction holds it across a whole destination; deletion holds it while
    it removes all three. Source ingestion held nothing, so its commit could
    land after a newer generation had already replaced every one of them.

    Both requests succeed, and no serial ordering produces the result:
    source-then-upload should end with the upload's chunks, and
    upload-then-source should have read the upload's bytes.
    """

    def test_a_newer_upload_is_not_replaced_by_what_the_source_read(self, client):
        import threading

        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE GENERATION THE SOURCE READ\n" * 12
        second = b"# report\nTHE GENERATION THE UPLOAD WROTE\n" * 12
        assert _upload(client, headers, "report.md", first).status_code == 200
        target = _files_dir(runtime, user_id) / "report.md"

        reached = threading.Event()
        may_continue = threading.Event()
        released = threading.Event()
        armed = {"on": True}
        real_commit = runtime.rag._commit_generation

        def gated(*args, **kwargs):
            # Entered with the bytes already read and chunked, and nothing
            # written yet: the widest form of the window.
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(30)
            return real_commit(*args, **kwargs)

        results: dict = {}

        def add_source():
            results["src"] = client.post(
                f"/v1/contexts/{context_id}/sources",
                headers=headers,
                json={"fs_path": str(target), "recursive": False},
            )

        def upload_and_record():
            results["up"] = _upload(client, headers, "report.md", second, context_id)
            # Recorded at the moment the upload returns: whether the source
            # request had let go of the name by then.
            results["waited_for_release"] = released.is_set()

        runtime.rag._commit_generation = gated
        source_thread = threading.Thread(target=add_source, daemon=True)
        upload_thread = threading.Thread(target=upload_and_record, daemon=True)
        try:
            source_thread.start()
            assert reached.wait(30), "the source ingestion never reached its commit"
            upload_thread.start()
            time.sleep(1.0)
            released.set()
            may_continue.set()
            source_thread.join(90)
            upload_thread.join(90)
        finally:
            may_continue.set()
            runtime.rag._commit_generation = real_commit

        assert results["up"].status_code == 200, results["up"].text
        assert results["src"].status_code in (200, 201), results["src"].text
        assert results["waited_for_release"], (
            "the upload published a new generation while the source request "
            "still owned the name"
        )
        assert target.read_bytes() == second
        assert _manifest(runtime, user_id)["report.md"]["checksum"] == (
            hashlib.sha256(second).hexdigest()
        )
        indexed = _text(runtime, context_id)
        assert "THE GENERATION THE UPLOAD WROTE" in indexed, (
            "the source ingestion committed over the newer generation's chunks"
        )
        assert "THE GENERATION THE SOURCE READ" not in indexed, (
            "the index describes a generation the file no longer holds"
        )


class TestAnArtifactIsInvisibleUntilItIsComplete:
    """`publish_artifacts` claimed the visible name and then filled it.

    `O_EXCL` on the final name makes the claim atomic, which is what stops
    two producers taking one name and what stops an artifact replacing an
    upload. It also makes the name appear before the bytes do. A copy that
    fails partway was caught and the loop moved on, so the user's file area
    kept a truncated file the tool reported publishing nothing about; and
    while a copy is in progress, anything listing or downloading that name
    sees a file that is still being written.

    This is the interpreter's form of the torn download 2E.3 removed from
    upload, and it takes the same answer: fill a name nobody can see, then
    make it visible in one step.
    """

    BODY = b"artifact bytes," * 20000

    def _publish(self, workdir, dest, names, allowed=None):
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
                    allowed_extensions=allowed or {".csv"},
                )
        finally:
            invocation.close()

    def test_a_failed_copy_leaves_no_file_behind(self, tmp_path, monkeypatch):
        from liminallm.service import interpreter

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        (workdir / "result.csv").write_bytes(self.BODY)

        real_copy = interpreter.shutil.copyfileobj

        def failing(src, out, *args, **kwargs):
            out.write(src.read(64 * 1024))
            raise OSError("no space left on device")

        monkeypatch.setattr(interpreter.shutil, "copyfileobj", failing)
        published = self._publish(workdir, dest, ["result.csv"])

        assert published == [], f"a failed copy was reported as published: {published}"
        assert not (dest / "result.csv").exists(), (
            "the destination kept a file the tool reported publishing nothing "
            "about"
        )
        assert sorted(p.name for p in dest.iterdir()) == [], (
            f"the failed publication left something behind: "
            f"{sorted(p.name for p in dest.iterdir())}"
        )
        assert real_copy is not None

    def test_the_name_appears_only_once_the_bytes_are_all_there(
        self, tmp_path, monkeypatch
    ):
        """Observed from inside the copy, which is the only moment it could
        be seen half-written."""
        from liminallm.service import interpreter

        workdir, dest = tmp_path / "w", tmp_path / "d"
        workdir.mkdir()
        dest.mkdir()
        (workdir / "result.csv").write_bytes(self.BODY)
        seen: list[int | None] = []
        real_copy = interpreter.shutil.copyfileobj

        def watched(src, out, *args, **kwargs):
            out.write(src.read(64 * 1024))
            out.flush()
            # What a reader would find at the published name right now.
            target = dest / "result.csv"
            seen.append(target.stat().st_size if target.exists() else None)
            return real_copy(src, out, *args, **kwargs)

        monkeypatch.setattr(interpreter.shutil, "copyfileobj", watched)
        published = self._publish(workdir, dest, ["result.csv"])

        assert published == ["result.csv"], published
        assert (dest / "result.csv").read_bytes() == self.BODY
        assert seen and all(
            size is None or size == len(self.BODY) for size in seen
        ), (
            f"the published name was visible holding {seen} bytes while the "
            f"copy was still running, out of {len(self.BODY)}"
        )


class TestAReplacedPathLeavesNoOlderGenerationBehind:
    """No race. Two ordinary uploads, one after the other.

    Upload starts a new context set when the checksum changes, so the
    contexts that described the previous generation are dropped from the
    manifest and left in the index. The record forgets them; the chunks do
    not. A context the user built by uploading a file then goes on
    answering with text the file has not held since.
    """

    def test_replacing_the_bytes_invalidates_the_other_contexts(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        first_ctx, second_ctx = _context(client, headers), _context(client, headers)
        first = b"# report\nTHE FIGURES FROM THE FIRST QUARTER\n" * 12
        second = b"# report\nTHE FIGURES FROM THE SECOND QUARTER\n" * 12

        assert _upload(client, headers, "report.md", first, first_ctx).status_code == 200
        assert "THE FIGURES FROM THE FIRST QUARTER" in _text(runtime, first_ctx)

        assert _upload(client, headers, "report.md", second, second_ctx).status_code == 200

        target = _files_dir(runtime, user_id) / "report.md"
        assert target.read_bytes() == second
        assert "THE FIGURES FROM THE SECOND QUARTER" in _text(runtime, second_ctx)
        assert "THE FIGURES FROM THE FIRST QUARTER" not in _text(runtime, first_ctx), (
            "the first context still describes a generation the file no "
            "longer holds"
        )
        assert str(target) not in _described_paths(runtime, first_ctx)

    def test_replacing_the_bytes_with_no_context_named_invalidates_too(self, client):
        """The simplest form: the second upload names no context at all."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# notes\nTHE ORIGINAL NOTE TEXT\n" * 12
        second = b"# notes\nTHE REPLACEMENT NOTE TEXT\n" * 12

        assert _upload(client, headers, "notes.md", first, context_id).status_code == 200
        assert "THE ORIGINAL NOTE TEXT" in _text(runtime, context_id)

        assert _upload(client, headers, "notes.md", second).status_code == 200

        assert (_files_dir(runtime, user_id) / "notes.md").read_bytes() == second
        assert "THE ORIGINAL NOTE TEXT" not in _text(runtime, context_id), (
            "the context describes a generation the file no longer holds"
        )

    def test_the_same_bytes_uploaded_again_keep_their_contexts(self, client):
        """A dedupe hit is not a replacement. Nothing changed, so nothing
        the other contexts say has stopped being true."""
        runtime = get_runtime()
        _user_id, headers = _account(client)
        first_ctx, second_ctx = _context(client, headers), _context(client, headers)
        body = b"# stable\nTHE UNCHANGED CONTENTS\n" * 12

        assert _upload(client, headers, "stable.md", body, first_ctx).status_code == 200
        assert _upload(client, headers, "stable.md", body, second_ctx).status_code == 200

        for context_id in (first_ctx, second_ctx):
            assert "THE UNCHANGED CONTENTS" in _text(runtime, context_id), (
                "re-uploading identical bytes dropped a context that was "
                "still describing them correctly"
            )

    def test_a_conversations_attachment_index_is_not_invalidated(self, client):
        """A conversation's implicit context is not a path-following source.

        §19.5 scopes an attachment to the chat that received it, so another
        chat's later upload of the same filename must not reach into it —
        not by replacing its chunks, and not by removing them either. The
        attachment's own staleness is the separate, recorded problem; what
        this holds is that the invalidation sweep does not become the way
        one chat changes another chat's state.
        """
        runtime = get_runtime()
        _user_id, headers = _account(client)
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        conversation_id = resp.json()["data"]["id"]
        attached = b"# attached\nTHE TEXT THIS CHAT WAS GIVEN\n" * 400

        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("shared-name.md", attached, "text/markdown")},
            data={"conversation_id": conversation_id},
        )
        assert resp.status_code == 200, resp.text
        from liminallm.service.attachments import find_conversation_context_id

        auto_ctx = find_conversation_context_id(
            runtime.store, user_id=_user_id, conversation_id=conversation_id
        )
        assert auto_ctx, "the attachment was not indexed, so nothing is being tested"
        assert "THE TEXT THIS CHAT WAS GIVEN" in _text(runtime, auto_ctx)

        # Another upload of the same global name, from outside the chat.
        assert _upload(
            client, headers, "shared-name.md", b"# other\nWRITTEN BY SOMETHING ELSE\n" * 12
        ).status_code == 200

        assert "WRITTEN BY SOMETHING ELSE" not in _text(runtime, auto_ctx), (
            "another upload's bytes were indexed into a conversation that "
            "never received them"
        )
        assert "THE TEXT THIS CHAT WAS GIVEN" in _text(runtime, auto_ctx), (
            "an unrelated upload removed a conversation's attachment index"
        )


class TestAnAttachmentNeverResolvesToOtherBytes:
    """An attachment record named a file, and the file was a moving target.

    `{"name": "notes.txt"}` is resolved against `/users/{u}/files/notes.txt`
    by every consumer: the inline reader, the interpreter's workdir, and the
    note importer. Another conversation uploading that filename replaces the
    global path, so one chat's attachment started resolving to another
    chat's bytes — and §19.5 scopes an attachment to the chat that received
    it. Deleting the file and later creating the name again is the same
    thing with a gap in the middle: the old record silently rebinds to bytes
    that were never attached to anything.

    These are the plain sequential cases. The interleavings the same defect
    also allowed, and the store that removes both, are in
    `TestAnAttachmentIsAnImmutableGeneration` below.
    """

    def _conversation(self, client, headers) -> str:
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["data"]["id"]

    def _attach(self, client, headers, conversation_id, name, body):
        return client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, body, "text/markdown")},
            data={"conversation_id": conversation_id},
        )

    def _inline(self, runtime, client, headers, user_id, conversation_id):
        from liminallm.service.attachments import (
            list_attachments,
            read_inline_contents,
        )

        conversation = runtime.store.get_conversation(
            conversation_id, user_id=user_id
        )
        return read_inline_contents(
            list_attachments(conversation),
            fs_root=runtime.settings.shared_fs_root,
            user_id=user_id,
        )

    def test_another_chats_upload_is_not_served_to_this_one(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        mine = self._conversation(client, headers)
        theirs = self._conversation(client, headers)

        assert self._attach(
            client, headers, mine, "notes.md", b"# notes\nTHE BYTES THIS CHAT ATTACHED\n"
        ).status_code == 200
        served = self._inline(runtime, client, headers, user_id, mine)
        assert any("THE BYTES THIS CHAT ATTACHED" in f["content"] for f in served), served

        assert self._attach(
            client, headers, theirs, "notes.md", b"# notes\nA DIFFERENT CHATS BYTES\n"
        ).status_code == 200

        served = self._inline(runtime, client, headers, user_id, mine)
        text = " ".join(f["content"] for f in served)
        assert "A DIFFERENT CHATS BYTES" not in text, (
            "one conversation was served the bytes another conversation "
            "attached, which §19.5 scopes to that chat"
        )

    def test_a_name_recreated_after_a_delete_does_not_rebind(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)

        assert self._attach(
            client, headers, conversation_id, "payroll.md", b"# payroll\nTHE ATTACHED FIGURES\n"
        ).status_code == 200
        assert client.delete("/v1/files/payroll.md", headers=headers).status_code == 200
        assert _upload(
            client, headers, "payroll.md", b"# payroll\nSOMETHING WRITTEN LATER\n"
        ).status_code == 200

        served = self._inline(runtime, client, headers, user_id, conversation_id)
        text = " ".join(f["content"] for f in served)
        assert "SOMETHING WRITTEN LATER" not in text, (
            "a deleted attachment came back to life pointing at bytes that "
            "were never attached to this conversation"
        )

    def test_the_interpreter_stages_no_substituted_bytes(self, client):
        """`run_python` rebuilds its workdir from the conversation's records."""
        from liminallm.service import interpreter
        from liminallm.service.attachments import list_attachments, resolved_sources

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        assert self._attach(
            client, headers, conversation_id, "data.md", b"# data\nTHE ATTACHED ROWS\n"
        ).status_code == 200
        assert _upload(
            client, headers, "data.md", b"# data\nROWS FROM SOMEWHERE ELSE\n"
        ).status_code == 200

        conversation = runtime.store.get_conversation(conversation_id, user_id=user_id)
        sources = resolved_sources(
            list_attachments(conversation),
            fs_root=runtime.settings.shared_fs_root,
            user_id=user_id,
        )
        workdir = interpreter.prepare_workdir(
            str(Path(runtime.settings.shared_fs_root) / "scratch"), sources
        )
        staged = " ".join(
            p.read_text(errors="replace") for p in Path(workdir).iterdir() if p.is_file()
        )
        assert "ROWS FROM SOMEWHERE ELSE" not in staged, (
            "the interpreter staged bytes that were never attached"
        )
        assert "THE ATTACHED ROWS" in staged

    def test_the_prompt_does_not_promise_text_it_leaves_out(self, client):
        """The listing and the envelope have to agree.

        An attachment whose generation is not there is simply not in the
        envelope. The listing above it said "full text included below"
        regardless, so the model was told to read text that was not there.
        A record from before the generation store is the case that still
        reaches this: its bytes cannot be reconstructed, so it resolves to
        nothing.
        """
        from liminallm.service.attachments import build_attachment_preamble

        runtime = get_runtime()
        user_id, headers = _account(client)
        assert _upload(
            client, headers, "legacy.md", b"# legacy\nTODAYS BYTES AT THAT NAME\n"
        ).status_code == 200

        legacy = {
            "name": "legacy.md",
            "size": 33,
            "inline": True,
            "searchable": False,
            "analyzable": True,
        }
        preamble = build_attachment_preamble(
            [legacy], fs_root=runtime.settings.shared_fs_root, user_id=user_id
        )
        assert "TODAYS BYTES AT THAT NAME" not in preamble
        assert "full text included below" not in preamble, (
            "the listing promised text the envelope does not contain"
        )
        assert "run_python" not in preamble, (
            "the listing offered a capability the tool will refuse"
        )
        assert "unavailable" in preamble, preamble


class TestARejectedRequestDoesNotMutateAnything:
    """The order of validation and mutation.

    `_publish` replaced the pathname and validated the named context
    afterwards, inside the ingestion step. So a request rejected for naming a
    context that does not exist had already overwritten the file, and the
    failure handler then unlinked it: the pathname was gone, the manifest and
    the chunks still described the generation it used to hold, and the user
    was told their request was refused.

    A parameter the route will refuse is knowable before anything durable
    happens, so it is checked there.
    """

    def test_an_unknown_context_leaves_the_previous_generation_alone(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE GENERATION ALREADY PUBLISHED\n" * 12
        assert _upload(client, headers, "report.md", first, context_id).status_code == 200
        target = _files_dir(runtime, user_id) / "report.md"

        resp = _upload(
            client,
            headers,
            "report.md",
            b"# report\nTHE GENERATION THAT WAS REFUSED\n" * 12,
            str(uuid.uuid4()),
        )
        assert resp.status_code == 404, resp.status_code

        assert target.exists(), (
            "a request refused for naming an unknown context deleted the "
            "file it was replacing"
        )
        assert target.read_bytes() == first
        assert _manifest(runtime, user_id)["report.md"]["checksum"] == (
            hashlib.sha256(first).hexdigest()
        )
        assert "THE GENERATION ALREADY PUBLISHED" in _text(runtime, context_id)

    def test_another_users_context_is_refused_before_the_write(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        _other_id, other_headers = _account(client)
        theirs = _context(client, other_headers)
        first = b"# report\nMY OWN PUBLISHED GENERATION\n" * 12
        assert _upload(client, headers, "report.md", first).status_code == 200
        target = _files_dir(runtime, user_id) / "report.md"

        resp = _upload(
            client, headers, "report.md", b"# report\nREFUSED BYTES\n" * 12, theirs
        )
        assert resp.status_code == 403, resp.status_code
        assert target.read_bytes() == first, (
            "a request refused for naming another account's context still "
            "replaced the file"
        )

    def test_a_failed_ingestion_leaves_a_generation_that_can_be_retried(self, client):
        """The other half: the context is real and the ingestion fails.

        Unlinking the destination does not restore what it replaced — the
        previous bytes are already gone — so it leaves the pathname absent
        while the manifest and the chunks still describe them. The new bytes
        are the only generation that exists by then, so they are what is
        kept, recorded, and left for the retry to finish.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE FIRST GENERATION\n" * 12
        second = b"# report\nTHE SECOND GENERATION\n" * 12
        assert _upload(client, headers, "report.md", first, context_id).status_code == 200
        target = _files_dir(runtime, user_id) / "report.md"

        armed = {"on": True}
        real_ingest = runtime.rag.ingest_file

        def failing(*args, **kwargs):
            if armed["on"]:
                armed["on"] = False
                raise OSError("the index is unreachable")
            return real_ingest(*args, **kwargs)

        runtime.rag.ingest_file = failing
        key = _unique("retry")
        try:
            with pytest.raises(OSError):
                client.post(
                    "/v1/files/upload",
                    headers={**headers, "Idempotency-Key": key},
                    files={"file": ("report.md", second, "text/markdown")},
                    data={"context_id": context_id},
                )
            assert target.exists(), (
                "the failed ingestion removed the pathname, leaving the "
                "manifest and the index describing bytes no file has"
            )
            assert target.read_bytes() == second
            assert _manifest(runtime, user_id)["report.md"]["checksum"] == (
                hashlib.sha256(second).hexdigest()
            ), "the surviving bytes are not the ones the manifest describes"
            assert "THE FIRST GENERATION" not in _text(runtime, context_id), (
                "the index still describes a generation the file no longer holds"
            )

            resp = client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": key},
                files={"file": ("report.md", second, "text/markdown")},
                data={"context_id": context_id},
            )
            assert resp.status_code == 200, resp.text
            assert "THE SECOND GENERATION" in _text(runtime, context_id), (
                "the retry did not finish the ingestion the first attempt failed"
            )
        finally:
            runtime.rag.ingest_file = real_ingest


class TestDedupeIsConfirmedByTheDiskNotTheRecord:
    """A manifest entry nominates a dedupe hit; the file confirms it.

    The manifest can outlive the bytes it describes — a publication that
    failed after writing them leaves exactly that. When the record alone
    decides, re-uploading the bytes it names skips the write and reports
    success over a file holding something else entirely, which is the false
    dedupe hit 2E.1 removed, arriving through a failed request instead of a
    race. Whoever abandons a failed upload reaches it; no retry is involved.
    """

    def test_an_abandoned_failure_cannot_make_a_later_upload_lie(
        self, client, monkeypatch
    ):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE FIRST GENERATION\n" * 12
        second = b"# report\nTHE SECOND GENERATION\n" * 12
        target = _files_dir(runtime, user_id) / "report.md"
        manifest_path = _files_dir(runtime, user_id) / ".checksums.json"

        assert _upload(client, headers, "report.md", first, context_id).status_code == 200

        armed = {"on": True}
        real_write = Path.write_text

        def gated(path, *args, **kwargs):
            if armed["on"] and path == manifest_path:
                armed["on"] = False
                raise OSError("no space left on device")
            return real_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", gated)
        with pytest.raises(OSError):
            _upload(client, headers, "report.md", second, context_id)
        monkeypatch.undo()

        # The state a client that walks away leaves behind.
        assert target.read_bytes() == second
        assert _manifest(runtime, user_id)["report.md"]["checksum"] == (
            hashlib.sha256(first).hexdigest()
        ), "the manifest was expected to be describing the first generation"

        # A fresh request for the first generation. The manifest nominates it
        # as a dedupe hit; the file on disk is not it.
        resp = _upload(client, headers, "report.md", first, context_id)
        assert resp.status_code == 200, resp.text
        assert target.read_bytes() == first, (
            "the upload reported success without writing, because a record "
            "nominated a dedupe hit that the disk did not confirm"
        )
        assert "THE FIRST GENERATION" in _text(runtime, context_id)
        assert "THE SECOND GENERATION" not in _text(runtime, context_id)


class TestTheIndexIsItsOwnReverseIndex:
    """Which contexts describe a path is a question the database answers.

    The invalidation swept `prior_contexts` out of `.checksums.json`, which
    only ever records the contexts an *upload* named. A context that acquired
    the path through `POST /contexts/{id}/sources` is not in it, and never
    becomes so — the source route ingests and takes the namespace lock, and
    writes nothing to the manifest. So the sweep walked past it, entirely
    sequentially, with no failure anywhere.

    The manifest's context set stays useful for deciding whether an upload
    needs to re-ingest. It is not the reverse index, because it cannot see
    every way a path gets indexed.
    """

    def _add_source(self, client, headers, context_id, path):
        return client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": str(path), "recursive": False},
        )

    def test_a_context_that_took_the_path_as_a_source_is_invalidated(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE GENERATION THE SOURCE INDEXED\n" * 12
        second = b"# report\nTHE GENERATION THAT REPLACED IT\n" * 12

        # No context named on the upload, so the manifest records none.
        assert _upload(client, headers, "report.md", first).status_code == 200
        target = _files_dir(runtime, user_id) / "report.md"
        assert self._add_source(
            client, headers, context_id, target
        ).status_code in (200, 201)
        assert "THE GENERATION THE SOURCE INDEXED" in _text(runtime, context_id)
        assert _manifest(runtime, user_id)["report.md"]["contexts"] == [], (
            "the manifest was expected to know nothing about this context"
        )

        assert _upload(client, headers, "report.md", second).status_code == 200

        assert target.read_bytes() == second
        assert "THE GENERATION THE SOURCE INDEXED" not in _text(runtime, context_id), (
            "a context that acquired the path as a source still describes the "
            "generation the file no longer holds"
        )
        assert str(target) not in _described_paths(runtime, context_id)

    def test_a_source_rooted_above_the_file_still_serializes(self, client):
        """The lock has to be taken where the mutation happens.

        One lock for the whole source works while the source *is* the file.
        A source rooted at `files/` locks `files/`, an upload of
        `files/report.md` locks `report.md`, and the two never meet — so the
        walk reads one generation while the upload publishes the next, and
        the walk's commit lands last. Every step succeeds.
        """
        import threading

        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# report\nTHE GENERATION THE WALK READ\n" * 12
        second = b"# report\nTHE GENERATION THE UPLOAD WROTE\n" * 12
        assert _upload(client, headers, "report.md", first).status_code == 200
        files_dir = _files_dir(runtime, user_id)

        reached = threading.Event()
        may_continue = threading.Event()
        released = threading.Event()
        armed = {"on": True}
        real_commit = runtime.rag._commit_generation

        def gated(*args, **kwargs):
            if armed["on"]:
                armed["on"] = False
                reached.set()
                may_continue.wait(30)
            return real_commit(*args, **kwargs)

        results: dict = {}

        def add_source():
            results["src"] = self._add_source(client, headers, context_id, files_dir)

        def upload_and_record():
            results["up"] = _upload(client, headers, "report.md", second)
            results["waited_for_release"] = released.is_set()

        runtime.rag._commit_generation = gated
        source_thread = threading.Thread(target=add_source, daemon=True)
        upload_thread = threading.Thread(target=upload_and_record, daemon=True)
        try:
            source_thread.start()
            assert reached.wait(30), "the walk never reached a commit"
            upload_thread.start()
            time.sleep(1.0)
            released.set()
            may_continue.set()
            source_thread.join(90)
            upload_thread.join(90)
        finally:
            may_continue.set()
            runtime.rag._commit_generation = real_commit

        assert results["up"].status_code == 200, results["up"].text
        assert results["src"].status_code in (200, 201), results["src"].text
        assert results["waited_for_release"], (
            "the upload replaced the file while the walk still owned it"
        )
        assert (files_dir / "report.md").read_bytes() == second
        indexed = _text(runtime, context_id)
        assert "THE GENERATION THE UPLOAD WROTE" in indexed, (
            "the walk committed over the newer generation's chunks"
        )
        assert "THE GENERATION THE WALK READ" not in indexed, (
            "the index describes a generation the file no longer holds"
        )

    def test_the_context_receiving_the_new_generation_keeps_it(self, client):
        """The one context that must not be swept is the one being written."""
        runtime = get_runtime()
        _user_id, headers = _account(client)
        first_ctx, second_ctx = _context(client, headers), _context(client, headers)
        first = b"# report\nTHE EARLIER GENERATION\n" * 12
        second = b"# report\nTHE INCOMING GENERATION\n" * 12

        assert _upload(client, headers, "report.md", first, first_ctx).status_code == 200
        assert _upload(client, headers, "report.md", second, second_ctx).status_code == 200

        assert "THE INCOMING GENERATION" in _text(runtime, second_ctx), (
            "the invalidation removed the generation the request was writing"
        )
        assert "THE EARLIER GENERATION" not in _text(runtime, first_ctx)

    def test_a_lost_manifest_does_not_lose_the_invalidation(self, client):
        """The manifest is an optimization for dedupe, not the record of
        which contexts describe a path. Deleting it must not make a stale
        generation survive."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        first = b"# notes\nTHE GENERATION BEFORE THE MANIFEST WENT\n" * 12
        second = b"# notes\nTHE GENERATION AFTER IT\n" * 12

        assert _upload(client, headers, "notes.md", first, context_id).status_code == 200
        (_files_dir(runtime, user_id) / ".checksums.json").unlink()

        assert _upload(client, headers, "notes.md", second).status_code == 200

        assert "THE GENERATION BEFORE THE MANIFEST WENT" not in _text(
            runtime, context_id
        ), "losing the manifest left a stale generation in the index"


class TestAnAttachmentIsAnImmutableGeneration:
    """Verifying a pathname and then reopening it is two moments again.

    `resolve_attachment` hashed `/users/{u}/files/{name}` and returned the
    *path*; the inline reader then reopened it to read the text, and
    `resolved_names` threw the object away entirely and returned a basename
    for `prepare_workdir` to reopen later. So a replacement landing between
    the check and the use was served exactly as before — the check noticed a
    replacement that had already happened, and nothing about one that had
    not happened yet.

    A hash is only a name for bytes if the bytes cannot move. Each attached
    generation is now copied into a content-addressed store the moment it is
    attached, and every consumer reads that object. The pathname a chat was
    given the file under can then be replaced, deleted, or recreated without
    the chat noticing.
    """

    def _conversation(self, client, headers) -> str:
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["data"]["id"]

    def _attach(self, client, headers, conversation_id, name, body):
        return client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, body, "text/markdown")},
            data={"conversation_id": conversation_id},
        )

    def _records(self, runtime, conversation_id, user_id):
        from liminallm.service.attachments import list_attachments

        return list_attachments(
            runtime.store.get_conversation(conversation_id, user_id=user_id)
        )

    def _inline(self, runtime, conversation_id, user_id):
        from liminallm.service.attachments import read_inline_contents

        return read_inline_contents(
            self._records(runtime, conversation_id, user_id),
            fs_root=runtime.settings.shared_fs_root,
            user_id=user_id,
        )

    def test_a_replacement_between_the_check_and_the_read_is_not_served(
        self, client, monkeypatch
    ):
        """The window `resolve_attachment` left open, closed at the source."""
        from liminallm.service import attachments as attachments_service

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        attached = b"# notes\nTHE BYTES THIS CHAT ATTACHED\n"
        assert self._attach(
            client, headers, conversation_id, "notes.md", attached
        ).status_code == 200

        armed = {"on": True}
        real_resolve = attachments_service.resolve_attachment

        def gated(fs_root, uid, record):
            resolved = real_resolve(fs_root, uid, record)
            if armed["on"] and resolved is not None:
                armed["on"] = False
                # The window: the attachment has been resolved and nothing
                # has read it yet.
                assert _upload(
                    client, headers, "notes.md", b"# notes\nBYTES FROM ELSEWHERE\n"
                ).status_code == 200
            return resolved

        monkeypatch.setattr(attachments_service, "resolve_attachment", gated)
        served = " ".join(
            item["content"] for item in self._inline(runtime, conversation_id, user_id)
        )
        assert not armed["on"], "the window never opened, so nothing was raced"
        assert "BYTES FROM ELSEWHERE" not in served, (
            "the reader was served bytes that replaced the attachment after "
            "the attachment had been verified"
        )
        assert "THE BYTES THIS CHAT ATTACHED" in served

    def test_a_replacement_between_resolution_and_staging_is_not_copied(
        self, client
    ):
        """The same window on the interpreter's side of the fence."""
        from liminallm.service import attachments as attachments_service
        from liminallm.service import interpreter

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        assert self._attach(
            client, headers, conversation_id, "data.md", b"# data\nTHE ATTACHED ROWS\n"
        ).status_code == 200

        sources = attachments_service.resolved_sources(
            self._records(runtime, conversation_id, user_id),
            fs_root=runtime.settings.shared_fs_root,
            user_id=user_id,
        )
        assert sources, "the attachment did not resolve at all"

        # Between the resolution and the copy, exactly where the basename
        # used to be re-read.
        assert _upload(
            client, headers, "data.md", b"# data\nROWS FROM SOMEWHERE ELSE\n"
        ).status_code == 200

        workdir = interpreter.prepare_workdir(
            str(Path(runtime.settings.shared_fs_root) / "scratch"), sources
        )
        staged = " ".join(
            p.read_text(errors="replace")
            for p in Path(workdir).iterdir()
            if p.is_file()
        )
        assert "ROWS FROM SOMEWHERE ELSE" not in staged, (
            "the interpreter staged bytes that replaced the attachment after "
            "it had been resolved"
        )
        assert "THE ATTACHED ROWS" in staged
        assert [p.name for p in Path(workdir).iterdir()] == ["data.md"], (
            "the workdir should hold the file under the name the chat knows"
        )

    def test_the_attachment_survives_the_pathname_being_replaced(self, client):
        """The payoff: the chat keeps its file when the global name moves."""
        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        assert self._attach(
            client, headers, conversation_id, "brief.md", b"# brief\nTHE ATTACHED BRIEF\n"
        ).status_code == 200
        assert _upload(
            client, headers, "brief.md", b"# brief\nA LATER UNRELATED BRIEF\n"
        ).status_code == 200

        served = " ".join(
            item["content"] for item in self._inline(runtime, conversation_id, user_id)
        )
        assert "THE ATTACHED BRIEF" in served, (
            "the chat lost its attachment because the global pathname moved"
        )
        assert "A LATER UNRELATED BRIEF" not in served

    def test_the_attachment_survives_the_pathname_being_deleted(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        assert self._attach(
            client, headers, conversation_id, "payroll.md", b"# payroll\nTHE FIGURES\n"
        ).status_code == 200
        assert client.delete("/v1/files/payroll.md", headers=headers).status_code == 200
        assert _upload(
            client, headers, "payroll.md", b"# payroll\nWRITTEN LATER\n"
        ).status_code == 200

        served = " ".join(
            item["content"] for item in self._inline(runtime, conversation_id, user_id)
        )
        assert "WRITTEN LATER" not in served, "a recreated name rebound the attachment"
        assert "THE FIGURES" in served, (
            "deleting the global pathname took the chat's attachment with it"
        )

    def test_a_record_from_before_generations_fails_closed(self, client):
        """An old record names a pathname and nothing else.

        Its generation cannot be reconstructed, so today's bytes at that
        pathname are not evidence of what was attached. Resolving them would
        be the cross-chat substitution this removed, kept alive by the
        upgrade.
        """
        from liminallm.service.attachments import resolve_attachment

        runtime = get_runtime()
        user_id, headers = _account(client)
        assert _upload(client, headers, "legacy.md", b"# legacy\nTODAYS BYTES\n").status_code == 200

        legacy = {"name": "legacy.md", "size": 22, "inline": True}
        assert resolve_attachment(
            runtime.settings.shared_fs_root, user_id, legacy
        ) is None, (
            "a record with no generation resolved against the live pathname"
        )


class TestUnreferencedGenerationsAreReclaimed:
    """The store is write-once, so something has to take things out of it.

    Mark and sweep, over the marks that already exist: every attachment
    record names its generation. A reference count would be a second record
    of the same fact, to be kept correct across every way a conversation is
    created, edited and deleted.
    """

    def _attach(self, client, headers, conversation_id, name, body):
        return client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, body, "text/markdown")},
            data={"conversation_id": conversation_id},
        )

    def _conversation(self, client, headers) -> str:
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["data"]["id"]

    def _age(self, path: Path, seconds: int) -> None:
        import os

        stamp = path.stat().st_mtime - seconds
        os.utime(path, (stamp, stamp))

    def test_the_generation_store_is_not_part_of_the_users_files(self, client):
        """It sits beside `files/`, so nothing that walks `files/` sees it.

        These objects are named by hash, are not files the user created, and
        are not files the user can delete without deleting the conversation
        that holds them. Listing them would offer all three.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        body = b"# hidden\nTHE ATTACHED TEXT\n"
        assert self._attach(
            client, headers, conversation_id, "hidden.md", body
        ).status_code == 200
        checksum = hashlib.sha256(body).hexdigest()

        resp = client.get("/v1/files", headers=headers)
        assert resp.status_code == 200, resp.text
        names = {f["name"] for f in resp.json()["data"]["files"]}
        assert names == {"hidden.md"}, names

        # And it cannot be reached through the download path either.
        resp = client.get(f"/v1/files/{checksum}/url", headers=headers)
        assert resp.status_code == 404, resp.status_code

    def test_reusing_an_old_generation_survives_a_concurrent_sweep(self, client):
        """The grace period protects a *new* object, not a reused old one.

        `store_generation` returns an existing object without touching it, so
        its age still says when it was first written. An object that has been
        unreferenced long enough to be swept can be reused by a new
        attachment, and the sweep then unlinks it during that attachment's
        own operation: the record lands naming bytes that are already gone.
        """
        import threading

        from liminallm.api import routes
        from liminallm.service.attachments import (
            generation_path,
            resolve_attachment,
            store_generation,
            sweep_generations,
        )

        runtime = get_runtime()
        user_id, headers = _account(client)
        body = b"# reused\nBYTES ATTACHED ONCE AND THEN AGAIN\n"
        checksum = hashlib.sha256(body).hexdigest()

        # An old object nothing references: what the sweep is looking for.
        blob = store_generation(
            runtime.settings.shared_fs_root, user_id, body, checksum
        )
        assert blob is not None and blob.is_file()
        self._age(blob, 10_000)

        swept: dict = {}
        armed = {"on": True}
        real_store = routes.store_generation

        def gated(*args, **kwargs):
            out = real_store(*args, **kwargs)
            if armed["on"]:
                armed["on"] = False

                def sweep():
                    swept["removed"] = sweep_generations(
                        runtime.store,
                        runtime.settings.shared_fs_root,
                        grace_seconds=60,
                    )

                sweeper = threading.Thread(target=sweep, daemon=True)
                sweeper.start()
                # Long enough for a sweep that takes no lock to finish.
                time.sleep(1.0)
                swept["thread"] = sweeper
            return out

        conversation_id = self._conversation(client, headers)
        original = routes.store_generation
        routes.store_generation = gated
        try:
            resp = self._attach(client, headers, conversation_id, "reused.md", body)
        finally:
            routes.store_generation = original
        assert resp.status_code == 200, resp.text
        assert not armed["on"], "the window never opened"
        swept["thread"].join(60)

        record = resp.json()["data"]["attachment"]
        assert record and record["checksum"] == checksum, record
        assert resolve_attachment(
            runtime.settings.shared_fs_root, user_id, record
        ) is not None, (
            "the sweep removed the generation during the attachment that was "
            "adopting it, so the record names bytes that are gone"
        )
        assert generation_path(
            runtime.settings.shared_fs_root, user_id, checksum
        ).is_file()

    def test_a_referenced_generation_survives_the_sweep(self, client):
        from liminallm.service.attachments import generation_path, sweep_generations

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        body = b"# kept\nTHE ATTACHED TEXT\n"
        assert self._attach(
            client, headers, conversation_id, "kept.md", body
        ).status_code == 200
        blob = generation_path(
            runtime.settings.shared_fs_root, user_id, hashlib.sha256(body).hexdigest()
        )
        assert blob is not None and blob.is_file()
        self._age(blob, 10_000)

        sweep_generations(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=60
        )
        assert blob.is_file(), (
            "a generation a conversation still names was reclaimed"
        )

    def test_a_generation_nothing_names_is_reclaimed(self, client):
        from liminallm.service.attachments import (
            generation_path,
            store_generation,
            sweep_generations,
        )

        runtime = get_runtime()
        user_id, _headers = _account(client)
        body = b"orphaned bytes nothing ever attached"
        checksum = hashlib.sha256(body).hexdigest()
        stored = store_generation(
            runtime.settings.shared_fs_root, user_id, body, checksum
        )
        assert stored is not None and stored.is_file()
        self._age(stored, 10_000)

        removed = sweep_generations(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=60
        )
        assert removed >= 1, removed
        assert generation_path(
            runtime.settings.shared_fs_root, user_id, checksum
        ).exists() is False

    def test_a_fresh_generation_is_inside_the_grace_period(self, client):
        """The window between storing a generation and recording the
        attachment that names it is exactly what the grace period covers."""
        from liminallm.service.attachments import store_generation, sweep_generations

        runtime = get_runtime()
        user_id, _headers = _account(client)
        body = b"just written, not yet recorded"
        stored = store_generation(
            runtime.settings.shared_fs_root,
            user_id,
            body,
            hashlib.sha256(body).hexdigest(),
        )
        assert stored is not None

        sweep_generations(
            runtime.store, runtime.settings.shared_fs_root, grace_seconds=3600
        )
        assert stored.is_file(), (
            "a generation written moments ago was reclaimed before anything "
            "had a chance to name it"
        )

    def test_an_unreadable_reference_set_sweeps_nothing(self, client):
        """An empty set means "no attachments"; an error means "unknown"."""
        from liminallm.service.attachments import store_generation, sweep_generations

        runtime = get_runtime()
        user_id, _headers = _account(client)
        body = b"kept because the marks could not be read"
        stored = store_generation(
            runtime.settings.shared_fs_root,
            user_id,
            body,
            hashlib.sha256(body).hexdigest(),
        )
        assert stored is not None
        self._age(stored, 10_000)

        class _Unreadable:
            def referenced_attachment_checksums(self, owner_user_id):
                raise OSError("the database is unreachable")

        removed = sweep_generations(
            _Unreadable(), runtime.settings.shared_fs_root, grace_seconds=60
        )
        assert removed == 0, removed
        assert stored.is_file(), (
            "the sweep deleted generations after failing to read what "
            "references them"
        )


class TestTheListingAgreesWithTheEnvelope:
    """Every line of the listing has to be true of the prompt it introduces.

    Two ways an inline attachment ends up outside the envelope, and they are
    not the same fact. Its generation may be gone, or the shared inline
    budget may have filled up before it. "Full text included below" was said
    in both cases, and "no longer stored" would be wrong in the second.
    """

    def test_a_file_that_did_not_fit_is_not_announced_as_included(self, client):
        from liminallm.service.attachments import (
            INLINE_TOTAL_BUDGET,
            build_attachment_preamble,
            store_generation,
        )

        runtime = get_runtime()
        user_id, _headers = _account(client)
        records = []
        # Enough small inline files to overrun the shared budget.
        for index in range(6):
            body = (f"file {index} " + "x" * 40).encode() * 200
            checksum = hashlib.sha256(body).hexdigest()
            assert store_generation(
                runtime.settings.shared_fs_root, user_id, body, checksum
            ) is not None
            records.append(
                {
                    "name": f"part{index}.txt",
                    "size": len(body),
                    "checksum": checksum,
                    "inline": True,
                    "searchable": False,
                    "analyzable": True,
                }
            )
        assert sum(r["size"] for r in records) > INLINE_TOTAL_BUDGET

        preamble = build_attachment_preamble(
            records, fs_root=runtime.settings.shared_fs_root, user_id=user_id
        )
        quoted = preamble.count("quoted below as [file ")
        assert 0 < quoted < len(records), (
            f"{quoted} of {len(records)} files were quoted; the budget was "
            "expected to stop some of them"
        )
        assert "full text included below" not in preamble, (
            "the listing promised text the envelope does not contain"
        )
        assert "no longer stored" not in preamble, (
            "a file that is stored was described as gone because its text "
            "did not fit"
        )
        assert "did not fit" in preamble, preamble
        # The capability that does not depend on the prompt is still offered.
        assert "run_python" in preamble


def _pdf_bytes(line: str) -> bytes:
    """A one-page PDF whose text only a PDF reader can recover.

    The content stream is Flate-compressed on purpose. An uncompressed one
    is mostly ASCII, so the marker survives a generic byte decode and a test
    built on it passes whether or not the format was recognised — measured,
    that is exactly what the first version of this did.
    """
    raw = f"BT /F1 24 Tf 72 700 Td ({line}) Tj ET".encode()
    content = zlib.compress(raw)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" /Filter /FlateDecode >>"
        b"\nstream\n" + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


class TestAGenerationKeepsItsFormat:
    """The object is named by its digest, and a digest has no extension.

    `extract_text` routes by `path.suffix`, so moving a searchable
    attachment into the content-addressed store took its format away with
    its name: a `.docx` arrived as an extensionless object, fell through to
    the generic byte decode, and was refused as binary. The upload reported
    success with nothing indexed.

    The extension does not go into the key — the key is the identity of the
    bytes and nothing else. The format travels beside it, as what the
    conversation calls the file.
    """

    def _conversation(self, client, headers) -> str:
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["data"]["id"]

    def test_a_document_attachment_is_still_read(self, client):
        from liminallm.service.attachments import (
            classify_attachment,
            find_conversation_context_id,
        )

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        body = _pdf_bytes("THE PARAGRAPH INSIDE THE DOCUMENT")
        assert classify_attachment("report.pdf", len(body))["searchable"], (
            "the test needs a format the upload routes into the index"
        )

        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("report.pdf", body, "application/pdf")},
            data={"conversation_id": conversation_id},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["chunk_count"], (
            f"the document was stored and never read: {resp.json()['data']}"
        )

        auto_ctx = find_conversation_context_id(
            runtime.store, user_id=user_id, conversation_id=conversation_id
        )
        assert auto_ctx, "no implicit context was created"
        assert "THE PARAGRAPH INSIDE THE DOCUMENT" in _text(runtime, auto_ctx)

    def test_the_extension_is_not_part_of_the_key(self, client):
        """Two names, one set of bytes, one object.

        The digest is the identity of the bytes. Putting a display name into
        it would give the same bytes two objects and defeat the dedupe the
        store gets for free.
        """
        from liminallm.service.attachments import generation_path

        runtime = get_runtime()
        user_id, headers = _account(client)
        first = self._conversation(client, headers)
        second = self._conversation(client, headers)
        body = b"# shared\nTHE SAME BYTES UNDER TWO NAMES\n"
        for conversation_id, name in ((first, "one.md"), (second, "two.md")):
            resp = client.post(
                "/v1/files/upload",
                headers={**headers, "Idempotency-Key": _unique("k")},
                files={"file": (name, body, "text/markdown")},
                data={"conversation_id": conversation_id},
            )
            assert resp.status_code == 200, resp.text

        blob = generation_path(
            runtime.settings.shared_fs_root, user_id, hashlib.sha256(body).hexdigest()
        )
        assert blob is not None and blob.is_file()
        assert blob.suffix == "", blob.name
        stored = sorted(p.name for p in blob.parent.iterdir())
        assert stored == [blob.name], (
            f"the same bytes were stored more than once: {stored}"
        )


class TestAChatSearchesOnlyWhatItStillHolds:
    """Replacing an attachment in one chat leaves two generations indexed.

    `replace_chunks_for_path` replaces the rows for the path it is given, and
    the path is now the generation. A second attachment under the same name
    is a *different* generation, so its ingestion replaced nothing: the
    conversation's record named the new bytes while its index held both.

    The record is the authority for what the chat holds. What the index
    happens to contain is not a capability.
    """

    def _conversation(self, client, headers) -> str:
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["data"]["id"]

    def _attach(self, client, headers, conversation_id, name, body):
        return client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": (name, body, "text/markdown")},
            data={"conversation_id": conversation_id},
        )

    def _search(self, runtime, conversation_id, user_id, query):
        text, _snippets = runtime.workflow._run_file_search(
            query,
            8,
            conversation_id=conversation_id,
            context_id=None,
            user_id=user_id,
            tenant_id=None,
        )
        return text

    def test_replacing_an_attachment_retires_the_one_it_replaced(self, client):
        from liminallm.service.attachments import (
            classify_attachment,
            find_conversation_context_id,
            list_attachments,
        )

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        first = b"# manual\nTHE INSTRUCTIONS FROM THE FIRST EDITION\n" * 400
        second = b"# manual\nTHE INSTRUCTIONS FROM THE SECOND EDITION\n" * 400
        assert classify_attachment("manual.md", len(first))["searchable"], (
            "the test needs an attachment large enough to be chunked"
        )

        assert self._attach(
            client, headers, conversation_id, "manual.md", first
        ).status_code == 200
        assert self._attach(
            client, headers, conversation_id, "manual.md", second
        ).status_code == 200

        conversation = runtime.store.get_conversation(conversation_id, user_id=user_id)
        records = list_attachments(conversation)
        assert [r["checksum"] for r in records] == [
            hashlib.sha256(second).hexdigest()
        ], records

        found = self._search(runtime, conversation_id, user_id, "instructions edition")
        editions = {
            name for name in ("FIRST EDITION", "SECOND EDITION") if name in found
        }
        assert editions == {"SECOND EDITION"}, (
            f"file_search returned {sorted(editions)}; the chat holds only the "
            "second edition"
        )

        auto_ctx = find_conversation_context_id(
            runtime.store, user_id=user_id, conversation_id=conversation_id
        )
        assert "THE INSTRUCTIONS FROM THE FIRST EDITION" not in _text(
            runtime, auto_ctx
        ), "the retired generation's chunks were left in the index"

    def test_a_generation_reclaimed_by_the_sweep_is_not_retrievable(self, client):
        """Chunks outlive the object they describe unless something says so.

        The sweeper removes an unreferenced blob; it does not touch Postgres.
        If retrieval were driven by what the index contains rather than by
        what the conversation holds, `file_search` would go on answering from
        a generation whose bytes are gone.
        """
        from liminallm.service.attachments import generation_path

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        body = b"# ledger\nTHE ENTRIES IN THE OLD LEDGER\n" * 400
        assert self._attach(
            client, headers, conversation_id, "ledger.md", body
        ).status_code == 200
        assert "OLD LEDGER" in self._search(
            runtime, conversation_id, user_id, "entries ledger"
        )

        # Drop the record, leaving the chunks: the shape a sweep produces.
        runtime.store.merge_conversation_meta(
            conversation_id, user_id=user_id, patch={"attachments": []}
        )
        blob = generation_path(
            runtime.settings.shared_fs_root, user_id, hashlib.sha256(body).hexdigest()
        )
        blob.unlink()

        found = self._search(runtime, conversation_id, user_id, "entries ledger")
        assert "OLD LEDGER" not in found, (
            "file_search answered from a generation the conversation no "
            f"longer holds and whose bytes are gone: {found[:120]!r}"
        )


class TestAConversationsIndexIsNotAUserManagedContext:
    """`meta.auto` is load-bearing, so it has to be true.

    The invalidation sweep skips auto contexts because they hold attachment
    generations, which are immutable and scoped to one chat. That reasoning
    only holds if nothing else can put a path-following source into one —
    and `POST /contexts/{id}/sources` checked ownership and nothing else.
    The id is not even hidden: a searchable attachment upload returns it.

    So a client could add an ordinary mutable file to a conversation's index
    and then replace that file, and the sweep would deliberately leave the
    stale generation alone. Making the architectural statement true is
    cheaper than making the sweep understand the exception.
    """

    def test_a_source_cannot_be_added_to_a_conversations_index(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        resp = client.post(
            "/v1/conversations", headers=headers, json={"title": _unique("chat")}
        )
        assert resp.status_code in (200, 201), resp.text
        conversation_id = resp.json()["data"]["id"]

        body = b"# manual\nTHE ATTACHED MANUAL TEXT\n" * 400
        resp = client.post(
            "/v1/files/upload",
            headers={**headers, "Idempotency-Key": _unique("k")},
            files={"file": ("manual.md", body, "text/markdown")},
            data={"conversation_id": conversation_id},
        )
        assert resp.status_code == 200, resp.text
        auto_ctx = resp.json()["data"]["context_id"]
        assert auto_ctx, "the searchable attachment did not report its context"

        assert _upload(client, headers, "ordinary.md", b"# ordinary\nBODY\n" * 12).status_code == 200
        resp = client.post(
            f"/v1/contexts/{auto_ctx}/sources",
            headers=headers,
            json={
                "fs_path": str(_files_dir(runtime, user_id) / "ordinary.md"),
                "recursive": False,
            },
        )
        assert resp.status_code == 404, (
            f"a conversation's index accepted a user-managed source: "
            f"{resp.status_code} {resp.text[:200]}"
        )
        assert "BODY" not in _text(runtime, auto_ctx)

    def test_an_ordinary_context_still_accepts_sources(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        context_id = _context(client, headers)
        assert _upload(
            client, headers, "corpus.md", b"# corpus\nTHE CORPUS TEXT\n" * 12
        ).status_code == 200

        resp = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={
                "fs_path": str(_files_dir(runtime, user_id) / "corpus.md"),
                "recursive": False,
            },
        )
        assert resp.status_code in (200, 201), resp.text
        assert "THE CORPUS TEXT" in _text(runtime, context_id)


def _bundle_zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buf.getvalue()


class TestARefusedExtractionPublishesNothing:
    """The upload ordering defect, in the archive route.

    `extract_archive_sandboxed` runs and *then* the named context is
    checked, so a request refused for naming a context that does not exist
    has already published the whole tree. Worse than the upload case: a
    retry without the bad parameter gets 409, because the destination the
    refused request created is now in the way.
    """

    def _publish_archive(self, client, headers):
        resp = _upload(
            client,
            headers,
            "bundle.zip",
            _bundle_zip({"a.txt": b"THE MEMBER TEXT\n" * 20}),
            media="application/zip",
        )
        assert resp.status_code == 200, resp.text

    def test_an_unknown_context_extracts_nothing(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        self._publish_archive(client, headers)

        resp = client.post(
            f"/v1/files/bundle.zip/extract?context_id={uuid.uuid4()}",
            headers=headers,
        )
        assert resp.status_code == 404, f"{resp.status_code}: {resp.text[:200]}"
        assert not (_files_dir(runtime, user_id) / "bundle").exists(), (
            "a request refused for naming an unknown context published the "
            "whole tree first"
        )

        # And the destination is still free, so the corrected request works.
        context_id = _context(client, headers)
        resp = client.post(
            f"/v1/files/bundle.zip/extract?context_id={context_id}", headers=headers
        )
        assert resp.status_code == 200, (
            f"the refused request left its destination behind: {resp.text[:200]}"
        )
        assert "THE MEMBER TEXT" in _text(runtime, context_id)

    def test_another_users_context_extracts_nothing(self, client):
        runtime = get_runtime()
        user_id, headers = _account(client)
        _other, other_headers = _account(client)
        theirs = _context(client, other_headers)
        self._publish_archive(client, headers)

        resp = client.post(
            f"/v1/files/bundle.zip/extract?context_id={theirs}", headers=headers
        )
        assert resp.status_code == 403, f"{resp.status_code}: {resp.text[:200]}"
        assert not (_files_dir(runtime, user_id) / "bundle").exists(), (
            "a request refused for naming another account's context still "
            "published the tree"
        )


class TestAnExtractedTreeIsInvisibleUntilItIsComplete:
    """`_write_member` opens each member at its final path and streams into it.

    The destination directory exists under its real name from the first
    member onward, so a download of a member still being written returns a
    truncated file with a content-length taken from the descriptor at open
    time — the client gets a short file and no reason to doubt it. This is
    the same harm the staged upload and the linked artifact removed, one
    level up, and it takes the same answer: fill something nobody can see,
    then make it visible in one step.

    Whole-tree staging rather than one temporary file per member, because
    the unit that has to appear at once is the tree: a listing that shows
    half a bundle is describing something that never existed.
    """

    def _publish_archive(self, client, headers, size: int) -> bytes:
        # Incompressible on purpose: a run of one byte compresses past the
        # extractor's 100:1 ratio guard and is refused as a bomb.
        member = os.urandom(size)
        resp = _upload(
            client,
            headers,
            "bundle.zip",
            _bundle_zip({"big.txt": member}),
            media="application/zip",
        )
        assert resp.status_code == 200, resp.text
        return member

    def test_no_member_is_reachable_while_the_tree_is_being_written(
        self, client, monkeypatch
    ):
        """Observed from inside the extraction, which is the only moment a
        member exists half-written.

        The extractor is replaced by one that writes a partial member, waits,
        and then finishes. What it writes into is whatever the route hands
        it, so the substitution changes nothing about the property under
        test: the route decides where members land before they are whole.
        """
        import threading

        from liminallm.api import routes

        runtime = get_runtime()
        user_id, headers = _account(client)
        member = self._publish_archive(client, headers, 400_000)

        reached = threading.Event()
        may_continue = threading.Event()
        seen: dict = {}

        def partial_then_whole(archive, destination, limits, *args, **kwargs):
            dest = Path(destination)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "big.txt").write_bytes(member[:64_000])
            reached.set()
            may_continue.wait(30)
            (dest / "big.txt").write_bytes(member)
            return {
                "extracted": ["big.txt"],
                "skipped": [],
                "total_bytes": len(member),
            }

        monkeypatch.setattr(
            routes, "extract_archive_sandboxed", partial_then_whole
        )
        results: dict = {}

        def extract():
            results["ex"] = client.post(
                "/v1/files/bundle.zip/extract", headers=headers
            )

        worker = threading.Thread(target=extract, daemon=True)
        worker.start()
        try:
            assert reached.wait(30), "the extraction never started"
            seen["listing"] = {
                f["name"]
                for f in client.get("/v1/files", headers=headers).json()["data"]["files"]
            }
            seen["url"] = client.get(
                "/v1/files/bundle/big.txt/url", headers=headers
            ).status_code
            seen["on_disk"] = (_files_dir(runtime, user_id) / "bundle").exists()
        finally:
            may_continue.set()
            worker.join(90)

        assert results["ex"].status_code == 200, results["ex"].text
        assert seen["url"] == 404, (
            f"a member of an unfinished tree was signable: {seen['url']}"
        )
        assert not [n for n in seen["listing"] if n.startswith("bundle/")], (
            f"an unfinished tree was listed: {sorted(seen['listing'])}"
        )
        assert not seen["on_disk"], (
            "the destination existed under its real name while it was still "
            "being filled"
        )

        # And it is whole once the request returns.
        resp = client.get("/v1/files/bundle/big.txt/url", headers=headers)
        assert resp.status_code == 200, resp.text
        assert (
            _files_dir(runtime, user_id) / "bundle" / "big.txt"
        ).read_bytes() == member

    def test_the_staging_area_is_not_inside_the_users_files(self, client):
        """It cannot be staged as a hidden sibling of the destination.

        `ingest_path` walks `**/*` and does not skip hidden components, so a
        context source covering `files/` would discover a half-written member
        under `files/.bundle-xxx.part/`. The staging root is outside every
        user's path authority instead.
        """
        runtime = get_runtime()
        user_id, headers = _account(client)
        self._publish_archive(client, headers, 2_000)
        resp = client.post("/v1/files/bundle.zip/extract", headers=headers)
        assert resp.status_code == 200, resp.text

        files_dir = _files_dir(runtime, user_id)
        leftovers = [p.name for p in files_dir.iterdir() if p.name.startswith(".")]
        assert leftovers == [".checksums.json"], (
            f"extraction left staging state inside the user's files: {leftovers}"
        )
        assert sorted(p.name for p in (files_dir / "bundle").iterdir()) == ["big.txt"]

    def test_stale_staging_is_reclaimed_but_a_live_one_is_left(self, tmp_path):
        """A staging tree outlives its extraction only if the process died.

        Nothing reads these directories, so age is the only signal there is —
        and the only one needed, since a finished extraction removes its own.
        """
        from liminallm.app import _sweep_archive_staging

        root = tmp_path / ".archive-staging" / "someone"
        root.mkdir(parents=True)
        stale, live = root / "old", root / "new"
        for tree in (stale, live):
            tree.mkdir()
            (tree / "member.txt").write_bytes(b"partial")
        stamp = stale.stat().st_mtime - 10_000
        os.utime(stale, (stamp, stamp))

        _sweep_archive_staging(tmp_path, max_age_hours=1)

        assert not stale.exists(), "a staging tree from a dead process was kept"
        assert live.exists(), "a staging tree still being filled was removed"
