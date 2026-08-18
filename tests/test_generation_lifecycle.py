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
import time
import uuid
import zipfile
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

    What is closed here is the safety half: an attachment never resolves to
    bytes that are not the ones attached. Keeping the attached generation
    available after the path moves needs somewhere to keep it, which is the
    open design question.
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
        """`run_python` rebuilds its workdir from the same global names."""
        from liminallm.service import interpreter

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        assert self._attach(
            client, headers, conversation_id, "data.md", b"# data\nTHE ATTACHED ROWS\n"
        ).status_code == 200
        assert _upload(
            client, headers, "data.md", b"# data\nROWS FROM SOMEWHERE ELSE\n"
        ).status_code == 200

        from liminallm.service.attachments import list_attachments, resolved_names

        conversation = runtime.store.get_conversation(conversation_id, user_id=user_id)
        records = list_attachments(conversation)
        # The same list the tool builds, through the same helper.
        names = resolved_names(
            records, fs_root=runtime.settings.shared_fs_root, user_id=user_id
        )
        workdir = interpreter.prepare_workdir(
            str(Path(runtime.settings.shared_fs_root) / "scratch"),
            str(_files_dir(runtime, user_id)),
            names,
        )
        staged = " ".join(
            p.read_text(errors="replace") for p in Path(workdir).iterdir() if p.is_file()
        )
        assert "ROWS FROM SOMEWHERE ELSE" not in staged, (
            "the interpreter staged bytes that were never attached"
        )

    def test_the_prompt_does_not_promise_text_it_leaves_out(self, client):
        """The listing and the envelope have to agree.

        An inline attachment that no longer resolves is simply not in the
        envelope. The listing above it said "full text included below"
        regardless, so the model was told to read text that was not there.
        """
        from liminallm.service.attachments import (
            build_attachment_preamble,
            list_attachments,
        )

        runtime = get_runtime()
        user_id, headers = _account(client)
        conversation_id = self._conversation(client, headers)
        assert self._attach(
            client, headers, conversation_id, "brief.md", b"# brief\nTHE ATTACHED BRIEF\n"
        ).status_code == 200
        assert _upload(
            client, headers, "brief.md", b"# brief\nA LATER UNRELATED BRIEF\n"
        ).status_code == 200

        conversation = runtime.store.get_conversation(conversation_id, user_id=user_id)
        preamble = build_attachment_preamble(
            list_attachments(conversation),
            fs_root=runtime.settings.shared_fs_root,
            user_id=user_id,
        )
        assert "A LATER UNRELATED BRIEF" not in preamble
        assert "full text included below" not in preamble, (
            "the listing promised text the envelope does not contain"
        )
        assert "unavailable" in preamble, preamble
