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
