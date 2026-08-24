"""A path the Files API calls internal must never become corpus content.

The Files API already draws this line and says so: any relative path with a
component beginning `.` is bookkeeping, omitted from listings, and treated as
absent by download and delete. `.checksums.json` — the upload manifest — is
the row that made the rule necessary.

Ingestion never learned it. The default extension list includes `.json`, so a
directory source rooted at a user's `files/` walked straight into the manifest
and chunked it: a retrieval could then answer out of a list of the user's own
filenames and their checksums, which is not a document anybody uploaded.

The rule is about *components*, not basenames. `bundle/.internal/secret.md` is
internal for the same reason `.checksums.json` is, and a fix that matched only
the manifest's name would leave that indexed.
"""

from __future__ import annotations

import json
import uuid

import pytest

from liminallm.service.fs import user_base
from liminallm.service.runtime import get_runtime

PASSWORD = "TestPassword123!"
DOCUMENT = (
    "Ferrothorn Bearing Notes. The bearing seats at 61 microns and is "
    "replaced every 900 running hours by the Wrenfield crew."
)


def account(client):
    email = f"hid_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": PASSWORD}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return {"Authorization": f"Bearer {data['access_token']}"}, data["user_id"]


def files_dir(user_id):
    runtime = get_runtime()
    return user_base(runtime.settings.shared_fs_root, user_id) / "files"


def upload(client, headers, name, body):
    """Through the real route, so the real manifest is written."""
    resp = client.post(
        "/v1/files/upload",
        headers=headers,
        files={"file": (name, body, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


def fresh_context(user_id):
    return get_runtime().store.upsert_context(
        name=f"corpus-{uuid.uuid4().hex[:6]}",
        description="internal-path witness",
        owner_user_id=user_id,
    )


def chunk_paths(context_id):
    return [c.fs_path or "" for c in get_runtime().store.list_chunks(context_id)]


def chunk_bodies(context_id):
    return [c.content or "" for c in get_runtime().store.list_chunks(context_id)]


@pytest.fixture
def corpus(client):
    """A real account whose files directory holds a real upload manifest."""
    headers, user_id = account(client)
    upload(client, headers, "ferrothorn.txt", DOCUMENT)
    root = files_dir(user_id)
    manifest = root / ".checksums.json"
    assert manifest.exists(), (
        "the fixture expected the upload route to write .checksums.json; "
        f"{root} holds {[p.name for p in root.iterdir()]}"
    )
    return headers, user_id, root


class TestTheManifestIsNotADocument:
    def test_a_directory_source_does_not_index_the_manifest(self, corpus):
        """The reported case, over the directory a user would actually name.

        `files/` is the obvious source to add: it is everything the user has
        uploaded. Walking it reached `.checksums.json`, whose suffix is in
        the default extension list.
        """
        _headers, user_id, root = corpus
        context = fresh_context(user_id)

        get_runtime().rag.ingest_path(context.id, str(root), recursive=True)

        assert any("ferrothorn.txt" in p for p in chunk_paths(context.id)), (
            f"the document itself was not indexed: {chunk_paths(context.id)}"
        )
        assert not any(".checksums.json" in p for p in chunk_paths(context.id)), (
            f"the manifest was indexed as content: {chunk_paths(context.id)}"
        )

    def test_no_chunk_carries_the_manifest_as_its_text(self, corpus):
        """Asserted on content as well as path, because either would leak.

        A chunk recorded under some other path but carrying the manifest's
        JSON is the same disclosure — the user's filenames and checksums in
        a retrieval — so the bytes are checked, not only the name.
        """
        _headers, user_id, root = corpus
        context = fresh_context(user_id)
        manifest_text = (root / ".checksums.json").read_text()
        recorded = next(iter(json.loads(manifest_text)), None)
        assert recorded, "the manifest is empty, so this case proves nothing"

        get_runtime().rag.ingest_path(context.id, str(root), recursive=True)

        assert not any("checksum" in body.lower() for body in chunk_bodies(context.id)), (
            f"manifest text reached a chunk: {chunk_bodies(context.id)}"
        )


class TestInternalIsAboutComponentsNotBasenames:
    def test_a_hidden_subtree_is_skipped_and_its_sibling_is_not(self, corpus):
        """Kills a fix that matches `.checksums.json` by name.

        A nested hidden directory is internal for the same reason the
        manifest is, and the rule the Files API already applies is about any
        component of the relative path.
        """
        _headers, user_id, root = corpus
        bundle = root / "bundle"
        (bundle / ".internal").mkdir(parents=True, exist_ok=True)
        (bundle / "public.md").write_text(
            "Public bundle note: the Wrenfield crew works nights."
        )
        (bundle / ".internal" / "secret.md").write_text(
            "Internal only: the Marrowgate override code is 7731."
        )
        context = fresh_context(user_id)

        get_runtime().rag.ingest_path(context.id, str(root), recursive=True)

        paths = chunk_paths(context.id)
        assert any("public.md" in p for p in paths), (
            f"the sibling content file was not indexed: {paths}"
        )
        assert not any("secret.md" in p for p in paths), (
            f"a file under a hidden directory was indexed: {paths}"
        )
        assert not any("Marrowgate" in b for b in chunk_bodies(context.id))


class TestTheApiAndTheWalkerAgree:
    def test_what_the_files_api_omits_is_what_ingestion_refuses(
        self, client, corpus
    ):
        """One predicate, asked twice.

        The listing and the walker answer the same question — is this path
        the user's content, or the server's bookkeeping — so they must not
        answer it differently. Driven through both surfaces over one tree
        rather than by reading the two implementations.
        """
        headers, user_id, root = corpus
        bundle = root / "bundle"
        (bundle / ".internal").mkdir(parents=True, exist_ok=True)
        (bundle / "public.md").write_text("Public bundle note.")
        (bundle / ".internal" / "secret.md").write_text("Internal only.")
        context = fresh_context(user_id)

        listed = {
            f["name"]
            for f in client.get("/v1/files", headers=headers).json()["data"]["files"]
        }
        get_runtime().rag.ingest_path(context.id, str(root), recursive=True)
        indexed = {
            p.split("/files/", 1)[-1] for p in chunk_paths(context.id) if p
        }

        assert listed, "the listing is empty, so this case compares nothing"
        assert indexed <= listed, (
            "ingestion indexed paths the Files API does not consider content: "
            f"{sorted(indexed - listed)}"
        )


class TestNamingAnInternalPathDirectlyIndexesNothing:
    def test_a_single_hidden_file_is_not_corpus(self, corpus):
        """The invariant is "never corpus", not "directory walks skip it".

        `authorize_path` grants authority over anything under the caller's own
        `users/{id}` directory and says nothing about bookkeeping, so a source
        naming `.checksums.json` outright passes authorization and reaches
        `ingest_path`'s single-file branch, which never walks a directory at
        all.
        """
        _headers, user_id, root = corpus
        context = fresh_context(user_id)

        written = get_runtime().rag.ingest_path(
            context.id, str(root / ".checksums.json"), recursive=False
        )

        assert written == 0, f"the manifest was ingested directly: {written} chunks"
        assert not chunk_paths(context.id), chunk_paths(context.id)
