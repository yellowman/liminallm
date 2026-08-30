"""A path the Files API calls internal must never become corpus content.

The Files API already draws this line and says so: any relative path with a
component beginning `.` is bookkeeping, omitted from listings, and treated as
absent by download and delete. `.checksums.json` - the upload manifest - is
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
        JSON is the same disclosure - the user's filenames and checksums in
        a retrieval - so the bytes are checked, not only the name.
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


class TestInternalEntriesDoNotStarveTheBudget:
    def test_a_tree_full_of_bookkeeping_still_indexes_the_document(self, corpus):
        """A property, not a reproduction - and worth stating either way.

        The walk stops after `max_files` documents. If internal entries
        counted against that, a directory carrying thousands of them would
        exhaust the budget before reaching anything a user wrote, and the
        turn would be grounded on nothing with no error to show for it.

        They do not count, and the reason is narrower than where the check
        sits: `files_processed` is incremented only after a successful
        ingest, so any path that `continue`s before that leaves the budget
        untouched wherever the test for it appears. What makes this hold is
        that the entries are refused at all, which is what the manifest case
        above already measures.
        """
        _headers, user_id, root = corpus
        for i in range(6):
            (root / f".bookkeeping-{i}.json").write_text('{"noise": %d}' % i)
        (root / "wrenfield.md").write_text(
            "Wrenfield rota: the bearing is seated by the night crew."
        )
        context = fresh_context(user_id)

        get_runtime().rag.ingest_path(
            context.id, str(root), recursive=True, max_files=1
        )

        paths = chunk_paths(context.id)
        assert any("wrenfield.md" in p or "ferrothorn.txt" in p for p in paths), (
            f"bookkeeping consumed the whole file budget: {paths}"
        )
        assert not any(".bookkeeping" in p for p in paths), paths


class TestTheApiAndTheWalkerAgree:
    def test_what_the_files_api_omits_is_what_ingestion_refuses(
        self, client, corpus
    ):
        """One predicate, asked twice.

        The listing and the walker answer the same question - is this path
        the user's content, or the server's bookkeeping - so they must not
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


class TestASourceBeneathAHiddenDirectoryIsStillInternal:
    """The basename is not the question; position under the namespace is.

    A source is classified by what it is called, and a path named outright
    arrives with its own basename in hand. `bundle/.internal/secret.md` has an
    ordinary basename and an internal position, so a check that asks only
    `path.name` admits it - while the very same file, reached by walking
    `files/`, is refused. One file, two answers.

    Reachable through the supported route: `POST /contexts/{id}/sources` calls
    `authorize_path`, which answers access and says nothing about content, and
    then hands the path to `ingest_path`.
    """

    @pytest.mark.parametrize("named", ["file", "directory"])
    def test_the_source_endpoint_indexes_nothing_internal(
        self, client, corpus, named
    ):
        headers, user_id, root = corpus
        hidden = root / "bundle" / ".internal"
        (hidden / "subdir").mkdir(parents=True, exist_ok=True)
        (hidden / "secret.md").write_text(
            "Internal only: the Marrowgate override code is 7731."
        )
        (hidden / "subdir" / "public-looking.md").write_text(
            "Internal only: the Redgrave bypass is 4409."
        )
        context = fresh_context(user_id)
        target = hidden / "secret.md" if named == "file" else hidden / "subdir"

        resp = client.post(
            f"/v1/contexts/{context.id}/sources",
            headers=headers,
            json={"fs_path": str(target), "recursive": True},
        )

        assert resp.status_code in (200, 201), resp.text
        assert not chunk_paths(context.id), (
            f"a source named beneath a hidden directory was indexed: "
            f"{chunk_paths(context.id)}"
        )
        bodies = " ".join(chunk_bodies(context.id))
        assert "Marrowgate" not in bodies and "Redgrave" not in bodies, bodies


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


class TestTheDurableQueueAsksTheSameQuestion:
    """The queue is not a caller of `ingest_path`, so it inherits none of it.

    Re-indexing calls `rag.ingest_file` directly. Every refusal added to the
    walk is therefore invisible here, and the queue is the durable machinery a
    replacement actually runs through - so an internal path that reaches
    `ingest_job` would be chunked on a schedule, long after whoever created it
    stopped watching.

    The job is closed rather than retried: nothing is owed now or later, so a
    terminal state with the reason is the honest record. A failure would be
    re-attempted five times to reach the same conclusion.
    """

    def test_a_queued_internal_path_indexes_nothing_and_is_closed(self, corpus):
        from liminallm.service import ingest_queue

        _headers, user_id, root = corpus
        runtime = get_runtime()
        context = fresh_context(user_id)
        internal = root / "bundle" / ".internal"
        internal.mkdir(parents=True, exist_ok=True)
        target = internal / "notes.md"
        target.write_text("Internal only: the Marrowgate override code is 7731.")

        # The generation the bytes actually have. A placeholder would be
        # declined as stale before ingestion was ever attempted, and the
        # "no chunks" assertion below would then pass for a reason that has
        # nothing to do with the path being internal.
        generation = ingest_queue.generation_of(target)
        assert generation, "the fixture file is not readable"
        runtime.store.enqueue_ingest_job(context.id, str(target), generation)
        attempted = ingest_queue.drain(
            runtime.store, runtime.rag, fs_root=str(runtime.settings.shared_fs_root)
        )

        assert attempted == 1, "the job was never attempted, so this proves nothing"
        assert not chunk_paths(context.id), (
            f"the queue indexed an internal path: {chunk_paths(context.id)}"
        )
        with runtime.store._connect() as conn:
            rows = conn.execute(
                "SELECT status, detail FROM ingest_job WHERE fs_path = %s",
                (str(target),),
            ).fetchall()
        assert len(rows) == 1, f"expected one job row, found {len(rows)}"
        status, detail = rows[0]["status"], rows[0]["detail"] or ""
        assert status not in ("queued", "running"), (
            f"the queue left work owed for an internal path: {status}"
        )
        assert "internal" in detail.lower(), (
            f"the reason it closed was not recorded: {detail!r}"
        )
