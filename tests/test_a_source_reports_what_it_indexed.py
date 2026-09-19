"""Adding a source reports how much of it was indexed.

`POST /contexts/{id}/sources` answered 201 with a source record that looked
identical whether the path held a library or nothing at all. `ingest_path`
returns the number of chunks it created - and answers 0 for a path that does
not exist, logging `ingest_path_not_found` and returning - and the route
discarded that value. The failure was visible in the server log and nowhere
a caller could reach.

Measured: a mistyped relative path produced 201, a stored source row, and a
knowledge context reporting "0 chunks loaded", with no error anywhere.

Refusing is the wrong repair, and this file says so with a witness. A source
row is the statement "this context covers this path": `contexts_covering_path`
reads that table alone, deliberately, so that coverage survives the index
being cleaned up - and every upload consults it to decide which contexts a
new file belongs in. Covering a directory that is empty today is therefore a
normal thing to do, and deleting the row would break "point a context at my
files, then upload into it". So the count is reported instead.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def account(client, runtime):
    """A signed-up user, their headers, and their own files directory."""
    email = f"{_unique('src')}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    files = (
        Path(runtime.settings.shared_fs_root) / "users" / data["user_id"] / "files"
    )
    files.mkdir(parents=True, exist_ok=True)
    return headers, files


@pytest.fixture
def context(client, account):
    headers, _files = account
    created = client.post(
        "/v1/contexts",
        headers=headers,
        json={"name": _unique("ctx"), "description": "fixture"},
    )
    assert created.status_code in (200, 201), created.text
    return created.json()["data"]["id"]


def _add_source(client, headers, context_id, fs_path):
    return client.post(
        f"/v1/contexts/{context_id}/sources",
        headers=headers,
        json={"fs_path": fs_path, "recursive": True},
    )


class TestTheCountTellsTheCallerWhatHappened:
    def test_a_path_that_does_not_exist_reports_zero(
        self, client, runtime, account, context
    ):
        """The shape that produced a silently empty knowledge context.

        The path is well-formed and inside the caller's own area, so
        authorization passes. It simply is not there.
        """
        headers, _files = account

        resp = _add_source(client, headers, context, "files/nothing-here")

        assert resp.status_code == 201, resp.text
        assert resp.json()["data"]["chunk_count"] == 0, (
            "a path that indexed nothing was indistinguishable from one that "
            f"worked: {resp.text}"
        )

    def test_a_directory_holding_nothing_indexable_reports_zero(
        self, client, runtime, account, context
    ):
        """The directory is real and readable, and nothing in it is a
        document. The row is still coverage, so it is kept."""
        headers, files = account
        opaque = files / _unique("opaque")
        opaque.mkdir(parents=True, exist_ok=True)
        (opaque / "payload.bin").write_bytes(b"\x00\x01\x02\x03")

        resp = _add_source(client, headers, context, f"files/{opaque.name}")

        assert resp.status_code == 201, resp.text
        assert resp.json()["data"]["chunk_count"] == 0, resp.text
        assert runtime.store.list_context_sources(context), (
            "the coverage row was dropped; a later upload here would not be "
            "indexed into this context"
        )


class TestCoveringAnEmptyDirectoryStillCatchesLaterUploads:
    """The reason this endpoint must not refuse a zero count.

    Refusing looked right until this was run: deleting the source row deletes
    the coverage, and `contexts_covering_path` is what every upload consults
    to decide which contexts a new file belongs in. A fresh account's
    `files/` is empty, so "cover my files, then upload into them" is exactly
    the sequence a refusal breaks.
    """

    def test_the_context_still_covers_its_files_directory(
        self, client, runtime, account, context
    ):
        headers, files = account
        user_id = files.parent.name

        covered = _add_source(client, headers, context, "files")
        assert covered.status_code == 201, covered.text
        assert covered.json()["data"]["chunk_count"] == 0, (
            "the fixture wanted an empty directory, so this proves nothing"
        )

        later = f"{_unique('later')}.md"
        uploaded = client.post(
            "/v1/files/upload",
            headers=headers,
            files={
                "file": (
                    later,
                    b"Datacenters are few, fixed, and already counted.\n",
                    "text/markdown",
                )
            },
        )
        assert uploaded.status_code == 200, uploaded.text

        covering = runtime.store.contexts_covering_path(
            str(files / later), owner_user_id=user_id
        )
        assert context in covering, (
            "the context no longer covers its own files directory, so an "
            "upload into it is indexed nowhere"
        )


class TestReAddingAnEmptiedDocumentReportsTheLoss:
    """Re-adding a document that has since been emptied really does change
    something, and the count is what says so.

    `_commit_generation` replaces what a context says about a named path
    **including by nothing** - deliberate, and documented there: chunks
    claiming to be a file's contents must not outlive those contents. So the
    chunks go, and a caller reading only the 201 would not know. Reading
    `chunk_count` they do.
    """

    def test_the_count_drops_to_zero_with_the_chunks(
        self, client, runtime, account, context
    ):
        headers, files = account
        document = files / f"{_unique('fading')}.md"
        document.write_text("Compute is the governable surface.\n")

        first = _add_source(client, headers, context, f"files/{document.name}")
        assert first.status_code == 201, first.text
        assert first.json()["data"]["chunk_count"] > 0, first.text

        document.write_text("   \n\t\n")
        second = _add_source(client, headers, context, f"files/{document.name}")

        assert second.status_code == 201, second.text
        assert second.json()["data"]["chunk_count"] == 0, second.text
        after = client.get(
            f"/v1/contexts/{context}/chunks?limit=20", headers=headers
        ).json()["data"]["items"]
        assert not after, (
            "the emptied generation was expected to replace the old chunks "
            f"and did not: {after}"
        )

    def test_a_missing_path_destroys_nothing(
        self, client, runtime, account, context
    ):
        """The control. A path that does not exist returns before
        `_commit_generation` is reached, so an unrelated generation in the
        same context survives."""
        headers, files = account
        document = files / f"{_unique('kept')}.md"
        document.write_text("Verification is the load-bearing problem.\n")
        assert _add_source(
            client, headers, context, f"files/{document.name}"
        ).status_code == 201

        missing = _add_source(client, headers, context, "files/never-existed")

        assert missing.status_code == 201, missing.text
        assert missing.json()["data"]["chunk_count"] == 0, missing.text
        kept = client.get(
            f"/v1/contexts/{context}/chunks?limit=20", headers=headers
        ).json()["data"]["items"]
        assert kept, "adding a missing path destroyed an unrelated generation"


class TestTextSuppliedAtContextCreationIsIndexedOrRefused:
    """The same shape one endpoint over.

    `POST /contexts` takes optional inline `text` and hands it to
    `ingest_text`, discarding the count that comes back. `ingest_text`
    answers 0 for text that is blank once stripped, so `text: "   "` created
    a context, indexed nothing, and said nothing about it.

    What makes it a lie rather than a quirk is that `text: ""` is already
    treated as no text at all - the route's `if body.text:` skips ingestion.
    So a caller who sent whitespace got the one outcome the endpoint does
    not have a word for: text supplied, text accepted, text nowhere.
    """

    def test_whitespace_only_text_is_refused(self, client, account):
        headers, _files = account

        resp = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": _unique("blank"), "description": "d", "text": "   \n\t "},
        )

        assert resp.status_code == 400, (
            f"whitespace was accepted as indexable text: {resp.text}"
        )
        message = (resp.json().get("error") or {}).get("message", "")
        assert message, f"no message for the caller: {resp.text}"

    def test_no_text_at_all_is_still_fine(self, client, account):
        """The control. Most contexts are created with no inline text, and
        `text: ""` already means the same thing - neither may start failing."""
        headers, _files = account

        for value in (None, ""):
            resp = client.post(
                "/v1/contexts",
                headers=headers,
                json={
                    "name": _unique("plain"),
                    "description": "d",
                    **({} if value is None else {"text": value}),
                },
            )
            assert resp.status_code in (200, 201), (
                f"text={value!r} was refused: {resp.text}"
            )

    def test_the_routes_predicate_matches_what_ingest_text_actually_does(
        self, client, runtime, account
    ):
        """The route decides before writing anything, so it cannot consult
        the count. That puts the same knowledge in two places, which is only
        safe while they agree - so this asserts they do, by running both.

        `ingest_text` has two early returns, one on a blank blob and one on
        an empty token list. The second is unreachable: every non-whitespace
        character is matched either by `\\w+` or by `[^\\w\\s]`, so a
        non-blank blob always tokenizes to something. That is why stripping
        is the whole predicate, and why this pins it rather than restating
        it.
        """
        headers, _files = account
        created = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": _unique("pin"), "description": "d"},
        )
        context_id = created.json()["data"]["id"]

        samples = [
            "",
            "   ",
            "\n\t  \n",
            " ",  # a non-breaking space is whitespace to str.strip
            ".",
            "...",
            "a",
            "-",
            "  real content  ",
        ]
        for text in samples:
            indexed = runtime.rag.ingest_text(context_id, text)
            assert bool(indexed) == bool(text.strip()), (
                f"the route refuses on {text!r} when text.strip() is falsy, "
                f"but ingest_text created {indexed} chunks from it"
            )

    def test_real_text_is_indexed(self, client, runtime, account):
        """The other control: text with content still reaches the index."""
        headers, _files = account

        resp = client.post(
            "/v1/contexts",
            headers=headers,
            json={
                "name": _unique("filled"),
                "description": "d",
                "text": "Verification is the load-bearing problem, not persuasion.",
            },
        )

        assert resp.status_code in (200, 201), resp.text
        context_id = resp.json()["data"]["id"]
        chunks = client.get(
            f"/v1/contexts/{context_id}/chunks?limit=5", headers=headers
        )
        assert chunks.json()["data"]["items"], "the supplied text was not indexed"


class TestARealSourceStillWorks:
    """The control. A check that refused everything would pass the class
    above and break the feature."""

    def test_a_readable_document_is_accepted_and_indexed(
        self, client, runtime, account, context
    ):
        headers, files = account
        corpus = files / _unique("corpus")
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "notes.md").write_text(
            "Frontier training runs need datacenters, which are few and fixed.\n"
        )

        resp = _add_source(client, headers, context, f"files/{corpus.name}")

        assert resp.status_code == 201, resp.text
        assert runtime.store.list_context_sources(context), (
            "the source that worked was not recorded"
        )
        chunks = client.get(f"/v1/contexts/{context}/chunks?limit=20", headers=headers)
        assert chunks.status_code == 200, chunks.text
        assert chunks.json()["data"]["items"], "nothing was indexed"

    def test_a_single_file_is_accepted(self, client, runtime, account, context):
        """A source may name one document rather than a tree, which takes a
        different branch in `ingest_path`."""
        headers, files = account
        document = files / f"{_unique('one')}.md"
        document.write_text("Compute is the governable surface.\n")

        resp = _add_source(client, headers, context, f"files/{document.name}")

        assert resp.status_code == 201, resp.text
        assert runtime.store.list_context_sources(context)
