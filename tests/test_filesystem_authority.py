"""Where a filesystem path gets its authority from.

SPEC §18 states the rule in one sentence: paths resolve through
`safe_join(base=/users/{user_id}, relative)` **unless** an artifact whose
persisted visibility is `shared` or `global` points into `/shared`. Two halves,
and only the first was implemented.

`POST /contexts/{id}/sources` accepted any absolute path underneath
`shared_fs_root/shared` because it was underneath that directory, and then
checked that the *destination context* belonged to the caller. That is the
wrong question asked of the wrong object: it establishes who receives the
content, never who was entitled to the source. Knowing a pathname became the
whole of the authority, which is exactly what the artifact row exists to stop.

So the tests below are about provenance, not syntax. A path is permitted
because something persisted says the caller may have it, or it is refused —
and an unprovable claim is refused, the same rule the workflow permission
model already follows.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from liminallm.service.fs import PathTraversalError, safe_join


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def tenants(runtime):
    """Two tenants, with two users inside the first.

    A test that used one tenant would pass against code that never compares
    tenants at all, which is the defect this file exists to catch one level up.
    """
    store = runtime.store
    left, right = _unique("tenant_a"), _unique("tenant_b")
    return {
        "owner": store.create_user(email=f"{_unique('a1')}@t.local", tenant_id=left),
        "colleague": store.create_user(
            email=f"{_unique('a2')}@t.local", tenant_id=left
        ),
        "outsider": store.create_user(
            email=f"{_unique('b1')}@t.local", tenant_id=right
        ),
    }


@pytest.fixture
def shared_object(runtime):
    """A real file under `shared_fs_root/shared`, with no artifact behind it.

    Nothing in the app writes here — `/shared` is populated by an operator or
    by an artifact's `fs_path` — so the fixture puts the bytes on disk exactly
    as a deployment would, and each test decides what provenance to give it.
    """
    root = Path(runtime.settings.shared_fs_root) / "shared" / _unique("corpus")
    root.mkdir(parents=True, exist_ok=True)
    document = root / "notes.md"
    document.write_text("turbine maintenance intervals are quarterly\n")
    return document


def _authorize(runtime, path, *, user):
    """The one predicate every path-consuming route is supposed to ask."""
    from liminallm.service.fs import authorize_path

    return authorize_path(
        runtime.store,
        runtime.settings,
        str(path),
        user_id=user.id,
        tenant_id=user.tenant_id,
    )


def _artifact(runtime, path, *, owner, visibility):
    """An artifact row whose `fs_path` names `path`.

    Written with SQL because `create_artifact` computes `fs_path` itself — it
    is where the artifact's own payload was persisted, always under
    `artifacts/{id}`. That is the finding underneath these tests: SPEC §18
    justifies `/shared` access with "an artifact whose visibility is shared or
    global points into `/shared`", and no code path produces such a row. The
    predicate is still the right one, and it is what these rows exercise;
    minting them through an API is separate work (see docs/ISSUES.md).
    """
    artifact = runtime.store.create_artifact(
        "tool",
        _unique("corpus"),
        {"kind": "tool.spec", "name": _unique("c"), "handler": "notes.search_v1"},
        owner_user_id=owner.id if owner else None,
        visibility=visibility,
    )
    with runtime.store.pool.connection() as conn:
        conn.execute(
            "UPDATE artifact SET fs_path = %s WHERE id = %s", (str(path), artifact.id)
        )
        conn.commit()
    return runtime.store.get_artifact(artifact.id)


# ---------------------------------------------------------------------------
# the hole: a pathname under /shared was the whole of the authority


class TestSharedNeedsAnArtifactNotAPathname:
    def test_a_shared_path_with_no_artifact_is_refused(
        self, runtime, tenants, shared_object
    ):
        """The red. Knowing the pathname was enough."""
        with pytest.raises(PermissionError):
            _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_a_global_artifact_authorizes_it(self, runtime, tenants, shared_object):
        _artifact(runtime, shared_object, owner=tenants["owner"], visibility="global")
        assert _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_a_shared_artifact_authorizes_it_within_one_tenant(
        self, runtime, tenants, shared_object
    ):
        _artifact(runtime, shared_object, owner=tenants["owner"], visibility="shared")
        assert _authorize(runtime, shared_object, user=tenants["colleague"])

    def test_a_shared_artifact_does_not_cross_tenants(
        self, runtime, tenants, shared_object
    ):
        _artifact(runtime, shared_object, owner=tenants["owner"], visibility="shared")
        with pytest.raises(PermissionError):
            _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_a_private_artifact_authorizes_nobody_else(
        self, runtime, tenants, shared_object
    ):
        _artifact(runtime, shared_object, owner=tenants["owner"], visibility="private")
        with pytest.raises(PermissionError):
            _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_a_private_artifact_confers_no_path_authority_at_all(
        self, runtime, tenants, shared_object
    ):
        """Not even to its own owner.

        §18 names two sources of filesystem authority and `private` is not one
        of them: the caller's authority is their `/users/{id}` root, and it is
        already exhausted there. Letting a private artifact confer path
        authority as well means an artifact row can widen a caller's reach
        beyond their own area, which the rule does not permit.
        """
        _artifact(runtime, shared_object, owner=tenants["owner"], visibility="private")
        with pytest.raises(PermissionError):
            _authorize(runtime, shared_object, user=tenants["owner"])

    def test_an_ownerless_shared_artifact_has_no_tenant_to_match(
        self, runtime, tenants, shared_object
    ):
        """`None` is the absence of the answer, never a wildcard."""
        _artifact(runtime, shared_object, owner=None, visibility="shared")
        with pytest.raises(PermissionError):
            _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_an_unrecognized_visibility_is_not_a_licence(
        self, runtime, tenants, shared_object
    ):
        _artifact(runtime, shared_object, owner=tenants["owner"], visibility="public")
        with pytest.raises(PermissionError):
            _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_a_caller_with_no_tenant_cannot_claim_a_shared_one(
        self, runtime, shared_object
    ):
        """`app_user.tenant_id` is NOT NULL, so this is not a user row without
        a tenant — it is a principal whose tenant did not resolve. `None` is
        the absence of the answer, and it must not match one."""
        from liminallm.service.fs import authorize_path

        store = runtime.store
        owner = store.create_user(email=f"{_unique('o')}@t.local", tenant_id="t-real")
        _artifact(runtime, shared_object, owner=owner, visibility="shared")
        with pytest.raises(PermissionError):
            authorize_path(
                store,
                runtime.settings,
                str(shared_object),
                user_id=owner.id,
                tenant_id=None,
            )

    def test_an_artifact_covers_the_tree_it_names(
        self, runtime, tenants, shared_object
    ):
        """A corpus is a directory, and its files are what get ingested."""
        _artifact(
            runtime, shared_object.parent, owner=tenants["owner"], visibility="global"
        )
        assert _authorize(runtime, shared_object, user=tenants["outsider"])

    def test_an_artifact_does_not_cover_a_name_it_is_a_prefix_of(
        self, runtime, tenants, shared_object
    ):
        """`/shared/corpus` must not answer for `/shared/corpus-2`.

        Coverage is an ancestor relation between paths, not a prefix relation
        between strings — and the cheap way to implement it (`LIKE fs_path ||
        '%'`) gets exactly this case wrong while passing every other test here.
        """
        directory = shared_object.parent
        neighbour = directory.parent / f"{directory.name}-2"
        neighbour.mkdir(parents=True, exist_ok=True)
        secret = neighbour / "payroll.md"
        secret.write_text("salaries\n")
        _artifact(runtime, directory, owner=tenants["owner"], visibility="global")

        with pytest.raises(PermissionError):
            _authorize(runtime, secret, user=tenants["outsider"])

    def test_an_artifact_does_not_cover_its_siblings(
        self, runtime, tenants, shared_object
    ):
        """Naming one directory does not name the one beside it."""
        sibling = shared_object.parent.parent / _unique("other")
        sibling.mkdir(parents=True, exist_ok=True)
        secret = sibling / "payroll.md"
        secret.write_text("salaries\n")
        _artifact(
            runtime, shared_object.parent, owner=tenants["owner"], visibility="global"
        )
        with pytest.raises(PermissionError):
            _authorize(runtime, secret, user=tenants["outsider"])


class TestTheExceptionIsForSharedAndNowhereElse:
    """§18 states the exception with a destination in it.

    "`artifact.visibility in ('shared','global')` **points into `/shared`**" —
    so an artifact is not a general-purpose grant that happens to name a path.
    Searching artifacts for any candidate under `shared_fs_root` made it one:
    a row covering `artifacts/{id}/v1.json`, or covering another user's files,
    would confer authority over paths §18 never opened.
    """

    @pytest.fixture
    def outside_shared(self, runtime, tenants):
        """A real path under the root but outside `/shared` — another user's
        file, which is the version of this that matters."""
        victim = tenants["colleague"]
        files = (
            Path(runtime.settings.shared_fs_root) / "users" / victim.id / "files"
        )
        files.mkdir(parents=True, exist_ok=True)
        document = files / "private.md"
        document.write_text("not yours\n")
        return document

    def test_a_shared_artifact_cannot_reach_outside_shared(
        self, runtime, tenants, outside_shared
    ):
        """Asked as `owner`, not as `colleague`: `colleague` owns that
        directory and would be granted it by the ownership rule, which would
        make this test pass without the artifact rule doing anything. `owner`
        is in the same tenant — so the artifact would grant it — and owns
        nothing here.
        """
        _artifact(
            runtime, outside_shared, owner=tenants["owner"], visibility="shared"
        )
        with pytest.raises(PermissionError):
            _authorize(runtime, outside_shared, user=tenants["owner"])

    def test_a_global_artifact_cannot_reach_outside_shared(
        self, runtime, tenants, outside_shared
    ):
        _artifact(
            runtime, outside_shared, owner=tenants["owner"], visibility="global"
        )
        with pytest.raises(PermissionError):
            _authorize(runtime, outside_shared, user=tenants["outsider"])

    def test_a_global_artifact_cannot_reach_the_artifact_store(
        self, runtime, tenants
    ):
        """The one path an artifact really does own — its own payload — is
        still not `/shared`, so it is still not this exception."""
        payload = (
            Path(runtime.settings.shared_fs_root)
            / "artifacts"
            / _unique("a")
            / "v1.json"
        )
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text("{}")
        _artifact(runtime, payload, owner=tenants["owner"], visibility="global")
        with pytest.raises(PermissionError):
            _authorize(runtime, payload, user=tenants["outsider"])

    def test_the_victims_own_access_is_unaffected(self, runtime, tenants, outside_shared):
        """The refusals above must come from the artifact rule, not from
        breaking ordinary ownership."""
        assert _authorize(runtime, outside_shared, user=tenants["colleague"])


# ---------------------------------------------------------------------------
# the half that did work, restated so a rewrite cannot lose it


class TestAUsersOwnAreaNeedsNoArtifact:
    def test_a_path_in_my_own_area_is_mine(self, runtime, tenants):
        user = tenants["owner"]
        mine = Path(runtime.settings.shared_fs_root) / "users" / user.id / "files"
        mine.mkdir(parents=True, exist_ok=True)
        document = mine / "notes.txt"
        document.write_text("mine\n")
        assert _authorize(runtime, document, user=user)

    def test_another_users_area_is_not(self, runtime, tenants):
        other = tenants["colleague"]
        theirs = Path(runtime.settings.shared_fs_root) / "users" / other.id / "files"
        theirs.mkdir(parents=True, exist_ok=True)
        document = theirs / "notes.txt"
        document.write_text("theirs\n")
        with pytest.raises(PermissionError):
            _authorize(runtime, document, user=tenants["owner"])

    def test_a_path_outside_the_root_entirely_is_refused(self, runtime, tenants):
        with pytest.raises(PermissionError):
            _authorize(runtime, "/etc/passwd", user=tenants["owner"])

    def test_traversal_out_of_my_own_area_is_refused(self, runtime, tenants):
        other = tenants["colleague"]
        with pytest.raises(PermissionError):
            _authorize(
                runtime,
                f"../{other.id}/files/notes.txt",
                user=tenants["owner"],
            )


class TestResolutionIsWhatCounts:
    """A name that lands somewhere else is that somewhere else.

    `..` is the case everyone tests. A symlink is the same escape written so
    the string looks innocent, and the check has to be about where the path
    resolves rather than how it reads.
    """

    def test_a_symlink_out_of_my_area_is_refused(self, runtime, tenants):
        mine = Path(runtime.settings.shared_fs_root) / "users" / tenants["owner"].id
        theirs = (
            Path(runtime.settings.shared_fs_root)
            / "users"
            / tenants["colleague"].id
            / "files"
        )
        mine.mkdir(parents=True, exist_ok=True)
        theirs.mkdir(parents=True, exist_ok=True)
        (theirs / "secret.txt").write_text("not yours\n")
        link = mine / "shortcut"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(theirs, link)

        with pytest.raises(PermissionError):
            _authorize(runtime, link / "secret.txt", user=tenants["owner"])

    def test_a_symlink_out_of_shared_is_refused(
        self, runtime, tenants, shared_object
    ):
        """An artifact naming a directory does not authorize wherever a link
        inside it happens to point."""
        theirs = (
            Path(runtime.settings.shared_fs_root)
            / "users"
            / tenants["colleague"].id
            / "files"
        )
        theirs.mkdir(parents=True, exist_ok=True)
        (theirs / "secret.txt").write_text("not yours\n")
        link = shared_object.parent / "shortcut"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(theirs, link)
        _artifact(
            runtime, shared_object.parent, owner=tenants["owner"], visibility="global"
        )

        with pytest.raises(PermissionError):
            _authorize(runtime, link / "secret.txt", user=tenants["outsider"])

    def test_safe_join_already_resolves_symlinks(self, tmp_path):
        """The primitive the rest of this relies on, stated as a test rather
        than as an assumption about `Path.resolve`."""
        base = tmp_path / "base"
        outside = tmp_path / "outside"
        base.mkdir()
        outside.mkdir()
        (outside / "x").write_text("x")
        os.symlink(outside, base / "link")
        with pytest.raises(PathTraversalError):
            safe_join(base, "link/x")


# ---------------------------------------------------------------------------
# and the route that had the hole


class TestTheContextSourceRoute:
    def _account(self, client, runtime):
        """A signed-up user and their headers, plus the row behind them."""
        email = f"{_unique('fs')}@example.com"
        resp = client.post(
            "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        headers = {"Authorization": f"Bearer {data['access_token']}"}
        return runtime.store.get_user(data["user_id"]), headers

    def _context(self, client, headers):
        created = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": _unique("ctx"), "description": "fixture"},
        )
        assert created.status_code in (200, 201), created.text
        return created.json()["data"]["id"]

    def test_a_stranger_cannot_point_a_context_at_shared_material(
        self, client, runtime, shared_object
    ):
        """The end-to-end form of the red: owning the destination is not
        authority over the source."""
        _user, headers = self._account(client, runtime)
        context_id = self._context(client, headers)

        resp = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": str(shared_object)},
        )
        assert resp.status_code == 403, resp.text
        assert runtime.store.list_context_sources(context_id) == []

    def test_an_authorized_shared_source_is_accepted(
        self, client, runtime, shared_object
    ):
        user, headers = self._account(client, runtime)
        _artifact(runtime, shared_object, owner=user, visibility="global")
        context_id = self._context(client, headers)

        resp = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": str(shared_object)},
        )
        assert resp.status_code == 201, resp.text
        assert runtime.store.list_context_sources(context_id)

    def test_a_relative_path_is_never_tried_against_shared(
        self, client, runtime, shared_object
    ):
        """The fallback chain is what made a name a licence.

        The route tried the caller's area, then `/shared`, then absolute forms
        under either. A relative name that means nothing in my own files must
        not quietly become a name in someone else's.
        """
        user, headers = self._account(client, runtime)
        context_id = self._context(client, headers)
        relative = str(
            shared_object.relative_to(Path(runtime.settings.shared_fs_root) / "shared")
        )

        resp = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": relative},
        )
        # Where it *resolved* is the property; a relative name means the
        # caller's own area whether or not anything is there, and must never
        # reach into `/shared` because the same name happens to exist in it.
        if resp.status_code == 201:
            stored = resp.json()["data"]["fs_path"]
            mine = str(
                Path(runtime.settings.shared_fs_root) / "users" / user.id
            )
            assert stored.startswith(mine), stored
            assert str(shared_object) != stored
        else:
            assert resp.status_code in (400, 403, 404), resp.text

    def test_a_users_own_file_still_works(self, client, runtime):
        """The ordinary path must survive the fix, or the refusal above is
        just a broken route."""
        user, headers = self._account(client, runtime)
        upload = client.post(
            "/v1/files/upload",
            headers=headers,
            files={"file": ("notes.md", b"turbine notes\n", "text/markdown")},
        )
        assert upload.status_code == 200, upload.text
        context_id = self._context(client, headers)

        resp = client.post(
            f"/v1/contexts/{context_id}/sources",
            headers=headers,
            json={"fs_path": "notes.md"},
        )
        assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------------------
# the census: every route that takes a path takes it against its own caller


class TestEveryPathRouteIsRootedAtItsCaller:
    """One property, asked of each surface that accepts a filename.

    The safe shape is that the base comes from the authenticated principal and
    the caller supplies only the leaf, so `safe_join` decides. These tests do
    not assume that: each one puts a real file in user A's area and has user B
    name it, by relative name and by absolute path, and asserts B does not get
    it. A route that grew a `/shared` fallback, or interpolated the caller's
    name from the request, fails here rather than in production.
    """

    def _account(self, client, runtime):
        email = f"{_unique('census')}@example.com"
        resp = client.post(
            "/v1/auth/signup", json={"email": email, "password": "TestPassword123!"}
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        return (
            runtime.store.get_user(data["user_id"]),
            {"Authorization": f"Bearer {data['access_token']}"},
        )

    @pytest.fixture
    def victim_file(self, client, runtime):
        """A real uploaded file belonging to somebody else."""
        user, headers = self._account(client, runtime)
        resp = client.post(
            "/v1/files/upload",
            headers=headers,
            files={"file": ("private.md", b"another user's notes\n", "text/markdown")},
        )
        assert resp.status_code == 200, resp.text
        path = (
            Path(runtime.settings.shared_fs_root)
            / "users"
            / user.id
            / "files"
            / "private.md"
        )
        assert path.is_file()
        return user, path

    def _names(self, victim_path):
        """The two ways to name someone else's file."""
        return [str(victim_path), f"../../{victim_path.parent.parent.name}/files/private.md"]

    def test_file_download_url_refuses_another_users_file(
        self, client, runtime, victim_file
    ):
        _victim, path = victim_file
        _user, headers = self._account(client, runtime)
        for name in self._names(path):
            resp = client.get(f"/v1/files/{name}/url", headers=headers)
            assert resp.status_code in (400, 403, 404), (name, resp.status_code)

    def test_file_delete_refuses_another_users_file(
        self, client, runtime, victim_file
    ):
        _victim, path = victim_file
        _user, headers = self._account(client, runtime)
        for name in self._names(path):
            resp = client.delete(f"/v1/files/{name}", headers=headers)
            assert resp.status_code in (400, 403, 404), (name, resp.status_code)
        assert path.is_file(), "another user's file was deleted"

    def test_note_from_file_refuses_another_users_file(
        self, client, runtime, victim_file
    ):
        """§19.5: joining the permanent vault is a deliberate act by the owner,
        not by whoever can spell the filename."""
        _victim, path = victim_file
        _user, headers = self._account(client, runtime)
        for name in self._names(path):
            resp = client.post(
                "/v1/notes/from-file", headers=headers, json={"name": name}
            )
            assert resp.status_code in (400, 403, 404), (name, resp.status_code)

    def test_extract_refuses_another_users_file(self, client, runtime, victim_file):
        _victim, path = victim_file
        _user, headers = self._account(client, runtime)
        for name in self._names(path):
            resp = client.post(f"/v1/files/{name}/extract", headers=headers)
            assert resp.status_code in (400, 403, 404), (name, resp.status_code)

    def test_context_sources_refuses_another_users_file(
        self, client, runtime, victim_file
    ):
        _victim, path = victim_file
        _user, headers = self._account(client, runtime)
        created = client.post(
            "/v1/contexts",
            headers=headers,
            json={"name": _unique("ctx"), "description": "census"},
        )
        context_id = created.json()["data"]["id"]
        for name in self._names(path):
            resp = client.post(
                f"/v1/contexts/{context_id}/sources",
                headers=headers,
                json={"fs_path": name},
            )
            assert resp.status_code in (400, 403, 404), (name, resp.status_code)
        assert runtime.store.list_context_sources(context_id) == []
