"""Who is allowed to run what - SPEC §18 and the artifact permission rule.

Five authority questions the source answered from the wrong place:

* **privileged provenance.** SPEC says `privileged:true` tools require
  *admin-owned artifacts*. The check asked only whether the caller is an
  admin. A `check_privileged_access` helper took an `artifact_owner_id`,
  never read it, and had no callers at all; it is deleted. Any authenticated
  user can create a `tool.spec` - `/v1/artifacts` depends on `get_user`, and
  the schema permits additional properties - so a user could author
  `privileged: true` and an admin invoking it would be granted the privileged
  sandbox for someone else's definition.

* **exact binding.** `POST /tools/{id}/invoke` authorized one artifact and
  handed the engine a bare `dict(schema)`, which resolved `schema["name"]`
  again. Names are not unique, so the row that ran need not be the row that
  was checked.

* **workflow ownership.** `/v1/chat` passes the caller's `workflow_id`
  straight into the engine, which loads it by artifact id with no owner or
  visibility check. A private workflow is an artifact like any other.

* **shared means one tenant.** The `shared` branch read a `tenant_id` off the
  artifact. `Artifact` has no such field, so the value was always `None` and
  the comparison accepted it as a match - every shared workflow was readable
  from every tenant.

* **registry scope.** Direct invocation did
  `self.tool_registry.setdefault(name, dict(schema))` on a process-global
  dict built from *globally visible* tool artifacts, so one user's private
  tool definition became resolvable for every later request in that process.

Today none of these grants arbitrary code execution: a tool artifact can only
name a built-in handler. They are authority-boundary defects, and the tranche
that gives workers brokered capabilities is exactly when they would stop
being latent.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from liminallm.service.sandbox import PrivilegedToolError, get_tool_sandbox_config
from liminallm.service.workflow import ToolDescriptor


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def runtime(client):
    from liminallm.service.runtime import get_runtime

    return get_runtime()


@pytest.fixture
def users(runtime):
    store = runtime.store
    owner = store.create_user(email=f"{_unique('owner')}@t.local")
    other = store.create_user(email=f"{_unique('other')}@t.local")
    admin = store.create_user(email=f"{_unique('admin')}@t.local")
    store.update_user_role(admin.id, "admin")
    return {"owner": owner, "other": other, "admin": store.get_user(admin.id)}


@pytest.fixture
def tenants(runtime):
    """Two users in two named tenants, plus a second user inside the first.

    `public` is every other fixture's tenant, so a test that only used it
    would pass against code that never compares tenants at all.
    """
    store = runtime.store
    left = _unique("tenant_a")
    right = _unique("tenant_b")
    return {
        "owner": store.create_user(email=f"{_unique('a1')}@t.local", tenant_id=left),
        "colleague": store.create_user(
            email=f"{_unique('a2')}@t.local", tenant_id=left
        ),
        "outsider": store.create_user(email=f"{_unique('b1')}@t.local", tenant_id=right),
    }


def _graph() -> dict:
    return {
        "kind": "workflow.graph",
        "entrypoint": "start",
        "nodes": [{"id": "start", "type": "tool_call", "tool": "llm.generic"}],
    }


def _assert_ran(result: dict) -> None:
    """A permitted invocation must reach its handler.

    Every "not forbidden" assertion here is also satisfied by `unknown tool`,
    which is what these tests returned while the specs named a handler
    (`note_search`) that is not a key in `_builtin_tool_handlers` - the real
    one is `notes.search_v1`. The refusals were still refusals, but the
    control proved nothing.
    """
    assert result.get("error") != "forbidden", result
    assert "unknown tool" not in (result.get("content") or ""), result


class TestPrivilegedToolsNeedAnAdminOwnedArtifact:
    def test_the_caller_is_only_half_the_question(self, runtime, users):
        """An ordinary user's `privileged: true` tool, invoked by an admin.

        The caller check passes - they really are an admin - and the artifact
        they are running was written by someone else. SPEC requires both.
        """
        store = runtime.store
        artifact = store.create_artifact(
            "tool",
            _unique("user_written"),
            {
                "kind": "tool.spec",
                "name": "notes.search_v1",
                "handler": "notes.search_v1",
                "privileged": True,
            },
            owner_user_id=users["owner"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "notes.search_v1",
                {"query": "x"},
                [],
                [],
                None,
                None,
                None,
                user_id=users["admin"].id,
                tenant_id=users["admin"].tenant_id,
                descriptor=runtime.workflow._describe_tool(artifact),
            )
        )
        assert result.get("error") == "forbidden", result

    def test_an_ordinary_caller_is_refused_on_an_admin_owned_tool(
        self, runtime, users
    ):
        store = runtime.store
        artifact = store.create_artifact(
            "tool",
            _unique("admin_written"),
            {
                "kind": "tool.spec",
                "name": "notes.search_v1",
                "handler": "notes.search_v1",
                "privileged": True,
            },
            owner_user_id=users["admin"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "notes.search_v1",
                {"query": "x"},
                [],
                [],
                None,
                None,
                None,
                user_id=users["other"].id,
                tenant_id=users["other"].tenant_id,
                descriptor=runtime.workflow._describe_tool(artifact),
            )
        )
        assert result.get("error") == "forbidden", result

    def test_admin_owned_and_admin_called_is_permitted(self, runtime, users):
        """The control: the rule is a conjunction, not a wall."""
        store = runtime.store
        artifact = store.create_artifact(
            "tool",
            _unique("admin_both"),
            {
                "kind": "tool.spec",
                "name": "notes.search_v1",
                "handler": "notes.search_v1",
                "privileged": True,
            },
            owner_user_id=users["admin"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "notes.search_v1",
                {"query": "x"},
                [],
                [],
                None,
                None,
                None,
                user_id=users["admin"].id,
                tenant_id=users["admin"].tenant_id,
                descriptor=runtime.workflow._describe_tool(artifact),
            )
        )
        _assert_ran(result)

    def test_schema_fields_cannot_supply_the_provenance(self, runtime, users):
        """`owner_user_id` copied into the schema is caller-controlled data.

        The persisted artifact row is the only thing that says who owns it.
        """
        store = runtime.store
        artifact = store.create_artifact(
            "tool",
            _unique("forged_owner"),
            {
                "kind": "tool.spec",
                "name": "notes.search_v1",
                "handler": "notes.search_v1",
                "privileged": True,
                "owner_user_id": users["admin"].id,  # claimed, not true
            },
            owner_user_id=users["owner"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "notes.search_v1",
                {"query": "x"},
                [],
                [],
                None,
                None,
                None,
                user_id=users["admin"].id,
                tenant_id=users["admin"].tenant_id,
                descriptor=runtime.workflow._describe_tool(artifact),
            )
        )
        assert result.get("error") == "forbidden", result

    def test_an_unverifiable_artifact_is_refused(self, runtime, users):
        """No artifact id means nothing to verify ownership against, and
        `privileged: true` from an unattributed spec is exactly the shape an
        injected registry entry would have."""
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "notes.search_v1",
                {"query": "x"},
                [],
                [],
                None,
                None,
                None,
                user_id=users["admin"].id,
                tenant_id=users["admin"].tenant_id,
                descriptor=ToolDescriptor(
                    name="notes.search_v1",
                    schema={
                        "kind": "tool.spec",
                        "name": "notes.search_v1",
                        "handler": "notes.search_v1",
                        "privileged": True,
                    },
                    artifact_id=None,
                    owner_user_id=None,
                    owner_role=None,
                ),
            )
        )
        assert result.get("error") == "forbidden", result


class TestTheSandboxConfigStillHoldsItsOwnLine:
    """`get_tool_sandbox_config` is reachable on its own, so it keeps the
    caller check - but it must no longer be the whole story."""

    def test_a_privileged_spec_still_needs_an_admin_caller(self):
        with pytest.raises(PrivilegedToolError):
            get_tool_sandbox_config({"privileged": True}, user_role="user")

    def test_an_admin_caller_gets_the_privileged_config(self):
        config = get_tool_sandbox_config({"privileged": True}, user_role="admin")
        assert config.privileged is True


class TestPrivateWorkflowsBelongToTheirOwner:
    def test_another_user_cannot_run_one_by_id(self, runtime, users):
        """`/v1/chat` takes `workflow_id` from the caller and the engine
        loaded it by artifact id alone."""
        store = runtime.store
        workflow = store.create_artifact(
            "workflow",
            _unique("private_flow"),
            {
                "kind": "workflow.graph",
                "entrypoint": "start",
                "nodes": [{"id": "start", "type": "tool_call", "tool": "llm.generic"}],
            },
            owner_user_id=users["owner"].id,
            visibility="private",
        )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=users["other"].id, tenant_id=users["other"].tenant_id
        )
        assert loaded is None, "another user's private workflow was loaded"

    def test_the_owner_can_run_it(self, runtime, users):
        store = runtime.store
        workflow = store.create_artifact(
            "workflow",
            _unique("own_flow"),
            {
                "kind": "workflow.graph",
                "entrypoint": "start",
                "nodes": [{"id": "start", "type": "tool_call", "tool": "llm.generic"}],
            },
            owner_user_id=users["owner"].id,
            visibility="private",
        )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=users["owner"].id, tenant_id=users["owner"].tenant_id
        )
        assert loaded is not None

    def test_the_store_refuses_to_answer_without_a_caller(self, runtime, users):
        """Grep the class: `get_latest_workflow` had two callers and neither
        passed an identity, so fixing one would have left the other as a way
        in. The rule lives in the store now, and `user_id` is a keyword with
        no default - a caller cannot omit the question by accident."""
        store = runtime.store
        workflow = store.create_artifact(
            "workflow",
            _unique("scoped"),
            {
                "kind": "workflow.graph",
                "entrypoint": "start",
                "nodes": [{"id": "start", "type": "tool_call", "tool": "llm.generic"}],
            },
            owner_user_id=users["owner"].id,
            visibility="private",
        )
        with pytest.raises(TypeError):
            store.get_latest_workflow(workflow.id)
        assert store.get_latest_workflow(workflow.id, user_id=users["other"].id) is None
        assert (
            store.get_latest_workflow(workflow.id, user_id=users["owner"].id)
            is not None
        )

    def test_a_global_workflow_is_available_to_anyone(self, runtime, users):
        store = runtime.store
        workflow = store.create_artifact(
            "workflow",
            _unique("shared_flow"),
            _graph(),
            owner_user_id=users["owner"].id,
            visibility="global",
        )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=users["other"].id, tenant_id=users["other"].tenant_id
        )
        assert loaded is not None


class TestSharedMeansSharedWithinOneTenant:
    """`Artifact` has no tenant column.

    The first fix read `getattr(artifact, "tenant_id", None)`, which was
    therefore always `None`, and then accepted the workflow when either side
    was `None` - so every `shared` workflow was readable from every tenant.
    The tenant of a shared artifact is its *owner's*; that is the only place
    it is recorded.
    """

    def _shared(self, store, owner):
        return store.create_artifact(
            "workflow", _unique("team_flow"), _graph(),
            owner_user_id=owner.id, visibility="shared",
        )

    def test_a_colleague_in_the_same_tenant_may_run_it(self, runtime, tenants):
        workflow = self._shared(runtime.store, tenants["owner"])
        loaded = runtime.workflow._load_workflow_for(
            workflow.id,
            user_id=tenants["colleague"].id,
            tenant_id=tenants["colleague"].tenant_id,
        )
        assert loaded is not None, "shared workflow denied inside its own tenant"

    def test_another_tenant_may_not(self, runtime, tenants):
        workflow = self._shared(runtime.store, tenants["owner"])
        loaded = runtime.workflow._load_workflow_for(
            workflow.id,
            user_id=tenants["outsider"].id,
            tenant_id=tenants["outsider"].tenant_id,
        )
        assert loaded is None, "shared workflow crossed a tenant boundary"

    def test_a_caller_with_no_tenant_may_not(self, runtime, tenants):
        """`None` is not a wildcard. It is the absence of the answer, and the
        `None in (...)` form treated it as a match against every tenant."""
        workflow = self._shared(runtime.store, tenants["owner"])
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=tenants["outsider"].id, tenant_id=None
        )
        assert loaded is None

    def test_an_ownerless_shared_workflow_has_no_tenant_to_match(
        self, runtime, tenants
    ):
        workflow = runtime.store.create_artifact(
            "workflow", _unique("orphan_shared"), _graph(),
            owner_user_id=None, visibility="shared",
        )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id,
            user_id=tenants["owner"].id,
            tenant_id=tenants["owner"].tenant_id,
        )
        assert loaded is None


class TestAnUnprovableClaimIsRefused:
    def test_an_ownerless_private_workflow_belongs_to_nobody(self, runtime, users):
        """The first form refused only when an owner was present and differed,
        so `owner_user_id IS NULL` fell through the check and served every
        caller."""
        workflow = runtime.store.create_artifact(
            "workflow", _unique("orphan_private"), _graph(),
            owner_user_id=None, visibility="private",
        )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=users["other"].id, tenant_id=users["other"].tenant_id
        )
        assert loaded is None

    def test_an_unrecognized_visibility_is_not_a_licence(self, runtime, users):
        """A visibility the code does not know about must fail closed. An
        if/elif chain that ends by falling through grants access to exactly
        the values nobody thought about."""
        workflow = runtime.store.create_artifact(
            "workflow", _unique("odd_vis"), _graph(),
            owner_user_id=users["owner"].id, visibility="private",
        )
        with runtime.store._connect() as conn:
            conn.execute(
                "UPDATE artifact SET visibility = %s WHERE id = %s",
                ("team", workflow.id),
            )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=users["other"].id, tenant_id=users["other"].tenant_id
        )
        assert loaded is None


class TestListingCarriesTheSameRule:
    """`get_latest_workflow` is not the only place that reads visibility.

    `list_artifacts` had the same fail-open default twice: asked for
    `private` with no owner it dropped the owner clause, and asked for
    `shared` with no tenant it dropped the tenant clause - returning
    everyone's rows in both cases. Neither is reachable from `/v1/artifacts`
    today, which is exactly why it would still be there later.
    """

    def test_private_without_an_owner_returns_nothing(self, runtime, users):
        runtime.store.create_artifact(
            "workflow", _unique("someones_private"), _graph(),
            owner_user_id=users["owner"].id, visibility="private",
        )
        found = runtime.store.list_artifacts(
            type_filter="workflow", visibility="private", owner_user_id=None
        )
        assert found == []

    def test_shared_without_a_tenant_returns_nothing(self, runtime, tenants):
        runtime.store.create_artifact(
            "workflow", _unique("someones_shared"), _graph(),
            owner_user_id=tenants["owner"].id, visibility="shared",
        )
        found = runtime.store.list_artifacts(
            type_filter="workflow", visibility="shared", tenant_id=None
        )
        assert found == []

    def test_the_scoped_forms_still_work(self, runtime, tenants):
        mine = runtime.store.create_artifact(
            "workflow", _unique("scoped_shared"), _graph(),
            owner_user_id=tenants["owner"].id, visibility="shared",
        )
        found = runtime.store.list_artifacts(
            type_filter="workflow",
            visibility="shared",
            tenant_id=tenants["owner"].tenant_id,
        )
        assert [a.id for a in found] == [mine.id]


class TestAnInvocationStaysBoundToTheArtifactItNamed:
    """`/tools/{id}/invoke` authorizes one row and must execute that row.

    Artifact names carry no uniqueness constraint, and `schema.name` - the
    key a tool is resolved by - is free text inside the spec. The route
    checked out artifact A and handed the engine a bare `dict(A.schema)`;
    the engine threw the id away and resolved `schema["name"]` again. Two
    tools may answer to one name, so the row that ran was whichever the
    second lookup returned, which is not the row the request authorized.
    """

    def _tool(self, store, name, owner, *, privileged):
        return store.create_artifact(
            "tool",
            _unique("dup"),
            {
                "kind": "tool.spec",
                "name": name,
                "handler": "notes.search_v1",
                "privileged": privileged,
            },
            owner_user_id=owner.id,
            visibility="global",  # both visible, so the name really collides
        )

    def _invoke(self, runtime, artifact, caller):
        return asyncio.run(
            runtime.workflow.invoke_tool(
                runtime.workflow._describe_tool(artifact),
                {"query": "x"},
                user_id=caller.id,
                tenant_id=caller.tenant_id,
            )
        )

    def test_two_tools_share_a_name_and_each_id_keeps_its_own_answer(
        self, runtime, users
    ):
        """The two rows differ only in who owns them, and SPEC §18 makes that
        the difference between permitted and forbidden. Re-resolving by name
        collapses both invocations onto one row, so one of these two
        assertions fails however the lookup happens to be ordered.
        """
        store = runtime.store
        shared_name = _unique("collide")
        admin_owned = self._tool(
            store, shared_name, users["admin"], privileged=True
        )
        user_owned = self._tool(store, shared_name, users["owner"], privileged=True)
        assert admin_owned.id != user_owned.id

        permitted = self._invoke(runtime, admin_owned, users["admin"])
        refused = self._invoke(runtime, user_owned, users["admin"])

        _assert_ran(permitted)
        assert refused.get("error") == "forbidden", refused

    def test_the_descriptor_that_executes_carries_the_id_that_was_named(
        self, runtime, users, monkeypatch
    ):
        """The outcome test proves the two ids diverge; this one names the
        mechanism, so a future refactor that reintroduces a by-name lookup
        fails here with the id it substituted rather than only with a role."""
        store = runtime.store
        shared_name = _unique("collide")
        first = self._tool(store, shared_name, users["owner"], privileged=False)
        second = self._tool(store, shared_name, users["admin"], privileged=False)

        seen: list = []
        original = runtime.workflow._invoke_tool

        async def record(*args, **kwargs):
            seen.append(kwargs.get("descriptor"))
            return await original(*args, **kwargs)

        monkeypatch.setattr(runtime.workflow, "_invoke_tool", record)

        for artifact in (first, second, first):
            self._invoke(runtime, artifact, users["admin"])

        assert [d.artifact_id for d in seen] == [first.id, second.id, first.id]

    def test_the_route_runs_the_id_in_its_path(
        self, runtime, client, admin_headers, users
    ):
        """End to end, and the escalation the substitution allows.

        The admin asks for `harmless.id`, a plain unprivileged tool. A second
        tool answering to the same `schema.name` is global and declares
        `privileged: true` with an ordinary owner. Resolving by name finds
        only the global one - the authorized row is another user's private
        artifact, which no by-name lookup for this caller can see - so the
        request for a harmless tool came back refused as a privileged one.
        """
        store = runtime.store
        shared_name = _unique("collide")
        harmless = store.create_artifact(
            "tool", _unique("harmless"),
            {
                "kind": "tool.spec", "name": shared_name,
                "handler": "notes.search_v1", "privileged": False,
            },
            owner_user_id=users["owner"].id, visibility="private",
        )
        store.create_artifact(
            "tool", _unique("impostor"),
            {
                "kind": "tool.spec", "name": shared_name,
                "handler": "notes.search_v1", "privileged": True,
            },
            owner_user_id=users["other"].id, visibility="global",
        )

        resp = client.post(
            f"/v1/tools/{harmless.id}/invoke",
            json={"inputs": {"query": "x"}},
            headers=admin_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert "privileged" not in (data.get("content") or ""), data
        assert data.get("status") != "error", data

    def test_a_bare_spec_still_works_and_claims_no_provenance(
        self, runtime, users
    ):
        """Callers that legitimately hold no artifact - a workflow node, a
        test - keep working, but an unattributed spec cannot be privileged."""
        result = asyncio.run(
            runtime.workflow.invoke_tool(
                {
                    "kind": "tool.spec",
                    "name": "notes.search_v1",
                    "handler": "notes.search_v1",
                    "privileged": True,
                },
                {"query": "x"},
                user_id=users["admin"].id,
                tenant_id=users["admin"].tenant_id,
            )
        )
        assert result.get("error") == "forbidden", result


class TestOneUsersToolDoesNotLeakIntoTheProcess:
    def test_direct_invocation_leaves_the_shared_registry_alone(
        self, runtime, users
    ):
        """The registry is built once per process from globally visible tool
        artifacts. A private invocation caching itself into it made that
        user's definition resolvable for every later request."""
        engine = runtime.workflow
        before = dict(engine.tool_registry)
        private_name = _unique("private_tool")

        asyncio.run(
            engine.invoke_tool(
                {
                    "kind": "tool.spec",
                    "name": private_name,
                    "handler": "notes.search_v1",
                },
                {"query": "x"},
                user_id=users["owner"].id,
                tenant_id=users["owner"].tenant_id,
            )
        )
        assert private_name not in engine.tool_registry
        assert set(engine.tool_registry) == set(before)
