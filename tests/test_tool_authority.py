"""Who is allowed to run what — SPEC §18 and the artifact permission rule.

Three authority questions the source answered from the wrong place:

* **privileged provenance.** SPEC says `privileged:true` tools require
  *admin-owned artifacts*. The check asked only whether the caller is an
  admin. `check_privileged_access` even takes an `artifact_owner_id` and never
  reads it. Any authenticated user can create a `tool.spec` — `/v1/artifacts`
  depends on `get_user`, and the schema permits additional properties — so a
  user could author `privileged: true` and an admin invoking it would be
  granted the privileged sandbox for someone else's definition.

* **workflow ownership.** `/v1/chat` passes the caller's `workflow_id`
  straight into the engine, which loads it by artifact id with no owner or
  visibility check. A private workflow is an artifact like any other.

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


class TestPrivilegedToolsNeedAnAdminOwnedArtifact:
    def test_the_caller_is_only_half_the_question(self, runtime, users):
        """An ordinary user's `privileged: true` tool, invoked by an admin.

        The caller check passes — they really are an admin — and the artifact
        they are running was written by someone else. SPEC requires both.
        """
        store = runtime.store
        artifact = store.create_artifact(
            "tool",
            _unique("user_written"),
            {
                "kind": "tool.spec",
                "name": "note_search",
                "handler": "note_search",
                "privileged": True,
            },
            owner_user_id=users["owner"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "note_search",
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
                "name": "note_search",
                "handler": "note_search",
                "privileged": True,
            },
            owner_user_id=users["admin"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "note_search",
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
                "name": "note_search",
                "handler": "note_search",
                "privileged": True,
            },
            owner_user_id=users["admin"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "note_search",
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
        assert result.get("error") != "forbidden", result

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
                "name": "note_search",
                "handler": "note_search",
                "privileged": True,
                "owner_user_id": users["admin"].id,  # claimed, not true
            },
            owner_user_id=users["owner"].id,
        )
        result = asyncio.run(
            runtime.workflow._invoke_tool(
                "note_search",
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
                "note_search",
                {"query": "x"},
                [],
                [],
                None,
                None,
                None,
                user_id=users["admin"].id,
                tenant_id=users["admin"].tenant_id,
                descriptor=ToolDescriptor(
                    name="note_search",
                    schema={
                        "kind": "tool.spec",
                        "name": "note_search",
                        "handler": "note_search",
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
    caller check — but it must no longer be the whole story."""

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
                "nodes": [{"id": "start", "type": "respond"}],
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
                "nodes": [{"id": "start", "type": "respond"}],
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
        no default — a caller cannot omit the question by accident."""
        store = runtime.store
        workflow = store.create_artifact(
            "workflow",
            _unique("scoped"),
            {
                "kind": "workflow.graph",
                "entrypoint": "start",
                "nodes": [{"id": "start", "type": "respond"}],
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
            {
                "kind": "workflow.graph",
                "entrypoint": "start",
                "nodes": [{"id": "start", "type": "respond"}],
            },
            owner_user_id=users["owner"].id,
            visibility="global",
        )
        loaded = runtime.workflow._load_workflow_for(
            workflow.id, user_id=users["other"].id, tenant_id=users["other"].tenant_id
        )
        assert loaded is not None


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
                    "handler": "note_search",
                },
                {"query": "x"},
                user_id=users["owner"].id,
                tenant_id=users["owner"].tenant_id,
            )
        )
        assert private_name not in engine.tool_registry
        assert set(engine.tool_registry) == set(before)
