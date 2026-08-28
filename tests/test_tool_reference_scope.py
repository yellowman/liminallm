"""A workflow tool reference resolves in the workflow's execution namespace.

Tranche 2, the tool half. The rule, frozen:

    A workflow tool reference resolves to exactly one tool spec in the
    workflow's execution namespace, and that spec names an executable
    handler. Resolution is independent of the runner for published
    workflows, current canonical state is consulted at execution, ambiguity
    fails closed, and process-local caches cannot create authority.

Precedence, by the *workflow's* visibility rather than the runner's identity:

    ===============  =============================================
    private          private(owner) > shared(tenant) > global
    shared           shared(tenant) > global
    global           global
    ===============  =============================================

Zero matches in a tier falls through to the next; two matches in one tier is
ambiguous and fails closed. Creation time decides nothing.

Every defect below was reproduced against the running engine first.

`_resolve_tool` scans `list_artifacts(owner_user_id=..., tenant_id=...)` and
takes the first name match, then falls back to a registry built once at
startup. Measured, that gives five distinct ways for a reference to mean
something other than what was published:

    ==================================  ====================================
    a global `foo` + Bob's private      Bob runs web.fetch_v1, everyone
    `foo`                               else runs notes.search_v1
    two global rows named `foo`         the newer wins; `created_at DESC`
                                        is the authority rule
    shared `foo` + global `foo`         whichever was created later
    110 newer tools created after       `foo` is not on page one and
    `foo`                               becomes unresolvable
    `foo` deleted from Postgres         the startup registry hands back the
                                        spec and it still executes
    ==================================  ====================================

And a sixth, one stage further in: a visible spec named `foo` whose
`handler` is `no.such.handler` resolves as a name. Name existence is not
handler executability.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime
from liminallm.service.tool_namespace import SYSTEM_SCOPE, ToolResolutionScope


def _u(p):
    return f"{p}_{uuid.uuid4().hex[:8]}"


def _tool(store, name, handler, *, owner, visibility, **extra):
    return store.create_artifact(
        "tool", _u("t"),
        {"kind": "tool.spec", "name": name, "handler": handler, **extra},
        owner_user_id=owner, visibility=visibility,
    )


def _wf(tool_name, entry="call"):
    return {"kind": "workflow.chat", "entrypoint": entry, "nodes": [
        {"id": entry, "type": "tool_call", "tool": tool_name, "next": "fin"},
        {"id": "fin", "type": "end"},
    ]}


@pytest.fixture
def store():
    return get_runtime().store


@pytest.fixture
def engine():
    return get_runtime().workflow


class TestAdmissionValidatesAgainstTheExecutionAudience:
    """Not the publisher's identity: the audience the artifact declares.

    Alice may own a private tool and a shared workflow that calls it. Asking
    "can Alice resolve this?" passes, and Bob — who may legitimately run the
    shared workflow — cannot resolve Alice's private tool. That admits a
    workflow known at publication time not to work for its declared audience.
    """

    def test_a_private_workflow_may_use_its_owners_private_tool(self, store):
        """The control, and the case that must keep working."""
        u = store.create_user(email=f"{_u('pa')}@t.local", tenant_id=_u("pt"))
        name = _u("mine")
        _tool(store, name, "llm.generic", owner=u.id, visibility="private")
        art = store.create_artifact(
            "workflow", _u("privwf"), _wf(name),
            owner_user_id=u.id, visibility="private",
        )
        assert art.id

    def test_a_shared_workflow_may_not_use_the_authors_private_tool(self, store):
        u = store.create_user(email=f"{_u('sa')}@t.local", tenant_id=_u("st"))
        name = _u("secret")
        _tool(store, name, "llm.generic", owner=u.id, visibility="private")
        with pytest.raises(Exception) as exc:
            store.create_artifact(
                "workflow", _u("sharedwf"), _wf(name),
                owner_user_id=u.id, visibility="shared",
            )
        errors = " ".join(getattr(exc.value, "errors", []) or []).lower()
        assert name.lower() in errors, errors

    def test_a_global_workflow_may_not_use_a_tenant_shared_tool(self, store):
        u = store.create_user(email=f"{_u('ga')}@t.local", tenant_id=_u("gt"))
        name = _u("tenantonly")
        _tool(store, name, "llm.generic", owner=u.id, visibility="shared")
        with pytest.raises(Exception):
            store.create_artifact(
                "workflow", _u("globalwf"), _wf(name),
                owner_user_id=u.id, visibility="global",
            )

    def test_a_shared_workflow_may_use_a_global_tool(self, store):
        """The other control. Refusing every published workflow would pass
        both refusals above."""
        u = store.create_user(email=f"{_u('sg')}@t.local", tenant_id=_u("sgt"))
        art = store.create_artifact(
            "workflow", _u("okwf"), _wf("llm.generic"),
            owner_user_id=u.id, visibility="shared",
        )
        assert art.id

    def test_an_ownerless_seeded_builtin_is_usable_and_unprivileged(self, store, engine):
        """SPEC distinguishes ownerless system artifacts. A seeded global tool
        resolves and can never be privileged — do not manufacture an owner for
        it to satisfy `privileged`."""
        d = engine._resolve_tool("llm.generic", SYSTEM_SCOPE)
        assert d is not None, "the seeded builtin stopped resolving"
        assert d.owner_role is None, (
            f"an ownerless system tool acquired authority: {d.owner_role!r}"
        )

    def test_an_ownerless_shared_tool_reaches_no_tenant(self, store):
        """`shared` is scoped through the owner's tenant, and `artifact` has no
        tenant column. An ownerless shared row therefore belongs to nobody —
        the same rule listing already applies."""
        u = store.create_user(email=f"{_u('os')}@t.local", tenant_id=_u("ost"))
        name = _u("detached")
        _tool(store, name, "llm.generic", owner=None, visibility="shared")
        with pytest.raises(Exception):
            store.create_artifact(
                "workflow", _u("detachedwf"), _wf(name),
                owner_user_id=u.id, visibility="shared",
            )


class TestAmbiguityFailsClosed:
    """Two rows in one precedence tier is not a tie the database may break.

    Measured: two global specs named `dup`, handlers `notes.search_v1` and
    `web.fetch_v1`. `_resolve_tool` picks the newer, because
    `list_artifacts` orders `created_at DESC` and it takes the first match.
    There is no name-uniqueness constraint behind it.
    """

    def test_two_rows_in_one_tier_are_ambiguous(self, store):
        u = store.create_user(email=f"{_u('am')}@t.local", tenant_id=_u("amt"))
        name = _u("dup")
        _tool(store, name, "llm.generic", owner=u.id, visibility="global")
        _tool(store, name, "agent.code_v1", owner=u.id, visibility="global")
        with pytest.raises(Exception):
            store.create_artifact(
                "workflow", _u("ambigwf"), _wf(name),
                owner_user_id=u.id, visibility="global",
            )

    @pytest.mark.parametrize("shared_first", [True, False],
                             ids=["shared-older", "shared-newer"])
    @pytest.mark.asyncio
    async def test_shared_beats_global_whichever_was_created_first(
        self, store, shared_first
    ):
        """Precedence is a rule, not a timestamp.

        Observed through execution rather than a resolver call, so this
        measures what the turn actually invokes. Both orders must give the
        same answer; today they give opposite ones. `shared-newer` therefore
        passes already — by accident, because `created_at DESC` happens to
        agree — so `shared-older` is the real red and the pair together is
        what says the rule stopped depending on insertion order.
        """
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()
        tenant = _u("prec")
        u = store.create_user(email=f"{_u('pa')}@t.local", tenant_id=tenant)
        name = _u("both")
        if shared_first:
            _tool(store, name, "agent.code_v1", owner=u.id, visibility="shared")
            _tool(store, name, "llm.generic", owner=u.id, visibility="global")
        else:
            _tool(store, name, "llm.generic", owner=u.id, visibility="global")
            _tool(store, name, "agent.code_v1", owner=u.id, visibility="shared")

        wf = store.create_artifact(
            "workflow", _u("precwf"), _wf(name),
            owner_user_id=u.id, visibility="shared",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        ran = []
        engine._tool_llm_generic = lambda *a, **k: (
            ran.append("llm.generic") or {"status": "ok", "content": "x"}
        )
        engine._tool_agent_code = lambda *a, **k: (
            ran.append("agent.code_v1") or {"status": "ok", "content": "x"}
        )
        await engine.run(wf.id, None, "hi", None,
                         user_id=u.id, tenant_id=u.tenant_id)
        assert ran == ["agent.code_v1"], (
            f"a shared workflow did not prefer the tenant-shared spec; "
            f"creation order decided instead: ran {ran}"
        )


class TestTheResolverIsNotAListing:
    """Paging and creation order are not authority rules."""

    def test_a_hundred_newer_tools_cannot_hide_the_referenced_one(
        self, store, engine
    ):
        """Measured: `list_artifacts` returns 100 rows ordered
        `created_at DESC`, so 110 newer tools push the referenced one off
        page one and it becomes unresolvable — visible, undeleted, and gone.
        """
        u = store.create_user(email=f"{_u('pg')}@t.local", tenant_id=_u("pgt"))
        name = _u("early")
        _tool(store, name, "llm.generic", owner=u.id, visibility="private")
        for _ in range(110):
            _tool(store, _u("filler"), "llm.generic", owner=u.id, visibility="private")
        d = engine._resolve_tool(
            name, ToolResolutionScope("private", u.id, u.tenant_id)
        )
        assert d is not None, (
            "a visible, undeleted tool became unresolvable because a listing "
            "page filled up"
        )

    def test_a_deleted_tool_cannot_resurrect_from_the_startup_cache(self, store):
        """The registry is built once in `__init__` from a listing. Measured:
        delete the artifact and the same engine still resolves it, with
        `artifact_id=None` — unattributed, and executable."""
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()
        u = store.create_user(email=f"{_u('gh')}@t.local", tenant_id=_u("ght"))
        name = _u("ghost")
        art = _tool(store, name, "llm.generic", owner=u.id, visibility="global")

        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        with store._connect() as conn:
            conn.execute("DELETE FROM artifact WHERE id = %s", (art.id,))
        assert store.get_artifact(art.id) is None, "the row survived"

        d = engine._resolve_tool(
            name, ToolResolutionScope("private", u.id, u.tenant_id)
        )
        assert d is None, (
            "a process-local cache proved a deleted artifact exists; Postgres "
            "is canonical and said it is gone"
        )


class TestAHandlerMustBeExecutable:
    """Name resolution and handler resolution are two stages of one claim.

    Measured: a spec named `badhandler` with `handler: "no.such.handler"`
    resolves as a name, and `_resolve_worker_tool` falls through to returning
    the tool's own name — which is neither a worker body nor a host handler.
    """

    def test_a_spec_whose_handler_names_nothing_is_refused(self, store):
        u = store.create_user(email=f"{_u('bh')}@t.local", tenant_id=_u("bht"))
        name = _u("bad")
        _tool(store, name, "no.such.handler", owner=u.id, visibility="private")
        with pytest.raises(Exception):
            store.create_artifact(
                "workflow", _u("badwf"), _wf(name),
                owner_user_id=u.id, visibility="private",
            )

    def test_one_definition_of_executable(self):
        """Admission cannot instantiate a `WorkflowEngine` to ask what is
        executable, so the host handler names live in a pure module — and the
        engine's own map is checked against it, or the two lists drift."""
        from liminallm.service import tool_worker
        from liminallm.service.tool_namespace import (
            EXECUTABLE_HANDLER_NAMES,
            HOST_TOOL_HANDLER_NAMES,
        )
        rt = get_runtime()
        assert set(rt.workflow._builtin_tool_handlers()) == set(
            HOST_TOOL_HANDLER_NAMES
        ), "the engine's host handlers and the declared set have drifted"
        assert EXECUTABLE_HANDLER_NAMES == (
            frozenset(tool_worker.BODY_NAMES) | frozenset(HOST_TOOL_HANDLER_NAMES)
        )


class TestExecutionCarriesThePublicationScope:
    """The runner's private namespace must not reach a published workflow.

    Measured, and the reason this tranche exists: a global `foo` handled by
    `notes.search_v1`, and Bob's private `foo` handled by `web.fetch_v1`.
    Alice runs the shared workflow and gets notes; Bob runs the same workflow
    and gets a web fetch. One published workflow, two capabilities.

    `_load_workflow_for` returns a bare schema dict, so the scope needed to
    prevent this is discarded at the load boundary.
    """

    @pytest.mark.asyncio
    async def test_a_shared_workflow_resolves_in_its_own_namespace(self, store):
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()
        tenant = _u("exec")
        alice = store.create_user(email=f"{_u('al')}@t.local", tenant_id=tenant)
        bob = store.create_user(email=f"{_u('bo')}@t.local", tenant_id=tenant)

        name = _u("foo")
        _tool(store, name, "llm.generic", owner=alice.id, visibility="global")
        _tool(store, name, "agent.code_v1", owner=bob.id, visibility="private")

        wf = store.create_artifact(
            "workflow", _u("sharedwf"), _wf(name),
            owner_user_id=alice.id, visibility="shared",
        )

        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        ran = []
        engine._tool_llm_generic = lambda *a, **k: (
            ran.append("llm.generic") or {"status": "ok", "content": "x"}
        )
        engine._tool_agent_code = lambda *a, **k: (
            ran.append("agent.code_v1") or {"status": "ok", "content": "x"}
        )

        await engine.run(wf.id, None, "hi", None,
                         user_id=bob.id, tenant_id=bob.tenant_id)
        assert ran == ["llm.generic"], (
            f"Bob's private tool captured a shared workflow: ran {ran}"
        )

    @pytest.mark.asyncio
    async def test_a_parallel_child_resolves_in_the_same_namespace(self, store):
        """A second descent, and a second chance to lose the scope.

        `_execute_parallel_nodes` calls `_execute_node_with_retry` itself,
        carrying only the runner's `user_id` and `tenant_id`. Carrying
        publication scope correctly through the driving loop and forgetting it
        here would leave children resolving under the runner — measured, Bob's
        child ran `agent.code_v1`.
        """
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()
        tenant = _u("par")
        alice = store.create_user(email=f"{_u('pa')}@t.local", tenant_id=tenant)
        bob = store.create_user(email=f"{_u('pb')}@t.local", tenant_id=tenant)

        name = _u("foo")
        _tool(store, name, "llm.generic", owner=alice.id, visibility="global")
        _tool(store, name, "agent.code_v1", owner=bob.id, visibility="private")

        wf = store.create_artifact(
            "workflow", _u("parwf"),
            {"kind": "workflow.chat", "entrypoint": "fan", "nodes": [
                {"id": "fan", "type": "parallel", "next": ["leaf"], "after": "done"},
                {"id": "leaf", "type": "tool_call", "tool": name},
                {"id": "done", "type": "end"},
            ]},
            owner_user_id=alice.id, visibility="shared",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        ran = []
        engine._tool_llm_generic = lambda *a, **k: (
            ran.append("llm.generic") or {"status": "ok", "content": "x"}
        )
        engine._tool_agent_code = lambda *a, **k: (
            ran.append("agent.code_v1") or {"status": "ok", "content": "x"}
        )
        await engine.run(wf.id, None, "hi", None,
                         user_id=bob.id, tenant_id=bob.tenant_id)
        assert ran == ["llm.generic"], (
            f"a parallel child resolved under the runner: ran {ran}"
        )


class TestStreamingSelectsTheCapabilityAfterResolving:
    """The `stream` flag must not change which capability a workflow runs.

    `run_streaming` reads `node.get("tool")` and compares the *literal string*
    against a hard-coded set before any tool artifact is resolved. So a
    tenant-shared spec that overrides the name `llm.generic` with handler
    `agent.code_v1` is honoured on one path and ignored on the other.

    Measured on the same workflow and the same caller, and this one is live
    on main rather than a hazard the fix would introduce:

    ==========  ==========================================================
    blocking    ran `agent.code_v1` — it already obeys the override
    streaming   ran `generate_stream` — it never looked
    ==========  ==========================================================

    The fix is to resolve the descriptor first and decide streamability from
    the resolved handler, not from the reference spelling. That also lets a
    custom name whose handler *is* `llm.generic` stream, which the current
    literal comparison cannot express.
    """

    @pytest.mark.asyncio
    async def test_both_paths_run_the_resolved_handler(self, store):
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()
        alice = store.create_user(email=f"{_u('sa')}@t.local", tenant_id=_u("stt"))
        # Overrides the seeded global builtin, for this tenant only.
        _tool(store, "llm.generic", "agent.code_v1",
              owner=alice.id, visibility="shared")
        wf = store.create_artifact(
            "workflow", _u("strwf"), _wf("llm.generic"),
            owner_user_id=alice.id, visibility="shared",
        )

        def build():
            engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
            ran = []
            engine._tool_llm_generic = lambda *a, **k: (
                ran.append("llm.generic") or {"status": "ok", "content": "x"}
            )
            engine._tool_agent_code = lambda *a, **k: (
                ran.append("agent.code_v1") or {"status": "ok", "content": "x"}
            )

            def generate_stream(*a, **k):
                ran.append("generate_stream")
                return iter([
                    {"event": "token", "data": "hi"},
                    {"event": "message_done",
                     "data": {"content": "hi", "usage": {}}},
                ])

            engine.llm.generate_stream = generate_stream
            return engine, ran

        engine, blocking_ran = build()
        await engine.run(wf.id, None, "hi", None,
                         user_id=alice.id, tenant_id=alice.tenant_id)

        engine, streaming_ran = build()
        [e async for e in engine.run_streaming(
            wf.id, None, "hi", None,
            user_id=alice.id, tenant_id=alice.tenant_id)]

        assert blocking_ran == ["agent.code_v1"], blocking_ran
        assert streaming_ran == ["agent.code_v1"], (
            f"the stream flag changed which capability ran: "
            f"blocking={blocking_ran} streaming={streaming_ran}"
        )

    @pytest.mark.asyncio
    async def test_a_custom_name_whose_handler_is_the_llm_still_streams(self, store):
        """The complement, and red today for its own reason.

        A custom name whose handler *is* `llm.generic` is not on the
        hard-coded list, so it never streams — measured, `generate_stream` was
        not called at all. Deciding by resolved handler fixes that too.

        It is the complement rather than a control because deleting the
        special case entirely — never stream — would satisfy the test above
        and fail this one.
        """
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()
        u = store.create_user(email=f"{_u('cn')}@t.local", tenant_id=_u("cnt"))
        name = _u("my.chat")
        _tool(store, name, "llm.generic", owner=u.id, visibility="private")
        wf = store.create_artifact(
            "workflow", _u("aliaswf"), _wf(name),
            owner_user_id=u.id, visibility="private",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        ran = []

        def generate_stream(*a, **k):
            ran.append("generate_stream")
            return iter([
                {"event": "token", "data": "hi"},
                {"event": "message_done", "data": {"content": "hi", "usage": {}}},
            ])

        engine.llm.generate_stream = generate_stream
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert ran == ["generate_stream"], (
            f"an aliased LLM tool did not stream: {ran}"
        )
        assert any(e.get("event") == "token" for e in events), events[-3:]


class TestEveryAdmissionPathAsksTheQuestion:
    """Create is not the only door. A valid workflow can be patched into an
    invalid one, and ConfigOps applies model-authored text."""

    def test_patching_a_reference_into_an_unreachable_tool_is_refused(
        self, client, admin_headers
    ):
        """A *private* workflow, deliberately. `PATCH /v1/artifacts` refuses a
        global one for an unrelated reason — published artifacts change
        through config ops — so patching a global row here would have measured
        that rule instead of this one."""
        made = client.post("/v1/artifacts", headers=admin_headers, json={
            "type": "workflow", "name": _u("patchwf"),
            "schema": _wf("llm.generic"), "visibility": "private",
        })
        assert made.status_code in (200, 201), made.text
        artifact = made.json()["data"]["id"]

        resp = client.patch(f"/v1/artifacts/{artifact}", headers=admin_headers, json={
            "patch": [{"op": "replace", "path": "/nodes/0/tool",
                       "value": "no.such.tool.v1"}],
        })
        assert resp.status_code == 400, resp.text
        kept = get_runtime().store.get_artifact(artifact)
        assert kept.schema["nodes"][0]["tool"] == "llm.generic", kept.schema

    def test_configops_apply_refuses_before_it_changes_anything(self, store):
        """The third door, and the one that writes an audit record.

        `apply_config_patch` builds `new_schema`, shape-validates it, writes a
        version and flips the patch to `applied`, all in one transaction while
        holding the artifact then patch locks. Measured on a *global*
        workflow patched to name Alice's *private* tool:

            patch status       applied
            workflow tool now  hidden_27c1362f

        So the audit asserts an applied configuration that the runtime will
        refuse. The refusal has to land before the version and before the
        status transition, which is why this asserts all three and not merely
        that apply raised.
        """
        from liminallm.service.config_ops import ConfigOpsService
        rt = get_runtime()
        alice = store.create_user(email=f"{_u('ca')}@t.local", tenant_id=_u("ct"))
        private_name = _u("hidden")
        _tool(store, private_name, "llm.generic",
              owner=alice.id, visibility="private")

        wf = store.create_artifact(
            "workflow", _u("cowf"), _wf("llm.generic"),
            owner_user_id=alice.id, visibility="global",
        )
        versions_before = len(store.list_artifact_versions(wf.id))

        patch = store.record_config_patch(
            artifact_id=wf.id,
            proposer="system_llm",
            patch={"ops": [{"op": "replace", "path": "/nodes/0/tool",
                            "value": private_name}]},
            justification="swap a global workflow onto a private tool",
        )
        store.update_config_patch_status(patch.id, "approved")

        svc = ConfigOpsService(store, rt.llm, rt.router, rt.training)
        with pytest.raises(Exception):
            svc.apply_patch(patch.id, approver_user_id=alice.id)

        after = store.get_artifact(wf.id)
        assert after.schema["nodes"][0]["tool"] == "llm.generic", (
            f"a refused apply still rewrote the workflow: {after.schema}"
        )
        assert len(store.list_artifact_versions(wf.id)) == versions_before, (
            "a refused apply still wrote an artifact version"
        )
        assert store.get_config_patch(patch.id).status != "applied", (
            "the audit records an applied configuration that was refused"
        )


class TestAuthorityIsCheckedOnBothPaths:
    """A resolved spec carries authority, and streaming must not skip it.

    `_invoke_tool` does real work before a body runs: input-schema validation,
    `timeout_seconds`, and SPEC §18's rule that `privileged: true` only counts
    when an *admin-owned artifact* stands behind it. `_stream_llm_node` takes
    a node, never a descriptor, and does none of it.

    Measured on one private workflow naming an ordinary user's own private
    spec that claims privilege:

    ==========  ==============================================
    blocking    refused; the model never ran
    streaming   ran `generate_stream`
    ==========  ==============================================

    Resolving the descriptor before dispatch — which the streaming path now
    does — is not enough on its own, because the descriptor is then dropped.
    The preflight has to be shared, and streaming may specialise token
    production only after it passes. `_stream_llm_node` must not become a
    second implementation of tool authorization.
    """

    @staticmethod
    def _both_paths(store, wf, runner):
        """Run one workflow on each path and report what executed."""
        from liminallm.service.workflow import WorkflowEngine
        rt = get_runtime()

        async def once(streaming):
            engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag,
                                    cache=rt.cache)
            ran = []
            engine._tool_llm_generic = lambda *a, **k: (
                ran.append("llm.generic") or {"status": "ok", "content": "x"}
            )

            def generate_stream(*a, **k):
                ran.append("generate_stream")
                return iter([
                    {"event": "token", "data": "hi"},
                    {"event": "message_done",
                     "data": {"content": "hi", "usage": {}}},
                ])

            engine.llm.generate_stream = generate_stream
            if streaming:
                [e async for e in engine.run_streaming(
                    wf.id, None, "hi", None,
                    user_id=runner.id, tenant_id=runner.tenant_id)]
            else:
                await engine.run(wf.id, None, "hi", None,
                                 user_id=runner.id, tenant_id=runner.tenant_id)
            return ran

        return once

    @pytest.mark.asyncio
    async def test_a_non_admins_privileged_spec_runs_on_neither_path(self, store):
        """SPEC §18: `privileged: true` is a property of an admin-owned
        artifact, not a key anyone may type into their own spec."""
        u = store.create_user(email=f"{_u('pv')}@t.local", tenant_id=_u("pvt"))
        _tool(store, "llm.generic", "llm.generic", owner=u.id,
              visibility="private", privileged=True)
        wf = store.create_artifact(
            "workflow", _u("pvwf"), _wf("llm.generic"),
            owner_user_id=u.id, visibility="private",
        )
        once = self._both_paths(store, wf, u)
        assert await once(streaming=False) == [], "blocking ran a claimed privilege"
        assert await once(streaming=True) == [], (
            "streaming ran a non-admin's privileged spec; the resolved "
            "descriptor was dropped before the body was entered"
        )

    @pytest.mark.asyncio
    async def test_an_admin_owned_privileged_tool_refuses_a_non_admin_runner(
        self, store
    ):
        """The other conjunction: real provenance, wrong caller."""
        admin = store.create_user(email=f"{_u('ad')}@t.local",
                                  tenant_id=_u("adt"), role="admin")
        runner = store.create_user(email=f"{_u('rn')}@t.local",
                                   tenant_id=admin.tenant_id)
        # A distinct name, not a second global `llm.generic`: publishing one
        # of those is now ambiguous against the seeded row, and this test is
        # about the caller check rather than about resolution.
        name = _u("admin.only")
        _tool(store, name, "llm.generic", owner=admin.id,
              visibility="global", privileged=True)
        wf = store.create_artifact(
            "workflow", _u("adwf"), _wf(name),
            owner_user_id=admin.id, visibility="global",
        )
        once = self._both_paths(store, wf, runner)
        assert await once(streaming=False) == [], "blocking served a non-admin"
        assert await once(streaming=True) == [], "streaming served a non-admin"

    @pytest.mark.asyncio
    async def test_an_unprivileged_tool_still_runs_on_both_paths(self, store):
        """The control. Refusing everything passes both witnesses above."""
        u = store.create_user(email=f"{_u('ok')}@t.local", tenant_id=_u("okt"))
        wf = store.create_artifact(
            "workflow", _u("okwf2"), _wf("llm.generic"),
            owner_user_id=u.id, visibility="private",
        )
        once = self._both_paths(store, wf, u)
        assert await once(streaming=False) == ["llm.generic"]
        assert await once(streaming=True) == ["generate_stream"]


class TestTheHandlerNamesTheBody:
    """The resolved spec's `handler` decides what runs, not the reference.

    `_resolve_worker_tool` checks `tool_name in tool_worker.BODY_NAMES` first
    and only then reads the spec's `handler`, so a reference whose *name*
    happens to match a worker body runs that body whatever the spec says.
    Measured on a private spec named `notes.search_v1` with handler
    `llm.generic`:

    ==========================  ================
    descriptor.handler          llm.generic
    body actually selected      notes.search_v1
    ==========================  ================

    Admission approves one executable handler and runtime executes another —
    the exact divergence this tranche exists to remove.
    """

    def test_the_handler_wins_over_a_name_that_matches_a_body(self, store, engine):
        from liminallm.service.tool_namespace import ToolResolutionScope

        u = store.create_user(email=f"{_u('hb')}@t.local", tenant_id=_u("hbt"))
        _tool(store, "notes.search_v1", "llm.generic", owner=u.id,
              visibility="private")
        scope = ToolResolutionScope("private", u.id, u.tenant_id)
        d = engine._resolve_tool("notes.search_v1", scope)
        assert d is not None and d.handler == "llm.generic", d
        body = engine._resolve_worker_tool("notes.search_v1", d.schema)
        assert body == "llm.generic", (
            f"the reference name beat the resolved handler: {body!r}"
        )

    def test_a_spec_with_no_handler_still_reaches_its_own_body(self, store, engine):
        """The compatibility case, and the control. A spec that names no
        handler means itself, so narrowing this rule must not strand the
        seeded tools that are their own body."""
        from liminallm.service.tool_namespace import SYSTEM_SCOPE

        d = engine._resolve_tool("notes.search_v1", SYSTEM_SCOPE)
        assert d is not None
        assert engine._resolve_worker_tool("notes.search_v1", d.schema) == (
            "notes.search_v1"
        )

    def test_one_function_answers_it_for_every_path(self, store):
        """Admission, blocking execution and streaming dispatch must ask the
        same question, or two of them can disagree about what will run."""
        from liminallm.service.tool_namespace import resolve_executable_handler

        assert resolve_executable_handler(
            "notes.search_v1", {"handler": "llm.generic"}
        ) == "llm.generic"
        assert resolve_executable_handler("notes.search_v1", None) == (
            "notes.search_v1"
        )
        assert resolve_executable_handler(
            "whatever", {"handler": "no.such.handler"}
        ) is None


class TestAStreamedNodeObeysTheNodeContract:
    """SPEC §9.2 and §18.3 apply to a streamed tool node too.

    §9.2: per-node `max_retries`, `backoff_ms` and `timeout_ms` are
    overridable, a node past its timeout fails, and tool inputs *and outputs*
    are JSON-Schema validated. §18.3 is the single normative home for the
    numbers — 2 retries by default, hard cap 3, 1s then 4s backoff, 15s node
    timeout capped at 60s.

    The streamable branch does not call `_execute_node_with_retry`; the branch
    directly below it does. Measured on one aliased tool that resolves to
    `llm.generic` and therefore streams:

    ==================  ==============  ==============================
    property            blocking        streaming
    ==================  ==============  ==============================
    `max_retries: 1`    2 attempts      1 attempt
    `timeout_ms: 200`   enforced        node ran 1.51s and completed
    `output_schema`     status error    tokens emitted, no error
    ==================  ==============  ==============================

    So the earlier claim that streaming specialises token production and
    nothing above it was too strong: the whole node contract above it was
    being skipped, not just the two things `tool_preflight` now covers.

    Retry and timeout are separate normative properties and get separate
    witnesses. Retrying a stream is only meaningful before the first token —
    after that a retry would emit a second answer, which is the boundary
    `emitted_tokens` already draws.
    """

    @staticmethod
    def _aliased(store, owner, **spec_extra):
        """A tool that resolves to `llm.generic`, so it takes the streamed
        path under a name of its own."""
        name = _u("my.chat")
        _tool(store, name, "llm.generic", owner=owner.id, visibility="private",
              **spec_extra)
        return name

    @pytest.mark.asyncio
    async def test_a_streamed_node_retries_per_policy(self, store):
        from liminallm.service.workflow import WorkflowEngine

        rt = get_runtime()
        u = store.create_user(email=f"{_u('rt')}@t.local", tenant_id=_u("rtt"))
        name = self._aliased(store, u)
        wf = store.create_artifact(
            "workflow", _u("rwf"),
            {"kind": "workflow.chat", "entrypoint": "call", "nodes": [
                {"id": "call", "type": "tool_call", "tool": name,
                 "max_retries": 1, "backoff_ms": 10, "next": "fin"},
                {"id": "fin", "type": "end"},
            ]},
            owner_user_id=u.id, visibility="private",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        attempts = []

        def generate_stream(*a, **k):
            attempts.append(1)
            raise RuntimeError("backend down")

        engine.llm.generate_stream = generate_stream
        [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert len(attempts) == 2, (
            f"`max_retries: 1` means two attempts; the streamed node made "
            f"{len(attempts)}"
        )

    @pytest.mark.asyncio
    async def test_a_streamed_node_past_its_timeout_fails(self, store):
        """A synchronous stream iterator that blocks must not outlive the
        node's `timeout_ms`. It also blocks the event loop while it does."""
        import time as _time

        from liminallm.service.workflow import WorkflowEngine

        rt = get_runtime()
        u = store.create_user(email=f"{_u('to')}@t.local", tenant_id=_u("tot"))
        name = self._aliased(store, u)
        wf = store.create_artifact(
            "workflow", _u("twf"),
            {"kind": "workflow.chat", "entrypoint": "call", "nodes": [
                {"id": "call", "type": "tool_call", "tool": name,
                 "timeout_ms": 200, "max_retries": 0, "next": "fin"},
                {"id": "fin", "type": "end"},
            ]},
            owner_user_id=u.id, visibility="private",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)

        def generate_stream(*a, **k):
            def slow():
                _time.sleep(1.5)
                yield {"event": "token", "data": "late"}
                yield {"event": "message_done",
                       "data": {"content": "late", "usage": {}}}

            return slow()

        engine.llm.generate_stream = generate_stream
        start = _time.monotonic()
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        elapsed = _time.monotonic() - start
        assert elapsed < 1.0, (
            f"a node with `timeout_ms: 200` ran for {elapsed:.2f}s; the "
            f"iterator was never interrupted"
        )
        assert not any(e.get("event") == "token" for e in events), events

    @pytest.mark.asyncio
    async def test_a_streamed_nodes_output_is_validated(self, store):
        """SPEC §9.2 validates outputs as well as inputs, and an aliased tool
        with an `output_schema` now legitimately reaches the streamed path."""
        from liminallm.service.workflow import WorkflowEngine

        rt = get_runtime()
        u = store.create_user(email=f"{_u('os')}@t.local", tenant_id=_u("ost"))
        name = self._aliased(store, u, output_schema={
            "type": "object", "required": ["impossible_field"],
        })
        wf = store.create_artifact(
            "workflow", _u("owf"), _wf(name),
            owner_user_id=u.id, visibility="private",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "token", "data": "hi"},
            {"event": "message_done", "data": {"content": "hi", "usage": {}}},
        ])
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        codes = [e["data"].get("code") for e in events
                 if e.get("event") == "error"]
        assert "validation_error" in codes, (
            f"a streamed node returned output its schema forbids: {events[-2:]}"
        )

    @pytest.mark.asyncio
    async def test_an_ordinary_streamed_node_still_streams(self, store):
        """The control. Enforcing the node contract must not stop tokens
        reaching the client on the path that exists to produce them."""
        from liminallm.service.workflow import WorkflowEngine

        rt = get_runtime()
        u = store.create_user(email=f"{_u('ok')}@t.local", tenant_id=_u("okt"))
        wf = store.create_artifact(
            "workflow", _u("okwf3"), _wf("llm.generic"),
            owner_user_id=u.id, visibility="private",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "token", "data": "hi"},
            {"event": "message_done", "data": {"content": "hi", "usage": {}}},
        ])
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert any(e.get("event") == "token" for e in events), events
        assert any(e.get("event") == "message_done" for e in events), events[-3:]
        assert not any(e.get("event") == "error" for e in events), events


class TestAmbiguityThatAppearsLater:
    """Admission cannot be the only altitude that refuses two rows.

    Tool names have no uniqueness constraint, and publishing a second `foo`
    does not revisit the workflows already naming it. This passes today, so it
    is a guard rather than a red: it stops a later change reintroducing a
    first-match at the execution altitude, which the deletion and cache
    witnesses would not catch — they prove runtime asks canonical state, not
    that it handles two canonical answers.
    """

    @pytest.mark.asyncio
    async def test_a_second_row_published_later_fails_closed(self, store):
        from liminallm.service.workflow import WorkflowEngine

        rt = get_runtime()
        u = store.create_user(email=f"{_u('al')}@t.local", tenant_id=_u("alt"))
        name = _u("foo")
        _tool(store, name, "llm.generic", owner=u.id, visibility="global")
        wf = store.create_artifact(
            "workflow", _u("alwf"), _wf(name),
            owner_user_id=u.id, visibility="global",
        )
        # Published after the workflow was admitted.
        _tool(store, name, "agent.code_v1", owner=u.id, visibility="global")

        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        ran = []
        engine._tool_llm_generic = lambda *a, **k: (
            ran.append("llm.generic") or {"status": "ok", "content": "x"}
        )
        engine._tool_agent_code = lambda *a, **k: (
            ran.append("agent.code_v1") or {"status": "ok", "content": "x"}
        )
        out = await engine.run(wf.id, None, "hi", None,
                               user_id=u.id, tenant_id=u.tenant_id)
        assert ran == [], f"an ambiguous reference executed a body: {ran}"
        statuses = [e.get("status") for e in out.get("workflow_trace") or []]
        assert "error" in statuses, statuses


class TestARefusedCreateWritesNothing:
    """A refusal must not leave a payload behind.

    `create_artifact` persists the payload inside the transaction before the
    row is inserted, so a reference check placed after `_persist_payload`
    would leave a file on disk for an artifact that never existed.
    """

    def test_no_payload_directory_survives_a_refusal(self, store):
        u = store.create_user(email=f"{_u('lk')}@t.local", tenant_id=_u("lkt"))
        artifacts_dir = store.fs_root / "artifacts"
        before = set(p.name for p in artifacts_dir.iterdir()) if (
            artifacts_dir.exists()
        ) else set()
        with pytest.raises(Exception):
            store.create_artifact(
                "workflow", _u("leak"), _wf("no.such.tool.v1"),
                owner_user_id=u.id, visibility="private",
            )
        after = set(p.name for p in artifacts_dir.iterdir()) if (
            artifacts_dir.exists()
        ) else set()
        assert after == before, f"a refused create left {after - before}"


class TestAFreshDatabaseStillBoots:
    """The seeded default workflow references four tools that are seeded
    *after* it. Once create-time reference validation exists, a fresh
    database refuses its own defaults unless bootstrap seeds tools first.

    A permanent witness rather than a unit test of the ordering, because what
    matters is that initialization succeeds, not how it is spelled.
    """

    def test_the_default_workflow_and_its_tools_are_present(self, store):
        arts = store.list_artifacts(page_size=500)
        wf = [a for a in arts if a.name == "default_chat_workflow"]
        assert wf, "the seeded default workflow is missing"
        referenced = sorted({
            n.get("tool") for n in wf[0].schema.get("nodes", []) if n.get("tool")
        })
        assert referenced, wf[0].schema
        names = {
            a.schema.get("name") for a in arts
            if isinstance(a.schema, dict) and a.schema.get("kind") == "tool.spec"
        }
        missing = [t for t in referenced if t not in names]
        assert not missing, f"the default workflow references unseeded tools: {missing}"


class TestAStreamedAttemptCanBeStopped:
    """The half of the node contract that a stream makes hard.

    A worker is a process and a kill ends it. A streamed producer is a thread,
    and `asyncio.wait_for` cancels the waiter rather than the work: the loop
    gets control back and the thread keeps producing the answer the next
    attempt is about to replace. So each property below is about the producer,
    not about the coroutine that was waiting for it.
    """

    @staticmethod
    def _aliased(store, owner, **spec_extra):
        name = _u("my.chat")
        _tool(store, name, "llm.generic", owner=owner.id, visibility="private",
              **spec_extra)
        return name

    def _engine_for(self, store, node, tag, **spec_extra):
        """An engine and a private workflow whose one node is `node`."""
        from liminallm.service.workflow import WorkflowEngine

        rt = get_runtime()
        u = store.create_user(email=f"{_u(tag)}@t.local", tenant_id=_u(tag))
        name = self._aliased(store, u, **spec_extra)
        wf = store.create_artifact(
            "workflow", _u(f"{tag}wf"),
            {"kind": "workflow.chat", "entrypoint": "call", "nodes": [
                {"id": "call", "type": "tool_call", "tool": name, **node},
                {"id": "fin", "type": "end"},
            ]},
            owner_user_id=u.id, visibility="private",
        )
        engine = WorkflowEngine(store, rt.llm, rt.router, rt.rag, cache=rt.cache)
        return engine, wf, u

    @pytest.mark.asyncio
    async def test_a_failure_after_the_first_token_is_not_retried(self, store):
        """SPEC §18.3 allows the retry; the transport forbids it.

        A second attempt would append a second answer to a bubble that already
        holds the first, so the boundary is the first token, not the budget.
        """
        engine, wf, u = self._engine_for(
            store, {"max_retries": 2, "backoff_ms": 10, "next": "fin"}, "aft"
        )
        attempts = []

        def generate_stream(*a, **k):
            attempts.append(1)
            yield {"event": "token", "data": "half"}
            raise RuntimeError("cut off mid-sentence")

        engine.llm.generate_stream = generate_stream
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert len(attempts) == 1, (
            f"the node retried after a token had reached the client: "
            f"{len(attempts)} attempts"
        )
        assert [e for e in events if e.get("event") == "token"], events
        assert [e for e in events if e.get("event") == "error"], events[-2:]

    @pytest.mark.asyncio
    async def test_a_timed_out_producer_is_told_to_stop(self, store):
        """Not merely abandoned. The producer must see the stop, or the node
        timeout is a description of what the caller stopped waiting for.

        The producer never ends by itself. A bounded one made this vacuous —
        mutation found it: the loop ran out during the wait, so the `finally`
        fired whether or not anything had asked it to stop.
        """
        import threading
        import time as _time

        engine, wf, u = self._engine_for(
            store, {"timeout_ms": 150, "max_retries": 0, "next": "fin"}, "stp"
        )
        stopped = threading.Event()

        def generate_stream(*a, **k):
            try:
                while True:
                    _time.sleep(0.02)
                    yield {"event": "token", "data": "."}
            finally:
                # Reached when the pump breaks out of its loop and closes the
                # iterator, which is the stop request arriving.
                stopped.set()

        engine.llm.generate_stream = generate_stream
        [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert stopped.wait(5.0), (
            "the producer was never told to stop; the timeout cancelled the "
            "waiter and left the thread producing"
        )

    @pytest.mark.asyncio
    async def test_a_retry_waits_for_the_previous_producer(self, store):
        """Two producers for one node is not a retry, it is a race whose
        winner writes the answer."""
        import threading
        import time as _time

        engine, wf, u = self._engine_for(
            store,
            {"timeout_ms": 120, "max_retries": 1, "backoff_ms": 1, "next": "fin"},
            "wait",
        )
        live = 0
        peak = 0
        lock = threading.Lock()

        def generate_stream(*a, **k):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            try:
                # Past its own timeout, and not interruptible until the sleep
                # returns — the case the confirmation exists for.
                _time.sleep(0.4)
                yield {"event": "token", "data": "late"}
            finally:
                with lock:
                    live -= 1

        engine.llm.generate_stream = generate_stream
        [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert peak == 1, (
            f"{peak} producers ran at once for one node; the retry started "
            f"beside the attempt it replaced"
        )

    @pytest.mark.asyncio
    async def test_a_node_with_an_output_schema_holds_its_tokens(self, store):
        """SPEC §9.2 validates outputs, and a token already on the screen
        cannot be withdrawn. So a node with a schema streams nothing until its
        finished answer passes — the refusal is not enough on its own."""
        engine, wf, u = self._engine_for(
            store, {"next": "fin"}, "hold",
            output_schema={"type": "object", "required": ["impossible_field"]},
        )
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "token", "data": "hi"},
            {"event": "message_done", "data": {"content": "hi", "usage": {}}},
        ])
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert not [e for e in events if e.get("event") == "token"], (
            f"a node whose output its schema forbids still streamed: {events}"
        )

    @pytest.mark.asyncio
    async def test_a_valid_output_still_reaches_the_client(self, store):
        """The complement. Holding tokens back must not mean losing them."""
        engine, wf, u = self._engine_for(
            store, {"next": "fin"}, "pass",
            output_schema={"type": "object", "required": ["content"]},
        )
        engine.llm.generate_stream = lambda *a, **k: iter([
            {"event": "token", "data": "hi"},
            {"event": "message_done", "data": {"content": "hi", "usage": {}}},
        ])
        events = [e async for e in engine.run_streaming(
            wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        assert [e for e in events if e.get("event") == "token"], events
        assert not [e for e in events if e.get("event") == "error"], events

    @pytest.mark.asyncio
    async def test_a_backend_that_cannot_be_stopped_does_not_stream(self, store):
        """`LocalJaxLoRABackend` runs the whole forward pass before its first
        yield, so no scheduling makes `timeout_ms` enforceable against it. The
        answer still arrives — through the executor that runs the body in a
        worker a kill does end."""
        engine, wf, u = self._engine_for(store, {"next": "fin"}, "nc")
        engine.llm.backend.supports_stream_cancel = False
        streamed = []
        engine.llm.generate_stream = lambda *a, **k: (
            streamed.append(1) or iter([
                {"event": "token", "data": "hi"},
                {"event": "message_done", "data": {"content": "hi", "usage": {}}},
            ])
        )
        try:
            events = [e async for e in engine.run_streaming(
                wf.id, None, "hi", None, user_id=u.id, tenant_id=u.tenant_id)]
        finally:
            del engine.llm.backend.supports_stream_cancel
        assert not streamed, "a backend that cannot be stopped was streamed anyway"
        assert not [e for e in events if e.get("event") == "error"], events
        done = [e for e in events if e.get("event") == "message_done"]
        assert done and done[-1]["data"].get("content"), (
            f"the fallback produced no answer: {events[-2:]}"
        )
