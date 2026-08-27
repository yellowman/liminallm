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


def _u(p):
    return f"{p}_{uuid.uuid4().hex[:8]}"


def _tool(store, name, handler, *, owner, visibility):
    return store.create_artifact(
        "tool", _u("t"),
        {"kind": "tool.spec", "name": name, "handler": handler},
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
        assert "tool" in str(exc.value).lower() or "reference" in str(exc.value).lower()

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
        d = engine._resolve_tool("llm.generic", user_id=None, tenant_id=None)
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
        d = engine._resolve_tool(name, user_id=u.id, tenant_id=u.tenant_id)
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

        d = engine._resolve_tool(name, user_id=u.id, tenant_id=u.tenant_id)
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
        from liminallm.service.tool_namespace import (
            EXECUTABLE_HANDLER_NAMES,
            HOST_TOOL_HANDLER_NAMES,
        )

        from liminallm.service import tool_worker
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
