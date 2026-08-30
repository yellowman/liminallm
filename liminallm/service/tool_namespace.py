"""What names are executable, and whose namespace a reference resolves in.

Pure, and shared by both altitudes on purpose. Admission cannot instantiate a
`WorkflowEngine` to ask what is executable, and the engine must not keep a
second copy of the answer - two lists of "executable" is the failure this
tranche exists to remove, not one to introduce.

The executable set really is split in two. Some tool bodies run in the worker
process (`tool_worker.BODY_NAMES`); the rest run in the parent, because they
are broad reads of the store with no model-chosen control flow in them. A
reference is executable if it reaches either.

Resolution scope is the other half. A tool reference is resolved in the
*workflow's* namespace, not the runner's: a published workflow that resolved
`foo` differently for each caller would name a different capability for each
of them. Measured before this existed - a global `foo` handled by
`llm.generic` and Bob's private `foo` handled by `agent.code_v1` gave Bob a
different body for the same shared workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from liminallm.service import tool_worker

#: Tool bodies that run in the parent. Kept beside the worker's set so the two
#: halves of "executable" are named in one place; `WorkflowEngine` is tested
#: against this rather than being its own authority.
HOST_TOOL_HANDLER_NAMES = frozenset({
    "llm.generic",
    "llm.generic_chat_v1",
    "rag.answer_with_context_v1",
    "llm.intent_classifier_v1",
    "agent.code_v1",
    "workflow.end",
})

#: Every handler a `tool.spec` may name and actually reach. A spec whose
#: `handler` is outside this set resolves as a name and executes as nothing:
#: name existence and handler executability are two stages of one claim.
EXECUTABLE_HANDLER_NAMES = frozenset(tool_worker.BODY_NAMES) | HOST_TOOL_HANDLER_NAMES

#: Handlers the streaming path can produce tokens for. Compared against the
#: *resolved* handler, never against the reference spelling - comparing the
#: spelling let a tenant-shared override of the name `llm.generic` stream the
#: model while the blocking path ran the override's real body, so the `stream`
#: flag chose the capability.
STREAMABLE_HANDLER_NAMES = frozenset({
    "llm.generic",
    "llm.generic_chat_v1",
    "agent.files_v1",
})

#: Precedence by the workflow's own visibility. A tier that matches nothing
#: falls through to the next; two matches in one tier is ambiguous and fails
#: closed, because otherwise `created_at` decides which body runs.
_TIERS: dict[str, Tuple[str, ...]] = {
    "private": ("private", "shared", "global"),
    "shared": ("shared", "global"),
    "global": ("global",),
}


@dataclass(frozen=True)
class ToolResolutionScope:
    """The namespace one workflow's tool references resolve in.

    Passed through every execution path rather than stored on the engine: one
    engine serves concurrent requests for different workflows, and a scope on
    the engine would be whichever request wrote it last.
    """

    visibility: str
    owner_user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    @property
    def tiers(self) -> Tuple[str, ...]:
        """Visibility tiers to search, in order.

        An unrecognized visibility gets the narrowest namespace rather than
        the widest: a row whose visibility this build does not understand must
        not thereby reach more tools.
        """
        return _TIERS.get(self.visibility, ("global",))


#: The namespace for workflows the system synthesises for itself. They are not
#: artifacts, so they have no publisher and no tenant - the global namespace is
#: the only one they can mean.
SYSTEM_SCOPE = ToolResolutionScope(visibility="global")


@dataclass(frozen=True)
class ResolvedWorkflow:
    """A workflow to execute, and the namespace its references mean.

    One object rather than two return values, because the scope must not be
    reconstructed from the runner further down: the whole point is that a
    published workflow means the same thing whoever runs it, and rebuilding
    the scope at the point of use is how it would stop.
    """

    schema: dict
    tool_scope: ToolResolutionScope


def resolve_executable_handler(
    tool_name: str, tool_spec: Optional[dict]
) -> Optional[str]:
    """The body `tool_name` runs, or ``None`` when nothing runs it.

    The one answer, asked by admission, by blocking execution and by streaming
    dispatch, because three implementations of "what will this run" is three
    chances to approve one body and execute another.

    A persisted spec's `handler` is authoritative. `_resolve_worker_tool` used
    to check `tool_name in BODY_NAMES` *first*, so a spec named
    `notes.search_v1` with handler `llm.generic` ran the notes body - the
    reference's spelling beat the row that was actually resolved, and
    admission had approved the other one.

    With no spec at all the literal name is the answer, which keeps a builtin
    reachable when nothing is persisted behind it.
    """
    handler = (tool_spec or {}).get("handler")
    if isinstance(handler, str) and handler:
        return handler if handler in EXECUTABLE_HANDLER_NAMES else None
    return tool_name if tool_name in EXECUTABLE_HANDLER_NAMES else None


@dataclass(frozen=True)
class ToolDescriptor:
    """A resolved tool and where its authority comes from.

    `artifact_id`/`owner_user_id`/`owner_role` are read from the persisted
    artifact row. SPEC §18 makes `privileged:true` a property of an
    *admin-owned artifact*, so the authority cannot be read out of the spec
    the caller supplied - a `privileged: true` key is only a claim until an
    admin-owned row is standing behind it.

    A seeded system tool is ownerless by design, so it resolves and is never
    privileged. That is correct, not a gap to close by inventing an owner.
    """

    name: str
    schema: dict
    artifact_id: Optional[str]
    owner_user_id: Optional[str]
    owner_role: Optional[str]

    @property
    def privileged(self) -> bool:
        return bool((self.schema or {}).get("privileged"))

    @property
    def admin_owned(self) -> bool:
        return bool(self.artifact_id) and self.owner_role == "admin"

    @property
    def handler(self) -> Optional[str]:
        """The body this spec runs, or ``None`` when nothing runs it."""
        return resolve_executable_handler(self.name, self.schema)

    @property
    def executable(self) -> bool:
        return self.handler is not None

    @property
    def streamable(self) -> bool:
        return self.handler in STREAMABLE_HANDLER_NAMES
