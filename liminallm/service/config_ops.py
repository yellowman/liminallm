from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from liminallm.logging import get_logger
from liminallm.service import json_patch
from liminallm.service.errors import BadRequestError, NotFoundError
from liminallm.service.llm import LLMService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.storage.models import Artifact, ConfigPatchAudit
from liminallm.storage.postgres import PostgresStore

logger = get_logger(__name__)



class ConfigOpsService:
    """ConfigOps helper for LLM-as-architect patch proposals and application."""

    def __init__(
        self,
        store: PostgresStore,
        llm: LLMService,
        router: RouterEngine,
        training: TrainingService,
    ) -> None:
        self.store = store
        self.llm = llm
        self.router = router
        self.training = training

    def auto_generate_patch(
        self, artifact_id: str, user_id: Optional[str], goal: Optional[str] = None
    ) -> ConfigPatchAudit:
        artifact = self.store.get_artifact(artifact_id)
        if not artifact:
            raise NotFoundError(
                "artifact not found", detail={"artifact_id": artifact_id}
            )
        prompt = self._build_prompt(artifact, goal)
        patch = self._run_llm_for_patch(prompt)
        proposer = "user" if user_id else "system_llm"
        return self.store.record_config_patch(
            artifact_id=artifact_id,
            proposer=proposer,
            patch=patch,
            justification=goal or "auto-proposed",
        )

    def decide_patch(
        self, patch_id: int, decision: str, reason: Optional[str] = None
    ) -> ConfigPatchAudit:
        # Validate patch exists and is in pending status
        patch = self.store.get_config_patch(patch_id)
        if not patch:
            raise NotFoundError("patch not found", detail={"patch_id": patch_id})
        if patch.status != "pending":
            raise BadRequestError(
                "patch not in pending status",
                detail={"patch_id": patch_id, "current_status": patch.status},
            )

        normalized = decision.lower()
        if normalized in {"approve", "approved"}:
            normalized = "approved"
        elif normalized in {"reject", "rejected"}:
            normalized = "rejected"
        else:
            raise BadRequestError("invalid decision", detail={"decision": decision})
        updated = self.store.update_config_patch_status(
            patch_id,
            normalized,
            meta={"reason": reason} if reason else None,
            mark_decided=True,
        )
        if not updated:
            raise NotFoundError("patch not found", detail={"patch_id": patch_id})
        return updated

    def apply_patch(
        self, patch_id: int, approver_user_id: Optional[str] = None
    ) -> dict:
        """Apply an approved patch to its target artifact.

        The schema update, the historical version and the patch's status are
        one transaction in the store, and the store validates the result
        before writing any of them - so there is no partial state to report
        and no invalid schema to persist. A failure here changed nothing.
        """
        patch = self.store.get_config_patch(patch_id)
        if not patch:
            raise NotFoundError("patch not found", detail={"patch_id": patch_id})
        # Security: Only apply approved patches
        if patch.status != "approved":
            raise BadRequestError(
                "patch must be approved before applying",
                detail={"patch_id": patch_id, "current_status": patch.status},
            )
        artifact = self.store.get_artifact(patch.artifact_id)
        if not artifact:
            raise NotFoundError(
                "artifact missing", detail={"artifact_id": patch.artifact_id}
            )

        # The patch is applied inside the store's transaction, against the
        # schema the store reads under the artifact lock - not against the
        # `artifact` above. That read answers the request; it is not what the
        # write is derived from, because anything committed between the two
        # would otherwise be overwritten by a document built from the older
        # row. SPEC §10.1 says apply loads the *current* schema.
        updated, applied_patch = self.store.apply_config_patch(
            patch,
            lambda current: self._apply_patch_to_schema(current, patch.patch),
            approver_user_id=approver_user_id,
        )
        return {"artifact": updated, "patch": applied_patch or patch}

    def _build_prompt(self, artifact: Artifact, goal: Optional[str]) -> str:
        insights = (
            self.training.summarize_preferences(artifact.owner_user_id)
            if artifact.owner_user_id
            else {}
        )
        if isinstance(insights, dict) and insights.get("status") == "error":
            summary_blob = f"preference retrieval error: {insights.get('error')}"
        elif isinstance(insights, dict) and insights.get("status") == "no_data":
            summary_blob = (
                "no preference data available; request feedback before risky changes"
            )
        else:
            summary_blob = (
                json.dumps(insights, indent=2)
                if insights
                else "no preference insights available"
            )
        description = artifact.description or artifact.name
        goal_line = goal or "improve routing quality and adapter selection accuracy"
        return (
            f"You are a config engineer for the artifact '{artifact.name}' ({artifact.type}).\n"
            f"Artifact description: {description}\n"
            f"Existing schema (truncated to 2KB): {self._safe_truncate_json(artifact.schema, 2000)}\n"
            f"Goal: {goal_line}\n"
            f"Preference insights: {summary_blob}\n"
            "Respond with only a JSON-patch style object."
        )

    def _run_llm_for_patch(self, prompt: str) -> dict:
        try:
            response = self.llm.generate(prompt, adapters=[], context_snippets=[])
            content = response.get("content", "{}")
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            logger.warning("config_patch_llm_error", error=str(exc))
        return self._fallback_patch()

    def _safe_truncate_json(self, obj: dict, max_chars: int) -> str:
        """Truncate JSON while maintaining valid syntax by removing trailing keys."""
        full_json = json.dumps(obj, indent=None, separators=(",", ":"))
        if len(full_json) <= max_chars:
            return full_json
        # Truncate by removing keys from a shallow copy until it fits
        if isinstance(obj, dict):
            truncated: dict = {}
            # Reserve space for the _truncated marker
            marker_overhead = len(',"_truncated":true')
            target_max = max_chars - marker_overhead
            for key, value in obj.items():
                # Test size before adding
                test_dict = {**truncated, key: value}
                test_json = json.dumps(test_dict, indent=None, separators=(",", ":"))
                if len(test_json) > target_max:
                    break
                truncated[key] = value
            truncated["_truncated"] = True
            return json.dumps(truncated, indent=None, separators=(",", ":"))
        return full_json[: max_chars - 3] + "..."

    def _fallback_patch(self) -> dict:
        """The patch proposed when the model does not produce one.

        One leaf op, deliberately. See `json_patch.meta_ops` for why this must
        not carry an `add /meta` alongside it.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "ops": json_patch.meta_ops(
                "llm_autopatch",
                {
                    "generated_at": timestamp,
                    "note": "Auto-tuned routing weights",
                },
            )
        }

    def _apply_patch_to_schema(self, schema: dict, patch: dict) -> dict:
        if not patch:
            return schema
        working = copy.deepcopy(schema)
        ops = patch.get("ops") if isinstance(patch, dict) else None
        if isinstance(ops, list):
            # This loops apply_op rather than apply_ops, so the whole-patch
            # rule has to be asked for explicitly or an empty `ops` reaches
            # the store as a patch marked applied.
            json_patch.validate_ops(ops)
            for op in ops:
                self._apply_single_op(working, op)
            return working
        if isinstance(patch, dict):
            return self._deep_merge(working, patch)
        return working

    def _apply_single_op(self, doc: dict, op: Dict[str, Any]) -> None:
        json_patch.apply_op(doc, op)

    def _deep_merge(self, base: dict, patch: dict) -> dict:
        return json_patch.deep_merge(base, patch, skip_keys=("ops",))
