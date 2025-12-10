from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Dict, Optional

from liminallm.logging import get_logger
from liminallm.service.errors import BadRequestError, NotFoundError
from liminallm.service.llm import LLMService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import Artifact, ConfigPatchAudit
from liminallm.storage.postgres import PostgresStore

logger = get_logger(__name__)

MAX_LIST_EXTENSION = 1024


class ConfigOpsService:
    """ConfigOps helper for LLM-as-architect patch proposals and application."""

    def __init__(
        self,
        store: PostgresStore | MemoryStore,
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

        This operation is performed in two steps:
        1. Update the artifact schema with the patch
        2. Mark the patch as applied

        Error Recovery:
            If step 2 fails after step 1 succeeds, the artifact is still modified
            but the patch status remains unchanged. In this case, we log the error
            and return with a warning flag to indicate partial success.
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

        # Step 1: Apply patch to artifact (pure function)
        new_schema = self._apply_patch_to_schema(artifact.schema, patch.patch)

        # Step 2: Persist schema and mark patch applied atomically when supported
        applied_patch = None
        status_update_failed = False
        try:
            if hasattr(self.store, "apply_config_patch"):
                updated, applied_patch = self.store.apply_config_patch(  # type: ignore[attr-defined]
                    patch,
                    new_schema,
                    artifact_description=artifact.description,
                    approver_user_id=approver_user_id,
                )
            else:
                updated = self.store.update_artifact(
                    artifact.id, new_schema, artifact.description
                )
                applied_patch = self.store.update_config_patch_status(
                    patch_id,
                    "applied",
                    meta={"applied_by": approver_user_id}
                    if approver_user_id
                    else None,
                    mark_applied=True,
                )
        except Exception as exc:
            status_update_failed = True
            logger.error(
                "config_patch_status_update_failed",
                patch_id=patch_id,
                artifact_id=artifact.id,
                error=str(exc),
                message="Artifact was updated but patch status could not be marked as applied",
            )
            if not applied_patch:
                applied_patch = self.store.get_config_patch(patch_id)

        result = {"artifact": updated, "patch": applied_patch or patch}
        if status_update_failed:
            result["warning"] = "Patch applied but status update failed - patch may appear unapplied"

        return result

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
            f"You are a config engineer. Given the artifact named '{artifact.name}' of type {artifact.type}, propose a JSON patch.\n"
            f"Artifact description: {description}\n"
            f"Existing schema (truncated to 2KB): {self._safe_truncate_json(artifact.schema, 2000)}\n"
            f"Goal: {goal_line}\n"
            f"Preference insights: {summary_blob}\n"
            "Respond with JSON representing a JSON-patch style object."
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
        timestamp = datetime.utcnow().isoformat()
        return {
            "ops": [
                {
                    "op": "add",
                    "path": "/meta/llm_autopatch",
                    "value": {
                        "generated_at": timestamp,
                        "note": "Auto-tuned routing weights",
                    },
                }
            ]
        }

    def _apply_patch_to_schema(self, schema: dict, patch: dict) -> dict:
        if not patch:
            return schema
        working = copy.deepcopy(schema)
        ops = patch.get("ops") if isinstance(patch, dict) else None
        if isinstance(ops, list):
            for op in ops:
                self._apply_single_op(working, op)
            return working
        if isinstance(patch, dict):
            return self._deep_merge(working, patch)
        return working

    def _apply_single_op(self, doc: dict, op: Dict[str, Any]) -> None:
        action = (op or {}).get("op")
        path = (op or {}).get("path", "")
        value = op.get("value")
        if not action or not path:
            return
        segments = [seg for seg in path.strip("/").split("/") if seg]
        parent = doc
        for seg in segments[:-1]:
            if isinstance(parent, list):
                try:
                    idx = int(seg)
                except ValueError:
                    return
                self._ensure_list_capacity(parent, idx, path)
                while len(parent) <= idx:
                    parent.append({})
                parent = parent[idx]
            else:
                parent = parent.setdefault(seg, {})
        key = segments[-1] if segments else ""
        if action == "add" or action == "replace":
            if isinstance(parent, list):
                if key == "-":
                    parent.append(value)
                else:
                    try:
                        idx = int(key)
                        self._ensure_list_capacity(parent, idx, path)
                        if idx < len(parent):
                            parent[idx] = value
                        else:
                            parent.append(value)
                    except ValueError:
                        return
            else:
                parent[key] = value
        elif action == "remove":
            if isinstance(parent, list):
                try:
                    idx = int(key)
                    self._ensure_list_capacity(parent, idx, path)
                    if 0 <= idx < len(parent):
                        parent.pop(idx)
                except ValueError:
                    return
            else:
                parent.pop(key, None)

    def _ensure_list_capacity(self, parent: list, idx: int, path: str) -> None:
        if idx < 0:
            raise BadRequestError(
                "negative list index", detail={"path": path, "index": idx}
            )
        if idx >= MAX_LIST_EXTENSION:
            raise BadRequestError(
                "list index too large",
                detail={
                    "path": path,
                    "index": idx,
                    "max_index": MAX_LIST_EXTENSION - 1,
                },
            )

    def _deep_merge(self, base: dict, patch: dict) -> dict:
        merged = dict(base)
        for key, value in patch.items():
            if key == "ops":
                continue
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merged[key] = self._deep_merge(base[key], value)
            else:
                merged[key] = value
        return merged
