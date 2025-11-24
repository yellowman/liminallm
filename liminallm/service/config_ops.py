from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from liminallm.service.llm import LLMService
from liminallm.service.router import RouterEngine
from liminallm.service.training import TrainingService
from liminallm.storage.errors import ConstraintViolation
from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import Artifact, ConfigPatchAudit
from liminallm.storage.postgres import PostgresStore

logger = logging.getLogger(__name__)


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

    def auto_generate_patch(self, artifact_id: str, user_id: Optional[str], goal: Optional[str] = None) -> ConfigPatchAudit:
        artifact = self.store.get_artifact(artifact_id)
        if not artifact:
            raise ConstraintViolation("artifact not found", {"artifact_id": artifact_id})
        prompt = self._build_prompt(artifact, goal)
        patch = self._run_llm_for_patch(prompt)
        return self.store.record_config_patch(artifact_id=artifact_id, proposer_user_id=user_id, patch=patch, justification=goal or "auto-proposed")

    def decide_patch(self, patch_id: str, decision: str, reason: Optional[str] = None) -> Optional[ConfigPatchAudit]:
        normalized = decision.lower()
        if normalized not in {"approved", "rejected"}:
            raise ConstraintViolation("invalid decision", {"decision": decision})
        return self.store.update_config_patch_status(
            patch_id,
            normalized,
            meta={"reason": reason} if reason else None,
            mark_decided=True,
        )

    def apply_patch(self, patch_id: str, approver_user_id: Optional[str] = None) -> Optional[dict]:
        patch = self.store.get_config_patch(patch_id)
        if not patch:
            return None
        artifact = self.store.get_artifact(patch.artifact_id)
        if not artifact:
            raise ConstraintViolation("artifact missing", {"artifact_id": patch.artifact_id})
        new_schema = self._apply_patch_to_schema(artifact.schema, patch.patch)
        updated = self.store.update_artifact(artifact.id, new_schema, artifact.description)
        applied_patch = self.store.update_config_patch_status(
            patch_id,
            "applied",
            meta={"applied_by": approver_user_id} if approver_user_id else None,
            mark_applied=True,
        )
        return {"artifact": updated, "patch": applied_patch or patch}

    def _build_prompt(self, artifact: Artifact, goal: Optional[str]) -> str:
        insights = self.training.summarize_preferences(artifact.owner_user_id) if artifact.owner_user_id else {}
        summary_blob = json.dumps(insights, indent=2) if insights else "no preference insights available"
        description = artifact.description or artifact.name
        goal_line = goal or "improve routing quality and adapter selection accuracy"
        return (
            f"You are a config engineer. Given the artifact named '{artifact.name}' of type {artifact.type}, propose a JSON patch.\n"
            f"Artifact description: {description}\n"
            f"Existing schema (truncated to 2KB): {json.dumps(artifact.schema)[:2000]}\n"
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
            logger.warning("Falling back to default patch after LLM error: %s", exc)
        return self._fallback_patch()

    def _fallback_patch(self) -> dict:
        timestamp = datetime.utcnow().isoformat()
        return {
            "ops": [
                {
                    "op": "add",
                    "path": "/meta/llm_autopatch",
                    "value": {"generated_at": timestamp, "note": "Auto-tuned routing weights"},
                }
            ]
        }

    def _apply_patch_to_schema(self, schema: dict, patch: dict) -> dict:
        if not patch:
            return schema
        working = json.loads(json.dumps(schema))
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
                    if 0 <= idx < len(parent):
                        parent.pop(idx)
                except ValueError:
                    return
            else:
                parent.pop(key, None)

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
