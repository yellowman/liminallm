from __future__ import annotations

import random
from typing import List, Optional

from liminallm.storage.models import Message

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAI = None  # type: ignore


class LLMService:
    """LLM executor with OpenAI-compatible calls and safe fallbacks."""

    def __init__(self, base_model: str, mode: str = "cloud", api_key: Optional[str] = None, base_url: Optional[str] = None, adapter_server_model: Optional[str] = None):
        self.base_model = base_model
        self.mode = mode
        self.adapter_server_model = adapter_server_model
        self.client = None
        if api_key and OpenAI:
            self.client = OpenAI(api_key=api_key, base_url=base_url)  # type: ignore[call-arg]

    def generate(self, prompt: str, adapters: List[str], context_snippets: List[str], history: Optional[List[Message]] = None) -> dict:
        messages = [{"role": "system", "content": "You are a concise assistant."}]
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages = self._inject_context(messages, context_snippets)
        else:
            messages.append({"role": "user", "content": self._format_user(prompt, context_snippets)})
        if self.client:
            extra_body = {}
            if self.mode == "adapter" and adapters:
                extra_body["adapter_id"] = adapters
            model = self.adapter_server_model or self.base_model
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                extra_body=extra_body or None,
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
                "total_tokens": getattr(completion.usage, "total_tokens", 0),
            }
            return {"content": content, "usage": usage}
        # fallback stub if no client or key
        history_txt = " | ".join([m.content for m in history]) if history else ""
        content = (
            f"[base={self.base_model} adapters={','.join(adapters)}] {history_txt} {prompt}"
            f"\nContext: {' | '.join(context_snippets)}"
        )
        return {
            "content": content,
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": random.randint(5, 20)},
        }

    def _format_user(self, prompt: str, context_snippets: List[str]) -> str:
        if context_snippets:
            return f"{prompt}\nContext: {' | '.join(context_snippets)}"
        return prompt

    def _inject_context(self, messages: List[dict], context_snippets: List[str]) -> List[dict]:
        if not context_snippets:
            return messages
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = f"{msg.get('content', '')}\nContext: {' | '.join(context_snippets)}"
                return messages
        messages.append({"role": "system", "content": f"Context: {' | '.join(context_snippets)}"})
        return messages
