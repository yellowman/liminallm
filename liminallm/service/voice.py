from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional


class VoiceService:
    """Minimal voice interface stub for transcription and synthesis.

    This placeholder skips actual ASR/TTS. It should be replaced by a
    hardened client that streams audio to a provider (or local model),
    enforces per-user quotas, and stores generated artifacts in durable
    object storage instead of the local filesystem.
    """

    def __init__(self, fs_root: str) -> None:
        self.fs_root = Path(fs_root)

    def transcribe(self, audio_bytes: bytes, *, user_id: Optional[str] = None) -> dict:
        transcript = audio_bytes.decode("utf-8", errors="ignore") or f"[audio {len(audio_bytes)} bytes]"
        duration_ms = min(len(audio_bytes) * 2, 120_000)
        return {"transcript": transcript, "duration_ms": duration_ms, "user_id": user_id}

    def synthesize(self, text: str, *, user_id: Optional[str] = None, voice: Optional[str] = None) -> dict:
        voice_dir = self.fs_root / "voice" / (user_id or "shared")
        voice_dir.mkdir(parents=True, exist_ok=True)
        file_path = voice_dir / f"{uuid.uuid4()}.txt"
        payload = f"voice={voice or 'default'}\n{text}"
        file_path.write_text(payload)
        duration_ms = max(500, len(text) * 20)
        return {
            "audio_path": str(file_path),
            "format": "text/placeholder",
            "sample_rate": 16000,
            "duration_ms": duration_ms,
            "voice": voice or "default",
        }
