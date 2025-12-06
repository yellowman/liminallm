from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Optional

import httpx

from liminallm.logging import get_logger

logger = get_logger(__name__)


class VoiceService:
    """Voice transcription and synthesis service with OpenAI API integration.

    Supports:
    - Transcription via OpenAI Whisper API
    - Synthesis via OpenAI TTS API
    - Fallback to placeholder mode when API key is not configured

    Audio files are stored in the shared filesystem under voice/{user_id}/.
    """

    OPENAI_API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        fs_root: str,
        *,
        api_key: Optional[str] = None,
        transcription_model: str = "whisper-1",
        synthesis_model: str = "tts-1",
        default_voice: str = "alloy",
    ) -> None:
        self.fs_root = Path(fs_root)
        self.api_key = api_key
        self.transcription_model = transcription_model
        self.synthesis_model = synthesis_model
        self.default_voice = default_voice
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if the service has a valid API key configured."""
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client for API calls."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        return self._client

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        user_id: Optional[str] = None,
        language: Optional[str] = None,
        format: str = "wav",
    ) -> dict:
        """Transcribe audio to text using OpenAI Whisper API.

        Args:
            audio_bytes: Raw audio data
            user_id: Optional user ID for tracking
            language: Optional language code (e.g., 'en', 'es')
            format: Audio format (wav, mp3, m4a, etc.)

        Returns:
            Dict with transcript, duration_ms, and metadata
        """
        if not self.is_configured:
            # Fallback placeholder mode
            logger.warning("voice_transcribe_no_api_key", user_id=user_id)
            return self._placeholder_transcribe(audio_bytes, user_id=user_id)

        try:
            client = await self._get_client()

            # Prepare multipart form data
            files = {
                "file": (f"audio.{format}", io.BytesIO(audio_bytes), f"audio/{format}"),
                "model": (None, self.transcription_model),
            }
            if language:
                files["language"] = (None, language)

            response = await client.post(
                f"{self.OPENAI_API_BASE}/audio/transcriptions",
                files=files,
            )
            response.raise_for_status()
            data = response.json()

            transcript = data.get("text", "")
            # OpenAI doesn't return duration, estimate from audio size
            duration_ms = self._estimate_duration(audio_bytes)

            logger.info(
                "voice_transcribe_success",
                user_id=user_id,
                transcript_length=len(transcript),
                duration_ms=duration_ms,
            )

            return {
                "transcript": transcript,
                "duration_ms": duration_ms,
                "user_id": user_id,
                "model": self.transcription_model,
                "language": language,
            }

        except httpx.HTTPStatusError as e:
            # Extract error details from response if available
            error_body = None
            try:
                error_body = e.response.json()
            except Exception:
                pass
            logger.error(
                "voice_transcribe_api_error",
                user_id=user_id,
                status_code=e.response.status_code,
                error=str(e),
                error_body=error_body,
                model=self.transcription_model,
            )
            raise ValueError(f"Transcription failed: {e.response.status_code}") from e
        except httpx.TimeoutException as e:
            logger.error(
                "voice_transcribe_timeout",
                user_id=user_id,
                audio_size=len(audio_bytes),
                model=self.transcription_model,
                error=str(e),
            )
            raise ValueError("Transcription timed out") from e
        except httpx.ConnectError as e:
            logger.error(
                "voice_transcribe_connect_error",
                user_id=user_id,
                api_base=self.OPENAI_API_BASE,
                error=str(e),
            )
            raise ValueError("Failed to connect to transcription service") from e
        except Exception as e:
            logger.error(
                "voice_transcribe_error",
                user_id=user_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise ValueError(f"Transcription failed: {str(e)}") from e

    async def synthesize(
        self,
        text: str,
        *,
        user_id: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> dict:
        """Synthesize text to speech using OpenAI TTS API.

        Args:
            text: Text to synthesize
            user_id: Optional user ID for tracking and storage
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speech speed (0.25 to 4.0)

        Returns:
            Dict with audio_path, format, sample_rate, duration_ms, voice
        """
        voice = voice or self.default_voice

        if not self.is_configured:
            # Fallback placeholder mode
            logger.warning("voice_synthesize_no_api_key", user_id=user_id)
            return self._placeholder_synthesize(text, user_id=user_id, voice=voice)

        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.OPENAI_API_BASE}/audio/speech",
                json={
                    "model": self.synthesis_model,
                    "input": text,
                    "voice": voice,
                    "speed": max(0.25, min(4.0, speed)),
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()

            # Store the audio file
            voice_dir = self.fs_root / "voice" / (user_id or "shared")
            voice_dir.mkdir(parents=True, exist_ok=True)
            file_id = uuid.uuid4()
            file_path = voice_dir / f"{file_id}.mp3"
            file_path.write_bytes(response.content)

            # Estimate duration from MP3 size (rough: ~16kbps for speech)
            duration_ms = len(response.content) * 8 // 16

            logger.info(
                "voice_synthesize_success",
                user_id=user_id,
                text_length=len(text),
                audio_size=len(response.content),
                voice=voice,
            )

            return {
                "audio_path": str(file_path),
                "audio_url": f"/voice/{user_id or 'shared'}/{file_id}.mp3",
                "format": "audio/mpeg",
                "sample_rate": 24000,
                "duration_ms": duration_ms,
                "voice": voice,
                "model": self.synthesis_model,
            }

        except httpx.HTTPStatusError as e:
            error_body = None
            try:
                error_body = e.response.json()
            except Exception:
                pass
            logger.error(
                "voice_synthesize_api_error",
                user_id=user_id,
                status_code=e.response.status_code,
                error=str(e),
                error_body=error_body,
                model=self.synthesis_model,
                voice=voice,
            )
            raise ValueError(f"Synthesis failed: {e.response.status_code}") from e
        except httpx.TimeoutException as e:
            logger.error(
                "voice_synthesize_timeout",
                user_id=user_id,
                text_length=len(text),
                model=self.synthesis_model,
                voice=voice,
                error=str(e),
            )
            raise ValueError("Synthesis timed out") from e
        except httpx.ConnectError as e:
            logger.error(
                "voice_synthesize_connect_error",
                user_id=user_id,
                api_base=self.OPENAI_API_BASE,
                error=str(e),
            )
            raise ValueError("Failed to connect to synthesis service") from e
        except OSError as e:
            logger.error(
                "voice_synthesize_file_error",
                user_id=user_id,
                fs_root=str(self.fs_root),
                error=str(e),
            )
            raise ValueError(f"Failed to store audio file: {str(e)}") from e
        except Exception as e:
            logger.error(
                "voice_synthesize_error",
                user_id=user_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise ValueError(f"Synthesis failed: {str(e)}") from e

    def _placeholder_transcribe(
        self, audio_bytes: bytes, *, user_id: Optional[str] = None
    ) -> dict:
        """Placeholder transcription when API key is not configured."""
        # Try to decode as text (for testing), otherwise return size info
        try:
            transcript = audio_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            transcript = f"[audio placeholder: {len(audio_bytes)} bytes]"

        return {
            "transcript": transcript,
            "duration_ms": self._estimate_duration(audio_bytes),
            "user_id": user_id,
            "model": "placeholder",
            "language": None,
        }

    def _placeholder_synthesize(
        self, text: str, *, user_id: Optional[str] = None, voice: str = "default"
    ) -> dict:
        """Placeholder synthesis when API key is not configured."""
        voice_dir = self.fs_root / "voice" / (user_id or "shared")
        voice_dir.mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4()
        file_path = voice_dir / f"{file_id}.txt"

        # Store text as placeholder
        payload = f"voice={voice}\n{text}"
        file_path.write_text(payload)

        return {
            "audio_path": str(file_path),
            "audio_url": f"/voice/{user_id or 'shared'}/{file_id}.txt",
            "format": "text/placeholder",
            "sample_rate": 16000,
            "duration_ms": max(500, len(text) * 20),
            "voice": voice,
            "model": "placeholder",
        }

    def _estimate_duration(self, audio_bytes: bytes) -> int:
        """Estimate audio duration from byte size.

        Assumes ~16kbps for typical speech audio.
        """
        return max(1000, len(audio_bytes) * 8 // 16)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
