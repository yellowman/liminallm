"""The voice half of a chat turn, and the signed download URL.

Both sat in the uncovered islands of otherwise-covered modules: `_transcribe`
is how spoken audio becomes a user message, and `validate_signed_url` is the
whole authorization check on a file download link.
"""

from __future__ import annotations

import base64
import time

import pytest
from fastapi import HTTPException

from liminallm.api import chat_turn
from liminallm.service.fs import generate_signed_url, validate_signed_url
from liminallm.service.runtime import get_runtime

# ---------------------------------------------------------------------------
# Voice: _transcribe
# ---------------------------------------------------------------------------


class _Voice:
    def __init__(self, reply):
        self._reply = reply
        self.received = None

    async def transcribe(self, audio_bytes, *, user_id):
        self.received = (audio_bytes, user_id)
        return self._reply


@pytest.fixture
def runtime():
    return get_runtime()


@pytest.mark.asyncio
async def test_audio_becomes_the_message_text(runtime, monkeypatch):
    voice = _Voice({"transcript": "hello from a microphone", "language": "en"})
    monkeypatch.setattr(runtime, "voice", voice, raising=False)
    payload = base64.b64encode(b"RIFF....WAVE").decode()

    text, meta = await chat_turn._transcribe(runtime, payload, "u1")

    assert text == "hello from a microphone"
    assert meta["mode"] == "voice"
    assert meta["transcript"]["language"] == "en"
    assert voice.received == (b"RIFF....WAVE", "u1")


@pytest.mark.asyncio
async def test_a_provider_that_answers_with_text_key_still_works(runtime, monkeypatch):
    """Whisper-style providers say "text"; ours says "transcript". Both are
    accepted so a provider swap does not silently produce empty turns."""
    monkeypatch.setattr(runtime, "voice", _Voice({"text": "alt shape"}), raising=False)

    text, _ = await chat_turn._transcribe(
        runtime, base64.b64encode(b"x").decode(), "u1"
    )
    assert text == "alt shape"


@pytest.mark.asyncio
async def test_garbage_base64_is_a_400_not_a_500(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "voice", _Voice({}), raising=False)

    with pytest.raises(HTTPException) as caught:
        await chat_turn._transcribe(runtime, "not!!valid@@base64::", "u1")
    assert caught.value.status_code == 400


@pytest.mark.asyncio
async def test_an_empty_transcript_refuses_the_turn(runtime, monkeypatch):
    """Silence must not become an empty user message that burns a model call."""
    monkeypatch.setattr(runtime, "voice", _Voice({"transcript": ""}), raising=False)

    with pytest.raises(HTTPException) as caught:
        await chat_turn._transcribe(runtime, base64.b64encode(b"x").decode(), "u1")
    assert caught.value.status_code == 400


@pytest.mark.asyncio
async def test_a_provider_returning_none_refuses_the_turn(runtime, monkeypatch):
    class NoneVoice:
        async def transcribe(self, audio_bytes, *, user_id):
            return None

    monkeypatch.setattr(runtime, "voice", NoneVoice(), raising=False)
    with pytest.raises(HTTPException):
        await chat_turn._transcribe(runtime, base64.b64encode(b"x").decode(), "u1")


# ---------------------------------------------------------------------------
# Signed URLs
# ---------------------------------------------------------------------------

KEY = "test-signing-key"


def _parts(url):
    from urllib.parse import parse_qs, urlparse

    q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    return q["path"], q["expires"], q["sig"]


def test_a_fresh_url_round_trips():
    path, expires, sig = _parts(generate_signed_url("docs/a.pdf", "u1", KEY))
    ok, reason = validate_signed_url(path, expires, sig, "u1", KEY)
    assert ok and reason is None


def test_another_user_cannot_use_the_link():
    """The signature binds the path to the user it was issued to."""
    path, expires, sig = _parts(generate_signed_url("docs/a.pdf", "u1", KEY))
    ok, reason = validate_signed_url(path, expires, sig, "u2", KEY)
    assert not ok
    assert reason == "invalid signature"


def test_the_link_does_not_open_a_different_file():
    path, expires, sig = _parts(generate_signed_url("docs/a.pdf", "u1", KEY))
    ok, reason = validate_signed_url("docs/b.pdf", expires, sig, "u1", KEY)
    assert not ok
    assert reason == "invalid signature"


def test_an_expired_link_is_refused():
    sig_time = int(time.time()) - 10
    import hashlib
    import hmac as hmac_mod

    message = f"docs/a.pdf|u1|{sig_time}"
    sig = hmac_mod.new(KEY.encode(), message.encode(), hashlib.sha256).hexdigest()
    ok, reason = validate_signed_url("docs/a.pdf", str(sig_time), sig, "u1", KEY)
    assert not ok
    assert reason == "URL has expired"


@pytest.mark.parametrize("expires", ["", "soon", "12.5", None])
def test_a_mangled_expiry_is_refused_not_crashed(expires):
    ok, reason = validate_signed_url("docs/a.pdf", expires, "sig", "u1", KEY)
    assert not ok
    assert reason == "invalid expiry format"


def test_extending_the_expiry_breaks_the_signature():
    """The expiry is inside the signed message, so it cannot be edited."""
    path, expires, sig = _parts(generate_signed_url("docs/a.pdf", "u1", KEY))
    later = str(int(expires) + 9999)
    ok, reason = validate_signed_url(path, later, sig, "u1", KEY)
    assert not ok
    assert reason == "invalid signature"
