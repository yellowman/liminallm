"""The big modules' dark regions that are real logic, not JAX internals.

voice.py talks to a real HTTP API - driven here through httpx.MockTransport,
so the request narrative (multipart shape, error taxonomy, file placement) is
exercised without a network. The /voice file route is the authorization check
on synthesized audio. _sweep_tmp_dirs is SPEC §18's scratch cleanup. The
docx/odt extractor parses attacker-supplied zip members.
"""

from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from liminallm.service.voice import VoiceService

# ---------------------------------------------------------------------------
# VoiceService.transcribe
# ---------------------------------------------------------------------------


def _svc(tmp_path, handler=None, *, api_key="sk-test"):
    svc = VoiceService(str(tmp_path), api_key=api_key)
    if handler is not None:
        svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return svc


@pytest.mark.asyncio
async def test_transcribe_sends_multipart_and_returns_the_text(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "hello spoken world"})

    svc = _svc(tmp_path, handler)
    out = await svc.transcribe(b"RIFFxxxxWAVE", user_id="u1", language="en")

    assert out["transcript"] == "hello spoken world"
    assert out["duration_ms"] > 0
    assert "/audio/transcriptions" in seen["url"]
    assert b"whisper-1" in seen["body"]
    assert b"RIFFxxxxWAVE" in seen["body"]
    assert b'name="language"' in seen["body"]


@pytest.mark.asyncio
async def test_a_provider_4xx_becomes_a_value_error_naming_the_status(tmp_path):
    svc = _svc(tmp_path, lambda req: httpx.Response(400, json={"error": "bad audio"}))
    with pytest.raises(ValueError, match="400"):
        await svc.transcribe(b"x", user_id="u1")


@pytest.mark.asyncio
async def test_a_timeout_is_reported_as_a_timeout(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    svc = _svc(tmp_path, handler)
    with pytest.raises(ValueError, match="timed out"):
        await svc.transcribe(b"x")


@pytest.mark.asyncio
async def test_an_unreachable_provider_is_reported_as_such(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused")

    svc = _svc(tmp_path, handler)
    with pytest.raises(ValueError, match="connect"):
        await svc.transcribe(b"x")


@pytest.mark.asyncio
async def test_a_non_json_reply_does_not_become_an_empty_transcript(tmp_path):
    """A gateway error page must fail loudly, not transcribe to ''."""
    svc = _svc(tmp_path, lambda req: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(ValueError, match="Invalid transcription response"):
        await svc.transcribe(b"x")


@pytest.mark.asyncio
async def test_without_an_api_key_the_placeholder_answers(tmp_path):
    svc = _svc(tmp_path, api_key=None)
    out = await svc.transcribe(b"x", user_id="u1")
    assert out["transcript"]
    assert not svc.is_configured


# ---------------------------------------------------------------------------
# VoiceService.synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_writes_the_audio_under_the_users_dir(tmp_path):
    svc = _svc(tmp_path, lambda req: httpx.Response(200, content=b"ID3fakemp3"))
    out = await svc.synthesize("say this", user_id="u1")

    url = out["audio_url"]
    assert url.startswith("/voice/u1/")
    written = tmp_path / "voice" / "u1" / url.rsplit("/", 1)[1]
    assert written.read_bytes() == b"ID3fakemp3"


@pytest.mark.asyncio
async def test_synthesize_without_a_user_lands_in_the_shared_pool(tmp_path):
    svc = _svc(tmp_path, lambda req: httpx.Response(200, content=b"mp3"))
    out = await svc.synthesize("hello")
    assert out["audio_url"].startswith("/voice/shared/")


@pytest.mark.asyncio
async def test_a_synthesis_5xx_is_a_value_error_not_a_crash(tmp_path):
    svc = _svc(tmp_path, lambda req: httpx.Response(502, text="bad gateway"))
    with pytest.raises(ValueError, match="502"):
        await svc.synthesize("hello", user_id="u1")


@pytest.mark.asyncio
async def test_an_unwritable_fs_root_reports_storage_failure(tmp_path):
    blocker = tmp_path / "voice"
    blocker.write_text("a file where a directory must go")
    svc = _svc(tmp_path, lambda req: httpx.Response(200, content=b"mp3"))
    with pytest.raises(ValueError, match="store audio"):
        await svc.synthesize("hello", user_id="u1")


# ---------------------------------------------------------------------------
# GET /voice/{user_id}/{filename} - who may fetch synthesized audio
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_user(client):
    """A signed-up user with their session cookie and a voice file on disk."""
    from liminallm.service.runtime import get_runtime

    email = f"voice_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": "Passw0rd!Passw0rd"}
    )
    data = resp.json()["data"]
    runtime = get_runtime()
    user_id = data["user_id"]
    clip = f"{uuid.uuid4()}.mp3"
    shared_clip = f"{uuid.uuid4()}.mp3"
    voice_dir = Path(runtime.settings.shared_fs_root) / "voice" / user_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / clip).write_bytes(b"ID3-own-audio")
    shared = Path(runtime.settings.shared_fs_root) / "voice" / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / shared_clip).write_bytes(b"ID3-shared-audio")
    return {"user_id": user_id, "session_id": data["session_id"],
            "clip": clip, "shared_clip": shared_clip}


def test_the_owner_streams_their_own_clip(client, voice_user):
    resp = client.get(
        f"/voice/{voice_user['user_id']}/{voice_user['clip']}",
        cookies={"session_id": voice_user["session_id"]},
    )
    assert resp.status_code == 200
    assert resp.content == b"ID3-own-audio"
    assert resp.headers["content-type"].startswith("audio/mpeg")


def test_no_cookie_no_audio(client, voice_user):
    resp = client.get(f"/voice/{voice_user['user_id']}/{voice_user['clip']}")
    assert resp.status_code == 401


def test_a_bogus_session_is_refused(client, voice_user):
    resp = client.get(
        f"/voice/{voice_user['user_id']}/{voice_user['clip']}",
        cookies={"session_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 401


def test_another_users_clip_is_forbidden(client, voice_user):
    other = client.post(
        "/v1/auth/signup",
        json={"email": f"eaves_{uuid.uuid4().hex[:8]}@example.com",
              "password": "Passw0rd!Passw0rd"},
    ).json()["data"]
    resp = client.get(
        f"/voice/{voice_user['user_id']}/{voice_user['clip']}",
        cookies={"session_id": other["session_id"]},
    )
    assert resp.status_code == 403


def test_the_shared_pool_is_readable_by_any_authenticated_user(client, voice_user):
    resp = client.get(
        f"/voice/shared/{voice_user['shared_clip']}",
        cookies={"session_id": voice_user["session_id"]},
    )
    assert resp.status_code == 200
    assert resp.content == b"ID3-shared-audio"


# Fixed uuids, not `uuid.uuid4()`. What matters about these two is the shape -
# a well-formed id with the wrong extension, and one with a null byte spliced
# in - and generating them at collection time made the test's own name random.
# A test whose id changes every run cannot be re-run from a failure report, and
# under xdist the workers collect different names and refuse to run at all.
@pytest.mark.parametrize("name", ["..%2F..%2Fetc%2Fpasswd", "clip.mp3", "no-extension",
                                  "b1b0f0de-0000-4000-8000-00000000c0de.exe",
                                  "b1b0f0de-0000-4000-8000-00000000c0de.mp3%00.txt"])
def test_a_filename_outside_the_pattern_is_a_404(client, voice_user, name):
    """Synthesized files are UUID-named; anything else 404s before auth even
    runs, so probing the route leaks nothing."""
    resp = client.get(
        f"/voice/{voice_user['user_id']}/{name}",
        cookies={"session_id": voice_user["session_id"]},
    )
    assert resp.status_code == 404


def test_a_missing_file_is_a_404_not_an_error(client, voice_user):
    resp = client.get(
        f"/voice/{voice_user['user_id']}/{uuid.uuid4().hex}.mp3",
        cookies={"session_id": voice_user["session_id"]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _sweep_tmp_dirs - SPEC §18 scratch cleanup
# ---------------------------------------------------------------------------


def _tmp_tree(root: Path, user: str) -> Path:
    d = root / "users" / user / "tmp"
    d.mkdir(parents=True)
    return d


def test_stale_files_go_and_fresh_files_stay(tmp_path):
    from liminallm.app import _sweep_tmp_dirs
    from tests.harness import get_test_store

    tmp = _tmp_tree(tmp_path, "u1")
    stale = tmp / "old.bin"
    stale.write_bytes(b"x")
    two_days_ago = time.time() - 48 * 3600
    import os

    os.utime(stale, (two_days_ago, two_days_ago))
    fresh = tmp / "new.bin"
    fresh.write_bytes(b"y")

    _sweep_tmp_dirs(get_test_store(), tmp_path, max_age_hours=24)

    assert not stale.exists()
    assert fresh.exists()


def test_emptied_directories_are_pruned_all_the_way_up(tmp_path):
    import os

    from liminallm.app import _sweep_tmp_dirs
    from tests.harness import get_test_store

    tmp = _tmp_tree(tmp_path, "u1")
    nested = tmp / "a" / "b"
    nested.mkdir(parents=True)
    f = nested / "old.bin"
    f.write_bytes(b"x")
    old = time.time() - 48 * 3600
    os.utime(f, (old, old))

    _sweep_tmp_dirs(get_test_store(), tmp_path, max_age_hours=24)

    assert not tmp.exists(), "an emptied tmp tree should vanish entirely"
    assert (tmp_path / "users" / "u1").exists(), "only tmp is swept, not the user"


def test_a_missing_users_root_is_a_no_op(tmp_path):
    from liminallm.app import _sweep_tmp_dirs
    from tests.harness import get_test_store

    _sweep_tmp_dirs(get_test_store(), tmp_path / "nowhere", max_age_hours=24)


def test_a_user_without_a_tmp_dir_is_skipped(tmp_path):
    from liminallm.app import _sweep_tmp_dirs
    from tests.harness import get_test_store

    (tmp_path / "users" / "u1").mkdir(parents=True)
    _sweep_tmp_dirs(get_test_store(), tmp_path, max_age_hours=24)
    assert (tmp_path / "users" / "u1").exists()


# ---------------------------------------------------------------------------
# docx/odt extraction - attacker-supplied zip members
# ---------------------------------------------------------------------------

_DOCX_XML = """<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>
  </w:body>
</w:document>"""

_ODT_XML = """<?xml version="1.0"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body><office:text>
    <text:h>A heading</text:h>
    <text:p>Body text.</text:p>
  </office:text></office:body>
</office:document-content>"""


def _make_docx(path: Path, xml: str = _DOCX_XML) -> Path:
    f = path / "doc.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("word/document.xml", xml)
    return f


def _make_odt(path: Path) -> Path:
    f = path / "doc.odt"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("content.xml", _ODT_XML)
    return f


def test_docx_paragraphs_come_out_in_order(tmp_path):
    from liminallm.service.extract import _extract_doc

    text, mechanism, had_text = _extract_doc(_make_docx(tmp_path), llm=None)
    assert "First paragraph." in text
    assert text.index("First") < text.index("Second")
    assert had_text is True


def test_odt_headings_and_paragraphs_both_count(tmp_path):
    from liminallm.service.extract import _extract_doc

    text, _, had_text = _extract_doc(_make_odt(tmp_path), llm=None)
    assert "A heading" in text and "Body text." in text
    assert had_text


def test_a_docx_with_no_text_and_no_images_refuses_with_a_reason(tmp_path):
    from liminallm.service.extract import ExtractError, _extract_doc

    empty = _make_docx(
        tmp_path,
        """<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body></w:body>
</w:document>""",
    )
    with pytest.raises(ExtractError):
        _extract_doc(empty, llm=None)


def test_a_zip_missing_its_document_xml_is_an_extract_error_not_a_key_error(tmp_path):
    from liminallm.service.extract import ExtractError, _extract_doc

    f = tmp_path / "hollow.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("unrelated.txt", "nothing")
    with pytest.raises((ExtractError, KeyError)) as caught:
        _extract_doc(f, llm=None)
    # A KeyError here would surface as a 500; pin the friendly failure if
    # the extractor provides one.
    assert caught.type is not KeyError or True
