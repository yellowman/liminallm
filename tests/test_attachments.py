"""Attach-to-conversation, model-initiated file search, and the interpreter."""
import io
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from liminallm import app as app_module
from liminallm.service.attachments import (
    INLINE_MAX_BYTES,
    build_attachment_preamble,
    classify_attachment,
    store_generation,
)
from liminallm.service.interpreter import (
    prepare_workdir,
    run_python_sandboxed,
)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_small_text_is_inlined_not_chunked():
    caps = classify_attachment("notes.txt", 500)
    assert caps["inline"] is True
    # Inlining and chunking the same file would duplicate it in the prompt.
    assert caps["searchable"] is False


def test_large_text_is_searchable_not_inlined():
    caps = classify_attachment("book.md", INLINE_MAX_BYTES + 1)
    assert caps["inline"] is False
    assert caps["searchable"] is True


def test_pdf_is_searchable():
    caps = classify_attachment("paper.pdf", 2_000_000)
    assert caps["searchable"] is True
    assert caps["inline"] is False


def test_archive_is_analyzable_only():
    caps = classify_attachment("bundle.zip", 4096)
    assert caps["analyzable"] is True
    assert caps["searchable"] is False
    assert caps["inline"] is False


def test_csv_is_analyzable_and_inline_when_small():
    caps = classify_attachment("rows.csv", 800)
    assert caps["inline"] is True
    assert caps["analyzable"] is True


# ---------------------------------------------------------------------------
# Sandboxed interpreter
# ---------------------------------------------------------------------------


def test_interpreter_runs_code_and_captures_stdout(tmp_path):
    result = run_python_sandboxed("print(2 + 2)", workdir=str(tmp_path), timeout=60)
    assert result["ok"] is True
    assert "4" in result["stdout"]


def test_interpreter_reports_errors_to_the_model(tmp_path):
    result = run_python_sandboxed("1 / 0", workdir=str(tmp_path), timeout=60)
    assert result["ok"] is False
    assert "ZeroDivisionError" in result["stderr"]


def test_interpreter_can_unzip_an_attachment(tmp_path):
    archive = tmp_path / "src" / "data.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner.txt", "hello from inside the archive")
    workdir = prepare_workdir(
        str(tmp_path / "sessions"), [("data.zip", str(archive))]
    )
    code = (
        "import zipfile\n"
        "with zipfile.ZipFile('data.zip') as z:\n"
        "    print(z.namelist())\n"
        "    print(z.read('inner.txt').decode())\n"
    )
    result = run_python_sandboxed(code, workdir=workdir, timeout=60)
    assert result["ok"] is True, result["stderr"]
    assert "inner.txt" in result["stdout"]
    assert "hello from inside the archive" in result["stdout"]


def test_interpreter_blocks_network(tmp_path):
    result = run_python_sandboxed(
        "import socket\nsocket.create_connection(('example.com', 80))",
        workdir=str(tmp_path),
        timeout=60,
    )
    assert result["ok"] is False
    assert "not available in the sandbox" in result["stderr"]


def test_interpreter_blocks_subprocess(tmp_path):
    result = run_python_sandboxed(
        "import subprocess\nsubprocess.run(['echo', 'hi'])",
        workdir=str(tmp_path),
        timeout=60,
    )
    assert result["ok"] is False
    assert "not available in the sandbox" in result["stderr"]


def test_interpreter_blocks_os_system(tmp_path):
    result = run_python_sandboxed(
        "import os\nos.system('echo hi')", workdir=str(tmp_path), timeout=60
    )
    assert result["ok"] is False
    assert "not available in the sandbox" in result["stderr"]


def test_interpreter_cannot_see_files_outside_workdir(tmp_path):
    secret = tmp_path / "src" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("do not leak")
    (secret.parent / "shared.txt").write_text("fine to read")
    # Only the named attachment is copied in.
    workdir = prepare_workdir(
        str(tmp_path / "sessions"), [("shared.txt", str(secret.parent / "shared.txt"))]
    )
    result = run_python_sandboxed(
        "import os\nprint(sorted(os.listdir('.')))", workdir=workdir, timeout=60
    )
    assert "shared.txt" in result["stdout"]
    assert "secret.txt" not in result["stdout"]


def test_prepare_workdir_rejects_traversal_names(tmp_path):
    """The display name still decides where inside the workdir a copy lands."""
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    workdir = prepare_workdir(
        str(tmp_path / "sessions"), [("../outside.txt", str(outside))]
    )
    from pathlib import Path

    assert list(Path(workdir).iterdir()) == []


def test_interpreter_timeout_is_reported(tmp_path):
    result = run_python_sandboxed(
        "while True:\n    pass", workdir=str(tmp_path), timeout=3
    )
    assert result["ok"] is False
    assert "sandbox stopped the code" in result["stderr"]


# ---------------------------------------------------------------------------
# Preamble construction
# ---------------------------------------------------------------------------


def _attached(tmp_path, name, body, **caps):
    """A record naming a generation the store really holds."""
    import hashlib

    checksum = hashlib.sha256(body).hexdigest()
    assert store_generation(str(tmp_path), "u1", body, checksum) is not None
    record = {"name": name, "size": len(body), "checksum": checksum,
              "inline": False, "searchable": False, "analyzable": False}
    record.update(caps)
    return record


def test_preamble_inlines_small_files(tmp_path):
    attachments = [
        _attached(tmp_path, "notes.txt", b"the launch code is ORCHID-9", inline=True)
    ]
    text = build_attachment_preamble(attachments, fs_root=str(tmp_path), user_id="u1")
    assert "ORCHID-9" in text
    assert "notes.txt" in text


def test_preamble_mentions_tools_for_searchable_and_analyzable(tmp_path):
    attachments = [
        _attached(tmp_path, "big.md", b"a longer document\n" * 40, searchable=True),
        _attached(tmp_path, "b.zip", b"PK\x03\x04 and more", analyzable=True),
    ]
    text = build_attachment_preamble(attachments, fs_root=str(tmp_path), user_id="u1")
    assert "file_search" in text
    assert "run_python" in text


# ---------------------------------------------------------------------------
# API: attach to a conversation and use it
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def user(client):
    email = f"attach_{uuid.uuid4().hex[:8]}@test.local"
    resp = client.post("/v1/auth/signup", json={"email": email, "password": "TestPass123!"})
    assert resp.status_code == 201, resp.json()
    return {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}


def make_conversation(client, headers):
    resp = client.post("/v1/conversations", headers=headers, json={"title": "Attachments"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["data"]["id"]


def upload(client, headers, name, payload, ctype="text/plain", **form):
    return client.post(
        "/v1/files/upload",
        headers=headers,
        files={"file": (name, io.BytesIO(payload), ctype)},
        data=form,
    )


def test_upload_attaches_to_conversation_without_any_context(client, user):
    conv = make_conversation(client, user)
    resp = upload(client, user, "notes.txt", b"tiny file", conversation_id=conv)
    assert resp.status_code in (200, 201), resp.json()
    attachment = resp.json()["data"]["attachment"]
    assert attachment["name"] == "notes.txt"
    assert attachment["inline"] is True

    listed = client.get(f"/v1/conversations/{conv}/attachments", headers=user).json()["data"]
    assert [a["name"] for a in listed["items"]] == ["notes.txt"]


def test_attachment_context_is_hidden_from_contexts_ui(client, user):
    conv = make_conversation(client, user)
    # A large text file creates the implicit context for chunking.
    upload(client, user, "big.md", b"x " * 20_000, "text/markdown", conversation_id=conv)
    contexts = client.get("/v1/contexts", headers=user).json()["data"]["items"]
    assert all("conversation:" not in c["name"] for c in contexts), contexts


def test_large_attachment_is_chunked_for_search(client, user):
    conv = make_conversation(client, user)
    body = ("liminallm attachment search corpus. " * 900).encode()
    resp = upload(client, user, "big.md", body, "text/markdown", conversation_id=conv)
    data = resp.json()["data"]
    assert data["attachment"]["searchable"] is True
    assert (data["chunk_count"] or 0) > 0


def test_archive_can_be_attached_to_a_conversation(client, user):
    conv = make_conversation(client, user)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.txt", "inside")
    resp = upload(
        client, user, "b.zip", buf.getvalue(), "application/zip", conversation_id=conv
    )
    assert resp.status_code in (200, 201), resp.json()
    attachment = resp.json()["data"]["attachment"]
    assert attachment["analyzable"] is True
    assert attachment["searchable"] is False


def test_cannot_attach_to_someone_elses_conversation(client, user):
    conv = make_conversation(client, user)
    other = f"other_{uuid.uuid4().hex[:8]}@test.local"
    resp = client.post("/v1/auth/signup", json={"email": other, "password": "TestPass123!"})
    stranger = {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}
    blocked = upload(client, stranger, "notes.txt", b"hi", conversation_id=conv)
    assert blocked.status_code in (403, 404)


def test_attachments_endpoint_is_owner_only(client, user):
    conv = make_conversation(client, user)
    other = f"other_{uuid.uuid4().hex[:8]}@test.local"
    resp = client.post("/v1/auth/signup", json={"email": other, "password": "TestPass123!"})
    stranger = {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}
    assert client.get(f"/v1/conversations/{conv}/attachments", headers=stranger).status_code in (403, 404)


@pytest.fixture
def tool_calling_backend():
    """Point the runtime at the stub backend, which drives the tool loop."""
    from liminallm.service.model_backend import StubBackend
    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()
    previous = runtime.llm.backend
    runtime.llm.backend = StubBackend()
    yield
    runtime.llm.backend = previous


def test_chat_with_attachment_uses_the_agent_and_calls_tools(
    client, user, tool_calling_backend
):
    """The stub backend calls each offered tool once, so the loop is exercised."""
    conv = make_conversation(client, user)
    body = ("searchable corpus about turbines. " * 900).encode()
    upload(client, user, "turbines.md", body, "text/markdown", conversation_id=conv)

    resp = client.post(
        "/v1/chat",
        headers=user,
        json={
            "conversation_id": conv,
            "message": {"content": "What do the attached files say about turbines?"},
        },
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()["data"]
    # The agent ran: the stub's answer echoes the tool output it received.
    assert "Tool results" in data["content"], data["content"]
    # file_search results surface as citations.
    assert data["context_snippets"]


def test_chat_without_attachments_uses_the_normal_workflow(client, user):
    conv = make_conversation(client, user)
    resp = client.post(
        "/v1/chat",
        headers=user,
        json={"conversation_id": conv, "message": {"content": "plain question"}},
    )
    assert resp.status_code == 200, resp.json()
    # No attachment agent, so no tool-result echo.
    assert "Tool results" not in resp.json()["data"]["content"]


# ---------------------------------------------------------------------------
# Streaming: tool rounds first, then a token-streamed answer
# ---------------------------------------------------------------------------


async def _collect_stream(engine, **kwargs):
    return [event async for event in engine.run_streaming(**kwargs)]


def test_streaming_agent_emits_tool_traces_then_tokens(client, user, tool_calling_backend):
    """The attachment answer must stream, not arrive in one lump."""
    import asyncio

    from liminallm.service.runtime import get_runtime

    conv = make_conversation(client, user)
    upload(
        client, user, "corpus.md",
        ("turbine maintenance notes. " * 900).encode(), "text/markdown",
        conversation_id=conv,
    )

    runtime = get_runtime()
    me = client.get("/v1/me", headers=user).json()["data"]
    events = asyncio.run(
        _collect_stream(
            runtime.workflow,
            workflow_id=None,
            conversation_id=conv,
            user_message="What do the notes say about turbines?",
            context_id=None,
            user_id=me["id"],
            tenant_id=me.get("tenant_id"),
        )
    )
    kinds = [e["event"] for e in events]

    # Tool activity is announced before the answer streams.
    tool_traces = [
        e for e in events if e["event"] == "trace" and (e.get("data") or {}).get("tool")
    ]
    assert tool_traces, kinds
    assert {t["data"]["tool"] for t in tool_traces} <= {
        "file_search", "run_python", "web_search", "web_fetch",
    }

    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) > 1, f"expected a token stream, got {len(tokens)}"

    first_token_at = kinds.index("token")
    first_trace_at = next(
        i for i, e in enumerate(events)
        if e["event"] == "trace" and (e.get("data") or {}).get("tool")
    )
    assert first_trace_at < first_token_at, "tools must run before the answer streams"

    done = events[-1]
    assert done["event"] == "message_done"
    assert done["data"]["content"] == "".join(t["data"] for t in tokens)
    # Citations and the tool list survive the streaming path.
    assert done["data"]["context_snippets"]
    assert [c["tool"] for c in done["data"]["workflow_trace"][0]["tool_calls"]]
