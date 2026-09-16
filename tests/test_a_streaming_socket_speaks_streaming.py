"""A streaming socket speaks the streaming vocabulary, to its last frame.

SPEC §13.7 gives `/v1/chat/stream` one shape - `{event, data, request_id}`,
with `error` among the five events - and reserves the bare `{status, data}`
envelope for the client that asked for `stream: false`.

The route kept two vocabularies instead, split by where the failure arose.
Failures the workflow yielded became `error` events; failures raised around it
were sent as envelopes by the outer handlers, and a completed turn replayed by
idempotency key was sent as an envelope too. Measured: deleting a chat from
inside its own streaming turn ended the socket with
`{"status": "error", "error": {"code": "server_error", ...}}` after 33 token
events - the non-streaming shape, and the wrong classification for a condition
the same platform answers `409 conflict` over HTTP and names
"conversation deleted during upload" on the upload route.

Two rules, and everything here is one of them:

* the shape follows the mode the client asked for, never where the failure
  arose;
* a chat that went away mid-turn is a `conflict`, not a `server_error`.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from liminallm.service.runtime import get_runtime

SEEDED = "the question asked before the socket was cut"
TERMINAL = ("message_done", "error", "cancel_ack")
#: Long enough for a slow runner, short enough that a stuck socket fails this
#: test rather than the job.
DRAIN_BUDGET_SECONDS = 90.0


@pytest.fixture
def auth(client):
    email = f"wsv_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/v1/auth/signup",
                       json={"email": email, "password": "TestPassword123!"})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return {"headers": {"Authorization": f"Bearer {data['access_token']}"},
            "access_token": data["access_token"]}


@pytest.fixture
def seeded_chat(client, auth):
    made = client.post("/v1/conversations", headers=auth["headers"],
                       json={"title": "about to be deleted"})
    conversation_id = made.json()["data"]["id"]
    assert client.post(
        "/v1/chat", headers=auth["headers"],
        json={"conversation_id": conversation_id,
              "message": {"content": SEEDED, "mode": "text"},
              "stream": False},
    ).status_code == 200
    return conversation_id


def _drain(ws, limit: int = 300) -> list[dict]:
    """Read until the socket says its last word, under a watchdog.

    Two bounds, because they answer different failures. Stopping on a terminal
    frame handles a socket that keeps talking. The watchdog handles a socket
    that says nothing at all: `TestClient.receive_json` has no timeout, so a
    server that neither sends nor closes blocks this thread for as long as the
    job lives.

    That is not hypothetical. An earlier version of this helper carried a
    docstring saying it could not hang, which was true only once a frame
    arrived - the blocking read itself was unbounded. A CI job then sat on a
    stuck worker until the six-hour ceiling cancelled it. A test that hangs
    costs a whole job; a test that fails costs a line.
    """
    frames: list[dict] = []
    finished = threading.Event()

    def _read() -> None:
        try:
            for _ in range(limit):
                try:
                    frame = ws.receive_json()
                except Exception:  # noqa: BLE001 - the close is an outcome
                    frames.append({"__closed__": True})
                    return
                frames.append(frame)
                if frame.get("event") in TERMINAL:
                    return
                if "event" not in frame and "status" in frame:
                    return
        finally:
            finished.set()

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    if not finished.wait(timeout=DRAIN_BUDGET_SECONDS):
        raise AssertionError(
            f"the socket sent no terminal frame within {DRAIN_BUDGET_SECONDS}s; "
            f"read so far: {[f.get('event') for f in frames]}"
        )
    return frames


def _talk(client, init: dict, *, raw: str | None = None) -> list[dict]:
    with client.websocket_connect("/v1/chat/stream") as ws:
        if raw is not None:
            ws.send_text(raw)
        else:
            ws.send_json(init)
        return _drain(ws)


def _last(frames: list[dict]) -> dict:
    assert frames, "the socket said nothing at all"
    return frames[-1]


def _kinds(frames: list[dict]) -> list:
    return [f.get("event") for f in frames]


def _end_it_mid_stream(client, auth, conversation_id, end_it):
    """Run `end_it` from inside the model call of a streaming turn.

    The ending lands after the turn has begun and before `chat_turn.finish()`
    is reached, which is the window production occupies. The hook does one
    thing and records one value, so a retried node cannot leave it half done.
    """
    runtime = get_runtime()
    real_stream = runtime.llm.generate_stream
    state: dict = {}

    def _end_then_stream(*args, **kwargs):
        if "ended" not in state:
            state["ended"] = end_it()
        return real_stream(*args, **kwargs)

    runtime.llm.generate_stream = _end_then_stream
    try:
        state["frames"] = _talk(client, {
            "access_token": auth["access_token"],
            "conversation_id": conversation_id,
            "message": "the doomed streamed question",
            "stream": True,
        })
    finally:
        runtime.llm.generate_stream = real_stream
    return state


def _rows(conversation_id) -> tuple[int, int]:
    store = get_runtime().store
    with store._connect() as conn:
        convo = conn.execute(
            "SELECT count(*) AS n FROM conversation WHERE id = %s",
            (conversation_id,),
        ).fetchone()["n"]
        msgs = conn.execute(
            "SELECT count(*) AS n FROM message WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()["n"]
    return convo, msgs


class TestAChatThatWentAwayIsAConflict:
    def test_deleting_the_chat_mid_stream_ends_in_an_error_event(
        self, client, auth, seeded_chat
    ):
        state = _end_it_mid_stream(
            client, auth, seeded_chat,
            lambda: client.delete(
                f"/v1/conversations/{seeded_chat}", headers=auth["headers"]
            ).status_code,
        )
        assert state["ended"] == 200, "the delete itself failed"

        frames = state["frames"]
        last = _last(frames)
        assert last.get("event") == "error", (
            f"the socket's last word was not an error event: {last}"
        )
        assert last["data"]["code"] == "conflict", last["data"]
        assert last["data"]["message"] == "conversation not found", last["data"]
        assert last["request_id"], "the terminal frame carries no request_id"

    def test_the_deleted_chat_gets_no_message_done(
        self, client, auth, seeded_chat
    ):
        """A turn that could not be persisted must not report completion."""
        state = _end_it_mid_stream(
            client, auth, seeded_chat,
            lambda: client.delete(
                f"/v1/conversations/{seeded_chat}", headers=auth["headers"]
            ).status_code,
        )
        assert "message_done" not in _kinds(state["frames"]), _kinds(state["frames"])

    def test_erasing_the_account_mid_stream_ends_the_same_way(
        self, client, auth, seeded_chat
    ):
        """The account's own lifetime, through the same terminal path.

        The chat cascades away with its owner, so the turn hits the same
        refusal - and must report it as the same condition rather than as an
        internal fault.
        """
        import asyncio

        runtime = get_runtime()
        user_id = client.get(
            "/v1/me", headers=auth["headers"]
        ).json()["data"]["id"]
        state = _end_it_mid_stream(
            client, auth, seeded_chat,
            lambda: asyncio.run(runtime.auth.delete_user(user_id)),
        )
        assert state["ended"], "the account was not erased"

        last = _last(state["frames"])
        assert last.get("event") == "error", last
        assert last["data"]["code"] == "conflict", last["data"]
        assert "message_done" not in _kinds(state["frames"])

    def test_nothing_of_the_erased_account_survives_the_turn(
        self, client, auth, seeded_chat
    ):
        import asyncio

        runtime = get_runtime()
        user_id = client.get(
            "/v1/me", headers=auth["headers"]
        ).json()["data"]["id"]
        _end_it_mid_stream(
            client, auth, seeded_chat,
            lambda: asyncio.run(runtime.auth.delete_user(user_id)),
        )
        convo, msgs = _rows(seeded_chat)
        assert (convo, msgs) == (0, 0), (
            f"the turn left {convo} conversation and {msgs} message row(s)"
        )


class TestEveryOuterFailureUsesTheStreamingShape:
    """The three handlers that used to build envelopes on a live stream."""

    def test_an_http_failure_is_an_error_event(self, client, auth):
        """A conversation the caller does not own: a real 404 from `begin`."""
        frames = _talk(client, {
            "access_token": auth["access_token"],
            "conversation_id": str(uuid.uuid4()),
            "message": "a chat that is not mine",
            "stream": True,
        })
        last = _last(frames)
        assert last.get("event") == "error", last
        assert last["data"]["code"] == "not_found", last["data"]

    def test_malformed_json_is_an_error_event(self, client, auth):
        """The failure that happens before `stream` can even be read.

        Streaming is this route's default, so the default shape is the
        streaming one - which is why `stream_enabled` starts True rather than
        being left undefined until `init` parses.
        """
        frames = _talk(client, {}, raw="{not json at all")
        last = _last(frames)
        assert last.get("event") == "error", last
        assert last["data"]["code"] == "validation_error", last["data"]

    def test_an_internal_failure_stays_a_server_error(
        self, client, auth, seeded_chat
    ):
        """The reclassification is narrow: only a storage conflict moves.

        Raised at `chat_turn.finish`, which is the seam the generic handler
        actually guards - a model-host failure never reaches it, because the
        workflow retries the node and yields its own `error` event, which was
        already event-shaped. The real `ConstraintViolation` comes from this
        same call, so the two cases are siblings on one seam.
        """
        from liminallm.api import chat_turn as chat_turn_module

        real_finish = chat_turn_module.finish

        async def _boom(*args, **kwargs):
            raise RuntimeError("the store fell over")

        chat_turn_module.finish = _boom
        try:
            frames = _talk(client, {
                "access_token": auth["access_token"],
                "conversation_id": seeded_chat,
                "message": "a turn that breaks after the stream",
                "stream": True,
            })
        finally:
            chat_turn_module.finish = real_finish

        last = _last(frames)
        assert last.get("event") == "error", last
        assert last["data"]["code"] == "server_error", last["data"]
        assert last["data"]["message"] == "An internal error occurred", (
            "the internal message leaked past the generic handler"
        )

    def test_a_conflict_is_scrubbed_on_its_way_out(
        self, client, auth, seeded_chat
    ):
        """The socket now carries a raised message, so it owes the scrubbing.

        `_error_response` calls itself the one place an error leaves the
        process, and the socket never went through it: the old
        `HTTPException` branch put `str(exc.detail)` on the wire untouched.
        Carrying `exc.message` and `exc.detail` would widen that, so the
        socket's error body runs the same two sanitizers.
        """
        from liminallm.api import chat_turn as chat_turn_module
        from liminallm.storage.errors import ConstraintViolation

        real_finish = chat_turn_module.finish

        async def _leak(*args, **kwargs):
            raise ConstraintViolation(
                "SELECT * FROM conversation WHERE id = 5",
                {"conversation_id": "c-1", "token": "sekrit"},
            )

        chat_turn_module.finish = _leak
        try:
            frames = _talk(client, {
                "access_token": auth["access_token"],
                "conversation_id": seeded_chat,
                "message": "a turn that raises something loud",
                "stream": True,
            })
        finally:
            chat_turn_module.finish = real_finish

        data = _last(frames)["data"]
        assert data["code"] == "conflict", data
        assert "SELECT" not in data["message"], data["message"]
        assert data["details"]["token"] != "sekrit", data["details"]
        assert data["details"]["conversation_id"] == "c-1", (
            "scrubbing removed the part that helps the caller"
        )


class TestTheNonStreamingShapeIsUntouched:
    """`stream: false` is the case §13.7 gives the envelope to."""

    def test_a_non_streaming_failure_is_still_an_envelope(self, client, auth):
        frames = _talk(client, {
            "access_token": auth["access_token"],
            "conversation_id": str(uuid.uuid4()),
            "message": "a chat that is not mine",
            "stream": False,
        })
        last = _last(frames)
        assert "event" not in last, f"a non-streaming error grew an event: {last}"
        assert last["status"] == "error", last
        assert last["error"]["code"] == "not_found", last["error"]


class TestTheWorkingPathsStillWork:
    """The door must not become a wall."""

    def test_a_streamed_turn_still_completes(self, client, auth, seeded_chat):
        frames = _talk(client, {
            "access_token": auth["access_token"],
            "conversation_id": seeded_chat,
            "message": "an ordinary question",
            "stream": True,
        })
        kinds = _kinds(frames)
        assert "token" in kinds, kinds
        assert kinds[-1] == "message_done", kinds
        assert frames[-1]["data"]["message_id"], frames[-1]
        assert frames[-1]["data"]["conversation_id"] == seeded_chat

    def test_a_replayed_streaming_turn_keeps_the_streaming_shape(
        self, client, auth, seeded_chat
    ):
        """The one exit that answers without running the turn.

        It sent the stored envelope verbatim, which ended a live stream in the
        shape reserved for a client that asked not to stream.
        """
        init = {
            "access_token": auth["access_token"],
            "conversation_id": seeded_chat,
            "message": "asked once, replayed once",
            "stream": True,
            "idempotency_key": f"k-{uuid.uuid4().hex[:12]}",
        }
        first = _talk(client, dict(init))
        assert _kinds(first)[-1] == "message_done", _kinds(first)

        second = _talk(client, dict(init))
        last = _last(second)
        assert last.get("event") == "message_done", (
            f"the replay left the streaming vocabulary: {last}"
        )
        assert last["data"]["message_id"] == first[-1]["data"]["message_id"]

    def test_a_replayed_non_streaming_turn_keeps_its_envelope(
        self, client, auth, seeded_chat
    ):
        init = {
            "access_token": auth["access_token"],
            "conversation_id": seeded_chat,
            "message": "asked once without streaming",
            "stream": False,
            "idempotency_key": f"k-{uuid.uuid4().hex[:12]}",
        }
        first = _last(_talk(client, dict(init)))
        assert first["status"] == "ok", first

        second = _last(_talk(client, dict(init)))
        assert "event" not in second, f"a non-streaming replay grew an event: {second}"
        assert second["status"] == "ok", second
