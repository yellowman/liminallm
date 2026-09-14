"""A turn that outlives its conversation does not put the chat back.

`cache_conversation_state` writes the conversation's own messages into
`chat:summary`, and it held one lifetime: the account's. A conversation
deletion is not an account deletion - the owner is still there - so the guard
passed and the write went through.

Measured on `main`: a turn loads history at T0; the owner deletes the chat at
T1, and the delete route retires `chat:summary:<id>` right after committing;
the turn's workflow finishes at T2 and writes the history it loaded back under
the same key, for the rest of the hour-long TTL. The turn itself fails - the
assistant message cannot be appended to a conversation that is gone, and the
caller gets a 409 - so the transport and the durable rows were already right.
What came back was the cache.

The same shape the account erasure closed, one lifetime down, and the fix is
the same: hold the lifetime the write depends on across both the decision and
the write, so only two histories remain. Either the write goes first and the
delete's retire removes what it wrote, or the delete goes first and the write
sees no conversation.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest

from liminallm.service.runtime import get_runtime

SEEDED = "the question asked before the chat was deleted"


@pytest.fixture
def auth(client):
    email = f"rc_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/v1/auth/signup",
                       json={"email": email, "password": "TestPassword123!"})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return {"headers": {"Authorization": f"Bearer {data['access_token']}"},
            "access_token": data["access_token"]}


def _summary(conversation_id):
    runtime = get_runtime()
    if not runtime.cache:
        pytest.skip("this behaviour is about the cache")
    return asyncio.run(
        runtime.cache.get_conversation_summary(conversation_id)
    )


def _cached_contents(conversation_id) -> list[str]:
    cached = _summary(conversation_id) or {}
    return [
        str(message.get("content") or "")
        for message in (cached.get("recent_messages") or [])
    ]


def _turn(client, headers, conversation_id, message):
    return client.post(
        "/v1/chat", headers=headers,
        json={"conversation_id": conversation_id,
              "message": {"content": message, "mode": "text"},
              "stream": False},
    )


@pytest.fixture
def seeded_chat(client, auth):
    """A conversation with one finished turn, and its summary cached."""
    made = client.post("/v1/conversations", headers=auth["headers"],
                       json={"title": "about to be deleted"})
    conversation_id = made.json()["data"]["id"]
    assert _turn(client, auth["headers"], conversation_id, SEEDED).status_code == 200
    assert SEEDED in _cached_contents(conversation_id), (
        "the fixture never cached the chat, so nothing could come back"
    )
    return conversation_id


def _delete_mid_turn(client, headers, conversation_id):
    """Delete the chat from inside the model call of the next turn.

    A real interleaving driven by real code: the delete lands after the turn
    has begun and before `chat_turn.finish()` is reached, which is where
    production's own window is. Nothing holds a lock open.
    """
    runtime = get_runtime()
    real_generate = runtime.llm.generate
    state = {}

    def _delete_then_answer(*args, **kwargs):
        if "status" not in state:
            state["status"] = client.delete(
                f"/v1/conversations/{conversation_id}", headers=headers
            ).status_code
            state["retired"] = not _cached_contents(conversation_id)
        return real_generate(*args, **kwargs)

    runtime.llm.generate = _delete_then_answer
    try:
        state["turn"] = _turn(
            client, headers, conversation_id, "the doomed question"
        )
    finally:
        runtime.llm.generate = real_generate
    return state


class TestTheChatDoesNotComeBack:
    def test_an_in_flight_turn_does_not_recache_a_deleted_chat(
        self, client, auth, seeded_chat
    ):
        state = _delete_mid_turn(client, auth["headers"], seeded_chat)

        assert state["status"] == 200, "the delete itself failed"
        assert state["retired"], "the delete route never retired the summary"
        assert _cached_contents(seeded_chat) == [], (
            "the deleted chat's messages are cached again"
        )

    def test_the_turn_still_fails_and_writes_nothing(
        self, client, auth, seeded_chat
    ):
        """The two halves that were already right, kept that way.

        A fix that silenced the turn, or that let it persist against a
        conversation that no longer exists, would be worse than the leak.
        """
        runtime = get_runtime()
        state = _delete_mid_turn(client, auth["headers"], seeded_chat)

        assert state["turn"].status_code == 409, state["turn"].text
        with runtime.store._connect() as conn:
            for table, column in (("conversation", "id"),
                                  ("message", "conversation_id"),
                                  ("knowledge_context", "conversation_id")):
                left = conn.execute(
                    f"SELECT count(*) c FROM {table} WHERE {column} = %s",
                    (seeded_chat,),
                ).fetchone()["c"]
                assert left == 0, f"{table} kept {left} row(s)"


class TestTheAccountHalfStillHolds:
    """The erasure this guard was built for, kept working.

    `cache_conversation_state` used to hold the account's lifetime directly.
    It now holds both through one transaction, so the older half needs its own
    witness rather than resting on the chat half passing.
    """

    def test_an_in_flight_turn_does_not_recache_an_erased_account(
        self, client, auth, seeded_chat
    ):
        runtime = get_runtime()
        me = client.get("/v1/me", headers=auth["headers"])
        user_id = me.json()["data"]["id"]
        real_generate = runtime.llm.generate
        erased = {}

        def _erase_then_answer(*args, **kwargs):
            if "done" not in erased:
                erased["done"] = asyncio.run(runtime.auth.delete_user(user_id))
            return real_generate(*args, **kwargs)

        runtime.llm.generate = _erase_then_answer
        try:
            _turn(client, auth["headers"], seeded_chat, "the doomed question")
        finally:
            runtime.llm.generate = real_generate

        assert erased.get("done"), "the account was not erased"
        assert _cached_contents(seeded_chat) == [], (
            "an erased account's chat is cached again"
        )

    def test_the_hold_refuses_a_chat_whose_owner_is_erased(
        self, client, auth, seeded_chat
    ):
        """The chat cascades with the account, so one row answers both - and
        this pins that, because a check that asked only about the account
        would pass here for the wrong reason."""
        runtime = get_runtime()
        me = client.get("/v1/me", headers=auth["headers"])
        user_id = me.json()["data"]["id"]

        assert asyncio.run(runtime.auth.delete_user(user_id))

        with runtime.store.hold_live_conversation(
            seeded_chat, user_id=user_id
        ) as live:
            assert live is False


class TestTheDeleteWaitsForTheWrite:
    """The locks, which no sequential test can tell from the checks.

    Removing either `pg_advisory_xact_lock` leaves every test above passing,
    because the existence check alone answers correctly when nothing runs at
    the same time. The lock is what closes the remaining window - the instant
    between deciding the chat is live and writing the cache - and witnessing
    it needs two threads that are genuinely inside it at once.

    These hold the guard open from a test rather than catching production at
    that instant, so they are evidence that the lock is taken and honoured,
    not that production reaches this interleaving. The measured defect above
    is what shows production reaches the window; this shows the lock closes
    it.
    """

    @staticmethod
    def _blocks_while_held(hold, delete):
        """True if `delete` cannot finish while `hold` is open."""
        inside, release = threading.Event(), threading.Event()
        outcome: dict = {}

        def _holder():
            with hold() as live:
                outcome["live"] = live
                inside.set()
                release.wait(timeout=30)

        def _deleter():
            outcome["deleted"] = delete()
            outcome["finished_at"] = time.monotonic()

        holder = threading.Thread(target=_holder, daemon=True)
        holder.start()
        assert inside.wait(timeout=30), "the hold never opened"

        deleter = threading.Thread(target=_deleter, daemon=True)
        deleter.start()
        deleter.join(timeout=1.5)
        blocked = deleter.is_alive()

        release.set()
        deleter.join(timeout=30)
        holder.join(timeout=30)
        assert not deleter.is_alive(), "the delete never completed"
        return blocked, outcome

    def test_deleting_a_chat_waits_for_a_write_in_progress(
        self, client, auth, seeded_chat
    ):
        store = get_runtime().store
        me = client.get("/v1/me", headers=auth["headers"])
        user_id = me.json()["data"]["id"]

        blocked, outcome = self._blocks_while_held(
            lambda: store.hold_live_conversation(seeded_chat, user_id=user_id),
            lambda: store.delete_conversation(seeded_chat, user_id=user_id),
        )

        assert outcome["live"] is True, "the hold did not see a live chat"
        assert blocked, (
            "the delete committed while a cache write held the chat's lifetime"
        )
        assert outcome["deleted"] is True

    def test_erasing_the_account_waits_for_a_write_in_progress(
        self, client, auth, seeded_chat
    ):
        """The account lock, which the chat's existence check cannot stand in
        for: `delete_user` removes the conversation without taking the chat's
        lock, so only this one makes it wait."""
        runtime = get_runtime()
        me = client.get("/v1/me", headers=auth["headers"])
        user_id = me.json()["data"]["id"]

        blocked, outcome = self._blocks_while_held(
            lambda: runtime.store.hold_live_conversation(
                seeded_chat, user_id=user_id
            ),
            lambda: asyncio.run(runtime.auth.delete_user(user_id)),
        )

        assert outcome["live"] is True
        assert blocked, (
            "the erasure committed while a cache write held the account"
        )
        assert outcome["deleted"] is True


class TestTheCacheStillWorks:
    """The door must not be a wall: an ordinary turn still warms the cache."""

    def test_a_turn_on_a_live_chat_caches_it(self, client, auth, seeded_chat):
        assert _turn(
            client, auth["headers"], seeded_chat, "a second question"
        ).status_code == 200

        contents = _cached_contents(seeded_chat)
        assert SEEDED in contents
        assert "a second question" in contents

    def test_the_hold_reports_a_live_conversation(self, client, auth, seeded_chat):
        """The store's own answer, so a guard that always refuses is visible
        here rather than only as a missing cache entry."""
        store = get_runtime().store
        with store.hold_live_conversation(seeded_chat) as live:
            assert live is True

    def test_the_hold_reports_a_deleted_conversation(
        self, client, auth, seeded_chat
    ):
        store = get_runtime().store
        assert client.delete(
            f"/v1/conversations/{seeded_chat}", headers=auth["headers"]
        ).status_code == 200

        with store.hold_live_conversation(seeded_chat) as live:
            assert live is False

    def test_a_name_that_is_not_a_conversation_id_is_not_refused(self):
        """`hold_live_user` accepts a non-UUID for the same reason: a name
        that could never have been a row has nothing to resurrect, and
        refusing it would break a caller the deletion has no claim on."""
        store = get_runtime().store
        with store.hold_live_conversation("not-a-uuid") as live:
            assert live is True
