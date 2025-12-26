"""Unit tests for the stub model backend.

Ensures the stub backend returns deterministic canned responses,
preventing accidental regression in smoke test infrastructure.
"""

from liminallm.service.model_backend import StubBackend


def test_stub_generate_returns_canned_response():
    """StubBackend.generate() returns the expected canned response."""
    backend = StubBackend()

    result = backend.generate(
        messages=[{"role": "user", "content": "Hello"}],
        adapters=[],
    )

    assert result["content"] == StubBackend.STUB_RESPONSE
    assert "usage" in result
    assert result["usage"]["total_tokens"] == 20


def test_stub_generate_stream_yields_tokens_then_done():
    """StubBackend.generate_stream() yields tokens then message_done."""
    backend = StubBackend()

    events = list(
        backend.generate_stream(
            messages=[{"role": "user", "content": "Hello"}],
            adapters=[],
        )
    )

    # Should have at least one token event
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) >= 1, "Expected at least one token event"

    # Last event should be message_done
    assert events[-1]["event"] == "message_done"
    assert events[-1]["data"]["content"] == StubBackend.STUB_RESPONSE
    assert "usage" in events[-1]["data"]
