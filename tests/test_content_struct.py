from liminallm.content_struct import normalize_content_struct


def test_normalize_preserves_known_segments():
    struct = {
        "segments": [
            {
                "type": "code",
                "text": "print('hi')",
                "language": "python",
                "unknown": "x",
            },
            {
                "type": "citation",
                "text": "cite",
                "source_id": "doc-1",
                "chunk_id": "chunk-1",
                "score": 0.9,
                "meta": {"note": "keep"},
            },
            {
                "type": "tool_call",
                "name": "lookup",
                "arguments": {"id": "123"},
                "result": {"status": "ok"},
            },
        ]
    }
    normalized = normalize_content_struct(struct)
    assert normalized
    segments = normalized["segments"]
    assert segments[0]["language"] == "python"
    assert "unknown" not in segments[0]
    assert segments[1]["chunk_id"] == "chunk-1"
    assert segments[2]["name"] == "lookup"
    assert "result" in segments[2]


def test_normalize_falls_back_to_text_segment_when_empty():
    normalized = normalize_content_struct({"segments": ["bad"]}, content="hello")
    assert normalized == {"segments": [{"type": "text", "text": "hello"}]}


def test_invalid_content_struct_is_dropped():
    assert normalize_content_struct(None) is None
    assert normalize_content_struct([], content="hi") is None
    assert normalize_content_struct({"segments": "not-a-list"}) is None
