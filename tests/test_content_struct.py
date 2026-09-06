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


class TestSegmentCoordinatesArePositionsInTheContent:
    """SPEC §2.2 defines one coordinate system for the whole structure:
    offsets into the same message's `content`, in Unicode code points, with
    `0 <= start <= end <= len(content)`.

    Normalization is where that becomes true of what is stored. A renderer
    indexes by these numbers, so an offset past the end or an inverted range
    is a coordinate into nothing - and a citation with no anchor and a
    redaction with no range are records of nothing, which is why the segment
    goes rather than the two keys.
    """

    CONTENT = "hello"

    def _kept(self, segment, content=CONTENT):
        """The stored segments that carry a coordinate.

        Not "segments of this type": when every segment is dropped the
        normalizer puts the content back as one text segment, and a witness
        that counted that as the survivor would pass whatever happened.
        """
        normalized = normalize_content_struct({"segments": [segment]}, content)
        return [
            item for item in (normalized or {}).get("segments") or []
            if "start" in item or "end" in item
        ]

    def test_a_span_inside_the_content_is_kept(self):
        kept = self._kept({"type": "text", "text": "he", "start": 0, "end": 2})
        assert kept and kept[0]["start"] == 0 and kept[0]["end"] == 2

    def test_the_end_of_the_content_is_a_position_in_it(self):
        kept = self._kept({
            "type": "citation", "source_id": "d", "start": 5, "end": 5,
        })
        assert kept and kept[0]["start"] == 5

    def test_a_negative_offset_is_dropped(self):
        assert self._kept({"type": "text", "text": "x", "start": -1}) == []

    def test_an_offset_past_the_end_is_dropped(self):
        assert self._kept({"type": "text", "text": "x", "end": 6}) == []

    def test_an_inverted_range_is_dropped(self):
        assert self._kept({
            "type": "redaction", "text": "x", "start": 4, "end": 2,
        }) == []

    def test_a_non_integer_offset_is_dropped(self):
        assert self._kept({"type": "text", "text": "x", "start": "0"}) == []
        assert self._kept({"type": "text", "text": "x", "start": 1.0}) == []
        # `bool` is an `int` in Python, so `True` would otherwise be stored as
        # the position 1.
        assert self._kept({"type": "text", "text": "x", "start": True}) == []

    def test_a_segment_with_no_coordinates_is_untouched(self):
        normalized = normalize_content_struct(
            {"segments": [{"type": "text", "text": "kept"}]}, self.CONTENT
        )
        assert normalized == {"segments": [{"type": "text", "text": "kept"}]}

    def test_the_message_still_gets_its_text_when_every_segment_goes(self):
        """Dropping the structure never drops the answer: the fallback puts
        the content back as one text segment."""
        normalized = normalize_content_struct(
            {"segments": [{"type": "text", "text": "x", "start": 99}]},
            self.CONTENT,
        )
        assert normalized == {"segments": [{"type": "text", "text": self.CONTENT}]}

    def test_without_the_content_the_shape_is_still_checked(self):
        """A caller that passes no content cannot have the upper bound
        checked, and can still be told that a negative offset is not one."""
        assert self._kept({"type": "text", "text": "x", "start": 99}, None)
        assert self._kept({"type": "text", "text": "x", "start": -1}, None) == []
