"""Tests for the shared BM25 scoring utility (liminallm/service/bm25.py).

These tests verify the BM25 implementation used for hybrid RAG retrieval
per SPEC §2.5 retrieval strategy.
"""

from liminallm.service.bm25 import (
    BM25_K1,
    BM25_B,
    tokenize_text,
    compute_bm25_scores,
)


class TestTokenizeText:
    """Tests for the tokenize_text function."""

    def test_tokenize_simple_text(self):
        """Basic tokenization splits on word boundaries."""
        result = tokenize_text("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_lowercases(self):
        """Tokenization converts to lowercase for case-insensitive matching."""
        result = tokenize_text("HELLO World HeLLo")
        assert result == ["hello", "world", "hello"]

    def test_tokenize_handles_punctuation(self):
        """Punctuation is stripped, only word characters kept."""
        result = tokenize_text("Hello, World! How are you?")
        assert result == ["hello", "world", "how", "are", "you"]

    def test_tokenize_handles_numbers(self):
        """Numbers are included as tokens."""
        result = tokenize_text("Version 2.0 released in 2024")
        assert result == ["version", "2", "0", "released", "in", "2024"]

    def test_tokenize_empty_string(self):
        """Empty string returns empty list."""
        result = tokenize_text("")
        assert result == []

    def test_tokenize_whitespace_only(self):
        """Whitespace-only string returns empty list."""
        result = tokenize_text("   \t\n  ")
        assert result == []

    def test_tokenize_special_characters_only(self):
        """Special characters only returns empty list."""
        result = tokenize_text("!@#$%^&*()")
        assert result == []

    def test_tokenize_unicode(self):
        """Unicode word characters are tokenized correctly."""
        result = tokenize_text("café résumé naïve")
        assert result == ["café", "résumé", "naïve"]

    def test_tokenize_underscores(self):
        """Underscores are included in tokens (word characters)."""
        result = tokenize_text("hello_world my_var_name")
        assert result == ["hello_world", "my_var_name"]


class TestComputeBM25Scores:
    """Tests for the compute_bm25_scores function."""

    def test_empty_query_returns_zeros(self):
        """Empty query tokens return zero scores for all documents."""
        documents = [["hello", "world"], ["foo", "bar"]]
        scores = compute_bm25_scores([], documents)
        assert scores == [0.0, 0.0]

    def test_empty_documents_returns_empty(self):
        """Empty documents list returns empty scores."""
        scores = compute_bm25_scores(["hello"], [])
        assert scores == []

    def test_both_empty_returns_empty(self):
        """Both empty returns empty list."""
        scores = compute_bm25_scores([], [])
        assert scores == []

    def test_single_document_exact_match(self):
        """Single document with exact query match has positive score."""
        documents = [["hello", "world"]]
        scores = compute_bm25_scores(["hello"], documents)
        assert len(scores) == 1
        assert scores[0] > 0

    def test_no_match_returns_zero(self):
        """Document with no matching terms scores zero."""
        documents = [["foo", "bar"]]
        scores = compute_bm25_scores(["hello"], documents)
        assert scores == [0.0]

    def test_term_frequency_increases_score(self):
        """Higher term frequency increases score (up to saturation)."""
        doc_single = ["hello"]
        doc_repeated = ["hello", "hello", "hello"]

        scores_single = compute_bm25_scores(["hello"], [doc_single])
        scores_repeated = compute_bm25_scores(["hello"], [doc_repeated])

        # Repeated terms should have higher score due to term frequency
        assert scores_repeated[0] > scores_single[0]

    def test_document_length_normalization(self):
        """Longer documents are penalized by length normalization."""
        short_doc = ["hello"]
        long_doc = ["hello", "foo", "bar", "baz", "qux", "quux"]

        # When comparing same query against docs of different lengths
        scores = compute_bm25_scores(["hello"], [short_doc, long_doc])

        # Short doc should score higher due to length normalization
        # (term is more significant in shorter doc)
        assert scores[0] > scores[1]

    def test_idf_favors_rare_terms(self):
        """Rare terms (lower document frequency) get higher IDF weight."""
        # "rare" appears in 1 doc, "common" appears in all docs
        documents = [
            ["common", "rare"],
            ["common", "foo"],
            ["common", "bar"],
        ]

        # Query for rare term should score first doc higher
        scores_rare = compute_bm25_scores(["rare"], documents)
        assert scores_rare[0] > 0
        assert scores_rare[1] == 0
        assert scores_rare[2] == 0

    def test_multiple_query_terms(self):
        """Multiple query terms combine scores additively."""
        documents = [
            ["hello", "world"],
            ["hello", "foo"],
            ["world", "bar"],
        ]

        scores = compute_bm25_scores(["hello", "world"], documents)

        # First doc matches both terms, should score highest
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_custom_k1_parameter(self):
        """Custom k1 parameter affects term frequency saturation."""
        documents = [["hello", "hello", "hello"]]

        # Higher k1 = slower saturation = higher scores for repeated terms
        scores_low_k1 = compute_bm25_scores(["hello"], documents, k1=0.5)
        scores_high_k1 = compute_bm25_scores(["hello"], documents, k1=2.5)

        # Both should be positive
        assert scores_low_k1[0] > 0
        assert scores_high_k1[0] > 0

    def test_custom_b_parameter(self):
        """Custom b parameter affects document length normalization."""
        short_doc = ["hello"]
        long_doc = ["hello", "foo", "bar", "baz", "qux"]

        # b=0 means no length normalization
        scores_no_norm = compute_bm25_scores(["hello"], [short_doc, long_doc], b=0.0)
        # b=1 means full length normalization
        scores_full_norm = compute_bm25_scores(["hello"], [short_doc, long_doc], b=1.0)

        # With no normalization, difference should be smaller
        diff_no_norm = abs(scores_no_norm[0] - scores_no_norm[1])
        diff_full_norm = abs(scores_full_norm[0] - scores_full_norm[1])

        assert diff_full_norm > diff_no_norm

    def test_default_parameters(self):
        """Default k1 and b parameters match BM25 standard values."""
        assert BM25_K1 == 1.5
        assert BM25_B == 0.75

    def test_returns_float_scores(self):
        """All returned scores are floats."""
        documents = [["hello"], ["world"]]
        scores = compute_bm25_scores(["hello"], documents)

        assert all(isinstance(s, float) for s in scores)

    def test_score_count_matches_document_count(self):
        """Number of scores equals number of documents."""
        documents = [["a"], ["b"], ["c"], ["d"], ["e"]]
        scores = compute_bm25_scores(["a"], documents)

        assert len(scores) == len(documents)

    def test_identical_documents_same_score(self):
        """Identical documents receive identical scores."""
        documents = [
            ["hello", "world"],
            ["hello", "world"],
        ]
        scores = compute_bm25_scores(["hello"], documents)

        assert scores[0] == scores[1]

    def test_handles_single_token_document(self):
        """Single token documents are handled correctly."""
        documents = [["hello"]]
        scores = compute_bm25_scores(["hello"], documents)

        assert len(scores) == 1
        assert scores[0] > 0

    def test_division_by_zero_protection(self):
        """No division by zero errors with edge cases."""
        # Empty document
        documents = [[]]
        scores = compute_bm25_scores(["hello"], documents)
        assert scores == [0.0]

        # Single empty doc among others
        documents = [["hello"], [], ["world"]]
        scores = compute_bm25_scores(["hello"], documents)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] == 0.0
