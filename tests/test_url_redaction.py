"""A URL names its password twice, and the log mask read only one spelling.

Same shape as the harness's `?db=` defect (docs/ISSUES.md, 2I.1 carry-over):
both drivers accept connection keywords in the query string, so
`redis://host/0?password=x` and `postgresql://host/db?password=x` carry the
secret outside the userinfo — measured, both drivers honour it — and a mask
that rewrites only the netloc publishes it to the log line it exists to
protect.
"""

from __future__ import annotations

from liminallm.service.runtime import _mask_url_password


class TestTheUserinfoSpelling:
    def test_a_userinfo_password_is_masked(self):
        assert (
            _mask_url_password("redis://:hunter2@cache:6379/0")
            == "redis://:***@cache:6379/0"
        )

    def test_a_username_survives_the_mask(self):
        masked = _mask_url_password("postgresql://app:hunter2@db:5432/prod")
        assert masked is not None and "hunter2" not in masked
        assert "app" in masked

    def test_a_url_with_no_password_is_untouched(self):
        for url in (
            "redis://cache:6379/0",
            "postgresql://app@db:5432/prod?sslmode=require",
            None,
            "",
        ):
            assert _mask_url_password(url) == url


class TestTheQuerySpelling:
    def test_a_query_password_is_masked(self):
        """redis-py and libpq both read `?password=` — measured."""
        for url in (
            "redis://cache.example.com:6379/0?password=hunter2",
            "postgresql://app@db.example.com:5432/prod?password=hunter2",
        ):
            masked = _mask_url_password(url)
            assert masked is not None and "hunter2" not in masked, (
                f"the query password reached the log line: {masked}"
            )

    def test_libpq_sslpassword_is_masked_too(self):
        masked = _mask_url_password(
            "postgresql://app@db:5432/prod?sslmode=require&sslpassword=hunter2"
        )
        assert masked is not None and "hunter2" not in masked
        assert "sslmode=require" in masked, "an innocent argument was lost"

    def test_both_spellings_at_once_are_both_masked(self):
        masked = _mask_url_password(
            "postgresql://app:first@db:5432/prod?password=second"
        )
        assert masked is not None
        assert "first" not in masked and "second" not in masked

    def test_innocent_query_arguments_survive(self):
        url = "redis://cache:6379/0?socket_timeout=5"
        assert _mask_url_password(url) == url

    def test_the_masked_value_is_literally_three_asterisks(self):
        """Exact output, because "the secret is gone" was already true.

        `urlencode` percent-encodes by default, so every masked query value
        came out as `%2A%2A%2A` — safe, and unreadable in the log line this
        function exists to produce. `*` is not special in a query string.
        Asserted as an exact string rather than a substring, since a
        substring check passes on the encoded form too.
        """
        assert (
            _mask_url_password("postgresql://app@db:5432/prod?password=hunter2")
            == "postgresql://app@db:5432/prod?password=***"
        )
        assert (
            _mask_url_password("redis://cache:6379/0?password=hunter2")
            == "redis://cache:6379/0?password=***"
        )
        assert (
            _mask_url_password(
                "postgresql://app@db/prod?sslmode=require&sslpassword=hunter2"
            )
            == "postgresql://app@db/prod?sslmode=require&sslpassword=***"
        )
