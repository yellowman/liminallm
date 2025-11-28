"""Tests for rate limit validation in runtime.py.

Per SPEC §18, rate limits use Redis token bucket with configurable defaults.
Invalid window_seconds should be logged and default to 60 seconds.
"""
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestCheckRateLimit:
    """Tests for the check_rate_limit function."""

    @pytest.fixture
    def mock_runtime(self):
        """Create a mock runtime with no Redis cache."""
        from liminallm.service.runtime import Runtime

        runtime = MagicMock(spec=Runtime)
        runtime.cache = None
        runtime._local_rate_limits = {}
        runtime._local_rate_limit_lock = asyncio.Lock()
        return runtime

    @pytest.fixture
    def mock_runtime_with_cache(self):
        """Create a mock runtime with Redis cache."""
        from liminallm.service.runtime import Runtime

        runtime = MagicMock(spec=Runtime)
        runtime.cache = AsyncMock()
        runtime.cache.check_rate_limit = AsyncMock(return_value=True)
        return runtime

    async def test_zero_limit_always_passes(self, mock_runtime):
        """Rate limit of 0 or negative always passes."""
        from liminallm.service.runtime import check_rate_limit

        result = await check_rate_limit(mock_runtime, "test_key", 0, 60)
        assert result is True

        result = await check_rate_limit(mock_runtime, "test_key", -1, 60)
        assert result is True

    async def test_invalid_window_logs_warning(self, mock_runtime):
        """Invalid window_seconds logs warning and defaults to 60."""
        from liminallm.service.runtime import check_rate_limit

        with patch("liminallm.service.runtime.logger") as mock_logger:
            result = await check_rate_limit(mock_runtime, "test_key", 10, 0)

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "rate_limit_invalid_window"
            assert "window_seconds" in call_args[1]
            assert call_args[1]["window_seconds"] == 0

    async def test_negative_window_logs_warning(self, mock_runtime):
        """Negative window_seconds logs warning and defaults to 60."""
        from liminallm.service.runtime import check_rate_limit

        with patch("liminallm.service.runtime.logger") as mock_logger:
            result = await check_rate_limit(mock_runtime, "test_key", 10, -5)

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[1]["window_seconds"] == -5

    async def test_valid_window_no_warning(self, mock_runtime):
        """Valid window_seconds doesn't log warning."""
        from liminallm.service.runtime import check_rate_limit

        with patch("liminallm.service.runtime.logger") as mock_logger:
            result = await check_rate_limit(mock_runtime, "test_key", 10, 60)

            mock_logger.warning.assert_not_called()

    async def test_rate_limit_increments_count(self, mock_runtime):
        """Each call increments the rate limit count."""
        from liminallm.service.runtime import check_rate_limit

        # First 5 calls should pass
        for i in range(5):
            result = await check_rate_limit(mock_runtime, "test_key", 5, 60)
            assert result is True, f"Call {i+1} should pass"

        # 6th call should fail
        result = await check_rate_limit(mock_runtime, "test_key", 5, 60)
        assert result is False

    async def test_different_keys_independent(self, mock_runtime):
        """Different keys have independent rate limits."""
        from liminallm.service.runtime import check_rate_limit

        # Exhaust limit for key1
        for _ in range(3):
            await check_rate_limit(mock_runtime, "key1", 3, 60)

        # key2 should still have capacity
        result = await check_rate_limit(mock_runtime, "key2", 3, 60)
        assert result is True

        # key1 should be exhausted
        result = await check_rate_limit(mock_runtime, "key1", 3, 60)
        assert result is False

    async def test_uses_redis_when_available(self, mock_runtime_with_cache):
        """Uses Redis cache when available."""
        from liminallm.service.runtime import check_rate_limit

        result = await check_rate_limit(mock_runtime_with_cache, "test_key", 10, 60)

        mock_runtime_with_cache.cache.check_rate_limit.assert_called_once_with(
            "test_key", 10, 60
        )

    async def test_fallback_to_local_without_redis(self, mock_runtime):
        """Falls back to local rate limiting without Redis."""
        from liminallm.service.runtime import check_rate_limit

        result = await check_rate_limit(mock_runtime, "test_key", 10, 60)

        assert result is True
        assert "test_key" in mock_runtime._local_rate_limits

    async def test_window_reset_after_expiry(self, mock_runtime):
        """Rate limit window resets after window_seconds expire."""
        from liminallm.service.runtime import check_rate_limit

        # Use very short window for testing
        # First exhaust the limit
        for _ in range(2):
            await check_rate_limit(mock_runtime, "test_key", 2, 1)

        # Should be rate limited
        result = await check_rate_limit(mock_runtime, "test_key", 2, 1)
        assert result is False

        # Manually simulate window expiry by backdating the window start
        window_start, count = mock_runtime._local_rate_limits["test_key"]
        from datetime import timedelta
        expired_start = datetime.utcnow() - timedelta(seconds=2)
        mock_runtime._local_rate_limits["test_key"] = (expired_start, count)

        # Should pass now (window reset)
        result = await check_rate_limit(mock_runtime, "test_key", 2, 1)
        assert result is True


class TestRateLimitIntegration:
    """Integration tests for rate limiting with actual runtime."""

    async def test_rate_limit_with_memory_runtime(self):
        """Rate limiting works with actual memory-based runtime."""
        from liminallm.service.runtime import get_runtime, check_rate_limit

        runtime = get_runtime()

        # Clear any existing rate limits for clean test
        runtime._local_rate_limits = {}

        key = f"test_integration_{datetime.utcnow().timestamp()}"

        # Should pass up to limit
        for _ in range(5):
            result = await check_rate_limit(runtime, key, 5, 60)
            assert result is True

        # Should fail after limit
        result = await check_rate_limit(runtime, key, 5, 60)
        assert result is False

    async def test_concurrent_rate_limit_calls(self):
        """Concurrent rate limit calls are handled correctly."""
        from liminallm.service.runtime import get_runtime, check_rate_limit

        runtime = get_runtime()
        runtime._local_rate_limits = {}

        key = f"test_concurrent_{datetime.utcnow().timestamp()}"
        limit = 10
        results = []

        async def make_request():
            result = await check_rate_limit(runtime, key, limit, 60)
            results.append(result)

        # Make more requests than the limit concurrently
        await asyncio.gather(*[make_request() for _ in range(15)])

        # Exactly `limit` should pass, rest should fail
        passes = sum(1 for r in results if r is True)
        fails = sum(1 for r in results if r is False)

        assert passes == limit
        assert fails == 5
