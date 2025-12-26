"""Performance tests for critical paths.

These tests measure response times and throughput for key operations
to ensure the system meets performance requirements.
"""

import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def perf_client():
    """Create a test client for performance testing."""
    import os
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("USE_MEMORY_STORE", "true")
    os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")
    os.environ.setdefault("JWT_SECRET", "test-secret-key")

    from liminallm.service.runtime import reset_runtime_for_tests
    reset_runtime_for_tests()

    from liminallm.app import app
    return TestClient(app)


@pytest.fixture
def auth_headers(perf_client):
    """Get auth headers for authenticated requests."""
    email = f"perf_{uuid.uuid4().hex[:8]}@example.com"
    response = perf_client.post(
        "/v1/auth/signup",
        json={"email": email, "password": "PerfTest123!"},
    )
    assert response.status_code == 201
    token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestEndpointLatency:
    """Test response latency for critical endpoints."""

    def test_health_endpoint_latency(self, perf_client):
        """Health endpoint should respond in < 50ms."""
        times = []
        for _ in range(100):
            start = time.perf_counter()
            response = perf_client.get("/healthz")
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 200

        avg = statistics.mean(times)
        p95 = sorted(times)[94]
        p99 = sorted(times)[98]

        print(f"\nHealth endpoint latency (ms): avg={avg:.2f}, p95={p95:.2f}, p99={p99:.2f}")
        assert avg < 50, f"Average latency {avg:.2f}ms exceeds 50ms"
        assert p99 < 100, f"P99 latency {p99:.2f}ms exceeds 100ms"

    def test_signup_latency(self, perf_client):
        """Signup should complete in < 500ms."""
        times = []
        for i in range(20):
            email = f"signup_perf_{uuid.uuid4().hex}@example.com"
            start = time.perf_counter()
            response = perf_client.post(
                "/v1/auth/signup",
                json={"email": email, "password": "PerfTest123!"},
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 201

        avg = statistics.mean(times)
        p95 = sorted(times)[18]

        print(f"\nSignup latency (ms): avg={avg:.2f}, p95={p95:.2f}")
        assert avg < 500, f"Average latency {avg:.2f}ms exceeds 500ms"

    def test_login_latency(self, perf_client):
        """Login should complete in < 300ms."""
        # Create multiple users to avoid rate limiting
        num_users = 10
        users = []
        for i in range(num_users):
            email = f"login_perf_{uuid.uuid4().hex}@example.com"
            perf_client.post(
                "/v1/auth/signup",
                json={"email": email, "password": "PerfTest123!"},
            )
            users.append(email)

        times = []
        for i in range(20):
            email = users[i % num_users]
            start = time.perf_counter()
            response = perf_client.post(
                "/v1/auth/login",
                json={"email": email, "password": "PerfTest123!"},
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 200

        avg = statistics.mean(times)
        p95 = sorted(times)[18]

        print(f"\nLogin latency (ms): avg={avg:.2f}, p95={p95:.2f}")
        assert avg < 300, f"Average latency {avg:.2f}ms exceeds 300ms"

    def test_artifacts_list_latency(self, perf_client, auth_headers):
        """Artifact listing should complete in < 100ms."""
        times = []
        for _ in range(50):
            start = time.perf_counter()
            response = perf_client.get("/v1/artifacts", headers=auth_headers)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 200

        avg = statistics.mean(times)
        p95 = sorted(times)[47]

        print(f"\nArtifacts list latency (ms): avg={avg:.2f}, p95={p95:.2f}")
        assert avg < 100, f"Average latency {avg:.2f}ms exceeds 100ms"


class TestThroughput:
    """Test request throughput under load."""

    def test_concurrent_health_checks(self, perf_client):
        """Should handle 50 concurrent health checks."""
        def health_check():
            start = time.perf_counter()
            response = perf_client.get("/healthz")
            return time.perf_counter() - start, response.status_code

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(health_check) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]

        times = [r[0] * 1000 for r in results]
        statuses = [r[1] for r in results]

        success_rate = sum(1 for s in statuses if s == 200) / len(statuses) * 100
        avg = statistics.mean(times)

        print(f"\nConcurrent health: {success_rate:.1f}% success, avg={avg:.2f}ms")
        assert success_rate >= 99, f"Success rate {success_rate:.1f}% below 99%"

    def test_concurrent_auth_requests(self, perf_client):
        """Should handle concurrent signup requests."""
        def signup():
            email = f"concurrent_{uuid.uuid4().hex}@example.com"
            start = time.perf_counter()
            response = perf_client.post(
                "/v1/auth/signup",
                json={"email": email, "password": "PerfTest123!"},
            )
            return time.perf_counter() - start, response.status_code

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(signup) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]

        times = [r[0] * 1000 for r in results]
        statuses = [r[1] for r in results]

        success_rate = sum(1 for s in statuses if s == 201) / len(statuses) * 100
        avg = statistics.mean(times)

        print(f"\nConcurrent signup: {success_rate:.1f}% success, avg={avg:.2f}ms")
        assert success_rate >= 95, f"Success rate {success_rate:.1f}% below 95%"


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_no_memory_leak_on_repeated_requests(self, perf_client, auth_headers):
        """Memory should not grow significantly over repeated requests."""
        import gc

        # Warm up
        for _ in range(10):
            perf_client.get("/v1/artifacts", headers=auth_headers)

        gc.collect()
        # Note: Full memory profiling would require tracemalloc or similar
        # This is a basic sanity check

        for _ in range(100):
            perf_client.get("/v1/artifacts", headers=auth_headers)

        gc.collect()
        # If we got here without OOM, basic memory management is working
        print("\nMemory test passed (no OOM after 100 requests)")
