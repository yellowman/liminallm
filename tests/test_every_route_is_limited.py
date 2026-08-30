"""Every endpoint enforces a rate limit, or says out loud why it does not.

The check exists because the preamble was six lines repeated fifty-seven
times, and a paragraph that long is one a new endpoint gets written without.
Nothing failed loudly when that happened - the endpoint was simply unlimited,
which looks exactly like an endpoint that works.
"""

import inspect
import re

import pytest
from fastapi.routing import APIRoute

from liminallm.api.routes import router

#: Endpoints that legitimately do not rate limit, each for a stated reason.
#: Adding a name here should be a decision someone reads, not a way to quiet
#: the test.
EXEMPT = {
    # Liveness and readiness: a probe that gets 429ed reads as an outage, and
    # these do no work worth limiting.
    "healthz", "readyz", "livez", "health_check", "readiness_check",
    "metrics", "prometheus_metrics",
    # Static pages and the OpenAPI document.
    "serve_index", "serve_admin", "serve_share", "openapi_json",
    # The version stamp: a constant.
    "version_info",
    # MCP's GET half: a constant 405 (this server offers no server-initiated
    # stream), the same class as version_info. The POST half, where all the
    # work happens, authenticates and limits.
    "mcp_get",
    # Delegates to update_system_settings, which limits. Limiting here as well
    # would charge one request against the bucket twice.
    "patch_system_settings",
}


def _endpoints():
    # Straight off the router: app.routes keeps an included router as one
    # object rather than flattening it, so iterating the app finds nine.
    return [r for r in router.routes if isinstance(r, APIRoute)]


def _enforces_a_limit(fn) -> bool:
    try:
        source = inspect.getsource(fn)
    except OSError:  # pragma: no cover - source always available in-tree
        return False
    return bool(re.search(r"await rate_limit\(|await enforce\(|await enforce_per_plan\(", source))


@pytest.mark.parametrize(
    "route", list(_endpoints()), ids=lambda r: r.endpoint.__name__
)
def test_route_enforces_a_rate_limit(route):
    name = route.endpoint.__name__
    if name in EXEMPT:
        pytest.skip("exempt by name")
    assert _enforces_a_limit(route.endpoint), (
        f"{name} ({','.join(sorted(route.methods))} {route.path}) enforces no "
        f"rate limit. Add `await rate_limit(runtime, \"<policy>\", subject)`, or "
        f"add it to EXEMPT with a reason."
    )


def test_every_policy_names_a_real_setting():
    """A policy pointing at a missing setting fails only when that route is hit."""
    from liminallm.api.limits import RATE_POLICIES
    from liminallm.config import Settings

    for policy, (limit_attr, window) in RATE_POLICIES.items():
        assert hasattr(Settings, "model_fields") and limit_attr in Settings.model_fields, (
            f"{policy} -> {limit_attr}"
        )
        if isinstance(window, str):
            assert window in Settings.model_fields, f"{policy} -> {window}"
