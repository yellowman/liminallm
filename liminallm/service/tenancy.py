"""Which tenant a request belongs to, decided by the site it arrived at.

``tenant_domains`` maps host to tenant; an empty map means one tenant for the
whole install. Nothing a caller sends can override it.

``Host`` is a client-supplied header, so ``X-Forwarded-Host`` is believed only
when ``trust_forwarded_host`` says a proxy sets it. That flag is the entire
trust boundary.
"""

from __future__ import annotations

from typing import Optional

from liminallm.service.errors import NotFoundError

#: Hosts that carry no tenant: a probe arrives by address, not by site name.
_INFRA_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "[::1]", "testserver"})


def normalize_host(value: Optional[str]) -> str:
    """Lowercase, no port, no trailing dot: a browser sends
    ``Acme.Example.com:8443`` where an operator typed ``acme.example.com``."""
    if not value:
        return ""
    host = str(value).split(",")[0].strip().lower().rstrip(".")
    if host.startswith("["):  # bracketed IPv6 literal, optional :port after ]
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.split(":")[0]


def host_of(headers, settings) -> str:
    """The site this request was sent to. ``headers`` is anything with a
    case-insensitive ``.get`` (a Request's or a WebSocket's)."""
    if settings.trust_forwarded_host:
        forwarded = headers.get("x-forwarded-host")
        if forwarded:
            return normalize_host(forwarded)
    return normalize_host(headers.get("host"))


def tenant_for_host(host: str, settings) -> str:
    """The tenant serving ``host``.

    Unlisted hosts are refused once any mapping exists: serving them the
    default tenant would mean any DNS name pointed at the box reaches it.
    """
    domains = settings.tenant_domains or {}
    if not domains:
        return settings.default_tenant_id
    if host in _INFRA_HOSTS:
        return settings.default_tenant_id
    tenant = domains.get(host)
    if not tenant:
        raise NotFoundError("no site is configured at this address")
    return tenant


def tenant_of(headers, settings) -> str:
    """The tenant for a request, from its headers. The only entry point."""
    return tenant_for_host(host_of(headers, settings), settings)
