"""Which tenant a request belongs to, decided by the site it arrived at.

A tenant is a property of *where you went*, not of anything the caller says. It
used to be neither: five text boxes labelled "Tenant (optional)" collected a
value that the server variously discarded (login), rejected with a 422 (signup,
OAuth) or refused with a 403 (admin create-user), while an ``X-Tenant-ID``
header echoed the server's own answer back at it. Nothing could set a tenant,
and typing in the box could only break the request.

Now the hostname decides. ``tenant_domains`` maps host to tenant; an empty map
means the install serves one tenant, which is every deployment until someone
adds a second site.

The hostname is only as trustworthy as whatever set it. ``Host`` is a request
header like any other — ``curl -H 'Host: acme.example.com'`` costs nothing — so
in front of a proxy the operator turns on ``trust_forwarded_host`` and the
proxy is responsible for setting ``X-Forwarded-Host`` from the real request and
for refusing hosts it does not serve. That is the whole trust boundary, and it
is deliberately one line rather than a heuristic: guessing which of several
forwarding headers to believe is how this kind of code gets it wrong.
"""

from __future__ import annotations

from typing import Optional

from liminallm.service.errors import NotFoundError

#: Hosts that never carry a tenant. A probe or a healthcheck reaches the box by
#: address, not by site name, and must not be refused for it.
_INFRA_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "[::1]", "testserver"})


def normalize_host(value: Optional[str]) -> str:
    """Lowercase, no port, no trailing dot — how a host is compared.

    A browser sends ``Acme.Example.com:8443``; an operator types
    ``acme.example.com``. Both have to land on the same key.
    """
    if not value:
        return ""
    host = str(value).split(",")[0].strip().lower().rstrip(".")
    if host.startswith("["):  # bracketed IPv6 literal, optional :port after ]
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.split(":")[0]


def host_of(headers, settings) -> str:
    """The site name this request was sent to.

    ``headers`` is anything with a case-insensitive ``.get`` — a Starlette
    Request's or a WebSocket's both qualify.
    """
    if settings.trust_forwarded_host:
        forwarded = headers.get("x-forwarded-host")
        if forwarded:
            return normalize_host(forwarded)
    return normalize_host(headers.get("host"))


def tenant_for_host(host: str, settings) -> str:
    """The tenant serving ``host``.

    With no mapping configured every host is the one tenant. Once a mapping
    exists an unlisted host is refused: serving it the default tenant would
    mean any DNS name pointed at the box gets that tenant's login page, which
    is the spoofing this module exists to prevent.
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
