"""Which tenant a request belongs to: the site it arrived at, and who is on it.

A tenanted request has two halves and **both** must agree.

*The site* comes from the host the web server was addressed as, resolved
through ``tenant_domains``. An empty map means one tenant for the whole
install; once any mapping exists, an unlisted host is refused rather than
served the default tenant, because otherwise any DNS name pointed at the box
reaches it. ``Host`` is client-supplied, so ``X-Forwarded-Host`` is believed
only when ``trust_forwarded_host`` says a proxy sets it — that flag is the
entire trust boundary on the site half.

*The account* comes from the authenticated session, never from the request
(CLAUDE.md: derive ``tenant_id`` from the token, never from user input).

Neither half is sufficient alone, which is why the check is a comparison and
not a lookup. The host is attacker-chosen on the unproxied path, so the site
half cannot stand by itself. A session is a bearer credential that stays valid
against whatever site it is replayed at, so the account half cannot either.
Requiring them to match means a stolen acme session is useless at globex, and
a forged ``Host`` reaches nothing the caller could not already reach.

No host is exempt. An earlier version let ``localhost`` and the test client's
``testserver`` through to the default tenant on the theory that a probe
arrives by address rather than by site name — but probes do not authenticate
and never ask for a tenant, while that exemption let anyone who could reach
the port name their own tenant. An operator who wants a bare hostname served
lists it like any other.
"""

from __future__ import annotations

from typing import Optional

from liminallm.service.errors import NotFoundError


def normalize_host(value: Optional[str]) -> str:
    """Lowercase, no port, no trailing dot: a browser sends
    ``Acme.Example.com:8443`` where an operator typed ``acme.example.com``.

    One spelling for one host, because this is also what ``tenant_domains``
    keys are normalized with. Two normalizers that agree most of the time
    produce a host an operator can type but never match.
    """
    if not value:
        return ""
    host = str(value).split(",")[0].strip().lower().rstrip(".")
    if host.startswith("["):  # bracketed IPv6 literal, optional :port after ]
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    if host.count(":") > 1:
        # A bare IPv6 literal, which is how an operator types it into settings
        # and never how a Host header carries it. Canonicalize to the bracketed
        # spelling the wire uses, so "::1" and "[::1]:8000" are one key.
        return f"[{host}]"
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
    """The tenant serving ``host`` — the site half of the decision.

    Unlisted hosts are refused once any mapping exists, with no exemption for
    bare addresses: serving them the default tenant would mean any DNS name
    pointed at the box reaches it, and ``Host`` is chosen by the caller.
    """
    domains = settings.tenant_domains or {}
    if not domains:
        return settings.default_tenant_id
    tenant = domains.get(host)
    if not tenant:
        raise NotFoundError("no site is configured at this address")
    return tenant


def tenant_of(headers, settings) -> str:
    """The tenant for a request, from its headers. The only entry point."""
    return tenant_for_host(host_of(headers, settings), settings)


def user_belongs_to_site(user_tenant: Optional[str], site_tenant: Optional[str]) -> bool:
    """The account half: does this session belong on the site it arrived at?

    A blank on either side is a mismatch, not a pass. Skipping the comparison
    when one value is missing is how the comparison goes missing — and the
    caller that has nothing to compare is exactly the one that resolved no
    site, which is the case that must not be trusted.
    """
    return bool(user_tenant) and bool(site_tenant) and user_tenant == site_tenant
