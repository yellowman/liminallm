# why tenancy and auth are shaped this way

SPEC §12 states the rules: tenant from the host, account from the session,
both halves must agree through one comparison method; no host exemptions;
one-time identity tokens are consumed, not observed. Each rule closed a
hole. This file records the holes. The account-erasure and identity-token
work has its full forensic record in docs/ISSUES.md (tranches 2G and 2H);
this file keeps only what a future editor needs before weakening a rule.

## the localhost exemption was a hole, not a convenience

`localhost`, `127.0.0.1`, `::1` and the test client's `testserver` used to
resolve to `default_tenant_id` even with a `tenant_domains` mapping
configured, on the theory that a probe arrives by address rather than by
site name. But `Host` is chosen by whoever can reach the port, so anyone
reaching the service directly named the default tenant — and with
`allow_signup` on, registered an account there. Probes do not authenticate
and never resolve a tenant, so nothing legitimate depended on the
exemption. An operator who wants a bare hostname served lists it like any
other.

## OAuth needed the site check said separately

`app_user.email` is globally unique, so resolving an account by provider id
or email finds it whatever site the flow began at. The provider proves who
someone is, not where they belong: signing in with Google at globex used to
mint acme's tokens, while the password path refused exactly that. The
check (`tenancy.user_belongs_to_site`, surfaced as
`AuthService._site_matches`) is one method with every entry point calling
it — password login, OAuth completion, refresh, and every authenticated
request — because the copy that gets missed on the next edit is an
authorization hole.

## one host normalizer

The request path and the `tenant_domains` validator normalized hostnames
separately and disagreed on bracketed IPv6 — settings split at the first
colon, requests kept the brackets — which made `::1` impossible to map and,
once bare addresses stopped being exempt, impossible to reach. One
normalizer, shared by both, and a bare IPv6 literal canonicalizes to the
bracketed spelling the wire uses.

## `default_tenant_id` refuses to be blank

A blank site tenant matches no account under the two-halves rule, so
clearing it in the console would 401 every user — including the admin who
would have to set it back, since that route authenticates too. The field
refuses (`min_length=1`) instead.

## TOTP verifies what its own QR promises

TOTP is HMAC-SHA-1, 6 digits, 30s, 160-bit secret (RFC 6238 / RFC 4226 §4
R6). These are the Key Uri Format defaults, so an authenticator app assumes
them whatever the `otpauth://` URI omits. The server once verified SHA-256
while every app computed SHA-1, so enrolment could never complete and
nothing said why. The URI now states `algorithm`, `digits` and `period`
explicitly rather than relying on the defaults holding — and the server
verifies the same thing its own QR code promises.

## identity tokens name accounts, not addresses

A password-reset token that recorded the email address followed the address
to whoever held it next: delete the account that asked for a reset,
register the same address, and the token resolved to the new account.
Nothing in the flow looked unusual — the attacker held a token their own
account was legitimately issued. Tokens record `user.id`; ids are never
reused, so the token expires with the account. Issuance runs inside the
account's lifetime guard, and consumption is a single atomic take
(GETDEL-style), so two racing completions cannot both act on one token.
Full record: docs/ISSUES.md tranche 2H.1.

## the session model

Refresh credentials stay out of JS-visible storage: the server sets
`refresh_token` and `session_id` as HttpOnly cookies, with a non-HttpOnly
`csrf_token` beside them. WebSockets authenticate with exactly one of
`access_token` or `session_id` in the first frame — both at once is refused
(`fresh_session_required`, close 4401), because a socket presenting two
credentials of different ages is how a rotated-out session sneaks back in.
The socket's tenant comes from the host it was opened against, like every
HTTP route; nothing in the frame can name one.
