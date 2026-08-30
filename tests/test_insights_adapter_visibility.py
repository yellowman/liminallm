"""Preference insights describe the adapters visible to their own subject.

`/v1/preferences/insights` is a user-scoped surface: the route always calls
`summarize_preferences(principal.user_id)`, and inside it the events and the
clusters are both read for that user. The adapter list alone was read with an
unscoped `list_artifacts(type_filter="adapter")`.

The store treats an unscoped artifact listing as a question about public
visibility, deliberately - caller identity is what adds that caller's private
rows, and tenant identity is what adds their tenant's shared ones. So the
panel did not leak: it collapsed to `visibility='global'` and never showed a
user the per-user adapter their own feedback had just created.

Every case below names a specific adapter. "The list is non-empty" would pass
against a fix that returned every adapter on the instance, which is the one
outcome worse than showing none.
"""

from __future__ import annotations

import uuid

import pytest

from liminallm.service.runtime import get_runtime

PASSWORD = "TestPassword123!"


def signup(client, prefix="ins"):
    """An account through the real route, plus the identity it was given."""
    email = f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/signup", json={"email": email, "password": PASSWORD}
    )
    assert resp.status_code == 201, resp.text
    headers = {"Authorization": f"Bearer {resp.json()['data']['access_token']}"}
    me = client.get("/v1/me", headers=headers).json()["data"]
    return headers, me


def insight_adapter_ids(client, headers):
    resp = client.get("/v1/preferences/insights", headers=headers)
    assert resp.status_code == 200, resp.text
    return [a["id"] for a in resp.json()["data"].get("adapters") or []]


@pytest.fixture
def matrix(client, store):
    """One subject, and an adapter of every visibility class around them.

    Five rows rather than two, because `owner_user_id` alone turns the
    obvious red green while leaving this call one argument short of the one
    that serves adapters at turn time.
    """
    subject_headers, subject = signup(client, "subject")
    home_tenant = subject["tenant_id"]
    tag = uuid.uuid4().hex[:6]

    # The subject's own row comes from the real code path, so this is the
    # reported case rather than an imitation of it: `ensure_user_adapter` is
    # what a thumbs-up calls, and it creates the adapter private and owned.
    mine = get_runtime().training.ensure_user_adapter(subject["id"])
    # Every other row is that same schema with a different owner. Copied from
    # the real artifact rather than written out here, so a change to the
    # adapter shape cannot leave this file asserting against a shape the
    # product no longer creates.
    template = dict(mine.schema)

    def like_it(name, *, owner, visibility):
        schema = dict(template)
        schema["user_id"] = owner
        return store.create_artifact(
            "adapter",
            name,
            schema,
            description="visibility witness",
            owner_user_id=owner,
            visibility=visibility,
        )

    neighbour = store.create_user(
        email=f"nb_{uuid.uuid4().hex[:8]}@example.com", tenant_id=home_tenant
    )
    outsider = store.create_user(
        email=f"out_{uuid.uuid4().hex[:8]}@example.com",
        tenant_id=f"other-{tag}",
    )

    ids = {
        "mine_private": mine.id,
        "neighbour_private": like_it(
            f"nb-private-{tag}", owner=neighbour.id, visibility="private"
        ).id,
        "neighbour_shared": like_it(
            f"nb-shared-{tag}", owner=neighbour.id, visibility="shared"
        ).id,
        "outsider_shared": like_it(
            f"out-shared-{tag}", owner=outsider.id, visibility="shared"
        ).id,
        "global": like_it(
            f"global-{tag}", owner=outsider.id, visibility="global"
        ).id,
    }
    return subject_headers, subject, ids


class TestInsightsShowTheSubjectTheirOwnAdapters:
    def test_the_subjects_private_adapter_is_listed(self, client, matrix):
        """The reported defect: a user's own persona adapter was invisible.

        `ensure_user_adapter` creates it private and owned by that user, so
        this is the row the panel exists to show and the only row the
        unscoped listing could never return.
        """
        headers, _subject, ids = matrix

        listed = insight_adapter_ids(client, headers)

        assert ids["mine_private"] in listed, listed

    def test_another_users_private_adapter_is_not_listed(self, client, matrix):
        """Scoping to the subject, not merely widening past global.

        Without this, "show the caller's adapters" and "show every adapter on
        the instance" are the same passing test, and the second is a
        cross-user disclosure rather than a fix.
        """
        headers, _subject, ids = matrix

        listed = insight_adapter_ids(client, headers)

        assert ids["neighbour_private"] not in listed, listed


class TestInsightsFollowTheOrdinaryVisibilityContract:
    """Not a private-only list: the same contract `_select_adapters` gets."""

    def test_a_shared_adapter_in_the_subjects_tenant_is_listed(
        self, client, matrix
    ):
        """Tenant identity is what admits a shared row.

        This is the case `owner_user_id=` alone does not reach, and the
        reason the fix passes the subject's tenant as well: a listing given
        only a caller sees their own private rows and the global ones, and
        silently drops everything their tenant shares with them.
        """
        headers, _subject, ids = matrix

        listed = insight_adapter_ids(client, headers)

        assert ids["neighbour_shared"] in listed, listed

    def test_a_shared_adapter_in_another_tenant_is_not_listed(
        self, client, matrix
    ):
        """The tenant argument scopes; it does not merely unlock.

        Paired with the case above so that passing the tenant through cannot
        be mistaken for admitting every shared adapter anywhere.
        """
        headers, _subject, ids = matrix

        listed = insight_adapter_ids(client, headers)

        assert ids["outsider_shared"] not in listed, listed

    def test_a_global_adapter_is_still_listed(self, client, matrix):
        """What the unscoped listing already returned must not be lost.

        The fix narrows a query that was too wide in one direction and too
        narrow in two others. This is the direction that was already right.
        """
        headers, _subject, ids = matrix

        listed = insight_adapter_ids(client, headers)

        assert ids["global"] in listed, listed


class TestSummarizingNobodyIsUnchanged:
    def test_no_subject_still_describes_the_public_set(self, client, matrix):
        """`summarize_preferences(None)` keeps the meaning it already had.

        The signature allows it and nothing in the product passes it, so this
        pins the existing behaviour rather than inventing one: with no
        identity to scope by, the store's answer is the public set, and the
        fix must not turn that into an error or into everything.
        """
        _headers, _subject, ids = matrix

        listed = [
            a["id"]
            for a in get_runtime().training.summarize_preferences(None)["adapters"]
        ]

        assert ids["global"] in listed, listed
        assert ids["mine_private"] not in listed, listed
        assert ids["neighbour_shared"] not in listed, listed
