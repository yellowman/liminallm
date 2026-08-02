"""Response schemas that mirror a storage model stay in step with it.

Three API schemas are field-for-field copies of a storage dataclass. They were
built by restating every field at each call site — ten such mappings — and had
already drifted: GET /me reported `tenant_id or "global"` while the admin user
listing reported the stored value, for the same user.

They are now built with `model_validate(obj)`, which fixes the restating but
introduces a quieter failure: add a field to the storage model and the mirror
simply does not carry it. Nothing raises; the field is just missing from the
API. This test is what makes that loud.

A mirror is allowed to diverge — but only on purpose, by being listed here with
the reason.
"""

from __future__ import annotations

import dataclasses

import pytest

from liminallm.api.schemas import (
    ConfigPatchAuditResponse,
    ContextSourceResponse,
    UserResponse,
)
from liminallm.storage.models import ConfigPatchAudit, ContextSource, User

#: (api schema, storage model, fields the API deliberately does not expose)
MIRRORS = [
    (UserResponse, User, frozenset()),
    (ConfigPatchAuditResponse, ConfigPatchAudit, frozenset()),
    (ContextSourceResponse, ContextSource, frozenset()),
]


@pytest.mark.parametrize(
    "schema, model, withheld", MIRRORS, ids=lambda x: getattr(x, "__name__", "")
)
def test_mirror_carries_every_stored_field(schema, model, withheld):
    stored = {f.name for f in dataclasses.fields(model)} - set(withheld)
    exposed = set(schema.model_fields)
    missing = stored - exposed
    assert not missing, (
        f"{model.__name__} has {sorted(missing)} that {schema.__name__} does not "
        f"carry, so the API silently drops it. Add the field, or list it as "
        f"deliberately withheld in MIRRORS."
    )


@pytest.mark.parametrize(
    "schema, model, withheld", MIRRORS, ids=lambda x: getattr(x, "__name__", "")
)
def test_mirror_invents_no_fields(schema, model, withheld):
    stored = {f.name for f in dataclasses.fields(model)}
    invented = set(schema.model_fields) - stored
    assert not invented, (
        f"{schema.__name__} declares {sorted(invented)} that {model.__name__} "
        f"has no field for; model_validate() will never populate it."
    )


@pytest.mark.parametrize(
    "schema, model, withheld", MIRRORS, ids=lambda x: getattr(x, "__name__", "")
)
def test_mirror_is_built_from_the_model(schema, model, withheld):
    """from_attributes is what lets model_validate() read the dataclass."""
    assert schema.model_config.get("from_attributes"), (
        f"{schema.__name__} cannot be built with model_validate(); call sites "
        f"will go back to restating every field by hand."
    )


def test_a_user_round_trips_without_being_restated():
    user = User(id="u1", email="a@b.c", handle="h", role="admin", tenant_id="acme")
    mirrored = UserResponse.model_validate(user)
    assert mirrored.tenant_id == "acme"  # not coerced to "global"
    assert mirrored.role == "admin"
    assert mirrored.handle == "h"
