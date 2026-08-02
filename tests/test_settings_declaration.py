"""One declaration per setting.

Twenty-nine settings used to be declared twice: once as a Settings field with
an env var, once again as a SYSTEM_SETTINGS_DEFAULTS entry for the admin UI.
Nine of the pairs had already drifted apart. These tests keep the derivation
honest so it cannot happen again.
"""

from enum import Enum

import pytest

from liminallm.config import SYSTEM_SETTINGS_DEFAULTS, Settings


def _admin_fields():
    for name, field in Settings.model_fields.items():
        extra = field.json_schema_extra or {}
        if isinstance(extra, dict) and extra.get("admin"):
            yield name, field


def test_every_admin_field_reaches_the_admin_defaults():
    for name, _field in _admin_fields():
        assert name in SYSTEM_SETTINGS_DEFAULTS, name


@pytest.mark.parametrize("name,field", list(_admin_fields()))
def test_admin_default_matches_the_settings_default(name, field):
    """The two can no longer disagree, because there is only one value."""
    expected = field.default
    if isinstance(expected, Enum):
        expected = expected.value
    elif expected is None:
        expected = ""
    assert SYSTEM_SETTINGS_DEFAULTS[name] == expected


def test_admin_only_settings_have_no_env_var():
    """Anything in the defaults but not on Settings is deliberately UI-only.

    If one of these ever needs an env var, it should be declared on Settings
    with admin=True rather than added here as a second declaration.
    """
    admin_names = {name for name, _ in _admin_fields()}
    ui_only = set(SYSTEM_SETTINGS_DEFAULTS) - admin_names
    for name in ui_only:
        assert name not in Settings.model_fields, (
            f"{name} is declared on Settings but not marked admin=True, so its "
            f"default is written out twice"
        )


def test_defaults_are_serialisable_as_the_admin_api_sends_them():
    """Enums and None do not survive a JSON round trip through the admin form."""
    for name, value in SYSTEM_SETTINGS_DEFAULTS.items():
        assert not isinstance(value, Enum), name
        assert value is not None, name
        assert isinstance(value, (str, int, float, bool)), (name, type(value))


def test_the_database_is_not_seeded_with_defaults(store):
    """A shipped default must never look like an explicit admin override.

    Seeding them made every default outrank the operator's environment, which
    is how notes_enabled once ignored its own env var.
    """
    assert store.get_system_settings_overrides() == {}
    # ...while the merged view still answers for every setting.
    merged = store.get_system_settings()
    for name in SYSTEM_SETTINGS_DEFAULTS:
        assert name in merged
