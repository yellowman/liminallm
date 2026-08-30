"""Where the data lives is a fact about the machine, not about the install.

`shared_fs_root` was a database-managed setting, and that is not something a
database can decide. `Runtime` must construct `PostgresStore` before it can
read managed settings at all, so the store is handed the shipped default while
every service built afterwards uses whatever Postgres says - and an admin edit
changes the second without the first.

That is the split-root condition a harness test already forbids under test.
This is the same condition in production, reached by a supported action.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from liminallm.config import Settings, get_settings
from liminallm.service.runtime import get_runtime


def test_shared_fs_root_comes_from_the_environment():
    """It is needed before the database is readable, so it cannot live there.

    `env_field` is reserved for secrets and for bootstrap - anything needed
    before Postgres is reachable, or that describes the machine rather than
    the install. A filesystem mount is both.
    """
    extra = Settings.model_fields["shared_fs_root"].json_schema_extra or {}
    assert extra.get("env") == "SHARED_FS_ROOT", (
        "shared_fs_root is not bound to an environment variable, so the store "
        "is constructed from the shipped default while everything built after "
        "it uses whatever the database says"
    )
    assert not extra.get("admin"), (
        "shared_fs_root is still admin-managed, so an admin edit can move the "
        "filesystem root out from under a store that is already constructed"
    )


def test_a_stored_setting_cannot_move_the_filesystem_root():
    """The database is not an authority on this one.

    A row saying `shared_fs_root=/mnt/elsewhere` must not take effect, because
    the store that writes artifact payloads was built before that row could be
    read and will go on writing where it started.
    """
    runtime = get_runtime()
    before = runtime.settings.shared_fs_root
    elsewhere = tempfile.mkdtemp(prefix="not_the_root_")

    runtime.store.set_system_settings({"shared_fs_root": elsewhere})
    runtime.refresh_settings()
    try:
        assert runtime.settings.shared_fs_root == before, (
            f"a stored setting moved the filesystem root to {elsewhere} while "
            f"the store keeps writing under {runtime.store.fs_root}"
        )
        assert (
            Path(runtime.store.fs_root).resolve()
            == Path(runtime.settings.shared_fs_root).resolve()
        )
    finally:
        runtime.store.set_system_settings({"shared_fs_root": None})
        runtime.refresh_settings()


def test_instance_settings_json_cannot_seed_it_either():
    """First-boot seeding is the other way into the settings table."""
    from liminallm.config import SYSTEM_SETTINGS_DEFAULTS

    assert "shared_fs_root" not in SYSTEM_SETTINGS_DEFAULTS, (
        "shared_fs_root is still a database-managed default, so "
        "INSTANCE_SETTINGS_JSON can seed it into the settings table - the "
        "same split root by another route"
    )


def test_the_suite_does_not_write_to_the_production_data_root():
    """`/srv/liminallm` is where a real install keeps its data.

    The suite writes artifact payloads, adapters, files and lock files under
    the shared root, and nothing removes it at session end. Running the tests
    on a machine that also runs the application must not put test data into
    that machine's real data directory.
    """
    root = Path(get_settings().shared_fs_root).resolve()
    assert root != Path("/srv/liminallm"), (
        "the test suite is writing into the production default data root"
    )
    assert str(root).startswith(tempfile.gettempdir()), (
        f"the suite's shared root is {root}, which is not a throwaway "
        "directory it created"
    )
