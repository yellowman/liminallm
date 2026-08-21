"""Where a setting is allowed to come from.

Env vars are for secrets and bootstrap — things needed before the database is
reachable, or that describe the machine rather than the install. Everything
else lives in the database, where it is auditable, changeable without a
restart, and identical across replicas.
"""

import json
import os

import pytest

from liminallm.config import (
    SYSTEM_SETTINGS_DEFAULTS,
    Settings,
    apply_managed_settings,
)
from liminallm.service.runtime import get_runtime, reset_runtime_for_tests

# Env vars are a standing invitation to configure per-container. Each name here
# is a deliberate exception; adding one should be a decision, not a reflex.
EXPECTED_ENV_SETTINGS = {
    # The one that cannot come from the database, because it is how you reach
    # the database.
    "database_url",
    # Not configuration: the build stamps this, nobody tunes it.
    "build_sha",
    # The test harness tells the process it is a test before anything else
    # happens, and it must not be flippable from a web form.
    "test_mode",
    # Where the data lives on this machine. Needed while the Postgres store
    # is being constructed, so a stored value could not take effect: the store
    # keeps the root it was built with while every service made afterwards
    # uses the refreshed one — artifact payloads under one tree, file and
    # adapter authority under another.
    "shared_fs_root",
    # A property of the schema that was applied, not of the running app;
    # scripts/migrate.sh needs the same value.
    "embedding_vector_dim",
    # Imports Python modules. Settable from the admin console would mean
    # remote code execution, which is the one thing genuinely too hot for a
    # row in a table.
    "extract_reader_plugins",
}


def _env_settings():
    return {
        name
        for name, field in Settings.model_fields.items()
        if isinstance(field.json_schema_extra, dict)
        and field.json_schema_extra.get("env")
        and not field.json_schema_extra.get("admin")
    }


def test_only_the_listed_settings_read_the_environment():
    assert _env_settings() == EXPECTED_ENV_SETTINGS


def test_every_other_setting_is_database_managed():
    managed = set(Settings.model_fields) - _env_settings()
    assert managed == set(SYSTEM_SETTINGS_DEFAULTS)


def test_a_managed_setting_does_not_absorb_a_matching_env_var(monkeypatch):
    """The old from_env fell back to NAME.upper() for fields with no env
    declaration, which would silently give every managed setting an env var
    back."""
    monkeypatch.setenv("MODEL_PATH", "sneaked-in")
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "1")
    settings = Settings.from_env()
    assert settings.model_path == "gpt-4o-mini"
    assert settings.chat_rate_limit_per_minute == 60


def test_env_settings_still_read_the_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")
    assert Settings.from_env().database_url == "postgresql://from-env/db"


class TestOverlay:
    def test_stored_values_win_over_declared_defaults(self):
        base = Settings()
        merged = apply_managed_settings(base, {"model_path": "llama-3"})
        assert merged.model_path == "llama-3"
        assert base.model_path == "gpt-4o-mini", "the base must not be mutated"

    def test_unknown_keys_are_ignored(self):
        merged = apply_managed_settings(Settings(), {"not_a_setting": 1})
        assert not hasattr(merged, "not_a_setting")

    def test_env_only_settings_cannot_be_overridden_from_the_database(self):
        """extract_reader_plugins imports code; a row must not be able to."""
        merged = apply_managed_settings(
            Settings(), {"extract_reader_plugins": "evil.module"}
        )
        assert merged.extract_reader_plugins == ""

    def test_one_bad_value_does_not_discard_the_good_ones(self):
        """A bad row must not make the instance unbootable — the admin UI that
        would fix it is served by this process."""
        merged = apply_managed_settings(
            Settings(),
            {"model_path": "llama-3", "chat_rate_limit_per_minute": "not-an-int"},
        )
        assert merged.model_path == "llama-3"
        assert merged.chat_rate_limit_per_minute == 60

    def test_empty_store_returns_the_base_unchanged(self):
        base = Settings()
        assert apply_managed_settings(base, {}) is base


class TestFirstBootSeed:
    """INSTANCE_SETTINGS_JSON: the one seam for declarative deploys."""

    @staticmethod
    def _boot(monkeypatch, value):
        monkeypatch.setenv("INSTANCE_SETTINGS_JSON", value)
        reset_runtime_for_tests()
        return get_runtime()

    def test_seeds_managed_settings_on_a_fresh_instance(self, monkeypatch):
        runtime = self._boot(monkeypatch, json.dumps({"model_path": "seeded-model"}))
        assert runtime.settings.model_path == "seeded-model"
        stored = runtime.store.get_system_settings_overrides()
        assert stored["model_path"] == "seeded-model"

    def test_does_not_overwrite_what_an_admin_already_saved(self, monkeypatch):
        """A stale container env must not silently revert an operator."""
        runtime = self._boot(monkeypatch, json.dumps({"model_path": "seeded-model"}))
        runtime.store.merge_instance_config(
            "system_settings", {"model_path": "admin-chose-this"}
        )
        runtime = self._boot(monkeypatch, json.dumps({"model_path": "seeded-model"}))
        assert runtime.settings.model_path == "admin-chose-this"

    def test_unknown_keys_are_dropped_not_stored(self, monkeypatch):
        runtime = self._boot(
            monkeypatch, json.dumps({"model_path": "m", "bogus_key": 1})
        )
        assert "bogus_key" not in runtime.store.get_system_settings_overrides()

    def test_env_only_settings_cannot_be_seeded(self, monkeypatch):
        """The seed writes to the database; env-only settings do not live there."""
        runtime = self._boot(
            monkeypatch, json.dumps({"extract_reader_plugins": "evil.module"})
        )
        stored = runtime.store.get_system_settings_overrides()
        assert "extract_reader_plugins" not in stored

    @pytest.mark.parametrize("bad", ["not json at all", "[1, 2, 3]", "null"])
    def test_malformed_seed_does_not_block_boot(self, monkeypatch, bad):
        runtime = self._boot(monkeypatch, bad)
        assert runtime.settings.model_path == "gpt-4o-mini"

    def test_absent_seed_is_the_normal_case(self, monkeypatch):
        """A fresh instance stores only its generated signing key."""
        monkeypatch.delenv("INSTANCE_SETTINGS_JSON", raising=False)
        reset_runtime_for_tests()
        stored = get_runtime().store.get_system_settings_overrides()
        assert set(stored) == {"jwt_secret"}


def test_no_stray_env_var_reads_outside_config(monkeypatch):
    """Settings should be the only thing reading configuration from os.environ.

    A direct getenv elsewhere is a setting that escaped the classification.
    """
    import subprocess

    # A *read* is the escaped setting. The sandbox child clears and rewrites
    # its own environment on the way into confinement — that is the opposite
    # move, and matching it here would push a security control into an
    # allowlist of exceptions.
    out = subprocess.run(
        [
            "rg", "-n", r"os\.getenv\(|os\.environ\.get\(|os\.environ\[",
            "--glob", "!**/config.py",
            "--glob", "!**/logging.py",
            "liminallm/",
        ],
        capture_output=True, text=True,
    )
    allowed = (
        # The one declarative-deploy seam, read once at boot.
        "INSTANCE_SETTINGS_JSON",
        # A constructor default that every real caller passes explicitly.
        "RAG_MODE",
        # Per-provider credentials, looked up by name so a Zhipu backend reads
        # ZHIPU_API_KEY rather than OPENAI_API_KEY. Secrets belong in env.
        "api_key_env",
        "_api_key_env",
        # Standard proxy protocol, not this application's configuration.
        # (the loop reads a tuple of HTTPS_PROXY/ALL_PROXY and their lowercase
        # spellings, so the variable name is not on the line itself)
        "HTTPS_PROXY",
        "os.getenv(var)",
        # Imports Python modules; an admin who could set it from the UI would
        # have remote code execution, so it stays env-only. See config.py.
        "EXTRACT_READER_PLUGINS",
    )
    offenders = [
        line for line in out.stdout.splitlines()
        if not any(x in line for x in allowed)
    ]
    assert offenders == [], "\n".join(offenders)


def test_no_setting_default_is_restated_outside_config():
    """`getattr(settings, "x", <default>)` is a default written a second time.

    Every managed setting is a declared field, so the attribute is always
    there; supplying a fallback only creates a value that can drift from the
    declaration without anything noticing. Two of them had already drifted:
    web_tools_enabled read False at the call site while the declaration said
    True, and history_budget_fraction was clamped in code to bounds the field
    did not state.
    """
    import subprocess

    out = subprocess.run(
        # getattr with a third argument — the fallback is the problem, not the
        # dynamic lookup.
        ["rg", "-n", r"getattr\((self\.|runtime\.)?settings, [^)]+,", "liminallm/"],
        capture_output=True, text=True,
    )
    assert out.stdout.strip() == "", out.stdout


def test_bounds_are_declared_not_clamped():
    """Numeric limits belong on the field, so the admin API rejects a bad
    value instead of the code silently correcting it later."""
    from pydantic import ValidationError

    for field, bad in (
        ("history_budget_fraction", 5.0),
        ("default_page_size", 10**6),
        ("session_ttl_minutes_web", 0),
        ("max_concurrent_workflows", 0),
        ("rag_chunk_size", 1),
        ("web_fetch_timeout", 0),
    ):
        with pytest.raises(ValidationError):
            Settings(**{field: bad})


def test_rate_limits_may_be_switched_off():
    """0 means unlimited, so these floor at 0 rather than 1."""
    assert Settings(chat_rate_limit_per_minute=0).chat_rate_limit_per_minute == 0


def test_managed_setting_validation_names_every_bad_field():
    from liminallm.config import validate_managed_settings

    errors = validate_managed_settings(
        {
            "default_page_size": 10**6,
            "notes_enabled": "yes please",
            "not_a_setting": 1,
            "model_path": "this-one-is-fine",
        }
    )
    assert set(errors) == {"default_page_size", "notes_enabled", "not_a_setting"}
    assert "not a managed setting" in errors["not_a_setting"]


def test_allowed_values_are_declared_on_the_field():
    """The admin API used to keep its own copy of these lists."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(voice_default_voice="not-a-voice")
    with pytest.raises(ValidationError):
        Settings(embedding_model_id="not-a-model")
