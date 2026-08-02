"""The admin console builds itself from the settings schema.

It used to mirror the field list by hand — every setting written out twice in
JavaScript plus a block of markup. Thirty settings had no control at all, and
smtp_password was still being posted after it became environment-only, which
made every save fail with a 400. These pin the schema the console renders from
and the invariants that stop the two drifting apart again.
"""

import json
import re
from pathlib import Path

import pytest

from liminallm.config import (
    MODEL_AFFECTING_SETTINGS,
    SYSTEM_SETTINGS_DEFAULTS,
    Settings,
    managed_settings_schema,
)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class TestSchema:
    def test_describes_every_managed_setting_and_nothing_else(self):
        described = {entry["name"] for entry in managed_settings_schema()}
        assert described == set(SYSTEM_SETTINGS_DEFAULTS)

    def test_every_entry_has_what_a_control_needs(self):
        for entry in managed_settings_schema():
            assert entry["type"] in {"int", "float", "bool", "text", "choice"}
            assert entry["group"]
            assert "default" in entry
            if entry["type"] == "choice":
                assert entry["choices"]
                # The shipped default has to be selectable, or the form opens
                # showing a value the operator never chose.
                assert str(entry["default"]) in entry["choices"]

    def test_bounds_reach_the_form(self):
        by_name = {entry["name"]: entry for entry in managed_settings_schema()}
        assert by_name["history_budget_fraction"]["min"] == 0.1
        assert by_name["history_budget_fraction"]["max"] == 0.9
        assert by_name["default_page_size"]["max"] == 1000
        # 0 disables a rate limit, so the control must allow it.
        assert by_name["chat_rate_limit_per_minute"]["min"] == 0

    def test_choices_come_from_the_field_type(self):
        by_name = {entry["name"]: entry for entry in managed_settings_schema()}
        assert set(by_name["voice_default_voice"]["choices"]) == {
            "alloy", "echo", "fable", "onyx", "nova", "shimmer"
        }
        assert "openai" in by_name["model_backend"]["choices"]

    def test_expensive_changes_are_flagged(self):
        """Saving one of these rebuilds the model stack; the console warns."""
        flagged = {
            entry["name"]
            for entry in managed_settings_schema()
            if entry["reloads_model"]
        }
        assert flagged == set(MODEL_AFFECTING_SETTINGS)

    def test_nothing_falls_into_the_unlabelled_bucket(self):
        """"Other" is a safety net so a new setting still appears, not a
        destination. Eighteen ended up there once and nobody noticed, because
        the console rendered them — just at the bottom, under no heading."""
        stranded = [
            entry["name"]
            for entry in managed_settings_schema()
            if entry["group"] == "Other"
        ]
        assert stranded == []

    def test_settings_are_grouped_contiguously(self):
        """The console prints a heading when the group changes, so a group
        appearing twice would render two headings for it."""
        groups = [entry["group"] for entry in managed_settings_schema()]
        seen = []
        for group in groups:
            if not seen or seen[-1] != group:
                assert group not in seen, f"{group} is split across the list"
                seen.append(group)

    def test_credentials_are_settable_but_write_only(self):
        """Rotating an SMTP password should not require restarting the app.

        They are managed settings like any other, so the console can set them
        — but they are marked secret, and a secret is never read back out.
        """
        by_name = {entry["name"]: entry for entry in managed_settings_schema()}
        for name in ("smtp_password", "jwt_secret", "web_search_api_key",
                     "adapter_openai_api_key", "voice_api_key",
                     "oauth_google_client_secret"):
            assert by_name[name]["secret"] is True, name
            assert by_name[name]["default"] == "", name

    def test_only_credentials_are_marked_secret(self):
        """A non-secret marked secret becomes invisible and unverifiable."""
        secret_names = {
            entry["name"] for entry in managed_settings_schema() if entry["secret"]
        }
        for name in secret_names:
            assert any(
                token in name
                for token in ("password", "secret", "api_key", "_key")
            ), name


class TestSchemaEndpoint:
    def test_requires_admin(self, client, auth_headers):
        assert client.get("/v1/admin/settings/schema", headers=auth_headers).status_code == 403

    def test_returns_the_fields(self, client, admin_headers):
        resp = client.get("/v1/admin/settings/schema", headers=admin_headers)
        assert resp.status_code == 200
        fields = resp.json()["data"]["fields"]
        assert len(fields) == len(SYSTEM_SETTINGS_DEFAULTS)

    def test_marks_which_settings_an_admin_changed(self, client, admin_headers):
        before = {
            entry["name"]: entry["overridden"]
            for entry in client.get(
                "/v1/admin/settings/schema", headers=admin_headers
            ).json()["data"]["fields"]
        }
        # The generated signing key is the only thing a fresh instance stores.
        assert [name for name, was in before.items() if was] == ["jwt_secret"]

        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"model_path": "something-else"},
        )
        after = {
            entry["name"]: entry["overridden"]
            for entry in client.get(
                "/v1/admin/settings/schema", headers=admin_headers
            ).json()["data"]["fields"]
        }
        assert after["model_path"] is True
        assert after["notes_enabled"] is False


class TestSaveValidation:
    def test_a_bad_value_is_rejected_with_the_offending_field(
        self, client, admin_headers
    ):
        resp = client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"default_page_size": 10**6},
        )
        assert resp.status_code == 400
        details = resp.json()["error"]["details"]
        assert "default_page_size" in details["fields"]

    def test_every_bad_field_is_named_not_just_the_first(
        self, client, admin_headers
    ):
        resp = client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={
                "default_page_size": 10**6,
                "session_rotation_hours": 0,
                "voice_default_voice": "not-a-voice",
            },
        )
        assert resp.status_code == 400
        fields = resp.json()["error"]["details"]["fields"]
        assert set(fields) == {
            "default_page_size", "session_rotation_hours", "voice_default_voice"
        }

    def test_a_rejected_save_changes_nothing(self, client, admin_headers):
        client.put(
            "/v1/admin/settings", headers=admin_headers, json={"model_path": "keep-me"}
        )
        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"model_path": "discard-me", "default_page_size": -5},
        )
        current = client.get("/v1/admin/settings", headers=admin_headers).json()
        assert current["data"]["model_path"] == "keep-me"

    def test_an_env_only_setting_cannot_be_written(self, client, admin_headers):
        """extract_reader_plugins imports Python modules — the console must
        not be a path to running code."""
        resp = client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"extract_reader_plugins": "evil.module"},
        )
        assert resp.status_code == 400

    def test_a_saved_value_survives_a_reload(self, client, admin_headers):
        client.put(
            "/v1/admin/settings", headers=admin_headers, json={"notes_enabled": False}
        )
        current = client.get("/v1/admin/settings", headers=admin_headers).json()
        assert current["data"]["notes_enabled"] is False

    @pytest.mark.parametrize("value", [0, 60, 1000])
    def test_rate_limits_accept_zero_for_unlimited(self, client, admin_headers, value):
        resp = client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"chat_rate_limit_per_minute": value},
        )
        assert resp.status_code == 200


class TestConsoleSource:
    """The console must not reacquire a hard-coded field list."""

    def test_the_form_is_not_a_hand_written_field_list(self):
        markup = (FRONTEND / "admin.html").read_text()
        hardcoded = re.findall(r'id="setting-[a-z-]+"', markup)
        assert hardcoded == [], hardcoded

    def test_the_script_does_not_enumerate_settings(self):
        script = (FRONTEND / "admin.js").read_text()
        # The old version listed each setting twice, once to load and once to
        # save. Anything resembling that list is the drift coming back.
        assert "getVal('setting-" not in script
        assert "setChecked('setting-" not in script

    def test_the_script_reads_the_schema_endpoint(self):
        script = (FRONTEND / "admin.js").read_text()
        assert "/admin/settings/schema" in script

    def test_field_errors_are_surfaced_per_control(self):
        script = (FRONTEND / "admin.js").read_text()
        # requestEnvelope flattened responses to a message string, so the
        # API's per-field detail could never reach the form.
        assert "extractErrorDetails" in script
        assert "setFieldError" in script

    def test_unsaved_changes_are_guarded(self):
        script = (FRONTEND / "admin.js").read_text()
        assert "beforeunload" in script


def test_schema_is_json_serialisable():
    """It goes over the wire; a stray Enum or set would 500 the endpoint."""
    json.dumps(managed_settings_schema())


def test_every_choice_is_actually_accepted_by_the_model():
    """A choice the form offers but the API rejects is a trap."""
    for entry in managed_settings_schema():
        if entry["type"] != "choice":
            continue
        for choice in entry["choices"]:
            Settings(**{entry["name"]: choice})


class TestSecretHandling:
    """A credential should be rotatable from the console and never readable."""

    def test_a_saved_secret_is_never_returned(self, client, admin_headers):
        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"smtp_password": "hunter2"},
        )
        body = client.get("/v1/admin/settings", headers=admin_headers).json()
        assert body["data"]["smtp_password"] == ""
        assert "hunter2" not in json.dumps(body)

    def test_the_schema_says_whether_one_is_set(self, client, admin_headers):
        def smtp_entry():
            fields = client.get(
                "/v1/admin/settings/schema", headers=admin_headers
            ).json()["data"]["fields"]
            return next(f for f in fields if f["name"] == "smtp_password")

        assert smtp_entry()["is_set"] is False
        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"smtp_password": "hunter2"},
        )
        entry = smtp_entry()
        assert entry["is_set"] is True
        assert entry["default"] == ""
        assert "hunter2" not in json.dumps(entry)

    def test_an_empty_secret_leaves_the_stored_one_alone(
        self, client, admin_headers, store
    ):
        """The console cannot show the current value, so it cannot round-trip
        one either — a blank field means 'not retyped', not 'erase it'."""
        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"smtp_password": "hunter2"},
        )
        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"smtp_password": "", "smtp_host": "mail.example.com"},
        )
        stored = store.get_system_settings_overrides()
        assert stored["smtp_password"] == "hunter2"
        assert stored["smtp_host"] == "mail.example.com"

    def test_a_secret_can_be_rotated(self, client, admin_headers, store):
        for value in ("first", "second"):
            client.put(
                "/v1/admin/settings",
                headers=admin_headers,
                json={"smtp_password": value},
            )
        assert store.get_system_settings_overrides()["smtp_password"] == "second"

    def test_the_signing_key_is_generated_once_and_shared(self, store):
        """Every replica reading the same database gets the same key. It used
        to be generated per process into a file, so a worker that could not
        read that file rejected tokens the others had issued."""
        from liminallm.service.runtime import get_runtime, reset_runtime_for_tests

        first = get_runtime().settings.jwt_secret
        assert len(first) >= 32
        reset_runtime_for_tests()
        assert get_runtime().settings.jwt_secret == first


class TestChangesTakeEffectWithoutRestart:
    """The point of moving configuration into the database.

    Restarting the process to change an SMTP password is the wrong answer, so
    these check that a save actually reaches the running services rather than
    only the settings object.
    """

    def test_rotating_the_smtp_password_reaches_the_mailer(
        self, client, admin_headers
    ):
        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={
                "smtp_host": "mail.example.com",
                "smtp_user": "postmaster",
                "smtp_password": "rotated-secret",
                "email_from_address": "noreply@example.com",
            },
        )
        runtime.refresh_settings()
        # EmailService captures its configuration at construction, so this only
        # holds because refresh rebuilds it.
        assert runtime.email.smtp_password == "rotated-secret"
        assert runtime.email.smtp_host == "mail.example.com"
        assert runtime.email.is_configured

    def test_a_feature_flag_reaches_the_workflow_engine(
        self, client, admin_headers
    ):
        from liminallm.service.runtime import get_runtime

        runtime = get_runtime()
        assert runtime.workflow._notes_enabled() is True
        client.put(
            "/v1/admin/settings", headers=admin_headers, json={"notes_enabled": False}
        )
        runtime.refresh_settings()
        assert runtime.workflow._notes_enabled() is False

    def test_cors_origins_follow_the_saved_setting(self, client, admin_headers):
        from liminallm import app as app_module
        from liminallm.service.runtime import get_runtime

        client.put(
            "/v1/admin/settings",
            headers=admin_headers,
            json={"cors_allow_origins": ["https://console.example"]},
        )
        get_runtime().refresh_settings()
        assert app_module._allowed_origins() == ["https://console.example"]
