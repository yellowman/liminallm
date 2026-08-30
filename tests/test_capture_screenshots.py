"""The screenshot harness owns its services, and names them explicitly.

Two properties, both learned the hard way. `redis_url` is a
database-managed setting with a `localhost:6379` default and no environment
variable behind it, so exporting `REDIS_URL` configures nothing: a capture
that relied on it ran against whichever Redis happened to be listening on
the developer's machine while believing it was isolated. And the scratch
Postgres and Redis are server processes with data directories, so a run that
starts them and does not stop them leaves both behind for the next one.

These test the script's own wiring, not a screenshot: no browser, no images.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / (
    "capture_screenshots.py"
)


def load_script():
    """Import the script by path; `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("capture_screenshots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["capture_screenshots"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


SCRATCH_REDIS = "redis://127.0.0.1:53219/0"


class TestTheScratchRedisIsTheOneTheRuntimeUses:
    """The setting is managed, so the URL has to be seeded, not exported."""

    def test_the_seeded_settings_name_the_scratch_redis(self, script):
        args = script.parse_args([])
        settings = script.instance_settings(args, SCRATCH_REDIS)
        assert settings["redis_url"] == SCRATCH_REDIS

    def test_the_shipped_localhost_default_is_not_what_runs(self, script):
        """The default is a real service on a developer's machine.

        A capture that silently used it would write to somebody's own Redis
        and read back state this script never seeded.
        """
        from liminallm.config import SYSTEM_SETTINGS_DEFAULTS

        default = SYSTEM_SETTINGS_DEFAULTS["redis_url"]
        args = script.parse_args([])
        settings = script.instance_settings(args, SCRATCH_REDIS)
        assert settings["redis_url"] != default, (
            "the capture would run against the shipped default Redis"
        )

    def test_the_offline_default_needs_no_credential(self, script):
        args = script.parse_args([])
        settings = script.instance_settings(args, SCRATCH_REDIS)
        assert settings["model_backend"] == "stub"
        assert "model_path" not in settings

    def test_live_without_a_credential_refuses(self, script, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        args = script.parse_args(["--live"])
        with pytest.raises(SystemExit):
            script.instance_settings(args, SCRATCH_REDIS)

    def test_live_names_the_requested_model_and_no_credential(
        self, script, monkeypatch
    ):
        """The key reaches the runtime through the environment alone.

        It must never be written into the settings the script seeds, which
        are logged and persisted.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
        args = script.parse_args(["--live", "--model", "some-model-42"])
        settings = script.instance_settings(args, SCRATCH_REDIS)
        assert settings["model_path"] == "some-model-42"
        assert "not-a-real-key" not in str(settings)


class TestEveryStartedServiceIsStopped:
    def test_a_failure_during_capture_still_stops_both_services(
        self, script, monkeypatch, tmp_path
    ):
        """Capture raises part way through, which is the interesting case:
        the two scratch servers are this script's to clean up either way.

        `main` points the process at throwaway state, and `shared_fs_root` is
        bound to an environment variable rather than a stored setting, so
        those writes would outlive this test and move the filesystem root for
        everything that ran after it in the same worker. The environment is a
        copy for the duration.
        """
        monkeypatch.setattr(script.os, "environ", dict(script.os.environ))
        stopped: list[str] = []

        class FakePostgres:
            def start(self):
                return "postgresql://scratch/db"

            def stop(self):
                stopped.append("postgres")

        class FakeRedis:
            def start(self):
                return SCRATCH_REDIS

            def stop(self):
                stopped.append("redis")

        class FakeServer:
            base_url = "http://127.0.0.1:1"

            def start(self):
                return self

            def stop(self):
                stopped.append("server")

        class FakeClient:
            """Stands in for the seeding client: no socket, and it records
            that the script closed it."""

            def __init__(self, *_args, **_kwargs):
                pass

            def post(self, *_args, **_kwargs):
                return None

            def close(self):
                stopped.append("client")

        import httpx

        import tests.browser as browser_mod
        import tests.harness as harness_mod

        monkeypatch.setattr(httpx, "Client", FakeClient)
        monkeypatch.setattr(harness_mod, "ScratchPostgres", FakePostgres)
        monkeypatch.setattr(harness_mod, "ScratchRedis", FakeRedis)
        monkeypatch.setattr(harness_mod, "apply_schema", lambda *a, **k: None)
        monkeypatch.setattr(browser_mod, "LiveServer", FakeServer)
        monkeypatch.setattr(script, "assert_isolated", lambda url: None)

        def boom(*_args, **_kwargs):
            raise RuntimeError("the browser died mid-capture")

        monkeypatch.setattr(script, "capture", boom)

        with pytest.raises(RuntimeError, match="mid-capture"):
            script.main(["--out", str(tmp_path)])

        assert "redis" in stopped, "the scratch Redis was left running"
        assert "postgres" in stopped, "the scratch Postgres was left running"
        assert "server" in stopped, "the app server was left running"
        assert "client" in stopped, "the seeding client was left open"


class TestTheIsolationCheckRefusesAForeignRedis:
    def test_a_runtime_pointed_elsewhere_is_refused(self, script, monkeypatch):
        """`TEST_MODE` lets the app run with no cache at all, so "it started"
        is not evidence that the scratch Redis was reached."""

        class Settings:
            redis_url = "redis://localhost:6379/0"

        class Runtime:
            settings = Settings()
            cache = object()

        import liminallm.service.runtime as runtime_mod

        monkeypatch.setattr(runtime_mod, "get_runtime", lambda: Runtime())
        with pytest.raises(SystemExit, match="does not own"):
            script.assert_isolated(SCRATCH_REDIS)

    def test_a_runtime_with_no_cache_is_refused(self, script, monkeypatch):
        class Settings:
            redis_url = SCRATCH_REDIS

        class Runtime:
            settings = Settings()
            cache = None

        import liminallm.service.runtime as runtime_mod

        monkeypatch.setattr(runtime_mod, "get_runtime", lambda: Runtime())
        with pytest.raises(SystemExit, match="never reached"):
            script.assert_isolated(SCRATCH_REDIS)
