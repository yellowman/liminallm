"""SPEC §18: what model-written Python can and cannot see.

The contract is a property, not a mechanism, so these tests name no
namespaces and no `unveil`. They state the view:

    can see:    staged inputs RO, the per-call workdir RW, the runtime RO
    cannot see: shared_fs_root, other users, service config, host paths

and they are written to run on any platform with a confinement backend. On a
platform with none, `run_python` is unavailable rather than degraded, and the
last class asserts exactly that.

The bug these exist for: the interpreter ran as the service uid with an
ordinary filesystem view, so `open("/srv/liminallm/users/<someone-else>/…")`
worked. `os.chdir()` is not confinement, and unix permissions cannot help when
the service owns every user's files. Each test therefore reads through the
real `run_python_sandboxed` rather than testing the backend directly — the
question is what the tool exposes, not what the mechanism intends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from liminallm.service import confine, interpreter

requires_backend = pytest.mark.skipif(
    confine.backend_name() is None,
    reason="no filesystem confinement backend on this platform",
)


def _run(code: str, workdir: Path) -> dict:
    return interpreter.run_python_sandboxed(code, workdir=str(workdir))


def _probe(path: str) -> str:
    """Code that reports whether one absolute path is readable."""
    return (
        "try:\n"
        f"    print('READABLE:' + open({path!r}).read()[:40])\n"
        "except OSError as exc:\n"
        "    print('blocked:%s' % exc.errno)\n"
    )


@pytest.fixture
def staged(tmp_path):
    """A workdir holding one staged attachment, as `prepare_workdir` leaves it."""
    workdir = tmp_path / "session-test"
    workdir.mkdir()
    (workdir / "input.csv").write_text("a,b\n1,2\n")
    return workdir


@pytest.fixture
def elsewhere(tmp_path):
    """Files that stand in for the rest of the host: another user's area under
    a shared root, and a service config file."""
    shared_root = tmp_path / "srv" / "liminallm"
    other_user = shared_root / "users" / "user-b" / "files"
    other_user.mkdir(parents=True)
    secret = other_user / "private.txt"
    secret.write_text("SECRET-BELONGING-TO-USER-B")
    config = tmp_path / "srv" / "config" / "settings.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"provider_api_key": "sk-should-not-be-readable"}))
    return {"other_user": secret, "config": config, "root": shared_root}


@requires_backend
class TestWhatItCannotSee:
    def test_system_files_are_absent(self, staged):
        result = _run(_probe("/etc/passwd"), staged)
        assert result["ok"], result["stderr"]
        assert "READABLE" not in result["stdout"], result["stdout"]

    def test_another_users_files_are_absent(self, staged, elsewhere):
        """The one that matters. The service uid owns this file, so nothing
        about permissions stops the read — the path has to not exist."""
        result = _run(_probe(str(elsewhere["other_user"])), staged)
        assert result["ok"], result["stderr"]
        assert "SECRET-BELONGING-TO-USER-B" not in result["stdout"]
        assert "READABLE" not in result["stdout"], result["stdout"]

    def test_service_configuration_is_absent(self, staged, elsewhere):
        result = _run(_probe(str(elsewhere["config"])), staged)
        assert result["ok"], result["stderr"]
        # On READABLE, not on the secret's text: the probe prints a prefix of
        # the file, and asserting the secret is absent passed unconfined
        # purely because the truncation happened to fall mid-string.
        assert "READABLE" not in result["stdout"], result["stdout"]
        assert "sk-should-not-be-readable" not in result["stdout"]

    def test_the_workdirs_own_host_path_is_absent(self, staged):
        """Even the staged directory is unreachable by the name the host knows
        it by: the code sees it where confinement put it, and nothing else."""
        result = _run(_probe(str(staged / "input.csv")), staged)
        assert result["ok"], result["stderr"]
        assert "READABLE" not in result["stdout"], result["stdout"]

    def test_the_shared_root_cannot_be_listed(self, staged, elsewhere):
        code = (
            "import os\n"
            "try:\n"
            f"    print('LISTED:%s' % os.listdir({str(elsewhere['root'])!r}))\n"
            "except OSError as exc:\n"
            "    print('blocked:%s' % exc.errno)\n"
        )
        result = _run(code, staged)
        assert result["ok"], result["stderr"]
        assert "LISTED" not in result["stdout"], result["stdout"]

    def test_walking_up_from_the_workdir_finds_nothing(self, staged, elsewhere):
        """A relative escape, since the absolute one is gone."""
        code = (
            "import os\n"
            "seen = []\n"
            "for depth in range(1, 8):\n"
            "    try:\n"
            "        seen.append(sorted(os.listdir('../' * depth))[:8])\n"
            "    except OSError:\n"
            "        seen.append('blocked')\n"
            "print(seen)\n"
        )
        result = _run(code, staged)
        assert result["ok"], result["stderr"]
        assert "SECRET" not in result["stdout"]
        assert "users" not in result["stdout"], result["stdout"]


@requires_backend
class TestWhatItCanSee:
    def test_the_staged_attachment_is_readable(self, staged):
        result = _run("print(open('input.csv').read())", staged)
        assert result["ok"], result["stderr"]
        assert "a,b" in result["stdout"]

    def test_the_workdir_is_writable_and_the_artifact_is_published(
        self, staged, tmp_path
    ):
        """The output has to reach the host, or the tool does nothing useful."""
        result = _run("open('report.txt', 'w').write('computed')", staged)
        assert result["ok"], result["stderr"]
        assert [f["name"] for f in result["created_files"]] == ["report.txt"]

        dest = tmp_path / "user_files"
        published = interpreter.publish_artifacts(
            str(staged), str(dest), result["created_files"], allowed_extensions={".txt"}
        )
        assert published == ["report.txt"]
        assert (dest / "report.txt").read_text() == "computed"

    def test_the_language_runtime_still_works(self, staged):
        """Confinement that breaks `import csv` would just make the tool
        useless — the runtime is read-only, not absent. Imported *after*
        confinement, so it exercises the mounted runtime rather than a module
        the child had already loaded."""
        code = (
            "import csv, io, json, zipfile, sqlite3\n"
            "rows = list(csv.reader(io.StringIO(open('input.csv').read())))\n"
            "print(json.dumps(rows))\n"
        )
        result = _run(code, staged)
        assert result["ok"], result["stderr"]
        assert '[["a", "b"], ["1", "2"]]' in result["stdout"]


@requires_backend
class TestTheOtherLayersStillHold:
    """Confinement replaced nothing; it sits under the existing defenses."""

    def test_networking_modules_stay_blocked(self, staged):
        result = _run("import socket", staged)
        assert not result["ok"]
        assert "not available in the sandbox" in result["stderr"]

    def test_process_spawning_stays_blocked(self, staged):
        result = _run("import os; os.system('id')", staged)
        assert not result["ok"]
        assert "process execution is not available" in result["stderr"]


class TestTheAvailabilityCheckIsSafeToAskAnywhere:
    """It is called from the API process, which has JAX loaded and threads
    running. The first version answered by forking and unsharing in a child —
    a correct answer obtained a dangerous way: a fork there can leave the
    child deadlocked on a lock held by a thread it did not inherit, which the
    interpreter warns about. It reads /proc now, and the authoritative check
    is `confine()` itself in the already-spawned single-threaded child.
    """

    def test_it_does_not_fork(self, monkeypatch):
        def _no_forking(*args, **kwargs):  # pragma: no cover - the assertion
            raise AssertionError("the availability check forked")

        monkeypatch.setattr("os.fork", _no_forking)
        assert confine.backend_name() in {None, "linux-namespaces", "openbsd-unveil"}

    @requires_backend
    def test_it_agrees_with_what_actually_happens(self, staged):
        """Cheap predicate, real outcome: if it says a backend exists, the
        confined child must actually run. A predicate that drifted optimistic
        would turn every call into a sandbox error."""
        result = _run("print('confined')", staged)
        assert result["ok"], result["stderr"]
        assert "confined" in result["stdout"]


class TestNoUnconfinedFallback:
    """A platform without a backend loses the tool; it does not get a weaker one."""

    def test_confine_refuses_rather_than_returning(self, monkeypatch, tmp_path):
        monkeypatch.setattr(confine, "_BACKENDS", ())
        with pytest.raises(confine.ConfinementUnavailable, match="no filesystem"):
            confine.confine(str(tmp_path))

    def test_the_tool_reports_it_instead_of_spawning(self, monkeypatch, staged):
        monkeypatch.setattr(interpreter, "backend_name", lambda: None)

        def _must_not_spawn(*args, **kwargs):  # pragma: no cover - the assertion
            raise AssertionError("untrusted code was spawned without confinement")

        monkeypatch.setattr(
            "liminallm.service.sandbox.run_in_sandbox", _must_not_spawn
        )
        result = interpreter.run_python_sandboxed("print(1)", workdir=str(staged))
        assert result["ok"] is False
        assert "no filesystem confinement backend" in result["stderr"]
