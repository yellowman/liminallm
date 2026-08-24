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
import os
from pathlib import Path

import pytest

from liminallm.service import confine, interpreter

requires_backend = pytest.mark.skipif(
    confine.backend_name() is None,
    reason="no filesystem confinement backend on this platform",
)


def test_a_lane_that_requires_confinement_actually_gets_it(tmp_path):
    """The guard on the guard: a skip must not be able to pass for a pass.

    `requires_backend` above silences every test in this file when the
    platform has no backend. That is right on a laptop and wrong in CI, and
    the difference is not academic — Ubuntu 24.04 turned on
    `kernel.apparmor_restrict_unprivileged_userns`, which lets `unshare`
    succeed and then refuses the identity mapping inside the namespace, so
    confinement stopped working on the hosted runners. Left alone, teaching
    the availability probe about that knob would have converted thirty-one
    failing confinement tests into thirty-one skips and reported a green lane.

    So a lane that is supposed to exercise the real boundary says so with
    `LIMINALLM_REQUIRE_CONFINEMENT`, and this fails loudly when it cannot.
    It runs code rather than reading a sysctl, because what matters is that
    the boundary engages, not that a knob looks encouraging.
    """
    if not os.environ.get("LIMINALLM_REQUIRE_CONFINEMENT"):
        pytest.skip("not a lane that claims to exercise real confinement")

    assert confine.backend_name() is not None, (
        "LIMINALLM_REQUIRE_CONFINEMENT is set, but no confinement backend is "
        "available — every test in this file would skip and the lane would "
        "still be green. On Ubuntu 24.04 check "
        "kernel.apparmor_restrict_unprivileged_userns."
    )
    result = _run("print('confined')", tmp_path)
    assert result["ok"] is True, result["stderr"]
    assert "confined" in result["stdout"]


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
class TestTheViewIsNotOnlyTheFilesystem:
    """`pivot_root` moves the filesystem and nothing else.

    The first version of this jail closed the path escape and left two ways
    out that never touch a path: the environment, inherited at process start
    and living in memory, and the network, which was denied only by refusing
    to import `socket`.
    """

    def test_service_secrets_are_not_in_the_environment(self, staged, monkeypatch):
        """Reproduced with a real DSN shape: the deployment passes secrets in
        the environment, and the confined child was reading them."""
        monkeypatch.setenv("DATABASE_URL", "postgres://u:SENTINEL-SECRET@db/l")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-SENTINEL-KEY")
        result = _run(
            "import os\n"
            "print('DSN=%s' % os.environ.get('DATABASE_URL'))\n"
            "print('KEY=%s' % os.environ.get('OPENAI_API_KEY'))\n"
            "print('ALL=%s' % sorted(os.environ))\n",
            staged,
        )
        assert result["ok"], result["stderr"]
        assert "SENTINEL-SECRET" not in result["stdout"], result["stdout"]
        assert "SENTINEL-KEY" not in result["stdout"], result["stdout"]

    def test_the_network_is_unreachable_below_the_import_denial(self, staged):
        """`_BlockedImportFinder` refuses `socket`; it did not refuse
        `_socket`, and an import denylist cannot be the wall anyway — an
        already-loaded module can hand out the same primitive. Proven against
        a listener in this process, so it needs no internet."""
        import socket as _sock
        import threading

        listener = _sock.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        reached = threading.Event()

        def _accept():
            try:
                conn, _ = listener.accept()
                reached.set()
                conn.close()
            except OSError:
                pass

        threading.Thread(target=_accept, daemon=True).start()
        try:
            result = _run(
                "import _socket\n"
                "s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)\n"
                "s.settimeout(3)\n"
                "try:\n"
                f"    s.connect(('127.0.0.1', {port}))\n"
                "    print('CONNECTED')\n"
                "except OSError as exc:\n"
                "    print('blocked:%s' % exc.errno)\n",
                staged,
            )
            assert result["ok"], result["stderr"]
            assert "CONNECTED" not in result["stdout"], result["stdout"]
            assert not reached.wait(1.0), "the confined child reached this process"
        finally:
            listener.close()


@requires_backend
class TestStagedInputsAreReadOnly:
    """SPEC §18 says inputs are read-only and the workdir is read-write.

    Both were read-write. The originals are safe — the workdir holds copies —
    but run 1 of a session could rewrite `input.csv` and run 2 would read the
    forgery as the user's attachment, which is the provenance the contract
    exists to state.
    """

    def test_a_staged_input_cannot_be_rewritten(self, staged):
        result = _run(
            "try:\n"
            "    open('input.csv', 'w').write('forged')\n"
            "    print('REWROTE')\n"
            "except OSError as exc:\n"
            "    print('blocked:%s' % exc.errno)\n",
            staged,
        )
        assert result["ok"], result["stderr"]
        assert "REWROTE" not in result["stdout"], result["stdout"]
        assert staged.joinpath("input.csv").read_text() == "a,b\n1,2\n"

    def test_it_cannot_be_appended_to_or_unlinked_either(self, staged):
        result = _run(
            "import os\n"
            "for label, fn in (('append', lambda: open('input.csv','a').write('x')),\n"
            "                  ('unlink', lambda: os.unlink('input.csv')),\n"
            "                  ('rename', lambda: os.rename('input.csv','other.csv'))):\n"
            "    try:\n"
            "        fn(); print('%s:SUCCEEDED' % label)\n"
            "    except OSError as exc:\n"
            "        print('%s:blocked' % label)\n",
            staged,
        )
        assert result["ok"], result["stderr"]
        assert "SUCCEEDED" not in result["stdout"], result["stdout"]
        assert staged.joinpath("input.csv").exists()

    def test_new_files_are_still_writable(self, staged):
        """The read-only inputs must not make the workdir read-only, or the
        tool cannot produce anything."""
        result = _run("open('out.txt','w').write('computed')", staged)
        assert result["ok"], result["stderr"]
        assert [f["name"] for f in result["created_files"]] == ["out.txt"]


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


class TestTheNativeWarmupListIsReal:
    """`pledge` without `prot_exec` denies new executable mappings, which is
    what `dlopen` needs — so a C extension not already resident when the
    promise drops can never be imported. The child is spawned, so it starts
    with none of them. They are loaded up front instead of granting the
    promise.

    The loader swallows `ImportError` (a build may genuinely lack `lzma`), so
    a misspelled name would warm nothing and say nothing, and the failure
    would appear only on OpenBSD as model code being killed mid-import. These
    run on whatever platform the suite runs on, which is the point.
    """

    def test_every_name_resolves(self):
        import importlib

        unresolved = []
        for name in confine._NATIVE_WARMUP:
            try:
                importlib.import_module(name)
            except ImportError:
                unresolved.append(name)
        assert unresolved == [], f"not importable on this build: {unresolved}"

    def test_it_covers_what_the_interpreter_advertises(self):
        """The module docstring promises parsing and unzipping attachments.
        `zipfile` is pure Python and reaches `zlib` for anything compressed,
        so the extension, not the wrapper, is the name that must be listed."""
        import sys

        confine._warm_native_modules()
        for extension in ("zlib", "bz2", "lzma", "_csv", "_json"):
            assert extension in sys.modules, extension


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
