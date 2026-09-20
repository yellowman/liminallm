"""A file the code wrote and the server refused is said out loud.

`publish_artifacts` returns exactly the names it wrote, skipping a created
file whose extension is outside the upload policy or whose size is over the
artifact ceiling. `run_python` checked that list for truthiness only, and
never against what the code had created - so a run that produced four files
and saved one reported the one, and the three that vanished appeared
nowhere: not in the tool result, not in a log, not in the model's view.

The model is not told the policy either. Its tool description is "Run
Python 3 in a sandbox whose working directory holds the attached files" and
says nothing about allowed types or sizes. So model-written code that prints
"saved report.xlsx" produces a turn asserting a file exists that was never
written, and the tool result contains nothing that would let the model
correct itself.

Counted rather than compared by name, and this is the part a name-based
check gets wrong: `_link_unused` renames on collision, so a file created as
`out.csv` is published as `out (2).csv` when that name is taken. Comparing
names would report a saved file as dropped. Dotfiles are skipped on purpose
and are left out of the count for the same reason.

The count also has to survive the interpreter result's MAX_ARTIFACTS bound.
Otherwise the first ten names look completely saved while an eleventh file
vanishes before the caller can compare created with published.
"""

from __future__ import annotations

import copy
import uuid

import pytest

from liminallm.service import agent_tools
from liminallm.service.interpreter import MAX_ARTIFACTS

pytestmark = pytest.mark.slow


def _settings(tmp_path):
    """The real settings object with two paths redirected.

    A namespace built from what this test believes `run_python` reads would
    encode that belief: it already missed `interpreter_scratch_dir`, and the
    next field added would break here rather than in the code under test.
    """
    from liminallm.config import get_settings

    settings = copy.copy(get_settings())
    settings.shared_fs_root = str(tmp_path / "fs")
    settings.interpreter_scratch_dir = str(tmp_path / "scratch")
    return settings


def _run(code: str, tmp_path):
    """The real tool, against a real sandbox and a real file area."""
    session: dict = {}
    return (
        agent_tools.run_python(
            code,
            [],
            settings=_settings(tmp_path),
            user_id=f"u{uuid.uuid4().hex[:8]}",
            session=session,
        ),
        session,
    )


class TestAFileTheServerRefused:
    def test_the_result_warns_that_files_were_not_saved(self, tmp_path):
        out, _session = _run(
            "open('summary.csv','w').write('a,b\\n1,2\\n')\n"
            "open('report.xlsx','w').write('x')\n"
            "open('chart.svg','w').write('<svg/>')\n"
            "print('made 3 files')\n",
            tmp_path,
        )

        if "the code interpreter is unavailable" in out:
            pytest.skip("no sandbox confinement backend on this host")

        assert "summary.csv" in out, f"the fixture saved nothing: {out}"
        assert "WARNING" in out, (
            "two files were refused and the result said nothing, so the model "
            f"cannot tell the user: {out}"
        )
        assert "2 of 3" in out, out

    def test_a_run_whose_files_were_all_saved_does_not_warn(self, tmp_path):
        """The control. A warning on every run would satisfy the test above
        and teach the model to doubt every file it writes."""
        out, _session = _run(
            "open('summary.csv','w').write('a,b\\n1,2\\n')\nprint('done')\n",
            tmp_path,
        )

        if "the code interpreter is unavailable" in out:
            pytest.skip("no sandbox confinement backend on this host")

        assert "summary.csv" in out, out
        assert "WARNING" not in out, (
            f"a run that saved everything it created still warned: {out}"
        )

    def test_the_publish_count_limit_is_reported(self, tmp_path):
        code = "".join(
            f"open('file-{i:02d}.csv','w').write('x')\n"
            for i in range(MAX_ARTIFACTS + 1)
        ) + "print('done')\n"
        out, _session = _run(code, tmp_path)

        if "the code interpreter is unavailable" in out:
            pytest.skip("no sandbox confinement backend on this host")

        assert "WARNING" in out, (
            "a file beyond MAX_ARTIFACTS disappeared before the result could "
            f"report it: {out}"
        )
        assert f"1 of {MAX_ARTIFACTS + 1}" in out, out
        assert "artifact count limit" in out, out

    def test_a_dotfile_is_not_counted_as_dropped(self, tmp_path):
        """Dotfiles are refused deliberately and are not a loss to report.
        Counting them would warn on every run that wrote a cache file."""
        out, _session = _run(
            "open('summary.csv','w').write('a,b\\n1,2\\n')\n"
            "open('.cache','w').write('x')\n"
            "print('done')\n",
            tmp_path,
        )

        if "the code interpreter is unavailable" in out:
            pytest.skip("no sandbox confinement backend on this host")

        assert "summary.csv" in out, out
        assert "WARNING" not in out, (
            f"a skipped dotfile was reported to the model as a loss: {out}"
        )
