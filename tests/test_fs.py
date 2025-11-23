import os
from pathlib import Path

import pytest

from liminallm.service.fs import PathTraversalError, safe_join


def test_safe_join_accepts_child_path(tmp_path: Path):
    base = tmp_path
    result = safe_join(base, "nested/file.txt")

    assert base in result.parents


def test_safe_join_rejects_traversal(tmp_path: Path):
    base = tmp_path

    with pytest.raises(PathTraversalError):
        safe_join(base, os.path.join("..", "escape.txt"))


def test_safe_join_rejects_absolute(tmp_path: Path):
    base = tmp_path

    with pytest.raises(PathTraversalError):
        safe_join(base, str(Path("/tmp/absolute.txt")))
