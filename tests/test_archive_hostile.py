"""Hostile archive members, judged by what ends up on disk.

§21.3 is four sentences and every clause is a property: un-archiving is
streamed and budgeted, never trusting headers; entry count, per-member size,
total size and compression ratio are enforced **as bytes are read**; every
member path is sanitized component-wise and re-joined through `safe_join`;
member type is checked with `stat.S_IFMT` because many writers store
permissions with no type bits.

The tests use real ZIP and TAR fixtures and assert on the filesystem
afterwards, not on the returned `skipped` list. A skip reason is the
extractor's opinion of what it did; the tree is what it actually did, and the
only one of those a zip-slip cares about.
"""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from liminallm.service.archive import (
    DEFAULT_MAX_RATIO,
    ArchiveExtractionError,
    extract_archive,
)

TEXT = {"allowed_extensions": [".txt", ".csv", ".md"]}


@pytest.fixture
def area(tmp_path):
    """An extraction root with a sibling nobody should be able to touch."""
    root = tmp_path / "area"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("original\n")
    return root, outside


def _zip(path: Path, members: list[tuple[str, bytes]], *, mode: int | None = None):
    """A zip built from explicit `ZipInfo`s, so member mode bits can be set.

    `compress_type` is set on each entry rather than on the `ZipFile`: passing
    a `ZipInfo` to `writestr` takes the compression from the *info*, which
    defaults to STORED — so the archive would be as large as its contents and a
    compression-ratio fixture would silently test nothing.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in members:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, body)
    return path


def _files_under(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()
    }


def _extract(src: Path, dest: Path, limits=None):
    return extract_archive(str(src), str(dest), limits or dict(TEXT))


def _extract_hostile(src: Path, dest: Path, limits=None) -> None:
    """Extract, tolerating the refusal.

    An archive whose every entry is skipped raises "nothing extractable",
    which is a correct answer and not the one under test. These tests are
    about the tree afterwards, so the error path is allowed and the
    filesystem is what gets asserted.
    """
    try:
        _extract(src, dest, limits)
    except ArchiveExtractionError:
        pass


# ---------------------------------------------------------------------------
# paths: nothing lands outside the extraction root


class TestNothingEscapesTheRoot:
    @pytest.mark.parametrize(
        "member",
        [
            "../victim.txt",
            "../../victim.txt",
            "a/../../victim.txt",
            "/etc/victim.txt",
            "//server/share/victim.txt",
            "C:\\Windows\\victim.txt",
            "..\\..\\victim.txt",
            "a\\..\\..\\victim.txt",
            "....//victim.txt",
            "a/./../../victim.txt",
        ],
    )
    def test_a_traversing_member_writes_nothing_outside(self, area, member, tmp_path):
        root, outside = area
        src = _zip(tmp_path / "t.zip", [(member, b"pwned\n")])
        dest = root / "out"
        _extract_hostile(src, dest)
        assert (outside / "victim.txt").read_text() == "original\n", member
        for landed in _files_under(dest):
            assert ".." not in landed, (member, landed)

    def test_an_over_deep_member_is_refused(self, area, tmp_path):
        root, _outside = area
        deep = "/".join(f"d{i}" for i in range(40)) + "/x.txt"
        src = _zip(tmp_path / "deep.zip", [(deep, b"x\n")])
        dest = root / "out"
        _extract_hostile(src, dest)
        assert _files_under(dest) == set()

    def test_a_tar_traversal_writes_nothing_outside(self, area, tmp_path):
        root, outside = area
        src = tmp_path / "t.tar"
        with tarfile.open(src, "w") as tf:
            info = tarfile.TarInfo("../victim.txt")
            body = b"pwned\n"
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
        dest = root / "out"
        _extract_hostile(src, dest)
        assert (outside / "victim.txt").read_text() == "original\n"

    def test_a_legitimate_nested_path_still_extracts(self, area, tmp_path):
        """The refusals must come from the rule, not from a broken extractor."""
        root, _outside = area
        src = _zip(tmp_path / "ok.zip", [("reports/2024/q1.txt", b"figures\n")])
        dest = root / "out"
        result = _extract(src, dest)
        assert _files_under(dest) == {"reports/2024/q1.txt"}
        assert result["extracted"] == ["reports/2024/q1.txt"]


# ---------------------------------------------------------------------------
# types: only regular files are ever materialized


class TestOnlyRegularFilesAreMaterialized:
    def _tar_with(self, path: Path, build) -> Path:
        with tarfile.open(path, "w") as tf:
            build(tf)
        return path

    def test_a_tar_symlink_is_never_created(self, area, tmp_path):
        root, outside = area

        def build(tf):
            info = tarfile.TarInfo("link.txt")
            info.type = tarfile.SYMTYPE
            info.linkname = str(outside / "victim.txt")
            tf.addfile(info)

        src = self._tar_with(tmp_path / "sym.tar", build)
        dest = root / "out"
        _extract_hostile(src, dest)
        assert _files_under(dest) == set()
        assert not (dest / "link.txt").is_symlink()

    def test_a_tar_hardlink_is_never_created(self, area, tmp_path):
        root, outside = area

        def build(tf):
            info = tarfile.TarInfo("hard.txt")
            info.type = tarfile.LNKTYPE
            info.linkname = str(outside / "victim.txt")
            tf.addfile(info)

        src = self._tar_with(tmp_path / "hard.tar", build)
        dest = root / "out"
        _extract_hostile(src, dest)
        assert _files_under(dest) == set()

    @pytest.mark.parametrize(
        "kind", [tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE]
    )
    def test_devices_and_fifos_are_never_created(self, area, tmp_path, kind):
        root, _outside = area

        def build(tf):
            info = tarfile.TarInfo("node.txt")
            info.type = kind
            tf.addfile(info)

        src = self._tar_with(tmp_path / f"node{kind}.tar", build)
        dest = root / "out"
        _extract_hostile(src, dest)
        assert _files_under(dest) == set()

    def test_a_zip_entry_carrying_a_symlink_type_is_skipped(self, area, tmp_path):
        root, outside = area
        src = _zip(
            tmp_path / "sym.zip",
            [("link.txt", str(outside / "victim.txt").encode())],
            mode=stat.S_IFLNK | 0o777,
        )
        dest = root / "out"
        _extract_hostile(src, dest)
        assert _files_under(dest) == set()

    def test_a_zip_entry_with_permission_bits_but_no_type_bits_extracts(
        self, area, tmp_path
    ):
        """§21.3 names this case: many writers store permissions with no type
        bits, and treating "no type bits" as "not a regular file" would refuse
        ordinary archives."""
        root, _outside = area
        src = _zip(tmp_path / "perm.zip", [("plain.txt", b"ok\n")], mode=0o644)
        dest = root / "out"
        _extract(src, dest)
        assert _files_under(dest) == {"plain.txt"}


# ---------------------------------------------------------------------------
# resources: budgets are enforced on bytes read, and failure leaves nothing


class TestBudgetsAreEnforcedOnBytesRead:
    def test_the_compression_ratio_cap_means_what_it_says(self, area, tmp_path):
        """The configured cap is 100:1. A 1 KiB archive expanding to ~600 KiB
        is roughly 850:1, and it used to extract: the ratio check only fired
        past a 1 MiB floor, so the cap was not a cap for small archives.

        The floor's stated reason was that "an empty-file tar is mostly header"
        and so expands past the ratio — which is backwards. Measured: an
        empty-file tar is 10240 bytes on disk and expands to 0.
        """
        root, _outside = area
        src = _zip(tmp_path / "bomb.zip", [("bomb.txt", b"\0" * (600 * 1024))])
        archive_bytes = src.stat().st_size
        assert archive_bytes * DEFAULT_MAX_RATIO < 600 * 1024, archive_bytes

        dest = root / "out"
        with pytest.raises(ArchiveExtractionError, match="ratio"):
            _extract(src, dest)
        assert not dest.exists(), "a refused bomb left its bytes behind"

    def test_a_compressible_file_within_the_ratio_still_extracts(
        self, area, tmp_path
    ):
        root, _outside = area
        src = _zip(tmp_path / "ok.zip", [("log.txt", b"repeat\n" * 200)])
        dest = root / "out"
        _extract(src, dest)
        assert _files_under(dest) == {"log.txt"}

    def test_the_entry_count_cap_is_enforced(self, area, tmp_path):
        root, _outside = area
        src = _zip(
            tmp_path / "many.zip", [(f"f{i}.txt", b"x") for i in range(20)]
        )
        dest = root / "out"
        with pytest.raises(ArchiveExtractionError, match="entries"):
            _extract(src, dest, {**TEXT, "max_entries": 5})
        assert not dest.exists()

    def test_one_oversized_member_is_refused(self, area, tmp_path):
        root, _outside = area
        src = _zip(tmp_path / "big.zip", [("big.txt", b"A" * 50_000)])
        dest = root / "out"
        with pytest.raises(ArchiveExtractionError):
            _extract(
                src,
                dest,
                {**TEXT, "max_member_bytes": 1000, "max_ratio": 10_000},
            )
        assert not dest.exists()

    def test_the_aggregate_cap_is_enforced_across_members(self, area, tmp_path):
        """No single member is over its own cap; together they are."""
        root, _outside = area
        src = _zip(
            tmp_path / "sum.zip",
            [(f"f{i}.txt", b"A" * 4000) for i in range(10)],
        )
        dest = root / "out"
        with pytest.raises(ArchiveExtractionError):
            _extract(
                src,
                dest,
                {
                    **TEXT,
                    "max_member_bytes": 8000,
                    "max_total_bytes": 10_000,
                    "max_ratio": 10_000,
                },
            )
        assert not dest.exists()

    def test_a_truncated_archive_leaves_nothing(self, area, tmp_path):
        root, _outside = area
        src = _zip(tmp_path / "trunc.zip", [("a.txt", b"hello world\n" * 100)])
        raw = src.read_bytes()
        src.write_bytes(raw[: len(raw) // 2])
        dest = root / "out"
        with pytest.raises(ArchiveExtractionError):
            _extract(src, dest)
        assert not dest.exists()

    def test_a_corrupt_archive_leaves_nothing(self, area, tmp_path):
        root, _outside = area
        src = tmp_path / "junk.zip"
        src.write_bytes(b"not an archive at all")
        dest = root / "out"
        with pytest.raises(ArchiveExtractionError):
            _extract(src, dest)
        assert not dest.exists()


class TestNestedArchivesStayOpaque:
    def test_an_inner_archive_is_written_but_not_expanded(self, area, tmp_path):
        """§21.3's recursion answer: each explicit extraction gets its own
        budget, so an archive inside an archive is just a file."""
        root, _outside = area
        inner = _zip(tmp_path / "inner.zip", [("secret.txt", b"deep\n")])
        src = _zip(tmp_path / "outer.zip", [("inner.zip", inner.read_bytes())])
        dest = root / "out"
        _extract(src, dest, {**TEXT, "allowed_extensions": [".txt", ".zip"]})
        assert _files_under(dest) == {"inner.zip"}
        assert "secret.txt" not in _files_under(dest)


class TestTheDestinationIsAllOrNothing:
    def test_a_later_budget_failure_removes_earlier_members(self, area, tmp_path):
        """Half an extraction is the failure mode that matters: the caller is
        told it failed while the files are already in the user's area."""
        root, _outside = area
        src = _zip(
            tmp_path / "partial.zip",
            [("first.txt", b"A" * 3000), ("second.txt", b"B" * 30_000)],
        )
        dest = root / "out"
        with pytest.raises(ArchiveExtractionError):
            _extract(
                src,
                dest,
                {**TEXT, "max_total_bytes": 10_000, "max_ratio": 10_000},
            )
        assert not dest.exists(), sorted(_files_under(dest))

    def test_a_policy_skip_is_not_all_or_nothing(self, area, tmp_path):
        """A disallowed extension is a per-entry policy skip, and the rest of
        the archive still arrives — that distinction is the module's own."""
        root, _outside = area
        src = _zip(
            tmp_path / "mixed.zip",
            [("keep.txt", b"kept\n"), ("drop.exe", b"MZ")],
        )
        dest = root / "out"
        result = _extract(src, dest)
        assert _files_under(dest) == {"keep.txt"}
        assert [s["name"] for s in result["skipped"]] == ["drop.exe"]
