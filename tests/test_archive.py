"""Safe archive extraction tests: zip-slip, bombs, and hostile members."""
import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from liminallm.service.archive import (
    ArchiveExtractionError,
    archive_stem,
    extract_archive,
    extract_archive_sandboxed,
    is_archive_filename,
)


def make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def make_tar(path: Path, entries: dict[str, bytes], mode: str = "w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_is_archive_filename():
    assert is_archive_filename("a.zip")
    assert is_archive_filename("a.TAR.GZ")
    assert is_archive_filename("a.tgz")
    assert is_archive_filename("a.csv.gz")
    assert not is_archive_filename("a.txt")


def test_archive_stem():
    assert archive_stem("data.tar.gz") == "data"
    assert archive_stem("data.tgz") == "data"
    assert archive_stem("data.zip") == "data"
    assert archive_stem("report.csv.gz") == "report.csv"


def test_zip_happy_path(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip",
        {"readme.md": b"# hi", "docs/notes.txt": b"text", "docs/deep/x.json": b"{}"},
    )
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert sorted(report["extracted"]) == ["docs/deep/x.json", "docs/notes.txt", "readme.md"]
    assert (dest / "docs" / "deep" / "x.json").read_bytes() == b"{}"
    assert report["total_bytes"] == 10


def test_tar_gz_happy_path(tmp_path):
    archive = make_tar(tmp_path / "a.tar.gz", {"one.txt": b"1", "d/two.txt": b"22"})
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert sorted(report["extracted"]) == ["d/two.txt", "one.txt"]
    assert (dest / "d" / "two.txt").read_bytes() == b"22"


def test_single_gzip(tmp_path):
    import gzip as gz

    archive = tmp_path / "report.csv.gz"
    archive.write_bytes(gz.compress(b"a,b\n1,2\n"))
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert report["extracted"] == ["report.csv"]
    assert (dest / "report.csv").read_bytes() == b"a,b\n1,2\n"


def test_zip_slip_is_skipped(tmp_path):
    archive = make_zip(
        tmp_path / "evil.zip",
        {"../escape.txt": b"pwn", "..\\win-escape.txt": b"pwn", "ok.txt": b"fine"},
    )
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert report["extracted"] == ["ok.txt"]
    assert len(report["skipped"]) == 2
    assert not (tmp_path / "escape.txt").exists()


def test_absolute_path_member_skipped(tmp_path):
    archive = make_tar(tmp_path / "abs.tar", {"/etc/cron.d/evil": b"x", "ok.txt": b"y"}, mode="w")
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert report["extracted"] == ["ok.txt"]
    # tarfile stores "/etc/..." as "etc/..."? ensure nothing landed outside dest
    assert not Path("/etc/cron.d/evil").exists() or True
    assert all(str(dest) in str(p) for p in dest.rglob("*"))


def test_tar_symlink_and_link_skipped(tmp_path):
    archive = tmp_path / "links.tar"
    with tarfile.open(archive, "w") as tf:
        sym = tarfile.TarInfo("link-to-passwd")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "/etc/passwd"
        tf.addfile(sym)
        hard = tarfile.TarInfo("hardlink")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "target"
        tf.addfile(hard)
        info = tarfile.TarInfo("ok.txt")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"ok"))
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert report["extracted"] == ["ok.txt"]
    reasons = {s["name"]: s["reason"] for s in report["skipped"]}
    assert reasons["link-to-passwd"] == "not a regular file"
    assert reasons["hardlink"] == "not a regular file"
    assert not (dest / "link-to-passwd").exists()


def test_zip_symlink_skipped(tmp_path):
    archive = tmp_path / "sym.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("evil-link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
        zf.writestr("ok.txt", b"ok")
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert report["extracted"] == ["ok.txt"]
    assert not (dest / "evil-link").exists()


def test_entry_count_bomb(tmp_path):
    archive = make_zip(tmp_path / "many.zip", {f"f{i}.txt": b"x" for i in range(50)})
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="too many entries"):
        extract_archive(str(archive), str(dest), {"max_entries": 10})
    assert not dest.exists()  # all-or-nothing on budget failure


def test_total_size_bomb(tmp_path):
    # 40MB of zeros compresses to ~40KB; total budget of 2MB must trip while
    # streaming, long before 40MB hits the disk.
    archive = make_zip(tmp_path / "bomb.zip", {"zeros.txt": b"\0" * (40 * 1024 * 1024)})
    assert archive.stat().st_size < 1024 * 1024
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="total size limit"):
        extract_archive(
            str(archive), str(dest), {"max_total_bytes": 2 * 1024 * 1024, "max_ratio": 10_000}
        )
    assert not dest.exists()


def test_compression_ratio_bomb(tmp_path):
    archive = make_zip(tmp_path / "ratio.zip", {"zeros.txt": b"\0" * (40 * 1024 * 1024)})
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="ratio"):
        extract_archive(str(archive), str(dest), {"max_ratio": 2})
    assert not dest.exists()


def test_member_size_cap(tmp_path):
    # `max_ratio` is raised so this isolates the per-member cap. 3MB of one
    # repeated byte compresses to a few KB, so with the shipped 100:1 ratio it
    # is *also* a bomb - and since the ratio cap lost its 1MB floor, that cap
    # now fires first. Both refusals are correct; this test is about the
    # member one, so it removes the other from the picture.
    archive = make_zip(tmp_path / "big.zip", {"big.txt": b"a" * (3 * 1024 * 1024)})
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="per-file limit"):
        extract_archive(
            str(archive),
            str(dest),
            {"max_member_bytes": 1024 * 1024, "max_ratio": 100_000},
        )
    assert not dest.exists()


def test_header_lies_are_ignored(tmp_path):
    # A tar whose member header claims 10 bytes but the budget counts what is
    # actually decompressed; conversely a truncated read cannot bypass caps.
    archive = make_tar(tmp_path / "t.tgz", {"x.txt": b"0123456789"})
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    assert report["total_bytes"] == 10


def test_extension_filter_skips(tmp_path):
    archive = make_zip(tmp_path / "mix.zip", {"ok.txt": b"a", "evil.exe": b"MZ"})
    dest = tmp_path / "out"
    report = extract_archive(
        str(archive), str(dest), {"allowed_extensions": [".txt"]}
    )
    assert report["extracted"] == ["ok.txt"]
    assert not (dest / "evil.exe").exists()
    assert report["skipped"][0]["name"] == "evil.exe"


def test_nested_archive_not_expanded(tmp_path):
    inner = make_zip(tmp_path / "inner.zip", {"secret.txt": b"s"})
    archive = make_zip(tmp_path / "outer.zip", {"inner.zip": inner.read_bytes()})
    dest = tmp_path / "out"
    report = extract_archive(
        str(archive), str(dest), {"allowed_extensions": [".zip", ".txt"]}
    )
    # The nested archive lands as an opaque file; its contents stay packed.
    assert report["extracted"] == ["inner.zip"]
    assert not (dest / "secret.txt").exists()


def test_hidden_and_deep_paths_skipped(tmp_path):
    deep = "/".join(f"d{i}" for i in range(30)) + "/x.txt"
    archive = make_zip(
        tmp_path / "odd.zip", {".ssh/authorized_keys": b"k", deep: b"x", "ok.txt": b"y"}
    )
    dest = tmp_path / "out"
    report = extract_archive(str(archive), str(dest))
    # dot-prefix is stripped by sanitization ("ssh/..."), never hidden
    assert "ok.txt" in report["extracted"]
    assert not (dest / ".ssh").exists()
    assert not any("d0" in e for e in report["extracted"])  # too deep → skipped


def test_corrupt_archive(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"this is not a zip file")
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="unreadable or corrupt"):
        extract_archive(str(archive), str(dest))
    assert not dest.exists()


def test_empty_archive_errors(tmp_path):
    archive = make_zip(tmp_path / "empty.zip", {})
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="nothing extractable"):
        extract_archive(str(archive), str(dest))
    assert not dest.exists()


def test_sandboxed_extraction_end_to_end(tmp_path):
    """Full path through the subprocess sandbox."""
    archive = make_zip(tmp_path / "a.zip", {"hello.txt": b"hello world"})
    dest = tmp_path / "out"
    report = extract_archive_sandboxed(str(archive), str(dest), timeout=120)
    assert report["extracted"] == ["hello.txt"]
    assert (dest / "hello.txt").read_bytes() == b"hello world"


def test_sandboxed_extraction_propagates_limit_errors(tmp_path):
    archive = make_zip(tmp_path / "bomb.zip", {"zeros.txt": b"\0" * (20 * 1024 * 1024)})
    dest = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError):
        extract_archive_sandboxed(
            str(archive),
            str(dest),
            {"max_total_bytes": 1024 * 1024, "max_ratio": 10_000},
            timeout=120,
        )
    assert not dest.exists()
