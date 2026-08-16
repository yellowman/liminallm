"""Safe archive extraction (SPEC §18 hardening).

Un-archives user uploads (.zip, .tar, .tar.gz, .tgz, single-file .gz) with
defense-in-depth against hostile archives:

- Entry-count, per-member, and total-size budgets are enforced while
  *streaming* the decompressed bytes — header sizes are never trusted, so a
  zip bomb is stopped at the byte limit no matter what its metadata claims.
- A compression-ratio cap compares bytes written against the on-disk archive
  size (again, not the headers).
- Member paths are sanitized component-wise: absolute paths, drive letters,
  ``..`` segments, over-deep trees, and hidden files are rejected, and the
  final path is re-checked with ``safe_join`` (zip-slip defense).
- Symlinks, hardlinks, devices, and FIFOs are skipped, never materialized.
- Nested archives are extracted as opaque files, never expanded, so there is
  no recursive blow-up; each explicit extraction gets its own budget.
- ``extract_archive_sandboxed`` runs the whole job in a resource-limited
  child process (memory/CPU/file-size rlimits plus a wall-clock kill).

On any budget violation the destination directory is removed — extraction is
all-or-nothing for resource failures, while per-entry policy skips (bad name,
disallowed extension, symlink) are reported in the result.
"""
from __future__ import annotations

import gzip
import re
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Optional

from liminallm.service.fs import PathTraversalError, safe_join


class ArchiveExtractionError(ValueError):
    """Raised when an archive violates extraction limits or is unreadable."""


# Suffixes recognized as extractable archives (longest first for stem logic).
TAR_SUFFIXES = (".tar.gz", ".tgz", ".tar")
ARCHIVE_SUFFIXES = (".zip",) + TAR_SUFFIXES + (".gz",)

DEFAULT_MAX_ENTRIES = 1_000
DEFAULT_MAX_MEMBER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_RATIO = 100
MAX_PATH_DEPTH = 16
_CHUNK = 64 * 1024


def is_archive_filename(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def archive_stem(name: str) -> str:
    """Archive filename without its archive suffix ("data.tar.gz" -> "data")."""
    lowered = name.lower()
    for suffix in (".zip",) + TAR_SUFFIXES + (".gz",):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _sanitize_member_path(name: str) -> Optional[str]:
    """Normalize an archive member name to a safe relative path, or None.

    Applies the same character policy as the upload endpoint to every path
    component; a None return means the entry must be skipped.
    """
    name = name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    parts: list[str] = []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return None
        cleaned = re.sub(r"[^\w\-. ]", "_", part).lstrip(".")[:255]
        if not cleaned:
            return None
        parts.append(cleaned)
    if not parts or len(parts) > MAX_PATH_DEPTH:
        return None
    return "/".join(parts)


class _Budget:
    """Streaming byte/entry accounting; raises the moment a cap is crossed."""

    def __init__(self, archive_bytes: int, limits: dict[str, Any]):
        self.archive_bytes = max(1, archive_bytes)
        self.max_entries = int(limits.get("max_entries", DEFAULT_MAX_ENTRIES))
        self.max_member = int(limits.get("max_member_bytes", DEFAULT_MAX_MEMBER_BYTES))
        self.max_total = int(limits.get("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES))
        self.max_ratio = int(limits.get("max_ratio", DEFAULT_MAX_RATIO))
        self.entries = 0
        self.total = 0

    def charge_entry(self) -> None:
        self.entries += 1
        if self.entries > self.max_entries:
            raise ArchiveExtractionError(
                f"archive has too many entries (max {self.max_entries})"
            )

    def charge_bytes(self, n: int) -> None:
        self.total += n
        if self.total > self.max_total:
            raise ArchiveExtractionError(
                f"archive expands past the total size limit "
                f"({self.max_total // (1024 * 1024)}MB)"
            )
        # No small-archive exemption. This used to be
        # `max(1 MiB, archive_bytes * max_ratio)`, which meant the configured
        # 100:1 cap was not a cap below a megabyte: measured, a 726-byte zip
        # expanded to 600 KiB — about 846:1 — and extracted. §21.3 states the
        # ratio cap with no floor in it.
        #
        # The floor's own justification was backwards. It read "an empty-file
        # tar is mostly header" and so expands past the ratio; measured, an
        # empty-file tar is 10240 bytes on disk and expands to 0 — a ratio of
        # zero, which no cap could trouble.
        if self.total > self.archive_bytes * self.max_ratio:
            raise ArchiveExtractionError(
                f"compression ratio exceeds {self.max_ratio}:1 — refusing to "
                "extract (archive bomb suspected)"
            )


def _write_member(
    src: BinaryIO, target: Path, budget: _Budget
) -> int:
    """Stream one member to disk under budget; returns bytes written."""
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > budget.max_member:
                raise ArchiveExtractionError(
                    f"an entry expands past the per-file limit "
                    f"({budget.max_member // (1024 * 1024)}MB)"
                )
            budget.charge_bytes(len(chunk))
            out.write(chunk)
    target.chmod(0o640)
    return written


def _resolve_target(
    dest: Path, member_name: str, allowed_extensions: Optional[set[str]]
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Map a member name to (target path, relative name, skip reason)."""
    rel = _sanitize_member_path(member_name)
    if rel is None:
        return None, None, "unsafe path"
    if allowed_extensions is not None:
        ext = Path(rel).suffix.lower()
        if ext not in allowed_extensions:
            return None, rel, f"extension {ext or 'none'} not allowed"
    try:
        target = safe_join(dest, rel)
    except PathTraversalError:
        return None, rel, "unsafe path"
    return target, rel, None


def _extract_zip(
    archive: Path, dest: Path, budget: _Budget, allowed: Optional[set[str]]
) -> tuple[list[str], list[dict[str, str]]]:
    extracted: list[str] = []
    skipped: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            budget.charge_entry()
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise ArchiveExtractionError("encrypted zip entries are not supported")
            # Only entries that explicitly carry a non-regular file type
            # (symlink, device, ...) are rejected; many zip writers store
            # permission bits with no type bits at all.
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) and not stat.S_ISREG(mode):
                skipped.append({"name": info.filename, "reason": "not a regular file"})
                continue
            target, rel, reason = _resolve_target(dest, info.filename, allowed)
            if target is None:
                skipped.append({"name": info.filename, "reason": reason or "skipped"})
                continue
            with zf.open(info) as src:
                _write_member(src, target, budget)
            extracted.append(rel)
    return extracted, skipped


def _extract_tar(
    archive: Path, dest: Path, budget: _Budget, allowed: Optional[set[str]]
) -> tuple[list[str], list[dict[str, str]]]:
    extracted: list[str] = []
    skipped: list[dict[str, str]] = []
    with tarfile.open(archive, mode="r:*") as tf:
        for member in tf:
            budget.charge_entry()
            if member.isdir():
                continue
            if not member.isreg():
                skipped.append(
                    {"name": member.name, "reason": "not a regular file"}
                )
                continue
            target, rel, reason = _resolve_target(dest, member.name, allowed)
            if target is None:
                skipped.append({"name": member.name, "reason": reason or "skipped"})
                continue
            src = tf.extractfile(member)
            if src is None:
                skipped.append({"name": member.name, "reason": "unreadable entry"})
                continue
            with src:
                _write_member(src, target, budget)
            extracted.append(rel)
    return extracted, skipped


def _extract_gzip(
    archive: Path, dest: Path, budget: _Budget, allowed: Optional[set[str]]
) -> tuple[list[str], list[dict[str, str]]]:
    """Single-file gzip (data.csv.gz -> data.csv)."""
    budget.charge_entry()
    target, rel, reason = _resolve_target(dest, archive_stem(archive.name), allowed)
    if target is None:
        raise ArchiveExtractionError(reason or "gzip target name not allowed")
    with gzip.open(archive, "rb") as src:
        _write_member(src, target, budget)
    return [rel], []


def extract_archive(
    archive_path: str, dest_dir: str, limits: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Extract an archive into ``dest_dir`` under strict budgets.

    Module-level and picklable-args by design so it can run through
    ``run_in_sandbox``. Removes ``dest_dir`` and raises
    ``ArchiveExtractionError`` on any budget violation or unreadable input.
    """
    archive = Path(archive_path)
    dest = Path(dest_dir)
    limits = limits or {}
    allowed_raw = limits.get("allowed_extensions")
    allowed = {e.lower() for e in allowed_raw} if allowed_raw is not None else None
    budget = _Budget(archive.stat().st_size, limits)

    lowered = archive.name.lower()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if lowered.endswith(".zip"):
            extracted, skipped = _extract_zip(archive, dest, budget, allowed)
        elif lowered.endswith(TAR_SUFFIXES):
            extracted, skipped = _extract_tar(archive, dest, budget, allowed)
        elif lowered.endswith(".gz"):
            extracted, skipped = _extract_gzip(archive, dest, budget, allowed)
        else:
            raise ArchiveExtractionError("unsupported archive type")
    except (zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, EOFError, OSError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise ArchiveExtractionError(f"unreadable or corrupt archive: {exc}") from exc
    except ArchiveExtractionError:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    if not extracted:
        shutil.rmtree(dest, ignore_errors=True)
        raise ArchiveExtractionError(
            "nothing extractable in archive"
            + (f" (skipped {len(skipped)} entries)" if skipped else "")
        )
    return {
        "extracted": extracted,
        "skipped": skipped,
        "total_bytes": budget.total,
        "entries": budget.entries,
    }


def extract_archive_sandboxed(
    archive_path: str,
    dest_dir: str,
    limits: Optional[dict[str, Any]] = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Run ``extract_archive`` in a resource-limited child process.

    The rlimits (memory, CPU, per-file size) back up the streaming budgets:
    even a decompressor bug cannot take down the API process.
    """
    from liminallm.service.sandbox import SandboxConfig, SandboxError, run_in_sandbox

    member_bytes = int(
        (limits or {}).get("max_member_bytes", DEFAULT_MAX_MEMBER_BYTES)
    )
    config = SandboxConfig(
        max_memory_mb=512,
        max_cpu_seconds=30,
        # rlimit backstop just above the streaming per-member cap
        max_file_size_mb=member_bytes // (1024 * 1024) + 8,
        scratch_dir=Path(dest_dir),
    )
    try:
        return run_in_sandbox(
            extract_archive,
            archive_path,
            dest_dir,
            limits,
            config=config,
            timeout=timeout,
        )
    except SandboxError:
        # The child may have died mid-write; never leave partial output.
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
