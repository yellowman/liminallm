"""What names one publication namespace, and what merely resembles one.

`publication_key` is the single answer to "which lock does a mutation of this
path take", and every side of a publication asks it: the upload route, the
delete route, and the re-index queue. If two of them disagree about the key
for one file, they take different locks and never see each other - which is
not hypothetical, it is the defect this function was extracted to fix.

Two ways to get the answer wrong, and they pull in opposite directions.

Too little canonicalisation and the answer depends on spelling: `fs_root` may
be a symlink, and `safe_join` resolves the paths it hands back, so a route
holding the logical root and a job row holding the resolved one describe the
same file with two names. Too much and the answer depends on *content*: an
extracted tree may contain `users/x/files/`, and anything that looks for the
nearest directory shaped like the layout finds the archive's copy.

So the rule has three parts, and this file pins all three:

* the root may be recognised through a symlink - resolve it to *match*;
* the key is built from the **logical** root - never the resolved spelling;
* the user's files directory is at a fixed depth below that root - never the
  nearest thing shaped like one.
"""

from __future__ import annotations

from pathlib import Path

from liminallm.service.fs import namespace_key, publication_key


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    """A physical root and a symlink to it, which is an ordinary deployment."""
    physical = tmp_path / "physical"
    (physical / "users" / "U" / "files").mkdir(parents=True)
    logical = tmp_path / "link"
    logical.symlink_to(physical)
    return physical, logical


def test_a_symlinked_root_and_a_lookalike_tree_still_name_one_namespace(
    tmp_path: Path,
):
    """Both rules at once, because both are about the same identity.

    The path is what a store row actually holds: `safe_join` resolved it, so
    it is spelled with the physical root. The tree inside it mirrors the
    layout, so the nearest `users/*/files` is the archive's. The key has to
    come out as the logical root's `bundle` either way - that is the name the
    route locks when it deletes the tree.
    """
    physical, logical = _layout(tmp_path)
    canonical_target = (
        physical / "users" / "U" / "files"
        / "bundle" / "users" / "fake" / "files" / "inner.md"
    )
    expected = namespace_key(
        logical / "users" / "U" / "files", "bundle/users/fake/files/inner.md"
    )

    assert publication_key(logical, canonical_target) == expected, (
        "the resolved path was not recognised as being under the configured "
        "root, so the job locks a name no route ever takes"
    )
    assert str(logical) in publication_key(logical, canonical_target), (
        "the key is spelled with the physical root, so it does not match the "
        "one the route builds from the configured root"
    )


def test_the_logical_spelling_and_the_resolved_one_agree(tmp_path: Path):
    """The same file reached two ways is one namespace, not two."""
    physical, logical = _layout(tmp_path)
    through_link = logical / "users" / "U" / "files" / "bundle" / "inner.md"
    through_real = physical / "users" / "U" / "files" / "bundle" / "inner.md"

    assert publication_key(logical, through_link) == publication_key(
        logical, through_real
    )


def test_a_path_outside_the_root_is_keyed_on_itself(tmp_path: Path):
    """An adapter or a shared object belongs to no user's tree."""
    _physical, logical = _layout(tmp_path)
    stray = tmp_path / "elsewhere" / "adapter.bin"
    assert publication_key(logical, stray) == str(stray)


def test_the_files_directory_is_at_a_fixed_depth(tmp_path: Path):
    """`users/<id>/files/<name>` and nothing shallower."""
    _physical, logical = _layout(tmp_path)
    # No name under `files`, so there is no publication to key.
    bare = logical / "users" / "U" / "files"
    assert publication_key(logical, bare) == str(bare)
