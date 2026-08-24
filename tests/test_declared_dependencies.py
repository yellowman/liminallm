"""Everything imported at module scope is a dependency somebody declared.

`httpx` was imported at module scope by five files in `liminallm/service` and
appeared in no dependency list. It worked for as long as it did because
`openai` depends on it, so every install happened to bring it along — a direct
import satisfied by somebody else's requirement. When CI resolved a set
without it, the application did not degrade: it failed to import at all, and
every test job died in the conftest before collecting a single test.

The rule this enforces is about *where* an import sits, not what it names.
A module-scope import is a hard requirement — the package cannot load without
it — so it has to be declared. A function-local one is this repository's idiom
for an optional capability (`numpy` inside the checkpoint loader, `tiktoken`
inside a `try:` that falls back to a heuristic count), and those are left
alone: their absence is a feature that turns off, not a service that will not
start.

`tests/` gets the same rule against a different list, because the failure
there is worse rather than milder: a missing import at module scope in a test
file is a *collection* error, and one collection error aborts the entire run
before a marker deselects anything. Two modules here imported `numpy` plainly
and took down the browser lane, whose install set is the narrowest in CI,
while 2694 tests it would have deselected never ran. So a test module may
import at module scope only what every lane installs, and reaches anything
else through `pytest.importorskip`.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys

import pytest

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10 — the floor this project supports
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Distribution name -> the name you import it by, where they differ. Only
#: the ones this project actually declares; a new entry belongs here when a
#: new dependency's two names disagree.
#:
#: Keyed by the bare distribution name, lowercased — `_DIST` below has already
#: stripped any extra by the time this is consulted, so `uvicorn[standard]`
#: would be an entry that never matches.
_IMPORT_NAME = {
    "python-dotenv": "dotenv",
    "python-multipart": "multipart",
    "argon2-cffi": "argon2",
    "pillow": "PIL",
}


#: The distribution name at the head of a requirement string, before any
#: extra, version specifier, or environment marker.
_DIST = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _declared() -> tuple[set[str], dict[str, set[str]], set[str]]:
    """(base, {extra: its dependencies}, marker-gated), as import names.

    Parsed with a regex rather than `packaging.requirements`, which is itself
    a transitively-supplied import — pytest happens to depend on it. A test
    about undeclared dependencies should not rest on one.

    The third set is every name whose requirement carries an environment
    marker, and it exists because declaring something is not the same as
    installing it. `tomli>=2.0; python_version < '3.11'` is declared in the
    dev extra and absent from every 3.11 environment, so a list of what is
    installed that reads markers as if they were not there names a package
    that is not present. Any marker at all disqualifies a name here, even one
    that happens to hold everywhere: the cost of being wrong that way is a
    guard that asks for one `importorskip` too many, and the cost of the
    other way is the collection abort this whole file is about.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    conditional: set[str] = set()

    def names(reqs) -> set[str]:
        out = set()
        for raw in reqs:
            match = _DIST.match(raw)
            assert match, raw
            dist = match.group(1)
            name = _IMPORT_NAME.get(dist.lower(), dist.lower().replace("-", "_"))
            out.add(name)
            # A marker follows a semicolon: `tomli>=2.0; python_version<"3.11"`.
            if ";" in raw:
                conditional.add(name)
        return out

    base = names(data["project"]["dependencies"])
    extras = {
        extra: names(reqs)
        for extra, reqs in (data["project"].get("optional-dependencies") or {}).items()
    }
    return base, extras, conditional


#: Package directories that are this repository's own, so an import of one is
#: not a dependency of anything.
_FIRST_PARTY = {"liminallm", "tests"}


def _module_scope_imports(package: str) -> dict[str, set[str]]:
    """Third-party names imported at module scope under `package`, and where."""
    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / package).rglob("*.py")):
        tree = ast.parse(path.read_text())
        # `tree.body` only: an import nested in a function or a `try` is a
        # deliberate soft dependency and not what this checks.
        for node in tree.body:
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".")[0]]
            for name in modules:
                if name in sys.stdlib_module_names or name in _FIRST_PARTY:
                    continue
                found.setdefault(name, set()).add(
                    str(path.relative_to(ROOT))
                )
    return found


def test_every_module_scope_import_is_declared():
    base, _extras, _conditional = _declared()
    imports = _module_scope_imports("liminallm")

    undeclared = {
        name: sorted(files) for name, files in imports.items() if name not in base
    }

    assert not undeclared, (
        "imported at module scope but not in [project] dependencies — the "
        "package cannot be imported without these:\n"
        + "\n".join(f"  {name}: {', '.join(files)}" for name, files in undeclared.items())
    )


def test_a_test_module_imports_only_what_every_lane_installs():
    """The same rule for `tests/`, measured against the narrowest CI lane.

    A test module that imports something at module scope does not merely fail
    itself when the package is absent: it fails *collection*, and a collection
    error aborts the whole run before any marker deselects anything. So a
    module-scope import in `tests/` has to be satisfied by every lane, and the
    narrowest is the browser lane — base plus the dev extra, less anything a
    marker gates, because a declaration is not an installation. See
    `test_a_marker_gated_dependency_is_not_treated_as_installed`.

    The `train` extra is the trap, because one lane hides it: the test job's
    install line names `jax`, which brings `numpy` with it, so both look
    available everywhere. They are not. Two of these modules imported `numpy`
    plainly and took the browser lane down at collection while deselecting
    2694 tests they never reached. Anything outside base + dev belongs behind
    `pytest.importorskip`, which is an import this walk does not see and a
    skip rather than an error when the package is missing.
    """
    base, extras, conditional = _declared()
    installed_everywhere = (base | extras["dev"]) - conditional
    imports = _module_scope_imports("tests")

    unavailable = {
        name: sorted(files)
        for name, files in imports.items()
        if name not in installed_everywhere
    }

    assert not unavailable, (
        "imported at module scope in tests/ but not installed by every CI "
        "lane — this aborts collection rather than skipping. Move each behind "
        "pytest.importorskip:\n"
        + "\n".join(f"  {name}: {', '.join(files)}" for name, files in unavailable.items())
    )


def test_a_marker_gated_dependency_is_not_treated_as_installed():
    """Declaring a package and installing it are different claims.

    The check above allows a module-scope import in `tests/` if the name is in
    base or dev. Read without markers, that list said `tomli` — declared in
    the dev extra as `tomli>=2.0; python_version < '3.11'`, and therefore
    installed on 3.10 and on nothing else. The browser lane runs 3.11, where
    it is absent. So a module-scope `import tomli` would have passed the
    guard and still aborted that lane's collection, which is the exact
    failure the guard was written to catch.

    Any marker disqualifies a name, not just this one — the parse cannot
    evaluate markers and should not pretend to.
    """
    base, extras, conditional = _declared()

    assert "tomli" in extras["dev"], sorted(extras["dev"])
    assert "tomli" in conditional, sorted(conditional)
    assert "tomli" not in (base | extras["dev"]) - conditional

    # Only the direction the marker guarantees. On 3.10 the dev extra promises
    # `tomli`, so its absence would mean the extra was not installed. On 3.11
    # nothing is claimed: the marker stops pip from installing it, but another
    # package is free to bring it, and this test is not about that.
    if sys.version_info < (3, 11):
        assert importlib.util.find_spec("tomli") is not None


@pytest.mark.parametrize(
    "package,expected", [("liminallm", 8), ("tests", 3)]
)
def test_the_check_can_see_something(package, expected):
    """A guard against the guard passing because it found nothing.

    If the walk stops working — a moved package, a parse that silently yields
    nothing — the assertions above become vacuously true and report a clean
    dependency list forever.
    """
    imports = _module_scope_imports(package)

    assert "httpx" in imports, sorted(imports)
    assert "fastapi" in imports, sorted(imports)
    assert len(imports) >= expected, sorted(imports)


def test_no_name_mapping_is_unreachable():
    """Every key in the map must survive `_DIST` unchanged, or it never fires.

    The map originally carried `uvicorn[standard]` and `psycopg[binary]`,
    which `_DIST` reduces to `uvicorn` and `psycopg` before the lookup — so
    those two entries could not match, and the default branch happened to
    produce the same answer. A wrong entry in that shape would be silent.
    """
    for key in _IMPORT_NAME:
        match = _DIST.match(key)
        assert match and match.group(1) == key, key


@pytest.mark.parametrize("name", ["numpy", "tiktoken"])
def test_a_soft_dependency_stays_out_of_module_scope(name):
    """The two this rule deliberately does not require.

    Both are imported inside a function, which is what makes them optional —
    `numpy` in the checkpoint loader that only the training extra reaches, and
    `tiktoken` inside a `try:` that falls back to a heuristic token count. If
    either moves to module scope it stops being optional, and the test above
    will start demanding it be declared. This says so out loud so that the
    move is a decision rather than an accident.
    """
    assert name not in _module_scope_imports("liminallm")
