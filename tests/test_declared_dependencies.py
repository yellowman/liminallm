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
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Distribution name -> the name you import it by, where they differ. Only
#: the ones this project actually declares; a new entry belongs here when a
#: new dependency's two names disagree.
_IMPORT_NAME = {
    "python-dotenv": "dotenv",
    "python-multipart": "multipart",
    "argon2-cffi": "argon2",
    "pillow": "PIL",
    "uvicorn[standard]": "uvicorn",
    "psycopg[binary]": "psycopg",
}


def _declared() -> tuple[set[str], set[str]]:
    """(base, base + every extra), as import names."""
    from packaging.requirements import Requirement

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    def names(reqs) -> set[str]:
        out = set()
        for raw in reqs:
            dist = Requirement(raw).name
            out.add(_IMPORT_NAME.get(dist.lower(), dist.lower().replace("-", "_")))
        return out

    base = names(data["project"]["dependencies"])
    everything = set(base)
    for reqs in (data["project"].get("optional-dependencies") or {}).values():
        everything |= names(reqs)
    return base, everything


def _module_scope_imports() -> dict[str, set[str]]:
    """Third-party names imported at module scope, and where."""
    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / "liminallm").rglob("*.py")):
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
                if name in sys.stdlib_module_names or name == "liminallm":
                    continue
                found.setdefault(name, set()).add(
                    str(path.relative_to(ROOT))
                )
    return found


def test_every_module_scope_import_is_declared():
    base, _everything = _declared()
    imports = _module_scope_imports()

    undeclared = {
        name: sorted(files) for name, files in imports.items() if name not in base
    }

    assert not undeclared, (
        "imported at module scope but not in [project] dependencies — the "
        "package cannot be imported without these:\n"
        + "\n".join(f"  {name}: {', '.join(files)}" for name, files in undeclared.items())
    )


def test_the_check_can_see_something():
    """A guard against the guard passing because it found nothing.

    If the walk stops working — a moved package, a parse that silently yields
    nothing — the assertion above becomes vacuously true and reports a clean
    dependency list forever.
    """
    imports = _module_scope_imports()

    assert "httpx" in imports, sorted(imports)
    assert "fastapi" in imports, sorted(imports)
    assert len(imports) >= 8, sorted(imports)


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
    assert name not in _module_scope_imports()
