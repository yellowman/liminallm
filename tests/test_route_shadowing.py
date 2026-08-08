"""No literal route may be shadowed by an earlier parameterized one.

FastAPI matches in declaration order, so `GET /notes/{note_id}` declared
before `GET /notes/sweeps` silently swallows "sweeps" and 404s. That bug is
easy to reintroduce and invisible until someone calls the endpoint, so this
checks the whole API rather than the one case that bit us.
"""

from __future__ import annotations

import re

from liminallm.api.routes import router

_PARAM = re.compile(r"^\{[^}]+\}$")


def _segments(path: str):
    return [s for s in path.split("/") if s]


def _shadows(earlier: str, later: str) -> bool:
    """True if `earlier` would capture requests intended for `later`."""
    a, b = _segments(earlier), _segments(later)
    if len(a) != len(b):
        return False
    saw_param_over_literal = False
    for seg_a, seg_b in zip(a, b):
        a_param, b_param = bool(_PARAM.match(seg_a)), bool(_PARAM.match(seg_b))
        if a_param and not b_param:
            saw_param_over_literal = True   # {id} eats a literal like "sweeps"
        elif seg_a != seg_b:
            return False                     # different literals: no overlap
    return saw_param_over_literal


def test_no_route_is_shadowed_by_an_earlier_parameterized_route():
    seen: list[tuple[str, str]] = []   # (method, path) in declaration order
    problems: list[str] = []
    for route in router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path:
            continue
        for method in methods:
            for prev_method, prev_path in seen:
                if prev_method == method and _shadows(prev_path, path):
                    problems.append(
                        f"{method} {path} is unreachable: "
                        f"{method} {prev_path} is declared earlier and captures it"
                    )
            seen.append((method, path))
    assert not problems, "\n".join(problems)


def test_the_detector_catches_the_bug_it_was_written_for():
    assert _shadows("/notes/{note_id}", "/notes/sweeps") is True
    assert _shadows("/notes/sweeps", "/notes/{note_id}") is False  # correct order
    assert _shadows("/notes/{note_id}", "/notes/{other}") is False
    assert _shadows("/notes/{id}/witness", "/notes/sweeps") is False  # different shape
    assert _shadows("/a/{x}/b", "/a/lit/b") is True
