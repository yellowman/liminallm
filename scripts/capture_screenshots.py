#!/usr/bin/env python3
"""Capture one screenshot per screen, from a real browser against a real server.

The images under `docs/screenshots/` are produced by this script. It starts a
throwaway Postgres and Redis, serves the real ASGI app on a real port, seeds a
workspace through the same HTTP API the SPA calls, and drives Chromium through
every screen. Nothing here is a mockup, and nothing is drawn by hand.

Usage:
    # Offline. No credential, no network, deterministic answers from the stub
    # backend. Use this to check the capture pipeline still works.
    python scripts/capture_screenshots.py

    # Live. Answers come from a real provider, which is what the images in the
    # README should be regenerated with.
    GEMINI_API_KEY=... python scripts/capture_screenshots.py --live

Options:
    --live          Answer with a real provider instead of the stub backend.
    --model NAME    Model for --live (default: the LIMINALLM_SCREENSHOT_MODEL
                    environment variable, else gemini-3.7-flash).
    --out DIR       Where to write the images (default: docs/screenshots).

The credential is read from `GEMINI_API_KEY` in the environment and nowhere
else. It is never written to a file, printed, or embedded in an image. Do not
add a key to this file or pass one on a shared command line.

Requires the dev extra for Playwright and a Chromium build; the browser test
lane has the same prerequisites.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_MODEL = "gemini-3.7-flash"
EMAIL = "ada@example.com"
PASSWORD = "Screenshot-Demo-2026!"
VIEWPORT = {"width": 1440, "height": 900}

#: Each row is (section, image, what to select in its pane first). The pane
#: navigates and the workspace shows the selection, so a section with a pane
#: is photographed with something selected: a shot of "Select a context to
#: view details" documents the empty state rather than the screen. Files,
#: Insights and Settings have no pane and select nothing.
#:
#: Each selector names its own list container. A bare `.row` would defeat
#: the point: the Tools pane holds a tool list and a workflow list, so an
#: empty tool list would still match a workflow row and photograph the wrong
#: thing without failing.
TABS = [
    ("notes-tab", "03-notes", "#note-list .note-item"),
    ("contexts-tab", "04-contexts", "#contexts-list .row"),
    ("files-tab", "05-files", None),
    ("artifacts-tab", "06-artifacts", "#artifacts-list .row"),
    ("tools-tab", "07-tools", "#tools-list .row"),
    ("insights-tab", "08-insights", None),
    ("settings-tab", "09-settings", None),
]

#: The second question names nobody. Answering it about the same subject is
#: the property the chat screenshot exists to show, so it must stay
#: unanchored: no "the pause", no "the 2023 calls".
FIRST_THREAD = [
    "Why did the 2023 calls to pause frontier AI training fail to stop "
    "any lab?",
    "Which of those obstacles does competition between countries make worst?",
]

DOC = (
    "# The pause debate, in one page\n\n"
    "In March 2023 an open letter asked labs to stop training systems more "
    "capable than GPT-4 for six months. No lab stopped. Its lasting effect "
    "was to move the argument from *whether* frontier training should be "
    "governed to *which lever does it*: compute thresholds, evaluation "
    "before deployment, or reporting requirements.\n"
)
FILES = [
    ("pause-debate-in-one-page.md", DOC.encode(), "text/markdown"),
    (
        "frontier-ai-reading-list.txt",
        b"Pause Giant AI Experiments: An Open Letter (2023)\n"
        b"The Bletchley Declaration (2023)\n"
        b"US Executive Order 14110 on AI (2023)\n"
        b"EU AI Act, final text (2024)\n"
        b"International AI Safety Report (2025)\n",
        "text/plain",
    ),
]
NOTES = [
    (
        "What the pause letter changed",
        "No lab stopped training, so by its own terms it failed. What "
        "changed was the default question: a frontier training run became "
        "something a lab might owe an account of, and every rule since "
        "argues over which account counts.",
    ),
    (
        "Competition is the standing objection",
        "Every proposal to slow the frontier meets the same reply: a "
        "one-sided slowdown moves capability rather than removing it. That "
        "reply is strongest where verification is weakest, which makes "
        "measurement the load-bearing problem rather than persuasion.",
    ),
    (
        "Compute is the governable surface",
        "Weights copy and researchers move, but large training runs need "
        "datacenters and fabrication that are few, fixed, and already "
        "counted. That is why thresholds get written in FLOP: not because "
        "the number means much, but because it is the part you can see.",
    ),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a screenshot of every screen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="answer with a real provider instead of the stub backend",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LIMINALLM_SCREENSHOT_MODEL", DEFAULT_MODEL),
        help=f"model for --live (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO / "docs" / "screenshots",
        help="directory to write the images into",
    )
    return parser.parse_args(argv)


def configure_paths(scratch: pathlib.Path) -> None:
    """Point the filesystem-shaped settings at throwaway state.

    These are read while the modules below are imported, so they are set
    before any of them is.
    """
    fs_root = scratch / "fs"
    fs_root.mkdir(parents=True, exist_ok=True)
    os.environ["SHARED_FS_ROOT"] = str(fs_root)
    os.environ["TEST_MODE"] = "true"
    os.environ.setdefault("EMBEDDING_VECTOR_DIM", "64")


def instance_settings(args: argparse.Namespace, redis_url: str) -> dict:
    """The managed settings this capture runs under.

    `redis_url` is a database-managed setting with a `localhost:6379`
    default and no environment variable behind it, so the scratch Redis has
    to be named here. Exporting `REDIS_URL` does nothing: the runtime never
    reads it, and a capture that relied on it would quietly use whichever
    Redis happened to be listening on the developer's own machine.
    """
    settings = {"redis_url": redis_url}
    if not args.live:
        # Deterministic and offline: canned answers, no credential, no
        # network. The images are for checking the pipeline, not for the
        # README.
        settings["model_backend"] = "stub"
        return settings

    # The credential comes from the environment and is never echoed.
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "--live needs GEMINI_API_KEY in the environment. Run without "
            "--live to capture with the stub backend instead."
        )
    settings["model_backend"] = "gemini_native"
    settings["model_path"] = args.model
    return settings


def assert_isolated(redis_url: str) -> None:
    """Refuse to run against anything but the Redis this script started.

    A tool that regenerates documentation must not reach a service it does
    not own, and `TEST_MODE` would otherwise let it proceed with no cache at
    all rather than say so.
    """
    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()
    actual = runtime.settings.redis_url
    if actual != redis_url:
        raise SystemExit(
            f"the runtime resolved Redis at {actual}, not the scratch "
            f"instance at {redis_url}; refusing to touch a service this "
            "script does not own"
        )
    if runtime.cache is None:
        raise SystemExit(
            "the runtime has no Redis cache, so the scratch instance was "
            "never reached"
        )


def unwrap(resp) -> dict:
    """The API answers `{"status": "ok", "data": {...}}`."""
    body = resp.json() or {}
    return body.get("data", body) if isinstance(body, dict) else {}


def seed(client, token: str) -> None:
    """Fill the workspace through the same API the SPA calls."""
    headers = {"Authorization": f"Bearer {token}"}
    user_id = unwrap(client.get("/v1/me", headers=headers)).get("id")

    context = unwrap(
        client.post(
            "/v1/contexts",
            headers=headers,
            json={
                "name": "Frontier AI policy",
                "description": (
                    "Sources on the pause debate, compute governance, and "
                    "international competition."
                ),
            },
        )
    )
    context_id = context.get("id")
    if not context_id:
        # Every later step needs this id. Skipping them quietly is what
        # produces a capture of empty screens that looks deliberate.
        raise SystemExit(f"the context was not created: {context}")

    for name, body, mime in FILES:
        uploaded = client.post(
            "/v1/files/upload", headers=headers, files={"file": (name, body, mime)}
        )
        if uploaded.status_code >= 400:
            # Checked per file. Chunks appearing at all does not say both
            # documents are there, so one failed upload would otherwise be
            # masked by the other one indexing successfully.
            raise SystemExit(
                f"could not upload {name} "
                f"(HTTP {uploaded.status_code}): {uploaded.text[:300]}"
            )

    # Index the uploads into the context. Without this the Contexts screen is
    # captured reading "No sources added yet" and "0 chunks loaded", which
    # documents an empty context rather than the retrieval the screen is for.
    # The path is relative: it is resolved against the caller's own root, so
    # "files" reaches `users/{user_id}/files`, where uploads land. Passing
    # the rooted path instead produces `users/{id}/users/{id}/files`, which
    # this endpoint accepts and indexes nothing from.
    added = client.post(
        f"/v1/contexts/{context_id}/sources",
        headers=headers,
        json={"fs_path": "files", "recursive": True},
    )
    if added.status_code != 201:
        raise SystemExit(
            f"could not index the uploads into the context "
            f"(HTTP {added.status_code}): {added.text[:300]}"
        )
    # 201 only means the path was accepted and recorded. A path that matched
    # no document is indexed to nothing and still answers 201, so the chunks
    # are what say the sources are really there. This reads the endpoint the
    # Contexts screen itself reads, rather than a count field that endpoint
    # does not return.
    chunks = unwrap(
        client.get(f"/v1/contexts/{context_id}/chunks?limit=20", headers=headers)
    ).get("items")
    if not chunks:
        raise SystemExit(
            "the context indexed no chunks, so the Contexts screen "
            "would document an empty context"
        )

    for title, content in NOTES:
        client.post(
            "/v1/notes", headers=headers, json={"title": title, "content": content}
        )

    # The admin console is a screen too, so the demo account needs the role.
    if user_id:
        from liminallm.service.runtime import get_runtime

        get_runtime().store.update_user_role(user_id, "admin")


def capture(args: argparse.Namespace, base: str) -> list[pathlib.Path]:
    from playwright.sync_api import sync_playwright

    from tests.browser import chromium_executable

    args.out.mkdir(parents=True, exist_ok=True)
    shots: list[pathlib.Path] = []

    # A settled answer is neither the typing placeholder nor a still-streaming
    # bubble. Live answers are long and a turn whose tool call returned no
    # prose renders "No response generated."; the stub's answer is one short
    # line, so the bar moves with the backend.
    floor = 120 if args.live else 10
    answers = """() => [...document.querySelectorAll(
        '.message.assistant:not(.typing):not(.streaming)')]
        .map(e => (e.innerText || '').trim())
        .filter(t => t.length > %d && !t.includes('No response generated'))
        .length""" % floor

    with sync_playwright() as play:
        launch = {"headless": True}
        executable = chromium_executable()
        if executable:
            launch["executable_path"] = executable
        browser = play.chromium.launch(**launch)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        def shot(name: str) -> None:
            path = args.out / f"{name}.png"
            page.screenshot(path=str(path))
            shots.append(path)
            print("captured", path.name, flush=True)

        def ask(questions: list[str], attempts: int = 3) -> None:
            """Run one exchange to completion in a single thread.

            A turn that produces no answer is retried as a whole exchange in
            a fresh thread, never by re-sending into the same one: a failed
            turn sitting above its own retry is what the screenshot must not
            show.
            """
            for _ in range(attempts):
                settled = True
                for turn, question in enumerate(questions, start=1):
                    page.fill("#message-input", question)
                    page.click("#send-btn")
                    try:
                        page.wait_for_function(
                            f"(n) => ({answers})() >= n", arg=turn, timeout=120000
                        )
                        time.sleep(1.5)
                    except Exception:  # noqa: BLE001 - retry in a new thread
                        settled = False
                        break
                if settled:
                    return
                page.click("#new-thread")
                time.sleep(1.2)
            raise RuntimeError(f"no answer for: {questions[0][:60]}")

        page.goto(f"{base}/", wait_until="domcontentloaded")
        page.wait_for_selector("#auth-form", state="visible")
        time.sleep(0.6)
        shot("01-sign-in")

        page.fill("#email", EMAIL)
        page.fill("#password", PASSWORD)
        page.click("#auth-form button[type=submit]")
        page.wait_for_function(
            "() => !!sessionStorage.getItem('liminal.accessToken')", timeout=30000
        )
        page.wait_for_selector("#main-tabs", state="visible")

        # The workspace was seeded before this sign-in, so the token the page
        # is holding already carries the admin role and the start-up requests
        # already see every note, file and context.
        time.sleep(1.5)

        ask(FIRST_THREAD)
        # Rate the answer, so the Insights screen summarises a real event.
        page.click("#thumbs-up")
        time.sleep(1.5)

        # The thread is scrolled back to its first question: the tail of a
        # long answer is not what the screen is for.
        page.evaluate(
            "() => { const m = document.querySelector('#messages');"
            " if (m) m.scrollTop = 0; window.scrollTo(0, 0); }"
        )
        time.sleep(0.8)
        shot("02-chat")

        # This is the only thread the capture opens. Extra threads would be
        # set dressing for a conversation list that no screen below shows:
        # each section has its own pane, and the chat shot is already taken.

        for tab_id, name, pick in TABS:
            page.click(f"#main-tabs .rail-btn[data-tab='{tab_id}']")
            page.wait_for_selector(f"#{tab_id}.active", state="visible")
            time.sleep(1.2)
            if pick:
                # Fatal, not a warning. A selector that no longer matches
                # anything is how these images went stale: the pane renders,
                # nothing is selected, and the capture succeeds with a shot
                # of the empty state that looks deliberate. Renaming a class
                # in the frontend has to break this script loudly.
                item = f".pane-view[data-pane='{tab_id}'] {pick}"
                try:
                    page.wait_for_selector(item, state="visible", timeout=15000)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    raise RuntimeError(
                        f"nothing matched {pick!r} in the {tab_id} pane, so "
                        f"{name} would document the empty state; check "
                        "whether the frontend renamed the class"
                    ) from exc
                page.locator(item).first.click()
                time.sleep(1.2)
            shot(name)

        page.goto(f"{base}/admin", wait_until="domcontentloaded")
        time.sleep(2.5)
        shot("10-admin")

        browser.close()
    return shots


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="liminallm-shots-") as tmp:
        configure_paths(pathlib.Path(tmp))

        import httpx

        from tests.browser import LiveServer
        from tests.harness import ScratchPostgres, ScratchRedis, apply_schema

        # Every start() below has its stop() in the finally, including a
        # failure part way through capture: this script owns two server
        # processes and a Postgres data directory of its own, and leaving
        # them behind is what makes a second run pick up a first run's mess.
        postgres = ScratchPostgres()
        redis = ScratchRedis()
        server = None
        client = None
        try:
            database_url = postgres.start()
            redis_url = redis.start()
            os.environ["DATABASE_URL"] = database_url
            # Read when the runtime first boots, which the server start
            # below triggers.
            os.environ["INSTANCE_SETTINGS_JSON"] = json.dumps(
                instance_settings(args, redis_url)
            )
            apply_schema(database_url, embedding_dim=64)

            server = LiveServer().start()
            assert_isolated(redis_url)

            client = httpx.Client(base_url=server.base_url, timeout=120.0)
            client.post(
                "/v1/auth/signup", json={"email": EMAIL, "password": PASSWORD}
            )
            # Seed before the browser signs in, never after. The workspace
            # ends with a promotion to admin, and a role change invalidates
            # the access token minted before it. Seeding against a token the
            # browser is also holding therefore breaks that browser session
            # mid-capture: the page recovers by refreshing, but the requests
            # already in flight fail, and whatever they were filling in stays
            # broken on screen. That is how "Unable to check" reached the
            # README.
            login = client.post(
                "/v1/auth/login", json={"email": EMAIL, "password": PASSWORD}
            )
            token = unwrap(login).get("access_token")
            if not token:
                # Seeding with no credential would be refused on every call
                # and still finish, leaving a capture of empty screens that
                # looks deliberate.
                raise SystemExit(
                    f"could not sign in as the demo account "
                    f"(HTTP {login.status_code}); nothing was captured"
                )
            seed(client, token)
            mode = f"live ({args.model})" if args.live else "stub backend"
            print(f"serving {server.base_url} with the {mode}", flush=True)
            shots = capture(args, server.base_url)
        finally:
            if client is not None:
                client.close()
            if server is not None:
                server.stop()
            redis.stop()
            postgres.stop()

    print(f"\n{len(shots)} screenshots in {args.out}")
    if not args.live:
        print(
            "These used the stub backend. Regenerate the README images with "
            "--live and a provider credential."
        )
    return 0


if __name__ == "__main__":
    # Everything runs under this guard on purpose: a tool call spawns a
    # worker process with the "spawn" start method, which re-imports
    # `__main__`, and module-level setup would start a second server and a
    # second database inside the child.
    sys.exit(main(sys.argv[1:]))
