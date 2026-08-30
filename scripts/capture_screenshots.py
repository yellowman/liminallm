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

#: The chat screen is captured before the extra threads are opened, so it
#: shows a conversation rather than an empty new thread.
TABS = [
    ("notes-tab", "03-notes"),
    ("contexts-tab", "04-contexts"),
    ("files-tab", "05-files"),
    ("artifacts-tab", "06-artifacts"),
    ("tools-tab", "07-tools"),
    ("insights-tab", "08-insights"),
    ("settings-tab", "09-settings"),
]

FIRST_THREAD = [
    "Who is Sergey Brin, and what is he best known for?",
    "What did he study before that, and where?",
]
#: Extra threads, so every screen after the chat shot shows a used workspace
#: in the conversation sidebar rather than a first-run one.
EXTRA_THREADS = [
    "Who was Hunter S. Thompson, and why does his reporting still get "
    "argued about?",
    "What is gonzo journalism, and which piece started it?",
]

DOC = (
    "# Gonzo, in one page\n\n"
    "Hunter S. Thompson filed *The Kentucky Derby Is Decadent and Depraved* "
    "in 1970 against a closing deadline, sending pages torn straight from "
    "his notebook. The reporter stopped pretending to be absent from the "
    "story, and the method got a name.\n"
)
FILES = [
    ("gonzo-in-one-page.md", DOC.encode(), "text/markdown"),
    (
        "thompson-reading-list.txt",
        b"Hell's Angels (1967)\n"
        b"The Kentucky Derby Is Decadent and Depraved (1970)\n"
        b"Fear and Loathing in Las Vegas (1971)\n"
        b"Fear and Loathing on the Campaign Trail '72 (1973)\n",
        "text/plain",
    ),
]
NOTES = [
    (
        "Gonzo starts at the Derby",
        "The 1970 Scanlan's piece is the origin point: no finished draft, "
        "notebook pages wired in as they were, the writer visibly inside the "
        "scene. What reads as style began as a deadline being missed.",
    ),
    (
        "The reporter is a character",
        "New Journalism let reporting borrow the novel's tools. Gonzo went "
        "further and made the reporter's own state part of the evidence, "
        "which is either the point or the flaw depending on the reader.",
    ),
    (
        "Fear and Loathing as reportage",
        "Read as a road book it is a comedy; read as reporting it is an "
        "argument about what the sixties turned into. Thompson's wave "
        "passage does the work an editorial would have.",
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


def configure_env(args: argparse.Namespace, scratch: pathlib.Path) -> None:
    """Point the runtime at throwaway state and choose the answering backend."""
    fs_root = scratch / "fs"
    fs_root.mkdir(parents=True, exist_ok=True)
    os.environ["SHARED_FS_ROOT"] = str(fs_root)
    os.environ["TEST_MODE"] = "true"
    os.environ.setdefault("EMBEDDING_VECTOR_DIM", "64")

    if not args.live:
        # Deterministic and offline: canned answers, no credential, no
        # network. The images are for checking the pipeline, not for the
        # README.
        os.environ["INSTANCE_SETTINGS_JSON"] = json.dumps(
            {"model_backend": "stub"}
        )
        return

    # The credential comes from the environment and is never echoed.
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "--live needs GEMINI_API_KEY in the environment. Run without "
            "--live to capture with the stub backend instead."
        )
    os.environ["INSTANCE_SETTINGS_JSON"] = json.dumps(
        {"model_backend": "gemini_native", "model_path": args.model}
    )


def unwrap(resp) -> dict:
    """The API answers `{"status": "ok", "data": {...}}`."""
    body = resp.json() or {}
    return body.get("data", body) if isinstance(body, dict) else {}


def seed(client, token: str) -> None:
    """Fill the workspace through the same API the SPA calls."""
    headers = {"Authorization": f"Bearer {token}"}

    client.post(
        "/v1/contexts",
        headers=headers,
        json={
            "name": "Gonzo",
            "description": "Sources on Hunter S. Thompson and the New Journalism.",
        },
    )
    for name, body, mime in FILES:
        client.post(
            "/v1/files/upload", headers=headers, files={"file": (name, body, mime)}
        )
    for title, content in NOTES:
        client.post(
            "/v1/notes", headers=headers, json={"title": title, "content": content}
        )

    # The admin console is a screen too, so the demo account needs the role.
    user_id = unwrap(client.get("/v1/me", headers=headers)).get("id")
    if user_id:
        from liminallm.service.runtime import get_runtime

        get_runtime().store.update_user_role(user_id, "admin")


def capture(args: argparse.Namespace, base: str, client) -> list[pathlib.Path]:
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

        # Seed with the credential the app itself holds, then reload so the
        # SPA renders the seeded workspace.
        seed(client, page.evaluate(
            "() => sessionStorage.getItem('liminal.accessToken')"
        ))
        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector("#main-tabs", state="visible")
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

        for question in EXTRA_THREADS:
            page.click("#new-thread")
            time.sleep(1.2)
            ask([question])

        for tab_id, name in TABS:
            page.click(f"#main-tabs .tab-btn[data-tab='{tab_id}']")
            page.wait_for_selector(f"#{tab_id}.active", state="visible")
            time.sleep(1.2)
            if tab_id == "notes-tab":
                # Open a note, so the screen shows the editor and not an
                # empty right-hand pane.
                page.click(f"#{tab_id} .note-item, #{tab_id} li")
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
        configure_env(args, pathlib.Path(tmp))

        from tests.harness import ScratchPostgres, ScratchRedis, apply_schema

        postgres = ScratchPostgres()
        os.environ["DATABASE_URL"] = postgres.start()
        redis = ScratchRedis()
        os.environ["REDIS_URL"] = redis.start()
        apply_schema(os.environ["DATABASE_URL"], embedding_dim=64)

        import httpx

        from tests.browser import LiveServer

        server = LiveServer().start()
        client = httpx.Client(base_url=server.base_url, timeout=120.0)
        try:
            client.post(
                "/v1/auth/signup", json={"email": EMAIL, "password": PASSWORD}
            )
            mode = f"live ({args.model})" if args.live else "stub backend"
            print(f"serving {server.base_url} with the {mode}", flush=True)
            shots = capture(args, server.base_url, client)
        finally:
            client.close()
            server.stop()

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
