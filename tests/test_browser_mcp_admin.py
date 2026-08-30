"""Publishing a tool server from the console an operator actually opens.

The API tests prove the route. This proves the page: that the form exists,
that what it sends is the shape the route accepts, and that a server published
by clicking the button is one a turn can then use. Those are separable - a
console that posts `visibility: private`, or the wrong `type`, passes every
API test in the suite and silently publishes nothing.

Run with the browser lane: `pytest -m browser`.
"""

from __future__ import annotations

import uuid

import pytest

from tests.browser import LiveServer, chromium_executable
from tests.mcpfixture import MCPFixture

pytest.importorskip(
    "playwright",
    reason="the browser lane needs the dev extra: uv pip install playwright",
)

pytestmark = pytest.mark.browser

PASSWORD = "TestPassword123!"


@pytest.fixture(scope="module")
def server():
    live = LiveServer().start()
    try:
        yield live
    finally:
        live.stop()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as play:
        launched = play.chromium.launch(executable_path=chromium_executable())
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    opened = context.new_page()
    try:
        yield opened
    finally:
        context.close()


def _admin(server) -> tuple[str, str]:
    """A real admin account, made the way the API makes one."""
    import httpx

    from liminallm.service.runtime import get_runtime

    email = f"mcpadm_{uuid.uuid4().hex[:8]}@example.com"
    resp = httpx.post(
        f"{server.base_url}/v1/auth/signup",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert resp.status_code == 201, resp.text
    get_runtime().store.update_user_role(
        resp.json()["data"]["user_id"], role="admin"
    )
    return email, PASSWORD


def _sign_in(page, server, email, password) -> None:
    page.goto(f"{server.base_url}/admin", wait_until="domcontentloaded")
    page.fill("#admin-email", email)
    page.fill("#admin-password", password)
    page.click("#admin-auth-form button[type=submit]")
    page.wait_for_selector("#admin-console:not(.hidden)", timeout=15000)


class TestAnAdminPublishesAServerFromTheConsole:
    def test_the_form_publishes_a_server_a_turn_can_use(self, page, server):
        """Clicked, not posted: the button is the thing under test.

        Checked against `servers_for_turn` rather than against the table the
        page redraws, because the page rendering a row it just typed is not
        evidence that anything was published - a private artifact would look
        identical in that table and reach no turn at all.
        """
        from liminallm.service import mcp_client
        from liminallm.service.runtime import get_runtime

        email, password = _admin(server)
        name = f"br{uuid.uuid4().hex[:6]}"
        with MCPFixture(name, {"lookup": "ok"}) as fixture:
            _sign_in(page, server, email, password)

            page.fill("#new-mcp-name", name)
            page.fill("#new-mcp-url", fixture.url)
            page.select_option("#new-mcp-taint", "local_read")
            page.fill("#new-mcp-description", "published from the console")
            page.click("#create-mcp-server")
            page.wait_for_selector(f"#mcp-table-wrapper td:text-is('{name}')",
                                   timeout=15000)

            servers = mcp_client.servers_for_turn(get_runtime().store)
            published = [s for s in servers if s["url"] == fixture.url]
            assert published, servers
            assert published[0]["taint_class"] == "local_read", (
                "the console's classification did not reach the artifact"
            )

    def test_the_table_lists_what_is_already_published(self, page, server):
        """The list is loaded on sign-in, not only after a create.

        An operator arriving at a console that shows nothing cannot tell an
        installation with no servers from a page that never asked.
        """
        import httpx

        email, password = _admin(server)
        login = httpx.post(
            f"{server.base_url}/v1/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        assert login.status_code == 200, login.text
        token = login.json()["data"]["access_token"]
        name = f"pre{uuid.uuid4().hex[:6]}"
        with MCPFixture(name) as fixture:
            created = httpx.post(
                f"{server.base_url}/v1/artifacts",
                json={
                    "type": "mcp",
                    "name": name,
                    "visibility": "global",
                    "schema": fixture.as_artifact_schema(),
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            assert created.status_code == 201, created.text

            _sign_in(page, server, email, password)

            page.wait_for_selector(f"#mcp-table-wrapper td:text-is('{name}')",
                                   timeout=15000)
            row = page.inner_text("#mcp-table-wrapper")
            assert fixture.url in row
            # Absent means enabled, in the resolver and so on the page.
            assert "enabled" in row

            # Again after a reload, which is the other way into the console:
            # a page opened with a session already in `sessionStorage` takes a
            # different branch from an interactive sign-in, and the two used
            # to load different things.
            page.reload(wait_until="domcontentloaded")

            page.wait_for_selector(f"#mcp-table-wrapper td:text-is('{name}')",
                                   timeout=15000)


class TestTheConsoleCannotBeAWayAround:
    def test_a_non_admin_never_reaches_the_console(self, page, server):
        """The form is admin-only because publishing is.

        The route refuses either way - this is about not showing an operator
        a control that cannot work, and about the gate being the role rather
        than the markup.
        """
        import httpx

        email = f"plain_{uuid.uuid4().hex[:8]}@example.com"
        resp = httpx.post(
            f"{server.base_url}/v1/auth/signup",
            json={"email": email, "password": PASSWORD},
            timeout=30,
        )
        assert resp.status_code == 201, resp.text

        page.goto(f"{server.base_url}/admin", wait_until="domcontentloaded")
        page.fill("#admin-email", email)
        page.fill("#admin-password", PASSWORD)
        page.click("#admin-auth-form button[type=submit]")
        page.wait_for_timeout(1500)

        assert page.is_hidden("#admin-console"), (
            "a non-admin was shown the publishing form"
        )


class TestTheTableSaysWhatTheResolverWouldSay:
    def test_a_server_whose_publisher_is_gone_reads_as_inert(self, page, server):
        """The console must not report a capability the turn does not have.

        `servers_for_turn` skips any artifact with no owner, because the admin
        attestation is what made it a capability. The table computed its state
        from `schema.enabled` alone, so a server whose publisher was deleted
        read as "enabled" while being offered to nobody - the one reading an
        operator would act on, and the opposite of the truth.
        """
        import httpx

        from liminallm.service import mcp_client
        from liminallm.service.runtime import get_runtime

        publisher_email, password = _admin(server)
        viewer_email, _ = _admin(server)
        login = httpx.post(
            f"{server.base_url}/v1/auth/login",
            json={"email": publisher_email, "password": password},
            timeout=30,
        )
        assert login.status_code == 200, login.text
        publisher_id = login.json()["data"]["user_id"]
        token = login.json()["data"]["access_token"]

        name = f"orph{uuid.uuid4().hex[:6]}"
        with MCPFixture(name) as fixture:
            created = httpx.post(
                f"{server.base_url}/v1/artifacts",
                json={
                    "type": "mcp",
                    "name": name,
                    "visibility": "global",
                    "schema": fixture.as_artifact_schema(),
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            assert created.status_code == 201, created.text
            store = get_runtime().store
            assert any(
                s["url"] == fixture.url for s in mcp_client.servers_for_turn(store)
            ), "it was never a capability, so losing it proves nothing"

            store.delete_user(publisher_id)
            assert not any(
                s["url"] == fixture.url for s in mcp_client.servers_for_turn(store)
            ), "the resolver still offers it, so the table is not wrong yet"

            _sign_in(page, server, viewer_email, password)

            page.wait_for_selector(f"#mcp-table-wrapper td:text-is('{name}')",
                                   timeout=15000)
            row = page.inner_text("#mcp-table-wrapper")
            assert "inert" in row, row
