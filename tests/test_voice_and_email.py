"""Voice and email: the logic around the third-party call, not the call.

Both were under 40% and both are mostly network I/O, so these stay on what is
actually ours - whether the service considers itself configured, what it does
when it is not, how addresses are redacted before they reach a log, and what
goes into the message. The provider itself is not worth pretending to test.

The unconfigured paths matter more than they look: a deployment with no API
key and no SMTP host is the normal case, not an edge one.
"""

from __future__ import annotations

import socket
import threading

import pytest

from liminallm.service.email import EmailService
from liminallm.service.voice import VoiceService

# ---------------------------------------------------------------------------
# Voice: configuration
# ---------------------------------------------------------------------------


def test_voice_without_a_key_is_not_configured(tmp_path):
    assert VoiceService(str(tmp_path)).is_configured is False


def test_voice_with_a_key_is_configured(tmp_path):
    assert VoiceService(str(tmp_path), api_key="sk-test").is_configured is True


def test_an_empty_key_does_not_count_as_configured(tmp_path):
    """An unset secret arrives as "" rather than None once it round-trips
    through the settings table."""
    assert VoiceService(str(tmp_path), api_key="").is_configured is False


# ---------------------------------------------------------------------------
# Voice: the placeholder path every keyless deployment runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcription_without_a_key_still_answers(tmp_path):
    """It must return a shaped result, not raise - chat_turn reads
    `transcript` and refuses the turn when it is missing."""
    out = await VoiceService(str(tmp_path)).transcribe(b"spoken words", user_id="u1")
    assert out["transcript"] == "spoken words"
    assert out["model"] == "placeholder"
    assert out["user_id"] == "u1"
    assert out["duration_ms"] > 0


@pytest.mark.asyncio
async def test_undecodable_audio_gets_a_described_placeholder(tmp_path):
    out = await VoiceService(str(tmp_path)).transcribe(b"\xff\xfe\x00\x01" * 50)
    assert "audio placeholder" in out["transcript"]
    assert "200 bytes" in out["transcript"]


@pytest.mark.asyncio
async def test_synthesis_without_a_key_writes_under_the_user(tmp_path):
    out = await VoiceService(str(tmp_path)).synthesize("hello there", user_id="u1")
    assert out["model"] == "placeholder"
    assert "/voice/u1/" in out["audio_url"]

    written = tmp_path / "voice" / "u1"
    assert written.is_dir()
    assert [f for f in written.iterdir() if "hello there" in f.read_text()]


@pytest.mark.asyncio
async def test_synthesis_without_a_user_is_shared_not_crashed(tmp_path):
    out = await VoiceService(str(tmp_path)).synthesize("anon speech")
    assert "/voice/shared/" in out["audio_url"]


@pytest.mark.asyncio
async def test_one_users_audio_does_not_land_in_anothers_directory(tmp_path):
    voice = VoiceService(str(tmp_path))
    await voice.synthesize("mine", user_id="alice")
    await voice.synthesize("theirs", user_id="bob")
    alice = list((tmp_path / "voice" / "alice").iterdir())
    assert len(alice) == 1
    assert "theirs" not in alice[0].read_text()


@pytest.mark.parametrize(
    "size, at_least", [(0, 1000), (16, 1000), (160_000, 80_000)]
)
def test_duration_is_estimated_from_size_with_a_floor(tmp_path, size, at_least):
    """A zero-length clip must not report zero and divide by it downstream."""
    assert VoiceService(str(tmp_path))._estimate_duration(b"x" * size) >= at_least


@pytest.mark.asyncio
async def test_closing_an_unused_voice_service_is_safe(tmp_path):
    await VoiceService(str(tmp_path)).close()


# ---------------------------------------------------------------------------
# Email: configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, configured",
    [
        ({}, False),
        ({"smtp_host": "smtp.example.com"}, False),          # no from address
        ({"from_email": "bot@example.com"}, False),          # no host
        ({"smtp_host": "smtp.example.com", "from_email": "bot@example.com"}, True),
        # smtp_user doubles as the from address when none is given.
        ({"smtp_host": "smtp.example.com", "smtp_user": "bot@example.com"}, True),
    ],
)
def test_email_knows_when_it_can_send(kwargs, configured):
    assert EmailService(**kwargs).is_configured is configured


# ---------------------------------------------------------------------------
# Email: addresses must not reach the logs intact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address, expected",
    [
        ("alice@example.com", "al***@example.com"),
        ("a@example.com", "a***@example.com"),
        ("verylongname@example.com", "ve***@example.com"),
        ("not-an-address", "redacted"),
        ("", "redacted"),
    ],
)
def test_an_address_is_redacted_for_logging(address, expected):
    assert EmailService()._redact_email(address) == expected


def test_redaction_keeps_no_more_than_two_local_characters():
    """Enough to correlate a support report, not enough to be the address."""
    for local in ("bob", "roberta", "r"):
        redacted = EmailService()._redact_email(f"{local}@example.com")
        assert redacted.split("*")[0] == local[:2]


# ---------------------------------------------------------------------------
# Email: what an unconfigured deployment does
# ---------------------------------------------------------------------------


def test_an_unconfigured_send_reports_success(caplog):
    """Dev mode logs the message instead of sending. It reports True on
    purpose: a signup must not fail because nobody set up SMTP."""
    assert EmailService().send_password_reset("alice@example.com", "tok-1") is True


def test_dev_mode_does_not_log_the_address(caplog):
    import logging

    with caplog.at_level(logging.INFO):
        EmailService().send_password_reset("alice@example.com", "tok-1")
    assert "alice@example.com" not in caplog.text


# ---------------------------------------------------------------------------
# Email: what goes in the message
# ---------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch):
    """Capture what _send_email was handed, without sending anything."""
    seen = {}

    def _capture(self, to_email, subject, html_body, text_body=None):
        seen.update(
            to=to_email, subject=subject, html=html_body, text=text_body or ""
        )
        return True

    monkeypatch.setattr(EmailService, "_send_email", _capture)
    return seen


def test_a_reset_link_carries_the_token_and_the_instance_url(captured):
    EmailService(base_url="https://chat.example.com").send_password_reset(
        "alice@example.com", "tok-abc"
    )
    assert captured["to"] == "alice@example.com"
    assert "tok-abc" in captured["html"] and "tok-abc" in captured["text"]
    assert "https://chat.example.com" in captured["html"]


def test_a_reset_email_offers_the_url_as_text_too(captured):
    """A button in HTML is useless in a plaintext client, and reset mail is
    exactly what strict clients strip."""
    EmailService().send_password_reset("alice@example.com", "tok-abc")
    assert "tok-abc" in captured["text"]
    assert captured["text"].strip()


def test_a_verification_link_carries_the_token(captured):
    EmailService(base_url="https://chat.example.com").send_email_verification(
        "alice@example.com", "verify-xyz"
    )
    assert "verify-xyz" in captured["html"]
    assert "verify-xyz" in captured["text"]


def test_the_mfa_notice_carries_no_secret(captured):
    """It confirms MFA was switched on. The TOTP secret belongs in the QR the
    user already scanned, never in mail."""
    EmailService().send_mfa_setup_confirmation("alice@example.com")
    body = captured["html"] + captured["text"]
    assert "otpauth" not in body
    assert "secret" not in body.lower()


@pytest.mark.parametrize(
    "send, needle",
    [
        (lambda s: s.send_password_reset("a@b.co", "t"), "reset"),
        (lambda s: s.send_email_verification("a@b.co", "t"), "verif"),
        (lambda s: s.send_mfa_setup_confirmation("a@b.co"), "authentication"),
    ],
)
def test_each_message_says_what_it_is_in_the_subject(captured, send, needle):
    send(EmailService())
    assert needle in captured["subject"].lower()


# ---------------------------------------------------------------------------
# Email: a failing server is reported, not raised
# ---------------------------------------------------------------------------


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_an_unreachable_server_returns_false_rather_than_raising():
    """A signup should not 500 because the mail host is down."""
    service = EmailService(
        smtp_host="127.0.0.1",
        smtp_port=_closed_port(),
        from_email="bot@example.com",
    )
    assert service.send_password_reset("alice@example.com", "tok") is False


# ---------------------------------------------------------------------------
# smtp_security: against a real SMTP server, not a mock
# ---------------------------------------------------------------------------


class FakeSMTP(threading.Thread):
    """A minimal SMTP responder, enough for smtplib to deliver one message.

    Real sockets rather than a patched smtplib: the point is that the message
    genuinely leaves, addressed correctly, with the body intact. It speaks no
    TLS, so it is the counterpart of smtp_security="none".
    """

    daemon = True

    def __init__(self):
        super().__init__()
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.received = None
        self.mail_from = None
        self.rcpt_to = None

    def run(self):
        try:
            self._serve()
        except OSError:
            # A client that fails closed (starttls or ssl against a server
            # that speaks neither) drops the connection mid-sentence. That is
            # the behaviour under test, not an error here.
            pass

    def _serve(self):
        conn, _ = self.sock.accept()
        with conn, conn.makefile("rwb") as stream:
            stream.write(b"220 fake ESMTP\r\n")
            stream.flush()
            for raw in stream:
                line = raw.decode("utf-8", "replace").strip()
                upper = line.upper()
                if upper.startswith("EHLO"):
                    stream.write(b"250-fake\r\n250 HELP\r\n")
                elif upper.startswith("HELO"):
                    stream.write(b"250 fake\r\n")
                elif upper.startswith("MAIL FROM"):
                    self.mail_from = line
                    stream.write(b"250 OK\r\n")
                elif upper.startswith("RCPT TO"):
                    self.rcpt_to = line
                    stream.write(b"250 OK\r\n")
                elif upper == "DATA":
                    stream.write(b"354 go ahead\r\n")
                    stream.flush()
                    body = []
                    for data_line in stream:
                        if data_line in (b".\r\n", b".\n"):
                            break
                        body.append(data_line)
                    self.received = b"".join(body).decode("utf-8", "replace")
                    stream.write(b"250 queued\r\n")
                elif upper == "QUIT":
                    stream.write(b"221 bye\r\n")
                    stream.flush()
                    return
                else:
                    stream.write(b"250 OK\r\n")
                stream.flush()


@pytest.fixture
def fake_smtp():
    server = FakeSMTP()
    server.start()
    yield server
    server.sock.close()


@pytest.mark.slow
def test_an_unencrypted_relay_actually_delivers(fake_smtp):
    """smtp_security="none" is the local-relay case - postfix on this box,
    which is the ordinary self-hosted arrangement. It used to be unreachable:
    the flag claiming to allow plaintext only removed a port guard in front of
    an SSL connection, so there was no plaintext path at all."""
    service = EmailService(
        smtp_host="127.0.0.1",
        smtp_port=fake_smtp.port,
        from_email="bot@example.com",
        smtp_security="none",
    )
    assert service.send_password_reset("alice@example.com", "tok-live") is True

    fake_smtp.join(timeout=10)
    assert "alice@example.com" in fake_smtp.rcpt_to
    assert "bot@example.com" in fake_smtp.mail_from
    assert "tok-live" in fake_smtp.received


@pytest.mark.slow
def test_an_unencrypted_relay_refuses_to_send_a_password(fake_smtp):
    """No encryption plus credentials means the password crosses the wire in
    the clear. Almost always a misconfiguration, so it is refused rather than
    quietly done."""
    service = EmailService(
        smtp_host="127.0.0.1",
        smtp_port=fake_smtp.port,
        from_email="bot@example.com",
        smtp_user="bot@example.com",
        smtp_password="hunter2",
        smtp_security="none",
    )
    assert service.send_password_reset("alice@example.com", "tok") is False
    assert fake_smtp.received is None, "the message went out anyway"


@pytest.mark.slow
def test_starttls_against_a_server_that_cannot_upgrade_fails_closed(fake_smtp):
    """The fake speaks no TLS. Failing to upgrade must not fall back to
    sending in the clear."""
    service = EmailService(
        smtp_host="127.0.0.1",
        smtp_port=fake_smtp.port,
        from_email="bot@example.com",
        smtp_security="starttls",
    )
    assert service.send_password_reset("alice@example.com", "tok") is False
    assert fake_smtp.received is None, "the message was sent unencrypted"


@pytest.mark.slow
def test_ssl_against_a_plaintext_server_fails_closed(fake_smtp):
    service = EmailService(
        smtp_host="127.0.0.1",
        smtp_port=fake_smtp.port,
        from_email="bot@example.com",
        smtp_security="ssl",
    )
    assert service.send_password_reset("alice@example.com", "tok") is False
    assert fake_smtp.received is None


def test_starttls_is_the_default():
    """An operator who sets only a host gets encryption, not plaintext."""
    assert EmailService(smtp_host="h", from_email="a@b.co").smtp_security == "starttls"
