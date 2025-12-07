from __future__ import annotations

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from liminallm.logging import get_logger

logger = get_logger(__name__)


class EmailService:
    """Email service for sending transactional emails.

    Supports:
    - SMTP with TLS/SSL
    - Password reset emails
    - Email verification emails
    - Fallback to logging when not configured (dev mode)
    """

    def __init__(
        self,
        *,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_use_tls: bool = True,
        from_email: Optional[str] = None,
        from_name: str = "LiminalLM",
        base_url: Optional[str] = None,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.smtp_use_tls = smtp_use_tls
        self.from_email = from_email or smtp_user
        self.from_name = from_name
        self.base_url = base_url or "http://localhost:8000"

    @property
    def is_configured(self) -> bool:
        """Check if email sending is properly configured."""
        return bool(self.smtp_host and self.from_email)

    def _redact_email(self, email: str) -> str:
        """Redact an email address for logging to avoid PII leakage."""
        if "@" not in email:
            return "redacted"
        local, domain = email.split("@", 1)
        return f"{local[:2]}***@{domain}"

    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
    ) -> bool:
        """Send an email via SMTP.

        Returns True if sent successfully, False otherwise.
        """
        if not self.is_configured:
            # Dev mode: log the email instead of sending
            logger.info(
                "email_dev_mode",
                to=self._redact_email(to_email),
                subject=subject,
                body_preview=text_body[:200] if text_body else html_body[:200],
            )
            return True

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email

            # Add text and HTML parts
            if text_body:
                msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Connect and send
            context = ssl.create_default_context()

            logger.debug(
                "email_connecting",
                host=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.smtp_use_tls,
                to=self._redact_email(to_email),
            )

            if self.smtp_use_tls:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.starttls(context=context)
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.from_email, to_email, msg.as_string())
            else:
                with smtplib.SMTP_SSL(
                    self.smtp_host, self.smtp_port, context=context, timeout=30
                ) as server:
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.from_email, to_email, msg.as_string())

            logger.info("email_sent", to=self._redact_email(to_email), subject=subject)
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(
                "email_auth_failed",
                to=self._redact_email(to_email),
                host=self.smtp_host,
                user=self.smtp_user,
                error=str(e),
                error_code=e.smtp_code if hasattr(e, "smtp_code") else None,
            )
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(
                "email_connect_failed",
                to=self._redact_email(to_email),
                host=self.smtp_host,
                port=self.smtp_port,
                error=str(e),
            )
            return False
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(
                "email_recipient_refused",
                to=self._redact_email(to_email),
                error=str(e),
                refused=list(e.recipients.keys()) if hasattr(e, "recipients") else None,
            )
            return False
        except smtplib.SMTPSenderRefused as e:
            logger.error(
                "email_sender_refused",
                to=self._redact_email(to_email),
                sender=self.from_email,
                error=str(e),
            )
            return False
        except smtplib.SMTPException as e:
            logger.error(
                "email_smtp_error",
                to=self._redact_email(to_email),
                host=self.smtp_host,
                error_type=type(e).__name__,
                error=str(e),
            )
            return False
        except ssl.SSLError as e:
            logger.error(
                "email_ssl_error",
                to=self._redact_email(to_email),
                host=self.smtp_host,
                port=self.smtp_port,
                error=str(e),
            )
            return False
        except TimeoutError as e:
            logger.error(
                "email_timeout",
                to=self._redact_email(to_email),
                host=self.smtp_host,
                port=self.smtp_port,
                error=str(e),
            )
            return False
        except Exception as e:
            logger.error(
                "email_send_failed",
                to=self._redact_email(to_email),
                error_type=type(e).__name__,
                error=str(e),
            )
            return False

    def send_password_reset(self, to_email: str, token: str) -> bool:
        """Send password reset email with reset link."""
        reset_url = f"{self.base_url}/?reset_token={token}"

        subject = "Reset your LiminalLM password"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2933; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 40px 20px; }}
        .button {{ display: inline-block; background: #10a37f; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; }}
        .footer {{ margin-top: 40px; font-size: 12px; color: #5b6470; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Reset your password</h1>
        <p>We received a request to reset your password. Click the button below to choose a new password:</p>
        <p style="margin: 30px 0;">
            <a href="{reset_url}" class="button">Reset Password</a>
        </p>
        <p>This link will expire in 15 minutes.</p>
        <p>If you didn't request this, you can safely ignore this email.</p>
        <div class="footer">
            <p>LiminalLM</p>
            <p>If the button doesn't work, copy and paste this URL: {reset_url}</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""Reset your LiminalLM password

We received a request to reset your password. Visit the link below to choose a new password:

{reset_url}

This link will expire in 15 minutes.

If you didn't request this, you can safely ignore this email.

---
LiminalLM
"""

        return self._send_email(to_email, subject, html_body, text_body)

    def send_email_verification(self, to_email: str, token: str) -> bool:
        """Send email verification link."""
        verify_url = f"{self.base_url}/?verify_token={token}"

        subject = "Verify your LiminalLM email"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2933; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 40px 20px; }}
        .button {{ display: inline-block; background: #10a37f; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; }}
        .footer {{ margin-top: 40px; font-size: 12px; color: #5b6470; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Verify your email</h1>
        <p>Thanks for signing up! Please verify your email address by clicking the button below:</p>
        <p style="margin: 30px 0;">
            <a href="{verify_url}" class="button">Verify Email</a>
        </p>
        <p>This link will expire in 24 hours.</p>
        <div class="footer">
            <p>LiminalLM</p>
            <p>If the button doesn't work, copy and paste this URL: {verify_url}</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""Verify your LiminalLM email

Thanks for signing up! Please verify your email address by visiting the link below:

{verify_url}

This link will expire in 24 hours.

---
LiminalLM
"""

        return self._send_email(to_email, subject, html_body, text_body)

    def send_mfa_setup_confirmation(self, to_email: str) -> bool:
        """Send confirmation that MFA was enabled."""
        subject = "Two-factor authentication enabled"

        html_body = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2933; }
        .container { max-width: 600px; margin: 0 auto; padding: 40px 20px; }
        .footer { margin-top: 40px; font-size: 12px; color: #5b6470; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Two-factor authentication enabled</h1>
        <p>Two-factor authentication has been successfully enabled on your LiminalLM account.</p>
        <p>You will now need to enter a code from your authenticator app when signing in.</p>
        <p>If you didn't make this change, please contact support immediately.</p>
        <div class="footer">
            <p>LiminalLM</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = """Two-factor authentication enabled

Two-factor authentication has been successfully enabled on your LiminalLM account.

You will now need to enter a code from your authenticator app when signing in.

If you didn't make this change, please contact support immediately.

---
LiminalLM
"""

        return self._send_email(to_email, subject, html_body, text_body)
