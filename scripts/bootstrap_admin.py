#!/usr/bin/env python3
"""Bootstrap an admin user for testing and initial setup.

Usage:
    # Using environment variables:
    ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD=SecurePassword123! python scripts/bootstrap_admin.py

    # Or with command line args:
    python scripts/bootstrap_admin.py --email admin@example.com --password SecurePassword123!

Environment Variables:
    ADMIN_EMAIL: Email for the admin user
    ADMIN_PASSWORD: Password for the admin user (must meet complexity requirements)
    DATABASE_URL: PostgreSQL connection string (optional, uses memory store if not set)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def validate_password(password: str) -> bool:
    """Check password meets complexity requirements."""
    if len(password) < 12:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password)
    return sum([has_upper, has_lower, has_digit, has_special]) >= 3


async def bootstrap_admin(email: str, password: str, dry_run: bool = False) -> dict:
    """Create or update an admin user.

    Returns:
        dict with user_id, email, and status ('created' or 'updated')
    """
    # Import here to avoid loading config before env vars are set
    from liminallm.service.runtime import get_runtime

    runtime = get_runtime()

    # Check if user already exists
    existing_user = runtime.store.get_user_by_email(email)

    if existing_user:
        if existing_user.role == "admin":
            print(f"User {email} already exists as admin (id: {existing_user.id})")
            return {
                "user_id": existing_user.id,
                "email": email,
                "status": "already_admin",
            }

        if dry_run:
            print(f"[DRY RUN] Would promote existing user {email} to admin")
            return {"user_id": existing_user.id, "email": email, "status": "dry_run"}

        # Promote existing user to admin
        runtime.store.update_user_role(existing_user.id, "admin")
        print(f"Promoted existing user {email} to admin (id: {existing_user.id})")
        return {
            "user_id": existing_user.id,
            "email": email,
            "status": "promoted",
        }

    if dry_run:
        print(f"[DRY RUN] Would create admin user: {email}")
        return {"user_id": None, "email": email, "status": "dry_run"}

    # Create new admin user
    user, session, tokens = await runtime.auth.signup(email, password)
    runtime.store.update_user_role(user.id, "admin")

    print(f"Created admin user: {email} (id: {user.id})")
    return {
        "user_id": user.id,
        "email": email,
        "status": "created",
        "access_token": tokens.get("access_token"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap an admin user for LiminalLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("ADMIN_EMAIL"),
        help="Admin email (or set ADMIN_EMAIL env var)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("ADMIN_PASSWORD"),
        help="Admin password (or set ADMIN_PASSWORD env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if not args.email:
        print("Error: --email or ADMIN_EMAIL environment variable required")
        sys.exit(1)

    if not args.password:
        print("Error: --password or ADMIN_PASSWORD environment variable required")
        sys.exit(1)

    if not validate_password(args.password):
        print("Error: Password must be at least 12 characters with 3+ character classes")
        print("       (uppercase, lowercase, digits, special characters)")
        sys.exit(1)

    # Set up minimal env for testing if not configured
    if not os.environ.get("JWT_SECRET"):
        # Generate a secure secret for bootstrap
        import secrets
        os.environ["JWT_SECRET"] = secrets.token_urlsafe(48)

    if not os.environ.get("SHARED_FS_ROOT"):
        os.environ["SHARED_FS_ROOT"] = "/tmp/liminallm-bootstrap"

    # Use memory store if no database configured
    if not os.environ.get("DATABASE_URL"):
        os.environ["USE_MEMORY_STORE"] = "true"
        print("Note: Using in-memory store (set DATABASE_URL for persistence)")

    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")

    try:
        result = asyncio.run(bootstrap_admin(args.email, args.password, args.dry_run))

        if result["status"] == "created":
            print("\nAdmin user created successfully!")
            print(f"  Email: {result['email']}")
            print(f"  User ID: {result['user_id']}")
            if result.get("access_token"):
                print(f"  Access Token: {result['access_token'][:50]}...")
        elif result["status"] == "promoted":
            print("\nExisting user promoted to admin!")
        elif result["status"] == "already_admin":
            print("\nNo changes needed - user is already an admin.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
