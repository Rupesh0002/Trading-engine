"""
Zerodha Kite Connect authentication.

Two paths:
  GitHub Actions  — KITE_ACCESS_TOKEN is set as an env var (from GitHub Secrets).
                    Used directly. No login flow, no input() calls.
  Local dev       — KITE_ACCESS_TOKEN not set. Falls back to interactive login
                    using request_token pasted from browser redirect.
"""
import os
import sys
from typing import Optional

from config.settings import KITE_API_KEY, KITE_API_SECRET, ACCESS_TOKEN_FILE


def get_kite_client():
    """Return an authenticated KiteConnect instance."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("[ERROR] kiteconnect is not installed. Run: pip install kiteconnect")
        sys.exit(1)

    if not KITE_API_KEY or KITE_API_KEY == "your_api_key_here":
        print("[ERROR] KITE_API_KEY not set. Cannot authenticate.")
        sys.exit(1)

    kite = KiteConnect(api_key=KITE_API_KEY)

    # ── GitHub Actions path: token already in environment ─────────────────
    env_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if env_token:
        kite.set_access_token(env_token)
        try:
            profile = kite.profile()
            print(f"[AUTH] Connected as: {profile['user_name']}")
            return kite
        except Exception as e:
            raise ValueError(
                f"[AUTH] KITE_ACCESS_TOKEN is invalid or expired. "
                f"Update it in GitHub Secrets. Error: {e}"
            )

    # ── Local dev path: check saved token file first ──────────────────────
    file_token = _load_token()
    if file_token:
        kite.set_access_token(file_token)
        try:
            kite.profile()
            print(f"[AUTH] Token loaded from {ACCESS_TOKEN_FILE}")
            return kite
        except Exception:
            print("[AUTH] Saved token expired. Re-authenticating...")

    # ── Local dev path: interactive login (never runs on GitHub Actions) ──
    token = _authenticate(kite)
    kite.set_access_token(token)
    return kite


def _load_token() -> Optional[str]:
    if os.path.exists(ACCESS_TOKEN_FILE):
        with open(ACCESS_TOKEN_FILE) as f:
            return f.read().strip() or None
    return None


def _save_token(token: str) -> None:
    os.makedirs(os.path.dirname(ACCESS_TOKEN_FILE), exist_ok=True)
    with open(ACCESS_TOKEN_FILE, "w") as f:
        f.write(token)


def _authenticate(kite) -> str:
    """Interactive login — local development only. Never called on GitHub Actions."""
    if not KITE_API_SECRET:
        raise ValueError("[AUTH] KITE_API_SECRET not set. Cannot authenticate.")

    login_url = kite.login_url()
    print()
    print("─" * 60)
    print("  Open this URL in your browser to log in to Zerodha:")
    print(f"  {login_url}")
    print("─" * 60)
    request_token = input("  Paste the request_token from the redirect URL: ").strip()
    data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = data["access_token"]
    _save_token(access_token)
    print("[AUTH] Authentication successful. Token saved.")
    return access_token
