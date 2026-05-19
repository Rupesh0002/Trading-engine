"""
Zerodha Kite Connect authentication.

GitHub Actions: KITE_ACCESS_TOKEN is set as env var from GitHub Secrets.
                Used directly — no login flow, no input() ever.
Local dev:      Falls back to saved token file, then interactive login.
"""
import os
import sys
from typing import Optional

from kiteconnect import KiteConnect

from config.settings import KITE_API_KEY, KITE_API_SECRET, ACCESS_TOKEN_FILE

# True when running inside GitHub Actions
_ON_CI = os.getenv("GITHUB_ACTIONS") == "true"


def get_kite_client() -> KiteConnect:
    """
    Returns an authenticated KiteConnect instance.

    GitHub Actions path  — KITE_ACCESS_TOKEN env var must be set (GitHub Secret).
    Local dev path       — tries saved token file, then interactive browser login.
    """
    api_key      = os.getenv("KITE_API_KEY") or KITE_API_KEY
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    api_secret   = os.getenv("KITE_API_SECRET") or KITE_API_SECRET

    if not api_key or api_key == "your_api_key_here":
        raise ValueError("[AUTH] KITE_API_KEY not found in environment")

    kite = KiteConnect(api_key=api_key)

    # ── Path 1: token already in environment (GitHub Actions) ─────────────
    if access_token:
        kite.set_access_token(access_token)
        try:
            profile = kite.profile()
            print(f"[AUTH] Connected: {profile['user_name']} ({profile['user_id']})")
            return kite
        except Exception as e:
            raise ValueError(
                f"[AUTH] KITE_ACCESS_TOKEN is expired or invalid.\n"
                f"       Go to GitHub → Repo → Settings → Secrets → update KITE_ACCESS_TOKEN.\n"
                f"       Error: {e}"
            )

    # ── Path 2: GitHub Actions but token not set ───────────────────────────
    if _ON_CI:
        raise ValueError(
            "[AUTH] Running on GitHub Actions but KITE_ACCESS_TOKEN is not set.\n"
            "       Add it under: Repo → Settings → Secrets and variables → Actions."
        )

    # ── Path 3: local dev — try saved token file ───────────────────────────
    file_token = _load_token()
    if file_token:
        kite.set_access_token(file_token)
        try:
            kite.profile()
            print(f"[AUTH] Token loaded from {ACCESS_TOKEN_FILE}")
            return kite
        except Exception:
            print("[AUTH] Saved token expired — re-authenticating...")

    # ── Path 4: local dev — interactive login (never runs on CI) ──────────
    return _interactive_login(kite, api_secret)


def _load_token() -> Optional[str]:
    if os.path.exists(ACCESS_TOKEN_FILE):
        with open(ACCESS_TOKEN_FILE) as f:
            return f.read().strip() or None
    return None


def _save_token(token: str) -> None:
    os.makedirs(os.path.dirname(ACCESS_TOKEN_FILE), exist_ok=True)
    with open(ACCESS_TOKEN_FILE, "w") as f:
        f.write(token)


def _interactive_login(kite: KiteConnect, api_secret: str) -> KiteConnect:
    """Browser login — local development only. Blocked on GitHub Actions by Path 2."""
    if not api_secret:
        raise ValueError("[AUTH] KITE_API_SECRET not set. Cannot authenticate.")

    is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
    if is_github_actions:
        raise ValueError(
            "[AUTH] KITE_ACCESS_TOKEN is not set in GitHub Secrets.\n"
            "Go to: GitHub repo → Settings → Secrets → Actions\n"
            "→ Update KITE_ACCESS_TOKEN with today's token"
        )

    print()
    print("─" * 60)
    print("  Open this URL in your browser to log in to Zerodha:")
    print(f"  {kite.login_url()}")
    print("─" * 60)
    request_token = input("  Paste the request_token from the redirect URL: ").strip()
    data = kite.generate_session(request_token, api_secret=api_secret)
    token = data["access_token"]
    _save_token(token)
    kite.set_access_token(token)
    print("[AUTH] Login successful. Token saved.")
    return kite
