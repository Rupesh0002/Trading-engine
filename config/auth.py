"""
Zerodha Kite Connect authentication.
Handles login flow and access token persistence.
All credentials are read from .env via settings.py.
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
        print("[ERROR] KITE_API_KEY not set in .env. Cannot authenticate.")
        sys.exit(1)

    kite = KiteConnect(api_key=KITE_API_KEY)
    token = _load_token()

    if token:
        kite.set_access_token(token)
        try:
            kite.profile()  # verify token is still valid
            print(f"[AUTH] Token loaded from {ACCESS_TOKEN_FILE}")
            return kite
        except Exception:
            print("[AUTH] Saved token expired. Re-authenticating...")

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
