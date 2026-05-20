#!/usr/bin/env python3
"""
Daily Kite token refresh — run this every morning before 09:15 IST.

Usage:
  python get_token.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("ERROR: kiteconnect not installed. Run: pip install kiteconnect")
        sys.exit(1)

    api_key    = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()

    if not api_key or api_key == "your_api_key_here":
        print("ERROR: KITE_API_KEY not set in .env")
        sys.exit(1)
    if not api_secret:
        print("ERROR: KITE_API_SECRET not set in .env")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)

    # ── Step 1: Login URL ─────────────────────────────────────────────────────
    login_url = kite.login_url()
    print()
    print("─" * 65)
    print("  Open this URL in your browser and log in to Zerodha:")
    print()
    print(f"  {login_url}")
    print()
    print("  After login you'll be redirected to a URL like:")
    print("  https://127.0.0.1/?request_token=XXXXXXXXXX&action=login&status=success")
    print("─" * 65)

    try:
        import webbrowser
        webbrowser.open(login_url)
        print("  (Browser opened automatically)")
    except Exception:
        pass

    # ── Step 2: Get request_token ─────────────────────────────────────────────
    print()
    request_token = input("  Paste the request_token from the redirect URL: ").strip()
    if not request_token:
        print("ERROR: No request_token provided.")
        sys.exit(1)

    # ── Step 3: Exchange for access_token ─────────────────────────────────────
    try:
        session   = kite.generate_session(request_token, api_secret=api_secret)
        token     = session["access_token"]
        user_name = session.get("user_name", "")
        user_id   = session.get("user_id", "")
    except Exception as exc:
        print(f"ERROR: Failed to generate session — {exc}")
        sys.exit(1)

    # ── Step 4: Save locally ──────────────────────────────────────────────────
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "access_token.txt")
    with open(token_file, "w") as f:
        f.write(token)

    print()
    print(f"  ✓ Token generated for {user_name} ({user_id})")
    print(f"  ✓ Saved to config/access_token.txt")
    print()
    print("  Now update GitHub Secret manually:")
    print("  GitHub → Repo → Settings → Secrets → Actions → KITE_ACCESS_TOKEN → Update")
    print()
    print(f"  Token: {token}")
    print("─" * 65)
    print()


if __name__ == "__main__":
    main()
