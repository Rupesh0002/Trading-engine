#!/usr/bin/env python3
"""
Daily Kite token refresh — run this every morning before 09:15 IST.

Steps:
  1. Opens your Zerodha login URL (in browser or copy-paste)
  2. You log in and paste the request_token from the redirect URL
  3. Script exchanges it for an access_token
  4. Saves token to config/access_token.txt  (local use)
  5. Updates KITE_ACCESS_TOKEN in GitHub Secrets (if `gh` CLI is installed)

Usage:
  python get_token.py
"""
import os
import subprocess
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
    print("  STEP 1 — Open this URL in your browser and log in to Zerodha:")
    print()
    print(f"  {login_url}")
    print()
    print("  After login you'll be redirected to a URL like:")
    print("  https://127.0.0.1/?request_token=XXXXXXXXXX&action=login&status=success")
    print("─" * 65)

    # Try to auto-open browser
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

    print()
    print(f"  ✓ Token generated for {user_name} ({user_id})")
    print(f"  Access token: {token}")

    # ── Step 4: Save locally ──────────────────────────────────────────────────
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "access_token.txt")
    os.makedirs(os.path.dirname(token_file), exist_ok=True)
    with open(token_file, "w") as f:
        f.write(token)
    print(f"  ✓ Saved to {token_file}")

    # ── Step 5: Update GitHub Secret ─────────────────────────────────────────
    print()
    gh_available = _gh_installed()
    if gh_available:
        try:
            subprocess.run(
                ["gh", "secret", "set", "KITE_ACCESS_TOKEN", "--body", token],
                check=True,
                capture_output=True,
                text=True,
            )
            print("  ✓ KITE_ACCESS_TOKEN updated in GitHub Secrets automatically.")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ GitHub secret update failed: {e.stderr.strip()}")
            print(f"    Run manually: gh secret set KITE_ACCESS_TOKEN --body \"{token}\"")
    else:
        print("  ─ `gh` CLI not found. Update GitHub Secret manually:")
        print(f"    gh secret set KITE_ACCESS_TOKEN --body \"{token}\"")
        print()
        print("  Or: GitHub → Repo → Settings → Secrets → Actions → KITE_ACCESS_TOKEN")

    print()
    print("  Done. Engine will use the new token on the next candle run.")
    print("─" * 65)
    print()


def _gh_installed() -> bool:
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    main()
