"""
Standalone DB connection test — run via GitHub Actions or locally.
Usage: python test_db.py
"""
import os
import sys

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

host   = os.getenv("DATABASE_HOST",     "")
port   = os.getenv("DATABASE_PORT",     "5432")
user   = os.getenv("DATABASE_USERNAME", "")
dbname = os.getenv("DATABASE_NAME",     "")

print(f"Connecting to: {host}:{port}/{dbname} (user={user})")

try:
    from database.connection import get_db_connection, init_db

    # ── 1. Verify raw connection ──────────────────────────────────────────────
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT version();")
    print("Server version:", cur.fetchone()[0])
    cur.close()
    conn.close()
    print("Connection OK.")

    # ── 2. Create tables if missing ───────────────────────────────────────────
    print("Running init_db()...")
    ok = init_db()
    print("init_db() returned:", ok)

    # ── 3. Verify tables now exist ────────────────────────────────────────────
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    print("Tables in DB:", tables if tables else "(none)")

    expected = {"trades", "signals", "engine_state"}
    missing  = expected - set(tables)
    if missing:
        print(f"ERROR: missing tables: {missing}")
        cur.close()
        conn.close()
        sys.exit(1)

    print("All required tables present.")
    cur.close()
    conn.close()

except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
