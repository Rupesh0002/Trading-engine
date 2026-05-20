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

host     = os.getenv("DATABASE_HOST",     "")
port     = os.getenv("DATABASE_PORT",     "5432")
user     = os.getenv("DATABASE_USERNAME", "")
password = os.getenv("DATABASE_PASSWORD", "")
dbname   = os.getenv("DATABASE_NAME",     "")

print(f"Connecting to: {host}:{port}/{dbname} (user={user})")

try:
    conn = psycopg2.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        dbname=dbname,
        sslmode="require",
        connect_timeout=10,
    )
    print("DB Connected successfully")

    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("Server version:", cur.fetchone()[0])

    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    print("Tables in DB:", tables if tables else "(none yet)")

    conn.close()
    print("Connection closed cleanly.")

except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
