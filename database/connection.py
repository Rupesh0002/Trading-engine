"""
Database connection and initialisation helpers.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_db_connection():
    """Return an open psycopg2 connection to Neon PostgreSQL."""
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DATABASE_HOST", ""),
        port=int(os.getenv("DATABASE_PORT", "5432")),
        user=os.getenv("DATABASE_USERNAME", ""),
        password=os.getenv("DATABASE_PASSWORD", ""),
        dbname=os.getenv("DATABASE_NAME", ""),
        sslmode="require",
        connect_timeout=10,
    )
    return conn


def init_db() -> bool:
    """
    Create tables if they don't exist.
    Returns True on success, False on failure (engine continues in CSV-only mode).
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed — skipping DB init.")
        return False

    if not os.getenv("DATABASE_HOST", ""):
        logger.warning("DATABASE_HOST not set — skipping DB init.")
        return False

    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              SERIAL PRIMARY KEY,
                created_at      TIMESTAMP DEFAULT NOW(),
                index           VARCHAR(20),
                direction       VARCHAR(10),
                signal_strength VARCHAR(10),
                strike          INTEGER,
                option_type     VARCHAR(5),
                expiry          DATE,
                entry_premium   FLOAT,
                sl_premium      FLOAT,
                tp_premium      FLOAT,
                exit_premium    FLOAT,
                exit_reason     VARCHAR(20),
                lots            INTEGER,
                qty             INTEGER,
                pnl             FLOAT,
                result          VARCHAR(10),
                adx             FLOAT,
                rsi             FLOAT,
                fib_level       VARCHAR(20),
                ml_confidence   FLOAT,
                pcr_bias        VARCHAR(10),
                paper_mode      BOOLEAN DEFAULT TRUE
            );

            CREATE TABLE IF NOT EXISTS signals (
                id             SERIAL PRIMARY KEY,
                created_at     TIMESTAMP DEFAULT NOW(),
                index          VARCHAR(20),
                direction      VARCHAR(10),
                conditions_met INTEGER,
                adx            FLOAT,
                rsi            FLOAT,
                vwap_distance  FLOAT,
                fib_level      VARCHAR(20),
                ml_confidence  FLOAT,
                fired          BOOLEAN,
                skip_reason    VARCHAR(50)
            );

            CREATE TABLE IF NOT EXISTS engine_state (
                id            SERIAL PRIMARY KEY,
                updated_at    TIMESTAMP DEFAULT NOW(),
                capital       FLOAT,
                total_trades  INTEGER,
                total_pnl     FLOAT,
                open_position JSONB,
                last_run_date DATE,
                last_run_time TIME,
                eod_sent      BOOLEAN DEFAULT FALSE
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("[DB] Tables created/verified successfully.")
        return True

    except Exception as exc:
        logger.warning("DB init failed: %s — continuing in CSV-only mode.", exc)
        return False
