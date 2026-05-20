"""
Database initialisation helper.
Called once at the start of each candle run to ensure all tables exist.
Safe to call multiple times — all DDL uses CREATE TABLE IF NOT EXISTS.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_db() -> bool:
    """
    Connect to Neon PostgreSQL and create tables if they don't exist.
    Returns True on success, False on failure (engine continues in CSV-only mode).
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed — skipping DB init.")
        return False

    host     = os.getenv("DATABASE_HOST",     "")
    port     = os.getenv("DATABASE_PORT",     "5432")
    user     = os.getenv("DATABASE_USERNAME", "")
    password = os.getenv("DATABASE_PASSWORD", "")
    dbname   = os.getenv("DATABASE_NAME",     "")

    if not host:
        logger.warning("DATABASE_HOST not set — skipping DB init.")
        return False

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
        conn.autocommit = False

        ddl = """
        CREATE TABLE IF NOT EXISTS trades (
            id                    SERIAL PRIMARY KEY,
            trade_id              VARCHAR(50)     UNIQUE NOT NULL,
            index_name            VARCHAR(20)     NOT NULL,
            trade_date            DATE            NOT NULL,
            entry_time            TIMESTAMP       NOT NULL,
            exit_time             TIMESTAMP,
            direction             VARCHAR(10)     NOT NULL,
            entry_type            VARCHAR(10)     NOT NULL DEFAULT 'BUY',
            symbol                VARCHAR(60)     NOT NULL,
            strike                INTEGER,
            option_type           VARCHAR(5),
            expiry                DATE,
            lot_size              INTEGER,
            lots                  INTEGER,
            quantity              INTEGER,
            entry_premium         NUMERIC(10,2),
            exit_premium          NUMERIC(10,2),
            spot_at_entry         NUMERIC(10,2),
            spot_at_exit          NUMERIC(10,2),
            stop_loss             NUMERIC(10,2),
            target_soft           NUMERIC(10,2),
            target_hard           NUMERIC(10,2),
            pnl_per_unit          NUMERIC(10,2),
            pnl_amount            NUMERIC(10,2),
            pnl_pct               NUMERIC(8,4),
            risk_reward_achieved  NUMERIC(8,4),
            exit_reason           VARCHAR(100),
            trade_quality         VARCHAR(20),
            capital_deployed      NUMERIC(10,2),
            risk_amount           NUMERIC(10,2),
            conditions_met        INTEGER,
            india_vix             NUMERIC(8,4),
            pcr                   NUMERIC(8,4),
            paper_mode            BOOLEAN         DEFAULT TRUE,
            strategy_details      JSONB,
            created_at            TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
            updated_at            TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS signals (
            id              SERIAL PRIMARY KEY,
            created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
            index_name      VARCHAR(20),
            direction       VARCHAR(10),
            conditions_met  INTEGER,
            adx             NUMERIC(8,4),
            rsi             NUMERIC(8,4),
            vwap_distance   NUMERIC(10,2),
            fib_level       VARCHAR(20),
            ml_confidence   NUMERIC(8,4),
            fired           BOOLEAN,
            skip_reason     VARCHAR(100)
        );

        CREATE TABLE IF NOT EXISTS engine_state (
            id              SERIAL PRIMARY KEY,
            updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
            capital         NUMERIC(12,2),
            total_trades    INTEGER,
            total_pnl       NUMERIC(12,2),
            open_position   JSONB,
            last_run_date   DATE,
            last_run_time   TIME
        );
        """

        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
        conn.close()
        logger.info("DB init OK — tables verified on %s/%s", host, dbname)
        return True

    except Exception as exc:
        logger.warning("DB init failed: %s — continuing in CSV-only mode.", exc)
        return False
