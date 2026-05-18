"""
Database manager — PostgreSQL.
All connection parameters come from config/settings.py → .env (DATABASE_* vars).
If PostgreSQL is unavailable, operations are no-ops and trades are stored in CSV only.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config.settings import (
    PG_DB,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_USER,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    PostgreSQL wrapper for trade storage.
    Usage:
        db = DatabaseManager()
        db.save_trade(trade_doc)
        db.get_trades({"index_name": "NIFTY"})
    """

    def __init__(self) -> None:
        self._conn = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                dbname=PG_DB,
            )
            self._conn.autocommit = False
            self._ensure_schema()
            logger.info(
                "PostgreSQL connected: %s:%s/%s", PG_HOST, PG_PORT, PG_DB
            )
        except ImportError:
            logger.warning(
                "psycopg2 not installed. Run: pip install psycopg2-binary"
            )
            self._conn = None
        except Exception as exc:
            logger.warning(
                "PostgreSQL connection failed (%s:%s/%s): %s — CSV-only mode.",
                PG_HOST, PG_PORT, PG_DB, exc,
            )
            self._conn = None

    def _ensure_schema(self) -> None:
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
        """
        with self._conn.cursor() as cur:
            cur.execute(ddl)
        self._conn.commit()
        logger.debug("PostgreSQL schema verified.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_trade(self, doc: Dict[str, Any]) -> Optional[str]:
        """
        Insert or update a trade document.
        Called twice per trade: once at entry (exit fields NULL),
        once at exit (all fields filled).
        Returns trade_id on success, None on failure.
        """
        if not self.is_connected():
            return None
        try:
            return self._upsert(doc)
        except Exception as exc:
            logger.error("PostgreSQL save_trade error: %s", exc)
            return None

    def get_trades(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch trades matching filters dict.
        Example: get_trades({"index_name": "NIFTY", "trade_quality": "GOOD"})
        """
        if not self.is_connected():
            return []
        try:
            import psycopg2.extras
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                where, vals = "", []
                if filters:
                    clauses = [f"{k} = %s" for k in filters]
                    where = "WHERE " + " AND ".join(clauses)
                    vals  = list(filters.values())
                cur.execute(
                    f"SELECT * FROM trades {where} ORDER BY entry_time DESC LIMIT %s",
                    vals + [limit],
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            logger.error("PostgreSQL get_trades error: %s", exc)
            return []

    def get_daily_summary(self, date_str: str) -> Dict[str, Any]:
        """Returns aggregated P&L stats for a given date (YYYY-MM-DD)."""
        if not self.is_connected():
            return {}
        try:
            import psycopg2.extras
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)                                        AS total_trades,
                        SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN pnl_amount <= 0 THEN 1 ELSE 0 END) AS losses,
                        COALESCE(SUM(pnl_amount), 0)                    AS total_pnl,
                        COALESCE(AVG(risk_reward_achieved), 0)          AS avg_rr
                    FROM trades
                    WHERE trade_date = %s AND exit_time IS NOT NULL
                    """,
                    (date_str,),
                )
                row = cur.fetchone()
                return dict(row) if row else {}
        except Exception as exc:
            logger.error("PostgreSQL daily summary error: %s", exc)
            return {}

    def is_connected(self) -> bool:
        if self._conn is None:
            return False
        try:
            if self._conn.closed:
                self._connect()
            return self._conn is not None and not self._conn.closed
        except Exception:
            return False

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upsert(self, doc: Dict[str, Any]) -> str:
        ctx   = doc.get("market_context", {})
        strat = doc.get("strategy", {})

        sql = """
        INSERT INTO trades (
            trade_id, index_name, trade_date, entry_time, exit_time,
            direction, entry_type, symbol, strike, option_type, expiry,
            lot_size, lots, quantity,
            entry_premium, exit_premium, spot_at_entry, spot_at_exit,
            stop_loss, target_soft, target_hard,
            pnl_per_unit, pnl_amount, pnl_pct, risk_reward_achieved,
            exit_reason, trade_quality, capital_deployed, risk_amount,
            conditions_met, india_vix, pcr, paper_mode, strategy_details,
            updated_at
        ) VALUES (
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT (trade_id) DO UPDATE SET
            exit_time            = EXCLUDED.exit_time,
            exit_premium         = EXCLUDED.exit_premium,
            spot_at_exit         = EXCLUDED.spot_at_exit,
            pnl_per_unit         = EXCLUDED.pnl_per_unit,
            pnl_amount           = EXCLUDED.pnl_amount,
            pnl_pct              = EXCLUDED.pnl_pct,
            risk_reward_achieved = EXCLUDED.risk_reward_achieved,
            exit_reason          = EXCLUDED.exit_reason,
            trade_quality        = EXCLUDED.trade_quality,
            updated_at           = CURRENT_TIMESTAMP
        """

        def _dt(v):
            if v is None:
                return None
            s = str(v)
            for sep in ("+", "Z"):
                if sep in s:
                    s = s.split(sep)[0]
            return s.replace("T", " ")[:19]

        vals = (
            doc["trade_id"],
            doc["index"],
            doc["date"],
            _dt(doc["entry_time"]),
            _dt(doc.get("exit_time")),
            doc["direction"],
            doc.get("entry_type", "BUY"),
            doc["symbol"],
            doc.get("strike"),
            doc.get("option_type"),
            doc.get("expiry"),
            doc.get("lot_size"),
            doc.get("lots"),
            doc.get("quantity"),
            doc.get("entry_premium"),
            doc.get("exit_premium"),
            doc.get("spot_at_entry"),
            doc.get("spot_at_exit"),
            doc.get("stop_loss"),
            doc.get("target_soft"),
            doc.get("target_hard"),
            doc.get("pnl_per_unit"),
            doc.get("pnl_amount"),
            doc.get("pnl_pct"),
            doc.get("risk_reward_achieved"),
            doc.get("exit_reason"),
            doc.get("trade_quality"),
            doc.get("capital_deployed"),
            doc.get("risk_amount"),
            strat.get("conditions_met"),
            ctx.get("india_vix"),
            ctx.get("pcr"),
            bool(doc.get("paper_mode", True)),
            json.dumps(strat),
        )

        with self._conn.cursor() as cur:
            cur.execute(sql, vals)
        self._conn.commit()
        logger.debug("PostgreSQL upsert: %s", doc["trade_id"])
        return doc["trade_id"]
