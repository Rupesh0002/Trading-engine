"""
Order execution layer.
PAPER_MODE=True  → simulate fills, log only, zero real API calls.
PAPER_MODE=False → live orders via Zerodha Kite Connect.

Live orders are gated by explicit authorization:
  executor.authorize_trade(trade_id, index)   ← must be called immediately before entry/exit
  executor.place_buy_order(...)               ← raises PermissionError if not authorized
Authorization is consumed (revoked) after ONE order, or on any error.
No other code path can place a live order.
"""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import INDEX_CONFIG, PAPER_MODE, PRODUCT_TYPE

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Places buy/sell orders in paper or live mode.

    Live-order flow (PAPER_MODE=False):
        executor.authorize_trade(trade_id, index)  # unlock for one order
        result = executor.place_buy_order(...)      # executes + auto-revokes
    """

    def __init__(self, kite=None) -> None:
        self.kite = kite
        self._paper_counter = 0
        self._paper_orders: list[dict] = []

        # Authorization state — live orders are locked by default
        self._authorized_trade_id: Optional[str] = None
        self._authorized_index: Optional[str] = None

    # ------------------------------------------------------------------
    # Authorization gate — MUST be called before every live order
    # ------------------------------------------------------------------

    def authorize_trade(self, trade_id: str, index: str) -> None:
        """
        Unlocks ONE live order for the given trade_id and index.
        Authorization is single-use: consumed immediately after the order.
        Call this in main.py immediately before place_buy_order / place_sell_order.
        """
        if not trade_id or not index:
            raise ValueError("authorize_trade requires a non-empty trade_id and index.")
        if index not in INDEX_CONFIG:
            raise ValueError(f"Unknown index '{index}'. Valid: {list(INDEX_CONFIG.keys())}")
        self._authorized_trade_id = trade_id
        self._authorized_index    = index
        logger.debug("Live order authorized: trade_id=%s index=%s", trade_id, index)

    def revoke_authorization(self) -> None:
        """Explicitly revoke pending authorization (e.g. after an error before placement)."""
        self._authorized_trade_id = None
        self._authorized_index    = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place_buy_order(
        self,
        symbol: str,
        quantity: int,
        index: str = "NIFTY",
        order_type: str = "MARKET",
        price: float = 0.0,
    ) -> dict:
        if PAPER_MODE:
            return self._paper_fill(symbol, quantity, price, side="BUY", index=index)
        self._assert_authorized()
        return self._live_order(symbol, quantity, order_type, price, side="BUY", index=index)

    def place_sell_order(
        self,
        symbol: str,
        quantity: int,
        index: str = "NIFTY",
        order_type: str = "MARKET",
        price: float = 0.0,
    ) -> dict:
        if PAPER_MODE:
            return self._paper_fill(symbol, quantity, price, side="SELL", index=index)
        self._assert_authorized()
        return self._live_order(symbol, quantity, order_type, price, side="SELL", index=index)

    def get_paper_orders(self) -> list[dict]:
        return list(self._paper_orders)

    # ------------------------------------------------------------------
    # Authorization check
    # ------------------------------------------------------------------

    def _assert_authorized(self) -> None:
        if self._authorized_trade_id is None:
            raise PermissionError(
                "Live order blocked: no active trade authorization. "
                "Call executor.authorize_trade(trade_id, index) immediately before placing an order. "
                "Set PAPER_MODE=True in .env to disable live trading entirely."
            )

    # ------------------------------------------------------------------
    # Paper execution
    # ------------------------------------------------------------------

    def _paper_fill(
        self, symbol: str, quantity: int, price: float, side: str, index: str
    ) -> dict:
        self._paper_counter += 1
        order_id    = f"PAPER-{self._paper_counter:05d}"
        fno_exchange = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["fno_exchange"]
        order = {
            "order_id":  order_id,
            "symbol":    symbol,
            "quantity":  quantity,
            "price":     price,
            "side":      side,
            "status":    "COMPLETE",
            "paper":     True,
            "exchange":  fno_exchange,
        }
        self._paper_orders.append(order)
        logger.info(
            "[PAPER] %s %s  qty=%d  price=%.2f  order_id=%s  exchange=%s",
            side, symbol, quantity, price, order_id, fno_exchange,
        )
        return order

    # ------------------------------------------------------------------
    # Live execution via Kite Connect
    # ------------------------------------------------------------------

    def _live_order(
        self,
        symbol: str,
        quantity: int,
        order_type: str,
        price: float,
        side: str,
        index: str,
    ) -> dict:
        if self.kite is None:
            self.revoke_authorization()
            raise RuntimeError(
                "Kite client not initialised. "
                "Pass a kite instance or set PAPER_MODE=True in .env."
            )

        fno_exchange = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["fno_exchange"]

        transaction = (
            self.kite.TRANSACTION_TYPE_BUY
            if side == "BUY"
            else self.kite.TRANSACTION_TYPE_SELL
        )
        product = (
            self.kite.PRODUCT_MIS
            if PRODUCT_TYPE == "MIS"
            else self.kite.PRODUCT_NRML
        )
        kite_order_type = (
            self.kite.ORDER_TYPE_MARKET
            if order_type == "MARKET"
            else self.kite.ORDER_TYPE_LIMIT
        )

        params: dict = {
            "tradingsymbol":    symbol,
            "exchange":         fno_exchange,
            "transaction_type": transaction,
            "quantity":         quantity,
            "product":          product,
            "order_type":       kite_order_type,
        }
        if order_type == "LIMIT" and price > 0:
            params["price"] = price

        # Consume authorization before the API call — one order, one auth token
        authorized_trade_id = self._authorized_trade_id
        self.revoke_authorization()

        try:
            order_id = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, **params)
        except Exception as exc:
            logger.error(
                "[LIVE] Order failed: %s %s  trade_id=%s  error=%s",
                side, symbol, authorized_trade_id, exc,
            )
            raise

        logger.info(
            "[LIVE] %s %s  qty=%d  order_id=%s  exchange=%s  trade_id=%s",
            side, symbol, quantity, order_id, fno_exchange, authorized_trade_id,
        )
        return {
            "order_id":  order_id,
            "symbol":    symbol,
            "quantity":  quantity,
            "price":     price,
            "side":      side,
            "status":    "PLACED",
            "paper":     False,
            "exchange":  fno_exchange,
            "trade_id":  authorized_trade_id,
        }
