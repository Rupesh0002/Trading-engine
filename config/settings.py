from dotenv import load_dotenv
import os

load_dotenv()

# ── Broker credentials ───────────────────────────────────────────────────────
KITE_API_KEY         = os.getenv("KITE_API_KEY")
KITE_API_SECRET      = os.getenv("KITE_API_SECRET")

# ── Trading mode ─────────────────────────────────────────────────────────────
PAPER_MODE           = os.getenv("PAPER_MODE", "True") == "True"

# ── Instruments ──────────────────────────────────────────────────────────────
ACTIVE_INDEX         = os.getenv("ACTIVE_INDEX", "NIFTY")
ACTIVE_INDICES       = [x.strip() for x in os.getenv("ACTIVE_INDICES", "NIFTY,BANKNIFTY,SENSEX").split(",") if x.strip()]
CANDLE_INTERVAL      = os.getenv("CANDLE_INTERVAL", "15minute")
CANDLE_MINUTES       = int(os.getenv("CANDLE_MINUTES", 15))

# ── Session times (IST, 24hr format) ─────────────────────────────────────────
MARKET_OPEN          = os.getenv("MARKET_OPEN", "09:15")
TRADE_START          = os.getenv("TRADE_START", "09:30")
SIGNAL_START         = os.getenv("SIGNAL_START", "10:00")
TRADE_END            = os.getenv("TRADE_END", "15:00")
MARKET_CLOSE         = os.getenv("MARKET_CLOSE", "15:30")

# ── Capital and risk ─────────────────────────────────────────────────────────
TRADING_CAPITAL      = float(os.getenv("TRADING_CAPITAL", 100000))
RISK_PER_TRADE_PCT   = float(os.getenv("RISK_PER_TRADE_PCT", 0.02))
STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT", 0.03))
TARGET_PCT           = float(os.getenv("TARGET_PCT", 0.09))
TRAILING_SL_PCT      = float(os.getenv("TRAILING_SL_PCT", 0.03))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0.05))
MAX_OPEN_POSITIONS   = int(os.getenv("MAX_OPEN_POSITIONS", 1))   # single trade at a time
MIN_PREMIUM          = float(os.getenv("MIN_PREMIUM", 30))
MAX_PREMIUM          = float(os.getenv("MAX_PREMIUM", 500))

# ── Flexible profit exit ──────────────────────────────────────────────────────
MIN_PROFIT_RATIO          = float(os.getenv("MIN_PROFIT_RATIO", 2.5))
MAX_PROFIT_RATIO          = float(os.getenv("MAX_PROFIT_RATIO", 3.0))
WEAK_SIGNAL_TARGET_RATIO  = float(os.getenv("WEAK_SIGNAL_TARGET_RATIO", 1.5))  # 4/5 signal target
PROFIT_LOCK_TIME          = os.getenv("PROFIT_LOCK_TIME", "14:00")              # close profitable trades after this

# ── Signal conditions ────────────────────────────────────────────────────────
MIN_CONDITIONS       = int(os.getenv("MIN_CONDITIONS", 4))
EMA_FAST_PERIOD      = int(os.getenv("EMA_FAST_PERIOD", 9))   # 9 × 15 min = 135 min
EMA_SLOW_PERIOD      = int(os.getenv("EMA_SLOW_PERIOD", 21))  # 21 × 15 min = 315 min (full session)
RSI_PERIOD           = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERSOLD         = float(os.getenv("RSI_OVERSOLD", 45))
RSI_OVERBOUGHT       = float(os.getenv("RSI_OVERBOUGHT", 55))
VOLUME_SPIKE_MULT    = float(os.getenv("VOLUME_SPIKE_MULT", 1.5))
ADX_PERIOD           = int(os.getenv("ADX_PERIOD", 14))
ADX_THRESHOLD        = float(os.getenv("ADX_THRESHOLD", 20.0))
FIB_PROXIMITY_PCT    = float(os.getenv("FIB_PROXIMITY_PCT", 0.10))   # kept for backcompat (unused by signal engine)
FIB_MIN_SWING_PCT    = float(os.getenv("FIB_MIN_SWING_PCT", 0.50))   # kept for backcompat (unused by signal engine)
PDH_PROXIMITY_PCT    = float(os.getenv("PDH_PROXIMITY_PCT", 0.005))  # 0.50% of price — entry zone around PDH/PDL
PDH_MIN_RANGE_PCT    = float(os.getenv("PDH_MIN_RANGE_PCT", 0.005))  # skip levels if previous day range < 0.50%
MAX_DAILY_TRADES     = int(os.getenv("MAX_DAILY_TRADES", 1))
MAX_DAILY_TRADES_OTM = int(os.getenv("MAX_DAILY_TRADES_OTM", 2))  # hard cap; 2nd trade only for 5/5 signals
VWAP_ZONE_PCT        = float(os.getenv("VWAP_ZONE_PCT", 0.20))  # % of price; replaces fixed VWAP_ZONE_POINTS

# ── Market filters ───────────────────────────────────────────────────────────
PCR_MIN              = float(os.getenv("PCR_MIN", 0.6))   # kept for backcompat
PCR_MAX              = float(os.getenv("PCR_MAX", 1.5))   # kept for backcompat
PCR_BULL_MIN         = float(os.getenv("PCR_BULL_MIN", 1.1))  # PCR above this → fear → CALL confirming
PCR_BEAR_MAX         = float(os.getenv("PCR_BEAR_MAX", 0.7))  # PCR below this → greed → PUT confirming
VIX_MAX              = float(os.getenv("VIX_MAX", 20.0))

# ── Strike selection ─────────────────────────────────────────────────────────
STRIKE_TYPE               = os.getenv("STRIKE_TYPE", "ATM")
PRODUCT_TYPE              = os.getenv("PRODUCT_TYPE", "MIS")
STRONG_SIGNAL_THRESHOLD   = int(os.getenv("STRONG_SIGNAL_THRESHOLD", 5))
OTM_STRIKES_AWAY          = int(os.getenv("OTM_STRIKES_AWAY", 1))
ENABLE_OPTION_SELLING     = os.getenv("ENABLE_OPTION_SELLING", "False") == "True"
WEAK_SIGNAL_THRESHOLD     = int(os.getenv("WEAK_SIGNAL_THRESHOLD", 3))
NIFTY_LOT_SIZE       = int(os.getenv("NIFTY_LOT_SIZE", 50))
BANKNIFTY_LOT_SIZE   = int(os.getenv("BANKNIFTY_LOT_SIZE", 15))
SENSEX_LOT_SIZE      = int(os.getenv("SENSEX_LOT_SIZE", 10))
NIFTY_STRIKE_STEP    = int(os.getenv("NIFTY_STRIKE_STEP", 50))
BANKNIFTY_STRIKE_STEP= int(os.getenv("BANKNIFTY_STRIKE_STEP", 100))
SENSEX_STRIKE_STEP   = int(os.getenv("SENSEX_STRIKE_STEP", 100))

# ── Database (PostgreSQL) ─────────────────────────────────────────────────────
DB_TYPE              = "postgresql"                            # fixed, not user-overridable
PG_HOST              = os.getenv("DATABASE_HOST",     "localhost")
PG_PORT              = int(os.getenv("DATABASE_PORT", 5432))
PG_USER              = os.getenv("DATABASE_USERNAME", "postgres")
PG_PASSWORD          = os.getenv("DATABASE_PASSWORD", "")
PG_DB                = os.getenv("DATABASE_NAME",     "trading_engine")

# ── Reports ──────────────────────────────────────────────────────────────────
REPORTS_DIR          = os.getenv("REPORTS_DIR", "reports")

# ── ML model (Phase 6) ───────────────────────────────────────────────────────
ML_MIN_CONFIDENCE    = float(os.getenv("ML_MIN_CONFIDENCE", 0.65))
ML_MODEL_PATH        = os.getenv("ML_MODEL_PATH", "ml/models/xgboost_model.pkl")

# ── Logging and dashboard ────────────────────────────────────────────────────
LOG_LEVEL            = os.getenv("LOG_LEVEL", "INFO")
DASHBOARD_PORT       = int(os.getenv("DASHBOARD_PORT", 8501))
DASHBOARD_REFRESH    = int(os.getenv("DASHBOARD_REFRESH", 5))

# ── Backtest ─────────────────────────────────────────────────────────────────
BACKTEST_START       = os.getenv("BACKTEST_START", "2023-01-01")
BACKTEST_END         = os.getenv("BACKTEST_END", "2024-12-31")
BACKTEST_VIX         = float(os.getenv("BACKTEST_VIX", 15.0))  # assumed VIX for premium simulation

# ── Derived values ────────────────────────────────────────────────────────────
RISK_REWARD_RATIO    = TARGET_PCT / STOP_LOSS_PCT
SOFT_TARGET_PCT      = STOP_LOSS_PCT * MIN_PROFIT_RATIO   # e.g. 7.5% at 2.5×
HARD_TARGET_PCT      = STOP_LOSS_PCT * MAX_PROFIT_RATIO   # e.g. 9.0% at 3.0×

ACCESS_TOKEN_FILE    = "config/access_token.txt"
TRADE_LOG_FILE       = "logs/trade_log.csv"
SIGNAL_LOG_FILE      = "logs/signal_log.csv"
ERROR_LOG_FILE       = "logs/error_log.txt"

# Exchange constants
NSE_EXCHANGE         = "NSE"
BSE_EXCHANGE         = "BSE"
NFO_EXCHANGE         = "NFO"    # NSE F&O (NIFTY, BANKNIFTY options)
BFO_EXCHANGE         = "BFO"    # BSE F&O (SENSEX options)

# ── Per-index config (exchange-defined + .env lot sizes) ──────────────────────
# Instrument tokens: verify against Kite instruments CSV each series.
# SENSEX token: fetch via kite.instruments("BSE") and search for "SENSEX".
INDEX_CONFIG = {
    "NIFTY": {
        "lot_size":       NIFTY_LOT_SIZE,
        "strike_step":    NIFTY_STRIKE_STEP,
        "token":          256265,       # NSE:NIFTY 50
        "spot_exchange":  NSE_EXCHANGE,
        "fno_exchange":   NFO_EXCHANGE,
        "expiry_day":     "thursday",
        "label":          "NIFTY 50",
        "adx_threshold":  float(os.getenv("NIFTY_ADX_THRESHOLD", 25)),   # stricter — NIFTY needs stronger trend
    },
    "BANKNIFTY": {
        "lot_size":       BANKNIFTY_LOT_SIZE,
        "strike_step":    BANKNIFTY_STRIKE_STEP,
        "token":          260105,       # NSE:BANKNIFTY
        "spot_exchange":  NSE_EXCHANGE,
        "fno_exchange":   NFO_EXCHANGE,
        "expiry_day":     "wednesday",
        "label":          "Bank Nifty",
        "adx_threshold":  float(os.getenv("BANKNIFTY_ADX_THRESHOLD", 20)),
    },
    "SENSEX": {
        "lot_size":       SENSEX_LOT_SIZE,
        "strike_step":    SENSEX_STRIKE_STEP,
        "token":          265,          # BSE:SENSEX — verify from instruments CSV
        "spot_exchange":  BSE_EXCHANGE,
        "fno_exchange":   BFO_EXCHANGE,
        "expiry_day":     "friday",
        "label":          "BSE Sensex",
        "adx_threshold":  float(os.getenv("SENSEX_ADX_THRESHOLD", 20)),
    },
}

LOT_SIZE    = INDEX_CONFIG.get(ACTIVE_INDEX, INDEX_CONFIG["NIFTY"])["lot_size"]
STRIKE_STEP = INDEX_CONFIG.get(ACTIVE_INDEX, INDEX_CONFIG["NIFTY"])["strike_step"]


# ── Validation ────────────────────────────────────────────────────────────────
def validate_settings() -> bool:
    warnings = []

    if KITE_API_KEY in {"your_api_key_here", "", None}:
        warnings.append("KITE_API_KEY is not set. Add your real key in .env.")
    if KITE_API_SECRET in {"your_api_secret_here", "", None}:
        warnings.append("KITE_API_SECRET is not set. Add your real secret in .env.")
    if TRADING_CAPITAL <= 10_000:
        warnings.append(f"TRADING_CAPITAL={TRADING_CAPITAL} too low. Min recommended: 10,000.")
    if not (0.005 <= RISK_PER_TRADE_PCT <= 0.05):
        warnings.append(f"RISK_PER_TRADE_PCT={RISK_PER_TRADE_PCT} outside safe range [0.005–0.05].")
    if STOP_LOSS_PCT >= TARGET_PCT:
        warnings.append(f"STOP_LOSS_PCT ({STOP_LOSS_PCT}) must be less than TARGET_PCT ({TARGET_PCT}).")
    if STOP_LOSS_PCT > 0 and (TARGET_PCT / STOP_LOSS_PCT) < 1.5:
        warnings.append(f"R:R ratio {TARGET_PCT/STOP_LOSS_PCT:.2f} below minimum 1:1.5.")
    if MAX_OPEN_POSITIONS != 1:
        warnings.append(f"MAX_OPEN_POSITIONS={MAX_OPEN_POSITIONS}. Set to 1 for single-trade mode.")
    if MIN_PROFIT_RATIO < 1.5:
        warnings.append(f"MIN_PROFIT_RATIO={MIN_PROFIT_RATIO} too low. Minimum 1.5× recommended.")
    unknown = [i for i in ACTIVE_INDICES if i not in INDEX_CONFIG]
    if unknown:
        warnings.append(f"Unknown indices: {unknown}. Valid: {list(INDEX_CONFIG.keys())}")
    if ENABLE_OPTION_SELLING and TRADING_CAPITAL < 200_000:
        warnings.append(
            f"ENABLE_OPTION_SELLING=True but TRADING_CAPITAL=₹{TRADING_CAPITAL:,.0f}. "
            "Option writing requires ~₹45K–₹1.1L margin per lot. "
            "Recommended capital for selling: ₹2,00,000+. "
            "NIFTY selling is NOT feasible below ₹1,50,000."
        )
    if ENABLE_OPTION_SELLING and WEAK_SIGNAL_THRESHOLD >= MIN_CONDITIONS:
        warnings.append(
            f"WEAK_SIGNAL_THRESHOLD ({WEAK_SIGNAL_THRESHOLD}) must be less than "
            f"MIN_CONDITIONS ({MIN_CONDITIONS}) for sell trades to fire."
        )

    print()
    print("─" * 60)
    print("  SETTINGS VALIDATION")
    print("─" * 60)
    if warnings:
        for w in warnings:
            print(f"  [WARNING] {w}")
    else:
        print("  [OK] Settings valid")
    print()
    print(f"  Indices        : {', '.join(ACTIVE_INDICES)}")
    print(f"  Mode           : {'PAPER' if PAPER_MODE else 'LIVE  ← REAL MONEY'}")
    print(f"  Trade limit    : {MAX_OPEN_POSITIONS} trade at a time")
    print(f"  Capital        : ₹{TRADING_CAPITAL:,.0f}")
    print(f"  Risk/trade     : {RISK_PER_TRADE_PCT*100:.1f}%  (₹{TRADING_CAPITAL*RISK_PER_TRADE_PCT:,.0f})")
    print(f"  Stop Loss      : {STOP_LOSS_PCT*100:.1f}%")
    print(f"  Exit (2.5×)    : {SOFT_TARGET_PCT*100:.1f}%  → GOOD trade")
    print(f"  Exit (3.0×)    : {HARD_TARGET_PCT*100:.1f}%  → EXCELLENT trade")
    print(f"  Risk:Reward    : 1:{RISK_REWARD_RATIO:.1f}")
    print(f"  Min Conditions : {MIN_CONDITIONS}/5")
    print(f"  Signal window  : {SIGNAL_START} – {TRADE_END} IST")
    print(f"  Strong signal  : {STRONG_SIGNAL_THRESHOLD}/5 → OTM +{OTM_STRIKES_AWAY} strike")
    print(f"  Option selling : {'ENABLED (weak signal ≥' + str(WEAK_SIGNAL_THRESHOLD) + '/5)' if ENABLE_OPTION_SELLING else 'DISABLED'}")
    print(f"  Database       : PostgreSQL  ({PG_HOST}:{PG_PORT}/{PG_DB})")
    print("─" * 60)
    print()
    return len(warnings) == 0
