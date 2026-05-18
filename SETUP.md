# Trading Engine — Setup Guide

NSE F&O intraday options engine (NIFTY + BANKNIFTY).
Zerodha Kite Connect. Paper mode by default. All config in `.env`.

---

## Requirements

- Python 3.9+
- Zerodha Kite Connect API subscription (₹2,000/month from kite.trade/connect)
- Active Zerodha trading account with F&O segment enabled
- Oracle Cloud Free Tier VM or any Ubuntu 22.04 server (for 24/7 deployment)

---

## Step 1 — Install dependencies

```bash
cd /path/to/engine
pip install -r requirements.txt
```

Verify:
```bash
python -c "import kiteconnect, schedule, pandas; print('OK')"
```

---

## Step 2 — Configure `.env`

Open `.env` and fill in your values:

```env
# Broker credentials (from kite.trade/connect → your app)
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here

# Trading mode — keep True until 8 profitable paper weeks
PAPER_MODE=True

# Active indices (comma-separated, no spaces)
ACTIVE_INDICES=NIFTY,BANKNIFTY

# Capital and risk
TRADING_CAPITAL=100000
RISK_PER_TRADE_PCT=0.03      # 3% per trade

# Telegram (optional — leave blank to disable)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

**Never share or commit `.env` — it is in `.gitignore` by default.**

---

## Step 3 — Generate Zerodha access token

Zerodha tokens expire every day at midnight IST. Generate one before each session:

**Option A — Interactive (first time)**
```bash
python -c "from config.auth import get_kite_client; get_kite_client()"
```
1. Open the printed URL in your browser
2. Log in to Zerodha
3. After redirect, copy the `request_token` from the URL
4. Paste it when prompted
5. Token is saved to `config/access_token.txt`

**Option B — Automate with kite-login script (recommended for servers)**

Use the Zerodha TOTP-based login helper:
```bash
pip install pyotp
```
Then write a `generate_token.py` using your TOTP secret from Zerodha's 2FA setup.
Run this script every morning at 8:45 AM via cron (see Step 6).

---

## Step 4 — Run the backtest first

Always backtest before paper trading:

```bash
# Default period from .env (BACKTEST_START / BACKTEST_END)
python main.py --backtest

# Custom period
python main.py --backtest --from 2026-01-01 --to 2026-05-01

# Specific index
python main.py --backtest --index BANKNIFTY --from 2026-01-01 --to 2026-05-01
```

Expected output: trade log with entry/exit prices, win rate, total P&L.

---

## Step 5 — Run in paper mode

Paper mode fetches **real live data** but **simulates** order fills. No real money.

```bash
python main.py
```

The engine will:
- Sleep until market hours (9:00 AM IST weekdays)
- Run morning setup at 9:00 (fetch PCR)
- Check for signals every 15 min from 10:00 to 15:00
- Hard-close any open position at 15:00
- Print day summary at 15:30

Monitor logs:
```bash
tail -f logs/trade_log.csv      # live trades
tail -f logs/signal_log.csv     # every candle evaluation
tail -f logs/error_log.txt      # errors and warnings
```

Check today's trades:
```bash
python main.py --status
```

Full stats:
```bash
python main.py --summary
```

---

## Step 6 — Switch to live mode

**Only after at least 8 consecutive profitable paper-trading weeks.**

Open `.env` and change:
```env
PAPER_MODE=False
```

Then restart the engine. There is no CLI command for this — the change must be made manually in `.env` so it is a deliberate, conscious action.

---

## Step 7 — Deploy on Oracle Cloud Free Tier (Ubuntu 22.04)

### 7a — Provision VM

1. Sign up at cloud.oracle.com (free tier — no credit card required)
2. Create an **Ampere A1** instance (4 OCPU, 24 GB RAM — all free)
   - Image: Ubuntu 22.04
   - Shape: VM.Standard.A1.Flex
3. Download the SSH key pair during setup
4. Open port 22 in the security list (for SSH access)

### 7b — Connect and install

```bash
ssh -i your_key.pem ubuntu@<your_instance_ip>

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y

git clone https://github.com/your-repo/trading-engine.git
cd trading-engine

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 7c — Configure the server

Copy your `.env` file to the server:
```bash
scp -i your_key.pem .env ubuntu@<ip>:~/trading-engine/.env
scp -i your_key.pem config/access_token.txt ubuntu@<ip>:~/trading-engine/config/access_token.txt
```

### 7d — Automate daily token refresh

Create `/home/ubuntu/generate_token.sh`:
```bash
#!/bin/bash
cd /home/ubuntu/trading-engine
source venv/bin/activate
python generate_token.py   # your TOTP-based token script
```

Add to crontab (`crontab -e`):
```cron
# Generate fresh Zerodha token at 8:45 AM IST every weekday
45 3 * * 1-5 /home/ubuntu/generate_token.sh >> /home/ubuntu/token_refresh.log 2>&1
```

*(Server time is UTC; IST = UTC+5:30, so 8:45 IST = 3:15 UTC)*

### 7e — systemd service for 24/7 running

Create `/etc/systemd/system/trading-engine.service`:

```ini
[Unit]
Description=NSE F&O Trading Engine
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-engine
ExecStart=/home/ubuntu/trading-engine/venv/bin/python main.py
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-engine
sudo systemctl start trading-engine
```

Check status:
```bash
sudo systemctl status trading-engine
journalctl -u trading-engine -f       # live logs
```

Stop the engine:
```bash
sudo systemctl stop trading-engine
```

---

## Telegram alerts setup (optional but recommended)

1. Open Telegram → message **@BotFather** → `/newbot`
2. Name your bot and get the `BOT_TOKEN`
3. Message your bot once, then open:
   `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
4. Find `"chat":{"id": ...}` — that is your `CHAT_ID`
5. Add both to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:AAHxxxx
   TELEGRAM_CHAT_ID=987654321
   ```

You will receive alerts for: signal fired, trade entered, SL hit, target hit, daily summary.

---

## File structure

```
.
├── main.py               ← entry point (python main.py)
├── scheduler.py          ← 15-min candle loop + schedule jobs
├── telegram_alerts.py    ← Telegram notifications
├── run_backtest.py        ← direct backtest runner (alternative to main.py --backtest)
├── .env                  ← all configuration (never commit)
├── requirements.txt
├── config/
│   ├── settings.py       ← loads .env, all constants
│   ├── auth.py           ← Zerodha token handling
│   └── events_calendar.py← holidays, expiry dates
├── signals/
│   ├── engine.py         ← 5-condition AND signal logic
│   └── indicators.py     ← EMA, RSI, VWAP, Fibonacci, ADX
├── data/
│   ├── feed.py           ← historical + live candles
│   └── option_chain.py   ← LTP, PCR, strike selection
├── execution/
│   └── order.py          ← paper/live order execution with auth gate
├── risk/
│   └── manager.py        ← position sizing, daily loss limit
├── backtest/
│   └── engine.py         ← 4-month backtest simulation
├── utils/
│   ├── logger.py         ← logging setup
│   ├── trade_logger.py   ← CSV trade + signal logs
│   └── time_checks.py    ← IST time helpers
└── logs/
    ├── trade_log.csv     ← all trades
    ├── signal_log.csv    ← every candle evaluation
    └── error_log.txt     ← warnings and errors
```

---

## Common issues

| Problem | Fix |
|---|---|
| `ACCESS_TOKEN expired` | Run `python -c "from config.auth import get_kite_client; get_kite_client()"` |
| `Position size = 0` | Premium too high for risk budget. Raise `RISK_PER_TRADE_PCT` or lower `MIN_PREMIUM` |
| No signals firing | Lower `ADX_THRESHOLD` in `.env` or check `signal_log.csv` for `skip_reason` |
| Telegram not working | Verify `BOT_TOKEN` and `CHAT_ID` in `.env`; check `error_log.txt` |
| Engine crashes at startup | Check `error_log.txt`; verify `.env` has valid `KITE_API_KEY` |

---

## Security checklist

- [ ] `.env` is in `.gitignore` — verify with `git status`
- [ ] `config/access_token.txt` is in `.gitignore`
- [ ] `PAPER_MODE=True` until 8 profitable paper weeks confirmed
- [ ] Live orders only go through `executor.authorize_trade()` gate
- [ ] Server SSH key is stored securely (not in the repo)
