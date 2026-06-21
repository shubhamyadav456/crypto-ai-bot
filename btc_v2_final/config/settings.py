# config/settings.py
import os
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV  = os.path.join(_ROOT, "config.env")
if os.path.exists(_ENV):
    load_dotenv(_ENV)

# ── Trading ─────────────────────────────────────────────────────
SYMBOL        = os.getenv("SYMBOL",    "BTCUSDT")
BASE_TF       = os.getenv("BASE_TF",   "1h")

# ── Model ────────────────────────────────────────────────────────
HORIZON       = int(os.getenv("HORIZON",        "3"))
THRESHOLD_PCT = float(os.getenv("THRESHOLD_PCT", "0.002"))
MIN_PROB_BUY  = float(os.getenv("MIN_PROB_BUY",  "0.52"))
MAX_PROB_SELL = float(os.getenv("MAX_PROB_SELL",  "0.48"))

_mp = os.getenv("MODEL_PATH", "models/saved/model_v4.pkl")
MODEL_PATH = _mp if os.path.isabs(_mp) else os.path.join(_ROOT, _mp)

# ── Risk ─────────────────────────────────────────────────────────
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "2.5"))
MIN_RR      = float(os.getenv("MIN_RR",      "1.5"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))

# ── API ──────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL  = os.getenv("API_URL", f"http://localhost:{int(os.getenv('API_PORT', '8000'))}")

# ── Telegram ─────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")
CHECK_INTERVAL   = int(os.getenv("CHECK_INTERVAL",  "30"))

# ── Binance ──────────────────────────────────────────────────────
BINANCE_SPOT    = "https://api.binance.com/api/v3"
BINANCE_FUTURES = "https://fapi.binance.com/fapi/v1"

# ── Storage ──────────────────────────────────────────────────────
_db = os.getenv("DB_PATH", "storage/trades.db")
DB_PATH = _db if os.path.isabs(_db) else os.path.join(_ROOT, _db)
