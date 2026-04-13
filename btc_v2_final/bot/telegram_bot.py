# bot/telegram_bot.py — v4.3 (news + technical combined signal)
"""
BTC AI Telegram Bot — v4.3
============================
- Technical signal from ML model
- News sentiment from CryptoPanic
- Combined signal: alert only when both agree (or strong individual)
- Commands: /start /status /news /last /help

Run:
    python bot/telegram_bot.py
"""

import os, sys, time, hashlib, logging, schedule
import requests, joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
load_dotenv(os.path.join(_ROOT, "config.env"))

from config.settings import (
    SYMBOL, BASE_TF, MODEL_PATH,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    CHECK_INTERVAL, MIN_PROB_BUY, MAX_PROB_SELL,
    ATR_SL_MULT, ATR_TP_MULT,
)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_ROOT, "btc_bot.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── State ──────────────────────────────────────────────────────
_artifact       = None
_last_direction = ""
_last_signal    = None
_sent_hashes    = set()
_chat_id        = TELEGRAM_CHAT_ID
_update_offset  = 0
_news_cache     = {}        # cache news for 30 min
_news_cache_ts  = 0

# Optional: set your CryptoPanic API key in config.env
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_API_KEY", None)


# ── Model ──────────────────────────────────────────────────────
def load_model() -> bool:
    global _artifact
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}  — run trainer first!")
        return False
    _artifact = joblib.load(MODEL_PATH)
    meta = _artifact.get("meta", {})
    log.info(f"Model loaded | AUC={meta.get('test_auc','?')} | "
             f"threshold={_artifact.get('threshold', 0.5)} | "
             f"version={meta.get('version','?')}")
    return True


# ── Technical prediction ───────────────────────────────────────
def get_technical_prediction() -> dict:
    from data.fetcher import build_raw_dataset
    from features.engineer import build_features

    df_raw  = build_raw_dataset(SYMBOL, BASE_TF, total=300)
    df_feat = build_features(df_raw, include_market_data=True)

    art  = _artifact
    sc   = art["scaler"]
    xgb  = art["xgb"]
    rf   = art["rf"]
    cal  = art["calibrator"]
    feat = art.get("all_features", art.get("features", []))
    idx  = art.get("sel_idx")
    thr  = art.get("threshold", 0.5)

    # Fill missing features with 0
    for f in feat:
        if f not in df_feat.columns:
            df_feat[f] = 0.0

    row_all = sc.transform(df_feat[feat].iloc[[-1]].fillna(0).values)
    row_sel = row_all[:, idx] if idx is not None else row_all

    raw  = (xgb.predict_proba(row_sel)[:, 1][0] +
             rf.predict_proba(row_sel)[:, 1][0]) / 2
    prob = float(cal.transform([raw])[0])

    direction = "BUY" if prob >= thr else "SELL"
    cp = prob if direction == "BUY" else 1 - prob

    if   cp >= 0.72: confidence = "HIGH"
    elif cp >= 0.62: confidence = "MEDIUM"
    elif cp >= 0.55: confidence = "LOW"
    else:            confidence = "SKIP"

    price  = float(df_feat["close"].iloc[-1])
    atr    = float(df_feat["atr_14"].iloc[-1]) if "atr_14" in df_feat.columns else price * 0.01
    rsi    = float(df_feat["rsi_14"].iloc[-1]) if "rsi_14" in df_feat.columns else 50.0
    regime = int(df_feat["market_regime"].iloc[-1]) if "market_regime" in df_feat.columns else 0
    oi_chg = float(df_feat.get("oi_change", pd.Series([0])).iloc[-1]) if "oi_change" in df_feat.columns else 0.0
    fr     = float(df_feat.get("funding_rate", pd.Series([0])).iloc[-1]) if "funding_rate" in df_feat.columns else 0.0

    sl = round(price - ATR_SL_MULT * atr, 2) if direction == "BUY" else round(price + ATR_SL_MULT * atr, 2)
    tp = round(price + ATR_TP_MULT * atr, 2) if direction == "BUY" else round(price - ATR_TP_MULT * atr, 2)
    rr = round(ATR_TP_MULT / ATR_SL_MULT, 2)

    regime_str = {1:"Uptrend", -1:"Downtrend", 0:"Sideways"}.get(regime, "?")

    return {
        "direction":    direction,
        "prob":         round(prob, 4),
        "confidence":   confidence,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "rr":           rr,
        "atr":          round(atr, 2),
        "rsi":          round(rsi, 1),
        "regime":       regime_str,
        "regime_int":   regime,
        "oi_change":    round(oi_chg * 100, 2),
        "funding_rate": round(fr * 100, 4),
        "threshold":    thr,
    }


# ── News sentiment (with 30-min cache) ────────────────────────
def get_news(force: bool = False) -> dict:
    global _news_cache, _news_cache_ts
    now = time.time()
    if not force and _news_cache and (now - _news_cache_ts) < 1800:
        return _news_cache

    try:
        from data.news import get_news_features, get_news_signal
        features = get_news_features("BTC", api_key=CRYPTOPANIC_KEY)
        signal   = get_news_signal(features)
        _news_cache    = {**features, **signal}
        _news_cache_ts = now
        return _news_cache
    except Exception as e:
        log.warning(f"News fetch failed: {e}")
        return {
            "signal": "NEUTRAL", "strength": 0, "score": 0,
            "sent_6h": 0, "sent_24h": 0, "fg": 50,
            "reasons": ["News unavailable"],
        }


# ── Combined signal logic ──────────────────────────────────────
def should_alert(tech: dict, news: dict) -> tuple:
    """
    Returns (alert: bool, reason: str, quality: str)

    Alert logic:
      HIGH quality  — strong tech signal AND news agrees
      MEDIUM quality — strong tech signal, news neutral
      Skip          — tech weak OR tech + news conflict
    """
    direction  = tech["direction"]
    confidence = tech["confidence"]
    regime     = tech["regime_int"]
    news_sig   = news.get("signal", "NEUTRAL")
    news_str   = news.get("strength", 0)

    # Must be trending
    if regime == 0:
        return False, "Sideways market", "SKIP"

    # Prob threshold
    prob = tech["prob"]
    if direction == "BUY"  and prob < MIN_PROB_BUY:
        return False, f"Prob too low ({prob:.2f})", "SKIP"
    if direction == "SELL" and (1-prob) < (1-MAX_PROB_SELL):
        return False, f"Prob too low ({1-prob:.2f})", "SKIP"

    # Confidence gate
    if confidence == "SKIP":
        return False, "Confidence SKIP", "SKIP"

    # News alignment check
    news_aligned = (
        (direction == "BUY"  and news_sig == "BULLISH") or
        (direction == "SELL" and news_sig == "BEARISH") or
        news_sig == "NEUTRAL"
    )
    news_conflict = (
        (direction == "BUY"  and news_sig == "BEARISH" and news_str > 0.5) or
        (direction == "SELL" and news_sig == "BULLISH" and news_str > 0.5)
    )

    if news_conflict:
        return False, f"News conflict: tech={direction}, news={news_sig}", "SKIP"

    # Quality
    if confidence == "HIGH" and news_sig in ("BULLISH","BEARISH") and news_aligned:
        quality = "A"
    elif confidence in ("HIGH","MEDIUM"):
        quality = "B"
    else:
        if confidence == "LOW":
            return False, "Confidence LOW", "SKIP"
        quality = "C"

    reason = f"Tech={direction}({confidence}), News={news_sig}"
    return True, reason, quality


# ── Telegram helpers ───────────────────────────────────────────
def _tg(method: str, **kwargs) -> dict:
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
        json=kwargs, timeout=10)
    r.raise_for_status()
    return r.json()

def send_msg(text: str, chat_id: str = None) -> bool:
    cid = chat_id or _chat_id
    if not cid:
        return False
    try:
        _tg("sendMessage", chat_id=cid, text=text, parse_mode="Markdown")
        return True
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


# ── Message formatters ─────────────────────────────────────────
def format_signal(tech: dict, news: dict, quality: str) -> str:
    d    = tech["direction"]
    prob = tech["prob"] if d == "BUY" else 1 - tech["prob"]
    conf = tech["confidence"]
    now  = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    grade = {"A": "[A]", "B": "[B]", "C": "[C]"}.get(quality, "[B]")
    arrow = "BUY  [LONG]" if d == "BUY" else "SELL [SHORT]"

    news_line = ""
    ns = news.get("signal","NEUTRAL")
    if ns != "NEUTRAL":
        emoji = "+" if ns == "BULLISH" else "-"
        news_line = f"News   : `{ns} ({emoji}{news.get('sent_6h',0):.1f})`\n"

    oi_line = ""
    if abs(tech.get("oi_change", 0)) > 0.5:
        oi_line = f"OI chg : `{tech['oi_change']:+.1f}%`\n"

    fr_line = ""
    if abs(tech.get("funding_rate", 0)) > 0.001:
        fr_line = f"Funding: `{tech['funding_rate']:+.4f}%`\n"

    return (
        f"*{SYMBOL} Signal* {grade}\n"
        f"_{now}_\n\n"
        f"*Direction : {arrow}*\n"
        f"Probability: `{prob*100:.1f}%`\n"
        f"Confidence : `{conf}`\n\n"
        f"Entry  : `${tech['price']:,.2f}`\n"
        f"Stop L : `${tech['sl']:,.2f}`\n"
        f"Take P : `${tech['tp']:,.2f}`\n"
        f"R:R    : `1 : {tech['rr']}`\n\n"
        f"RSI    : `{tech['rsi']}`\n"
        f"Market : `{tech['regime']}`\n"
        f"ATR    : `${tech['atr']:,.2f}`\n"
        f"{oi_line}{fr_line}{news_line}\n"
        f"_Technical + sentiment analysis. Not financial advice._"
    )


def format_status(tech: dict, news: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    ns  = news.get("signal", "NEUTRAL")
    fg  = news.get("fg", 50)
    return (
        f"*{SYMBOL} Status* _{now}_\n\n"
        f"Price   : `${tech['price']:,.2f}`\n"
        f"Signal  : `{tech['direction']}` ({tech['confidence']})\n"
        f"Prob    : `{tech['prob']*100:.1f}%`\n"
        f"Market  : `{tech['regime']}`\n"
        f"RSI     : `{tech['rsi']}`\n\n"
        f"News    : `{ns}` (6h: {news.get('sent_6h',0):.1f})\n"
        f"F&G     : `{fg}` — {news.get('fg_class','Neutral')}\n"
        f"OI chg  : `{tech.get('oi_change',0):+.2f}%`\n"
        f"Funding : `{tech.get('funding_rate',0):+.4f}%`"
    )


def format_news(news: dict) -> str:
    now     = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    ns      = news.get("signal", "NEUTRAL")
    reasons = news.get("reasons", [])
    reason_text = "\n".join(f"  • {r}" for r in reasons) if reasons else "  • No specific drivers"
    return (
        f"*BTC News Sentiment* _{now}_\n\n"
        f"Signal  : `{ns}` (score: {news.get('score',0):+d})\n"
        f"6h sent : `{news.get('sent_6h',0):.2f}`\n"
        f"24h sent: `{news.get('sent_24h',0):.2f}`\n"
        f"Articles: `{int(news.get('news_count_6h',0))}` in last 6h\n"
        f"F&G     : `{news.get('fg',50)}` — {news.get('fg_class','Neutral')}\n\n"
        f"*Drivers:*\n{reason_text}"
    )


# ── Command polling ────────────────────────────────────────────
def poll_commands():
    global _update_offset, _chat_id
    if not TELEGRAM_TOKEN:
        return
    try:
        data = _tg("getUpdates", offset=_update_offset, timeout=5)
        for upd in data.get("result", []):
            _update_offset = upd["update_id"] + 1
            msg  = upd.get("message", {})
            text = msg.get("text", "").strip()
            cid  = str(msg.get("chat", {}).get("id", ""))
            if not cid or not text:
                continue
            if not _chat_id:
                _chat_id = cid
                log.info(f"Chat ID saved: {_chat_id}")
            cmd = text.split()[0].lower()
            if cmd == "/start":
                send_msg(
                    "✅ *BTC AI Bot v4.3 active!*\n\n"
                    "You'll receive trading signals combining:\n"
                    "• ML model (technical)\n"
                    "• News sentiment\n"
                    "• Open Interest\n\n"
                    "Type /help for commands.", cid)
            elif cmd == "/status":
                try:
                    tech = get_technical_prediction()
                    news = get_news()
                    send_msg(format_status(tech, news), cid)
                except Exception as e:
                    send_msg(f"❌ Error: {e}", cid)
            elif cmd == "/news":
                try:
                    news = get_news(force=True)
                    send_msg(format_news(news), cid)
                except Exception as e:
                    send_msg(f"❌ News error: {e}", cid)
            elif cmd == "/last":
                if _last_signal:
                    send_msg(format_signal(*_last_signal), cid)
                else:
                    send_msg("No signal sent yet this session.", cid)
            elif cmd == "/help":
                send_msg(
                    "*BTC AI Bot Commands*\n\n"
                    "/start — subscribe to alerts\n"
                    "/status — current prediction + sentiment\n"
                    "/news — detailed news sentiment\n"
                    "/last — last signal sent\n"
                    "/help — this message", cid)
    except Exception as e:
        log.warning(f"Poll error: {e}")


# ── Main run ───────────────────────────────────────────────────
def run():
    global _last_direction, _last_signal, _chat_id

    if not _chat_id:
        log.warning("No chat ID — send /start to your bot first")

    log.info("Getting prediction...")
    try:
        tech = get_technical_prediction()
        news = get_news()

        alert, reason, quality = should_alert(tech, news)

        log.info(
            f"Tech={tech['direction']}({tech['confidence']}) "
            f"prob={tech['prob']:.3f} regime={tech['regime']} | "
            f"News={news.get('signal','?')}({news.get('sent_6h',0):.1f}) | "
            f"Alert={alert} Q={quality}"
        )

        if not alert:
            log.info(f"Filtered: {reason}")
            return

        # Dedup
        d = tech["direction"]
        sig_hash = hashlib.md5(
            f"{d}:{quality}:{round(tech['price'],-2)}".encode()
        ).hexdigest()[:8]

        if sig_hash in _sent_hashes:
            log.info("Duplicate — skip")
            return
        if d == _last_direction:
            log.info(f"Same direction ({d}) — skip")
            return

        msg = format_signal(tech, news, quality)
        if send_msg(msg):
            _last_direction = d
            _last_signal    = (tech, news, quality)
            _sent_hashes.add(sig_hash)
            if len(_sent_hashes) > 100:
                _sent_hashes.pop()
            log.info(f"Signal sent! {d} quality={quality}")

    except Exception as e:
        log.error(f"Run error: {e}")
        import traceback; traceback.print_exc()


# ── Entry ──────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("BTC AI Bot v4.3 starting...")
    log.info(f"  Symbol   : {SYMBOL}")
    log.info(f"  Interval : {CHECK_INTERVAL} min")
    log.info(f"  Model    : {MODEL_PATH}")
    log.info(f"  News     : CryptoPanic ({'API key set' if CRYPTOPANIC_KEY else 'free tier'})")

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN missing in config.env!")
        sys.exit(1)
    if not load_model():
        sys.exit(1)

    run()

    schedule.every(CHECK_INTERVAL).minutes.do(run)
    schedule.every(1).minutes.do(poll_commands)

    log.info(f"Scheduler running — every {CHECK_INTERVAL} min")
    while True:
        schedule.run_pending()
        time.sleep(15)
