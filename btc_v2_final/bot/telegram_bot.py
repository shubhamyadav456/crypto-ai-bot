# bot/telegram_bot.py — v4.4 (High Confidence Strategy)
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
from features.signal_filter import score_signal, SignalScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_ROOT, "btc_bot.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

CRYPTOPANIC_KEY     = os.getenv("CRYPTOPANIC_API_KEY", "")
PROB_BUY_THR        = float(os.getenv("MIN_PROB_BUY",  "0.70"))
PROB_SELL_THR       = float(os.getenv("MAX_PROB_SELL", "0.30"))
MIN_SCORE           = int(os.getenv("MIN_SIGNAL_SCORE", "4"))
MAX_TRADES_PER_DAY  = int(os.getenv("MAX_TRADES_PER_DAY", "2"))

_artifact           = None
_last_signal        = None
_sent_hashes        = set()
_chat_id            = TELEGRAM_CHAT_ID
_update_offset      = 0
_today_trade_count  = 0
_today_date         = ""
_news_cache         = {}
_news_cache_ts      = 0


def load_model() -> bool:
    global _artifact
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        return False
    _artifact = joblib.load(MODEL_PATH)
    meta = _artifact.get("meta", {})
    log.info(f"Model loaded | AUC={meta.get('test_auc','?')} | "
             f"threshold={_artifact.get('threshold',0.5)} | "
             f"version={meta.get('version','?')}")
    return True


def get_prediction() -> tuple:
    """Returns (SignalScore, raw_features_dict)"""
    from data.fetcher import build_raw_dataset
    from features.engineer import build_features, _ema, _rsi, _bb, _atr

    df_raw  = build_raw_dataset(SYMBOL, BASE_TF, total=300)
    df_feat = build_features(df_raw, include_market_data=True)

    art     = _artifact
    sc      = art["scaler"]
    xgb     = art["xgb"]
    rf      = art["rf"]
    cal     = art["calibrator"]
    feats   = art.get("all_features", art.get("features", []))
    idx     = art.get("sel_idx")

    for f in feats:
        if f not in df_feat.columns:
            df_feat[f] = 0.0

    row_all = sc.transform(df_feat[feats].iloc[[-1]].fillna(0).values)
    row_sel = row_all[:, idx] if idx is not None else row_all
    raw     = (xgb.predict_proba(row_sel)[:,1][0] + rf.predict_proba(row_sel)[:,1][0]) / 2
    prob    = float(cal.transform([raw])[0])

    direction = "BUY" if prob >= PROB_BUY_THR else ("SELL" if prob <= PROB_SELL_THR else None)

    # Indicators for scoring
    raw_ri = df_raw.reset_index(drop=True)
    cs = raw_ri["close"]
    hs = raw_ri["high"]
    ls = raw_ri["low"]
    vs = raw_ri["volume"]

    price   = float(cs.iloc[-1])
    e50     = float(_ema(cs, 50).iloc[-1])
    e200    = float(_ema(cs, 200).iloc[-1])
    rsi14   = float(_rsi(cs, 14).iloc[-1])
    at14    = float(_atr(hs, ls, cs, 14).iloc[-1])
    bu, bm, bl, bp, bw = _bb(cs, 20, 2)
    bw_ma   = float(bw.rolling(20).mean().iloc[-1])
    vm20    = float(vs.rolling(20).mean().iloc[-1])
    vol_r   = float(vs.iloc[-1] / (vm20 + 1e-9))
    fg_val  = float(df_feat.get("fg_value", pd.Series([50])).iloc[-1])
    oi_chg  = float(df_feat.get("oi_change", pd.Series([0])).iloc[-1]) if "oi_change" in df_feat.columns else 0
    fr      = float(df_feat.get("funding_rate", pd.Series([0])).iloc[-1]) if "funding_rate" in df_feat.columns else 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if direction is None:
        # Not strong enough — still score for /status
        direction = "BUY" if prob >= 0.5 else "SELL"

    sig = score_signal(
        direction    = direction,
        prob         = prob,
        price        = price,
        ema50        = e50,
        ema200       = e200,
        atr          = at14,
        rsi          = rsi14,
        bb_upper     = float(bu.iloc[-1]),
        bb_lower     = float(bl.iloc[-1]),
        bb_width     = float(bw.iloc[-1]),
        bb_width_ma  = bw_ma,
        fg_value     = fg_val,
        vol_ratio    = vol_r,
        prob_buy_thr = PROB_BUY_THR,
        prob_sell_thr= PROB_SELL_THR,
        min_score    = MIN_SCORE,
        sl_atr_mult  = ATR_SL_MULT,
        tp_atr_mult  = ATR_TP_MULT,
        timestamp    = now,
    )

    extras = {"oi_change": round(oi_chg*100,2), "funding_rate": round(fr*100,4)}
    return sig, extras


def get_news(force=False) -> dict:
    global _news_cache, _news_cache_ts
    now = time.time()
    if not force and _news_cache and (now - _news_cache_ts) < 1800:
        return _news_cache
    try:
        from data.news import get_news_features, get_news_signal
        key = CRYPTOPANIC_KEY if len(CRYPTOPANIC_KEY) > 20 else None
        features = get_news_features("BTC", api_key=key)
        signal   = get_news_signal(features)
        _news_cache = {**features, **signal}
        _news_cache_ts = now
        return _news_cache
    except Exception as e:
        log.warning(f"News error: {e}")
        return {"signal":"NEUTRAL","strength":0,"sent_6h":0,"fg":50,"reasons":[]}


def _tg(method, **kwargs):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
        json=kwargs, timeout=10)
    r.raise_for_status()
    return r.json()

def send_msg(text, chat_id=None):
    cid = chat_id or _chat_id
    if not cid: return False
    try:
        _tg("sendMessage", chat_id=cid, text=text, parse_mode="Markdown")
        return True
    except Exception as e:
        log.error(f"Send failed: {e}")
        return False


def format_signal(sig: SignalScore, extras: dict, news: dict) -> str:
    d    = sig.direction
    prob = sig.prob if d == "BUY" else 1 - sig.prob
    grade = "A" if sig.score >= 6 else "B" if sig.score >= 4 else "C"
    arrow = "BUY  [LONG]" if d == "BUY" else "SELL [SHORT]"
    now   = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    # Filter breakdown (only show passed/failed counts)
    passed_filters = [k.split("_",1)[1] for k, (ok,_) in sig.filters.items() if ok]
    failed_filters = [k.split("_",1)[1] for k, (ok,_) in sig.filters.items() if not ok]

    ns = news.get("signal","NEUTRAL")
    news_line = f"News   : `{ns}` (6h: {news.get('sent_6h',0):.1f})\n" if ns != "NEUTRAL" else ""
    oi_line   = f"OI chg : `{extras.get('oi_change',0):+.1f}%`\n" if abs(extras.get('oi_change',0)) > 0.5 else ""
    fr_line   = f"Funding: `{extras.get('funding_rate',0):+.4f}%`\n" if abs(extras.get('funding_rate',0)) > 0.001 else ""

    return (
        f"*{SYMBOL} Signal* [{grade}] Score:{sig.score}/7\n"
        f"_{now}_\n\n"
        f"*Direction : {arrow}*\n"
        f"Probability: `{prob*100:.1f}%`\n"
        f"Score      : `{sig.score}/7` ({'HIGH' if sig.score>=6 else 'GOOD' if sig.score>=4 else 'LOW'})\n\n"
        f"Entry  : `${sig.entry:,.2f}`\n"
        f"Stop L : `${sig.sl:,.2f}`\n"
        f"Take P : `${sig.tp:,.2f}`\n"
        f"R:R    : `1:{sig.rr}`\n\n"
        f"RSI    : `{sig.rsi}`\n"
        f"Regime : `{sig.regime}`\n"
        f"ATR%   : `{sig.atr/sig.entry*100:.2f}%`\n"
        f"F&G    : `{sig.fg_value:.0f}`\n"
        f"{oi_line}{fr_line}{news_line}\n"
        f"Passed : `{', '.join(passed_filters)}`\n\n"
        f"_High confidence filter. Not financial advice._"
    )


def format_status(sig: SignalScore, extras: dict, news: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    prob = sig.prob if sig.direction == "BUY" else 1 - sig.prob
    return (
        f"*{SYMBOL} Status* _{now}_\n\n"
        f"Price   : `${sig.entry:,.2f}`\n"
        f"Signal  : `{sig.direction}` — Score `{sig.score}/7`\n"
        f"Prob    : `{prob*100:.1f}%`\n"
        f"Regime  : `{sig.regime}`\n"
        f"RSI     : `{sig.rsi}`\n"
        f"ATR%    : `{sig.atr/sig.entry*100:.2f}%`\n"
        f"F&G     : `{sig.fg_value:.0f}`\n\n"
        f"News    : `{news.get('signal','NEUTRAL')}` (6h: {news.get('sent_6h',0):.1f})\n"
        f"OI chg  : `{extras.get('oi_change',0):+.2f}%`\n"
        f"Funding : `{extras.get('funding_rate',0):+.4f}%`\n\n"
        f"Alert?  : `{'YES — passes all filters' if sig.passed else 'NO — score '+str(sig.score)+'/7 (need '+str(MIN_SCORE)+')'}`"
    )


def poll_commands():
    global _update_offset, _chat_id
    if not TELEGRAM_TOKEN: return
    try:
        data = _tg("getUpdates", offset=_update_offset, timeout=5)
        for upd in data.get("result", []):
            _update_offset = upd["update_id"] + 1
            msg  = upd.get("message", {})
            text = msg.get("text","").strip()
            cid  = str(msg.get("chat",{}).get("id",""))
            if not cid or not text: continue
            if not _chat_id:
                _chat_id = cid
                log.info(f"Chat ID: {_chat_id}")
            cmd = text.split()[0].lower()
            if cmd == "/start":
                send_msg(
                    f"✅ *BTC AI Bot v4.4 — High Confidence*\n\n"
                    f"Filters: Prob≥{PROB_BUY_THR} | Score≥{MIN_SCORE}/7 | Max {MAX_TRADES_PER_DAY}/day\n"
                    f"SL=1×ATR | TP=2×ATR | R:R 1:2\n\n"
                    f"/status — current prediction\n/news — sentiment\n/last — last signal\n/help", cid)
            elif cmd == "/status":
                try:
                    sig, extras = get_prediction()
                    news = get_news()
                    send_msg(format_status(sig, extras, news), cid)
                except Exception as e:
                    send_msg(f"❌ {e}", cid)
            elif cmd == "/news":
                try:
                    news = get_news(force=True)
                    ns = news.get("signal","NEUTRAL")
                    reasons = "\n".join(f"  • {r}" for r in news.get("reasons",[]))
                    send_msg(
                        f"*BTC News Sentiment*\n\n"
                        f"Signal : `{ns}`\n"
                        f"6h     : `{news.get('sent_6h',0):.2f}`\n"
                        f"24h    : `{news.get('sent_24h',0):.2f}`\n"
                        f"F&G    : `{news.get('fg',50)}`\n\n"
                        f"*Drivers:*\n{reasons or '  • No specific drivers'}", cid)
                except Exception as e:
                    send_msg(f"❌ {e}", cid)
            elif cmd == "/last":
                if _last_signal:
                    sig, extras, news = _last_signal
                    send_msg(format_signal(sig, extras, news), cid)
                else:
                    send_msg("No signal sent yet this session.", cid)
            elif cmd == "/help":
                send_msg(
                    "*Commands*\n"
                    "/start — info\n/status — live prediction + score\n"
                    "/news — sentiment\n/last — last signal\n/help", cid)
    except Exception as e:
        log.warning(f"Poll error: {e}")


def run():
    global _last_signal, _today_trade_count, _today_date, _chat_id

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _today_date:
        _today_date = today
        _today_trade_count = 0

    if _today_trade_count >= MAX_TRADES_PER_DAY:
        log.info(f"Max trades/day reached ({MAX_TRADES_PER_DAY}) — skipping")
        return

    if not _chat_id:
        log.warning("No chat ID — send /start to bot")

    log.info("Getting prediction...")
    try:
        sig, extras = get_prediction()
        news        = get_news()

        log.info(
            f"{sig.direction} prob={sig.prob:.3f} score={sig.score}/7 "
            f"passed={sig.passed} regime={sig.regime} rsi={sig.rsi:.0f}"
        )

        if not sig.passed:
            log.info(f"Score {sig.score}/{MIN_SCORE} required — filtered")
            return

        # News conflict check
        ns = news.get("signal","NEUTRAL")
        if ((sig.direction == "BUY"  and ns == "BEARISH" and news.get("strength",0) > 0.6) or
            (sig.direction == "SELL" and ns == "BULLISH" and news.get("strength",0) > 0.6)):
            log.info(f"News conflict: tech={sig.direction}, news={ns} — skip")
            return

        # Dedup
        sig_hash = hashlib.md5(
            f"{sig.direction}:{sig.score}:{round(sig.entry,-2)}".encode()
        ).hexdigest()[:8]
        if sig_hash in _sent_hashes:
            log.info("Duplicate — skip")
            return

        msg = format_signal(sig, extras, news)
        if send_msg(msg):
            _last_signal       = (sig, extras, news)
            _today_trade_count += 1
            _sent_hashes.add(sig_hash)
            if len(_sent_hashes) > 100: _sent_hashes.pop()
            log.info(f"Signal sent! {sig.direction} score={sig.score}/7 trades_today={_today_trade_count}")

    except Exception as e:
        log.error(f"Run error: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    log.info("BTC AI Bot v4.4 — High Confidence Strategy")
    log.info(f"  Prob filter : BUY>={PROB_BUY_THR} | SELL<={PROB_SELL_THR}")
    log.info(f"  Score filter: >={MIN_SCORE}/7")
    log.info(f"  Max trades  : {MAX_TRADES_PER_DAY}/day")
    log.info(f"  SL/TP       : {ATR_SL_MULT}x / {ATR_TP_MULT}x ATR")

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN missing!")
        sys.exit(1)
    if not load_model():
        sys.exit(1)

    run()
    schedule.every(CHECK_INTERVAL).minutes.do(run)
    schedule.every(1).minutes.do(poll_commands)
    log.info(f"Scheduler: every {CHECK_INTERVAL} min")
    while True:
        schedule.run_pending()
        time.sleep(15)
