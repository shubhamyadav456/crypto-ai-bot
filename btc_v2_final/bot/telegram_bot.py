# bot/telegram_bot.py — v5.0 (Multi-AI: Debate + Narrative + Shadow)
"""
Signal pipeline:
  Layer 1: ML model (XGB + RF)
  Layer 2: 7-filter scoring (score >= 4/7)
  Layer 3: Claude Debate Panel (score >= 7/10)
  Layer 4: Narrative alignment check
  Layer 5: Final Telegram alert

Shadow Trader runs in parallel — paper trades every signal.
Weekly Claude review improves filters automatically.
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
from features.signal_filter import score_signal, SignalScore
from ai.debate_panel import DebatePanel, build_snapshot_from_features
from ai.narrative_engine import NarrativeEngine
from ai.shadow_trader import ShadowTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_ROOT, "btc_bot.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
CRYPTOPANIC_KEY    = os.getenv("CRYPTOPANIC_API_KEY", "")
ANTHROPIC_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
PROB_BUY_THR       = float(os.getenv("MIN_PROB_BUY",  "0.70"))
PROB_SELL_THR      = float(os.getenv("MAX_PROB_SELL", "0.30"))
MIN_SCORE          = int(os.getenv("MIN_SIGNAL_SCORE",   "4"))
MIN_DEBATE_SCORE   = int(os.getenv("MIN_DEBATE_SCORE",   "7"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
USE_DEBATE         = os.getenv("USE_DEBATE_PANEL", "true").lower() == "true"
USE_NARRATIVE      = os.getenv("USE_NARRATIVE", "true").lower() == "true"

# ── State ───────────────────────────────────────────────────────
_artifact          = None
_last_signal       = None
_sent_hashes       = set()
_chat_id           = TELEGRAM_CHAT_ID
_update_offset     = 0
_today_count       = 0
_today_date        = ""
_news_cache        = {}
_news_cache_ts     = 0

# ── AI agents ───────────────────────────────────────────────────
_debate    = DebatePanel(min_judge_score=MIN_DEBATE_SCORE) if ANTHROPIC_KEY else None
_narrative = NarrativeEngine() if ANTHROPIC_KEY else None
_shadow    = ShadowTrader()


# ── Model ───────────────────────────────────────────────────────
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


# ── Technical prediction ────────────────────────────────────────
def get_prediction() -> tuple:
    """Returns (SignalScore, pred_dict, df_feat)"""
    from data.fetcher import build_raw_dataset
    from features.engineer import build_features, _ema, _rsi, _bb, _atr

    df_raw  = build_raw_dataset(SYMBOL, BASE_TF, total=300)
    df_feat = build_features(df_raw, include_market_data=True)

    art   = _artifact
    sc    = art["scaler"]
    xgb   = art["xgb"]
    rf    = art["rf"]
    cal   = art["calibrator"]
    feats = art.get("all_features", art.get("features", []))
    idx   = art.get("sel_idx")
    thr   = art.get("threshold", 0.5)

    for f in feats:
        if f not in df_feat.columns:
            df_feat[f] = 0.0

    row_all = sc.transform(df_feat[feats].iloc[[-1]].fillna(0).values)
    row_sel = row_all[:, idx] if idx is not None else row_all
    raw     = (xgb.predict_proba(row_sel)[:,1][0] +
               rf.predict_proba(row_sel)[:,1][0]) / 2
    prob    = float(cal.transform([raw])[0])

    direction = "BUY" if prob >= PROB_BUY_THR else ("SELL" if prob <= PROB_SELL_THR else None)

    # Indicators for scoring
    raw_ri = df_raw.reset_index(drop=True)
    cs, hs, ls, vs = raw_ri["close"], raw_ri["high"], raw_ri["low"], raw_ri["volume"]
    price   = float(cs.iloc[-1])
    at14    = float(_atr(hs, ls, cs, 14).iloc[-1])
    bu, bm, bl, bp, bw = _bb(cs, 20, 2)

    # Build signal score
    eff_dir = direction or ("BUY" if prob >= 0.5 else "SELL")
    sig = score_signal(
        direction    = eff_dir,
        prob         = prob,
        price        = price,
        ema50        = float(_ema(cs, 50).iloc[-1]),
        ema200       = float(_ema(cs, 200).iloc[-1]),
        atr          = at14,
        rsi          = float(_rsi(cs, 14).iloc[-1]),
        bb_upper     = float(bu.iloc[-1]),
        bb_lower     = float(bl.iloc[-1]),
        bb_width     = float(bw.iloc[-1]),
        bb_width_ma  = float(bw.rolling(20).mean().iloc[-1]),
        fg_value     = float(df_feat.get("fg_value", pd.Series([50])).iloc[-1]),
        vol_ratio    = float(vs.iloc[-1] / (vs.rolling(20).mean().iloc[-1] + 1e-9)),
        prob_buy_thr = PROB_BUY_THR,
        prob_sell_thr= PROB_SELL_THR,
        min_score    = MIN_SCORE,
        sl_atr_mult  = ATR_SL_MULT,
        tp_atr_mult  = ATR_TP_MULT,
    )

    pred = {
        "direction":  eff_dir,
        "prob":       prob,
        "confidence": sig.confidence if hasattr(sig, "confidence") else "LOW",
        "entry":      sig.entry,
        "sl":         sig.sl,
        "tp":         sig.tp,
        "rr":         sig.rr,
        "oi_change":  float(df_feat.get("oi_change", pd.Series([0])).iloc[-1]) if "oi_change" in df_feat.columns else 0,
        "funding_rate": float(df_feat.get("funding_rate", pd.Series([0])).iloc[-1]) if "funding_rate" in df_feat.columns else 0,
    }
    return sig, pred, df_feat


# ── News ────────────────────────────────────────────────────────
def get_news(force=False) -> dict:
    global _news_cache, _news_cache_ts
    now = time.time()
    if not force and _news_cache and (now - _news_cache_ts) < 1800:
        return _news_cache
    try:
        from data.news import get_news_features, get_news_signal
        key      = CRYPTOPANIC_KEY if len(CRYPTOPANIC_KEY) > 20 else None
        features = get_news_features("BTC", api_key=key)
        signal   = get_news_signal(features)
        _news_cache    = {**features, **signal}
        _news_cache_ts = now
        return _news_cache
    except Exception as e:
        log.warning(f"News error: {e}")
        return {"signal":"NEUTRAL","strength":0,"sent_6h":0,"fg":50,"reasons":[]}


# ── Telegram helpers ────────────────────────────────────────────
def _tg(method, **kw):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
        json=kw, timeout=10)
    r.raise_for_status()
    return r.json()

def send_msg(text, chat_id=None):
    cid = chat_id or _chat_id
    if not cid: return False
    try:
        _tg("sendMessage", chat_id=cid, text=text, parse_mode="Markdown")
        return True
    except Exception as e:
        log.error(f"Send failed: {e}"); return False


# ── Message formatters ──────────────────────────────────────────
def format_signal(sig, pred, debate, narrative, news) -> str:
    d     = pred["direction"]
    prob  = pred["prob"] if d == "BUY" else 1 - pred["prob"]
    dscore = debate.get("score", 0) if debate else 0
    grade  = "A" if dscore >= 8 else "B" if dscore >= 6 else "C"
    arrow  = "BUY  [LONG]" if d == "BUY" else "SELL [SHORT]"
    now    = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    ns      = news.get("signal","NEUTRAL")
    nar_str = narrative.get("theme","") if narrative else ""
    ns_line = f"News   : `{ns}` — _{nar_str}_\n" if nar_str else f"News   : `{ns}`\n"

    debate_line = ""
    if debate and debate.get("primary_reason"):
        debate_line = f"\n💡 *AI Verdict*: _{debate['primary_reason']}_\n"
        if debate.get("key_risk"):
            debate_line += f"⚠️ *Risk*: _{debate['key_risk']}_\n"

    oi_line = f"OI chg : `{pred.get('oi_change',0)*100:+.1f}%`\n" if abs(pred.get('oi_change',0)) > 0.005 else ""
    fr_line = f"Funding: `{pred.get('funding_rate',0)*100:+.4f}%`\n" if abs(pred.get('funding_rate',0)) > 0.0001 else ""

    return (
        f"*{SYMBOL} Signal* [{grade}]\n"
        f"_{now}_\n\n"
        f"*Direction : {arrow}*\n"
        f"ML Prob    : `{prob*100:.1f}%`\n"
        f"Filter     : `{getattr(sig,'score',0)}/7`\n"
        f"AI Debate  : `{dscore}/10`\n\n"
        f"Entry  : `${pred['entry']:,.2f}`\n"
        f"Stop L : `${pred['sl']:,.2f}`\n"
        f"Take P : `${pred['tp']:,.2f}`\n"
        f"R:R    : `1:{pred['rr']}`\n\n"
        f"RSI    : `{getattr(sig,'rsi',0):.1f}`\n"
        f"Regime : `{getattr(sig,'regime','?')}`\n"
        f"ATR%   : `{sig.atr/sig.entry*100:.2f}%` \n"
        f"F&G    : `{sig.fg_value:.0f}`\n"
        f"{oi_line}{fr_line}{ns_line}"
        f"{debate_line}\n"
        f"_Multi-AI analysis. Not financial advice._"
    )


def format_status(sig, pred, debate, narrative, news) -> str:
    now    = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    prob   = pred["prob"] if pred["direction"] == "BUY" else 1 - pred["prob"]
    dscore = debate.get("score",0) if debate else "N/A (no API key)"
    nar    = narrative.get("theme","") if narrative else "N/A"
    return (
        f"*{SYMBOL} Status* _{now}_\n\n"
        f"Price   : `${sig.entry:,.2f}`\n"
        f"Signal  : `{pred['direction']}` ({prob*100:.1f}%)\n"
        f"Filter  : `{getattr(sig,'score',0)}/7`\n"
        f"Debate  : `{dscore}/10`\n"
        f"Narrative: `{nar}`\n\n"
        f"Regime  : `{getattr(sig,'regime','?')}`\n"
        f"RSI     : `{getattr(sig,'rsi',0):.1f}`\n"
        f"F&G     : `{sig.fg_value:.0f}`\n"
        f"News    : `{news.get('signal','NEUTRAL')}`\n\n"
        f"Shadow  : {_shadow.get_stats().get('win_rate','?')}% WR "
        f"({_shadow.get_stats().get('total',0)} trades)\n"
        f"Alert?  : `{'YES' if getattr(sig,'passed',False) else 'NO — score '+str(getattr(sig,'score',0))+'/'+str(MIN_SCORE)}`"
    )


# ── Command polling ─────────────────────────────────────────────
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
            if not _chat_id: _chat_id = cid
            cmd = text.split()[0].lower()

            if cmd == "/start":
                debate_status = f"✓ Active (min score {MIN_DEBATE_SCORE}/10)" if ANTHROPIC_KEY else "✗ No API key"
                send_msg(
                    f"✅ *BTC AI Bot v5.0 — Multi-AI Strategy*\n\n"
                    f"*Pipeline:*\n"
                    f"  ML Model → Filter ({MIN_SCORE}/7) → Debate → Narrative → Alert\n\n"
                    f"*AI Debate Panel:* {debate_status}\n"
                    f"*Prob threshold:* BUY≥{PROB_BUY_THR} | SELL≤{PROB_SELL_THR}\n"
                    f"*Max trades/day:* {MAX_TRADES_PER_DAY}\n\n"
                    f"/status /debate /narrative /shadow /news /last /help", cid)

            elif cmd == "/status":
                try:
                    sig, pred, df_feat = get_prediction()
                    news = get_news()
                    debate = {"score": "N/A"}
                    narrative = {}
                    send_msg(format_status(sig, pred, debate, narrative, news), cid)
                except Exception as e:
                    send_msg(f"❌ {e}", cid)

            elif cmd == "/debate":
                try:
                    if not _debate:
                        send_msg("❌ Set ANTHROPIC_API_KEY in config.env", cid); continue
                    send_msg("🤔 Running 4-agent debate... (30-60 sec)", cid)
                    sig, pred, df_feat = get_prediction()
                    news    = get_news()
                    snap    = build_snapshot_from_features(df_feat, pred, sig, news)
                    result  = _debate.debate(snap)
                    emoji   = "✅" if result["passed"] else "❌"
                    send_msg(
                        f"*Debate Panel Result* {emoji}\n\n"
                        f"Verdict    : `{result['verdict']}`\n"
                        f"Score      : `{result['score']}/10`\n"
                        f"Confidence : `{result['confidence']}`\n\n"
                        f"*Reason*: {result['primary_reason']}\n"
                        f"*Key Risk*: {result['key_risk']}\n\n"
                        f"Bull={result.get('bull_score',0)}/10  "
                        f"Bear={result.get('bear_score',0)}/10  "
                        f"Risk={result.get('risk_score',0)}/10", cid)
                except Exception as e:
                    send_msg(f"❌ Debate error: {e}", cid)

            elif cmd == "/narrative":
                try:
                    if not _narrative:
                        send_msg("❌ Set ANTHROPIC_API_KEY in config.env", cid); continue
                    send_msg("📰 Analyzing market narrative...", cid)
                    news = get_news(force=True)
                    headlines = news.get("recent_headlines", [])
                    if not headlines:
                        from data.news import fetch_cryptopanic
                        df_n = fetch_cryptopanic("BTC", pages=2)
                        headlines = df_n["title"].tolist()[:15] if not df_n.empty else []
                    result = _narrative.analyze(headlines, {"fg_value": news.get("fg", 50)})
                    send_msg(
                        f"*Market Narrative*\n\n"
                        f"Type    : `{result['narrative_type']}`\n"
                        f"Theme   : _{result['theme']}_\n"
                        f"Strength: `{result['strength']}/10`\n"
                        f"Bias    : `{result['trade_bias']}`\n\n"
                        f"*Summary*: {result['summary']}", cid)
                except Exception as e:
                    send_msg(f"❌ Narrative error: {e}", cid)

            elif cmd == "/shadow":
                stats = _shadow.get_stats()
                send_msg(
                    f"*Shadow Trader Stats*\n\n"
                    f"Total  : {stats.get('total',0)} paper trades\n"
                    f"Open   : {stats.get('open',0)}\n"
                    f"Win Rate: {stats.get('win_rate','?')}%\n"
                    f"Avg Win : {stats.get('avg_win',0):+.2f}%\n"
                    f"Avg Loss: {stats.get('avg_loss',0):+.2f}%", cid)

            elif cmd == "/review":
                try:
                    if not ANTHROPIC_KEY:
                        send_msg("❌ Set ANTHROPIC_API_KEY in config.env", cid); continue
                    send_msg("🔍 Claude reviewing paper trades...", cid)
                    review = _shadow.weekly_review(last_n=20)
                    suggestions = review.get("suggestions", [])
                    sug_text = "\n".join(
                        f"  • *{s['filter']}*: {s['change']}" for s in suggestions[:5]
                    ) or "  No specific suggestions"
                    send_msg(
                        f"*Weekly AI Review*\n\n"
                        f"_{review.get('summary','')}_\n\n"
                        f"*Suggestions:*\n{sug_text}", cid)
                except Exception as e:
                    send_msg(f"❌ Review error: {e}", cid)

            elif cmd == "/news":
                news = get_news(force=True)
                reasons = "\n".join(f"  • {r}" for r in news.get("reasons",[]))
                send_msg(
                    f"*BTC News Sentiment*\n\n"
                    f"Signal : `{news.get('signal','?')}`\n"
                    f"6h     : `{news.get('sent_6h',0):.2f}`\n"
                    f"24h    : `{news.get('sent_24h',0):.2f}`\n"
                    f"F&G    : `{news.get('fg',50)}`\n\n"
                    f"*Drivers:*\n{reasons or 'None'}", cid)

            elif cmd == "/last":
                if _last_signal:
                    sig, pred, debate, narrative, news = _last_signal
                    send_msg(format_signal(sig, pred, debate, narrative, news), cid)
                else:
                    send_msg("No signal sent yet this session.", cid)

            elif cmd == "/help":
                send_msg(
                    "*BTC AI Bot v5.0 Commands*\n\n"
                    "/start     — bot info\n"
                    "/status    — live prediction\n"
                    "/debate    — run 4-agent AI debate now\n"
                    "/narrative — current market narrative\n"
                    "/shadow    — paper trade stats\n"
                    "/review    — weekly AI review of trades\n"
                    "/news      — news sentiment\n"
                    "/last      — last signal sent\n"
                    "/help      — this message", cid)
    except Exception as e:
        log.warning(f"Poll error: {e}")


# ── Main run loop ───────────────────────────────────────────────
def run():
    global _last_signal, _today_count, _today_date, _chat_id

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _today_date:
        _today_date = today
        _today_count = 0

    if _today_count >= MAX_TRADES_PER_DAY:
        log.info(f"Max trades/day ({MAX_TRADES_PER_DAY}) reached")
        return

    if not _chat_id:
        log.warning("No chat ID — send /start to bot first")

    log.info("="*50)
    log.info("Getting prediction...")

    try:
        # Layer 1+2: ML + scoring
        sig, pred, df_feat = get_prediction()
        news = get_news()

        log.info(
            f"ML: {pred['direction']} prob={pred['prob']:.3f} | "
            f"Filter: {getattr(sig,'score',0)}/{MIN_SCORE} passed={getattr(sig,'passed',False)} | "
            f"Regime: {getattr(sig,'regime','?')} | "
            f"News: {news.get('signal','?')}"
        )

        if not getattr(sig, "passed", False):
            log.info(f"Layer 2 failed: score {getattr(sig,'score',0)}/{MIN_SCORE}")
            _shadow.update_prices({SYMBOL: pred["entry"]})
            return

        # Layer 3: Claude Debate Panel
        debate_result = {}
        if _debate and ANTHROPIC_KEY:
            log.info("Running debate panel...")
            snap         = build_snapshot_from_features(df_feat, pred, sig, news)
            debate_result = _debate.debate(snap)
            log.info(f"Debate: {debate_result['verdict']} score={debate_result['score']}/10")

            if not debate_result.get("passed", False):
                log.info(f"Layer 3 (debate) failed: score={debate_result['score']}/{MIN_DEBATE_SCORE}")
                # Still open shadow trade to track
                _shadow.open_trade({**pred, "signal_score": sig.score,
                                    "debate_score": debate_result.get("score",0)})
                _shadow.update_prices({SYMBOL: pred["entry"]})
                return
        else:
            log.info("Debate panel skipped (no ANTHROPIC_API_KEY)")

        # Layer 4: Narrative check
        narrative_result = {}
        if _narrative and ANTHROPIC_KEY:
            try:
                from data.news import fetch_cryptopanic
                df_n      = fetch_cryptopanic("BTC", pages=1)
                headlines = df_n["title"].tolist()[:15] if not df_n.empty else []
                narrative_result = _narrative.analyze(headlines, {"fg_value": sig.fg_value})
                delta, reason = _narrative.get_narrative_score(
                    narrative_result, pred["direction"])
                log.info(f"Narrative: {narrative_result.get('narrative_type','?')} "
                         f"delta={delta} reason={reason}")
                if delta < -1:
                    log.info(f"Layer 4 (narrative) conflict — skip")
                    return
            except Exception as e:
                log.warning(f"Narrative check error: {e}")

        # Dedup
        sig_hash = hashlib.md5(
            f"{pred['direction']}:{debate_result.get('score',0)}:"
            f"{round(pred['entry'],-2)}".encode()
        ).hexdigest()[:8]
        if sig_hash in _sent_hashes:
            log.info("Duplicate — skip")
            return

        # Send signal
        msg = format_signal(sig, pred, debate_result, narrative_result, news)
        if send_msg(msg):
            _last_signal   = (sig, pred, debate_result, narrative_result, news)
            _today_count  += 1
            _sent_hashes.add(sig_hash)
            if len(_sent_hashes) > 100: _sent_hashes.pop()

            # Open shadow trade
            _shadow.open_trade({
                **pred,
                "signal_score":  sig.score,
                "debate_score":  debate_result.get("score", 0),
                "narrative_theme": narrative_result.get("theme", ""),
            })

            log.info(f"✅ Signal sent! {pred['direction']} "
                     f"debate={debate_result.get('score',0)}/10 "
                     f"trades_today={_today_count}")

        _shadow.update_prices({SYMBOL: pred["entry"]})

    except Exception as e:
        log.error(f"Run error: {e}")
        import traceback; traceback.print_exc()


# ── Weekly review job ───────────────────────────────────────────
def weekly_review_job():
    log.info("[WEEKLY] Running Claude review of paper trades...")
    review = _shadow.weekly_review(last_n=20)
    suggestions = review.get("suggestions", [])
    if suggestions and _chat_id:
        sug_text = "\n".join(
            f"  • *{s['filter']}*: {s['change']}" for s in suggestions[:5])
        send_msg(
            f"🔍 *Weekly AI Review*\n\n"
            f"_{review.get('summary','')}_\n\n"
            f"*Suggestions:*\n{sug_text}"
        )


# ── Entry ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("BTC AI Bot v5.0 — Multi-AI Strategy")
    log.info(f"  ML + Filter ({MIN_SCORE}/7) + Debate ({MIN_DEBATE_SCORE}/10) + Narrative")
    log.info(f"  Debate Panel : {'ACTIVE' if ANTHROPIC_KEY else 'INACTIVE (add ANTHROPIC_API_KEY)'}")
    log.info(f"  Narrative    : {'ACTIVE' if ANTHROPIC_KEY else 'INACTIVE'}")
    log.info(f"  Shadow Trader: ACTIVE (paper trades always running)")
    log.info(f"  Check every  : {CHECK_INTERVAL} min | Max {MAX_TRADES_PER_DAY}/day")

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN missing!")
        sys.exit(1)
    if not load_model():
        sys.exit(1)

    run()  # immediate check

    schedule.every(CHECK_INTERVAL).minutes.do(run)
    schedule.every(1).minutes.do(poll_commands)
    schedule.every().sunday.at("08:00").do(weekly_review_job)

    log.info(f"Scheduler running — checking every {CHECK_INTERVAL} min")
    while True:
        schedule.run_pending()
        time.sleep(15)
