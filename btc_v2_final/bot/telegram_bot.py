# bot/telegram_bot.py — v5.1 (CoinGecko + AWS)
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

CRYPTOPANIC_KEY    = os.getenv("CRYPTOPANIC_API_KEY", "")
ANTHROPIC_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
PROB_BUY_THR       = float(os.getenv("MIN_PROB_BUY",  "0.70"))
PROB_SELL_THR      = float(os.getenv("MAX_PROB_SELL", "0.0"))
MIN_SCORE          = int(os.getenv("MIN_SIGNAL_SCORE",   "5"))
MIN_DEBATE_SCORE   = int(os.getenv("MIN_DEBATE_SCORE",   "7"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
USE_DEBATE         = os.getenv("USE_DEBATE_PANEL", "true").lower() == "true"

_artifact          = None
_last_signal       = None
_sent_hashes       = set()
_chat_id           = TELEGRAM_CHAT_ID
_update_offset     = 0
_today_count       = 0
_today_date        = ""
_news_cache        = {}
_news_cache_ts     = 0

_debate    = None
_narrative = None

def _init_ai():
    global _debate, _narrative
    if ANTHROPIC_KEY:
        try:
            from ai.debate_panel import DebatePanel
            from ai.narrative_engine import NarrativeEngine
            _debate    = DebatePanel(min_judge_score=MIN_DEBATE_SCORE)
            _narrative = NarrativeEngine()
            log.info("AI agents initialized")
        except Exception as e:
            log.warning(f"AI agents init failed: {e}")

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


def _safe_predict(df_feat: pd.DataFrame) -> float:
    """
    Safe inference — handles feature mismatch between
    model (trained on Binance) and live data (CoinGecko).
    """
    art       = _artifact
    sc        = art["scaler"]
    xgb       = art["xgb"]
    rf        = art["rf"]
    cal       = art["calibrator"]
    all_feats = art.get("all_features", art.get("features", []))
    sel_idx   = art.get("sel_idx")

    # Build feature vector — fill missing with 0
    row = []
    for f in all_feats:
        if f in df_feat.columns:
            val = float(df_feat[f].iloc[-1])
            val = 0.0 if (np.isnan(val) or np.isinf(val)) else val
        else:
            val = 0.0
        row.append(val)

    X = np.array([row], dtype=np.float32)

    # Scale
    X_sc = sc.transform(X)

    # Select features — SAFE: use only valid indices
    if sel_idx is not None:
        max_idx = X_sc.shape[1] - 1
        valid_idx = [i for i in sel_idx if i <= max_idx]
        if len(valid_idx) < len(sel_idx):
            log.warning(f"sel_idx: {len(sel_idx)} → {len(valid_idx)} valid (feature mismatch)")
        X_sel = X_sc[:, valid_idx]
    else:
        X_sel = X_sc

    # Predict
    try:
        p_xgb = float(xgb.predict_proba(X_sel)[:, 1][0])
        p_rf  = float(rf.predict_proba(X_sel)[:, 1][0])
    except Exception as e:
        log.warning(f"predict_proba failed: {e} — using 0.5")
        return 0.5

    raw  = (p_xgb + p_rf) / 2
    prob = float(cal.transform([raw])[0])
    return prob


def get_prediction() -> tuple:
    from data.fetcher import build_raw_dataset
    from features.engineer import build_features, _ema, _rsi, _bb, _atr

    df_raw  = build_raw_dataset(SYMBOL, BASE_TF, total=300)
    df_feat = build_features(df_raw, include_market_data=True)

    prob = _safe_predict(df_feat)

    direction = "BUY" if prob >= PROB_BUY_THR else (
                "SELL" if prob <= PROB_SELL_THR and PROB_SELL_THR > 0 else None)

    raw_ri = df_raw.reset_index(drop=True)
    cs = raw_ri["close"]; hs = raw_ri["high"]
    ls = raw_ri["low"];   vs = raw_ri["volume"]

    price  = float(cs.iloc[-1])
    at14   = float(_atr(hs, ls, cs, 14).iloc[-1])
    bu, bm, bl, bp, bw = _bb(cs, 20, 2)
    bw_ma  = float(bw.rolling(20).mean().iloc[-1])
    vm20   = float(vs.rolling(20).mean().iloc[-1])
    vol_r  = float(vs.iloc[-1] / (vm20 + 1e-9))
    fg_val = float(df_feat["fg_value"].iloc[-1]) if "fg_value" in df_feat.columns else 50.0

    eff_dir = direction or ("BUY" if prob >= 0.5 else "SELL")

    sig = score_signal(
        direction     = eff_dir,
        prob          = prob,
        price         = price,
        ema50         = float(_ema(cs, 50).iloc[-1]),
        ema200        = float(_ema(cs, 200).iloc[-1]),
        atr           = at14,
        rsi           = float(_rsi(cs, 14).iloc[-1]),
        bb_upper      = float(bu.iloc[-1]),
        bb_lower      = float(bl.iloc[-1]),
        bb_width      = float(bw.iloc[-1]),
        bb_width_ma   = bw_ma,
        fg_value      = fg_val,
        vol_ratio     = vol_r,
        prob_buy_thr  = PROB_BUY_THR,
        prob_sell_thr = PROB_SELL_THR,
        min_score     = MIN_SCORE,
        sl_atr_mult   = ATR_SL_MULT,
        tp_atr_mult   = ATR_TP_MULT,
    )

    pred = {
        "direction":    eff_dir,
        "prob":         prob,
        "confidence":   sig.confidence,
        "entry":        sig.entry,
        "sl":           sig.sl,
        "tp":           sig.tp,
        "rr":           sig.rr,
        "oi_change":    float(df_feat["oi_change"].iloc[-1]) if "oi_change" in df_feat.columns else 0,
        "funding_rate": float(df_feat["funding_rate"].iloc[-1]) if "funding_rate" in df_feat.columns else 0,
    }
    return sig, pred, df_feat


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


def format_signal(sig, pred, debate, narrative, news) -> str:
    d      = pred["direction"]
    prob   = pred["prob"] if d == "BUY" else 1 - pred["prob"]
    dscore = debate.get("score", 0) if debate else 0
    grade  = "A" if dscore >= 8 else "B" if dscore >= 6 else "C"
    arrow  = "BUY  [LONG]" if d == "BUY" else "SELL [SHORT]"
    now    = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    ns     = news.get("signal", "NEUTRAL")
    nar    = narrative.get("theme", "") if narrative else ""
    ns_line= f"News   : `{ns}` — _{nar}_\n" if nar else f"News   : `{ns}`\n"
    debate_line = ""
    if debate and debate.get("primary_reason"):
        debate_line = f"\n💡 *AI*: _{debate['primary_reason']}_\n"
        if debate.get("key_risk"):
            debate_line += f"⚠️ *Risk*: _{debate['key_risk']}_\n"
    oi_line = f"OI chg : `{pred.get('oi_change',0)*100:+.1f}%`\n" if abs(pred.get('oi_change',0)) > 0.005 else ""
    fr_line = f"Funding: `{pred.get('funding_rate',0)*100:+.4f}%`\n" if abs(pred.get('funding_rate',0)) > 0.0001 else ""
    return (
        f"*{SYMBOL} Signal* [{grade}]\n_{now}_\n\n"
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
        f"F&G    : `{sig.fg_value:.0f}`\n"
        f"{oi_line}{fr_line}{ns_line}"
        f"{debate_line}\n"
        f"_Multi-AI analysis. Not financial advice._"
    )

def format_status(sig, pred, news) -> str:
    now  = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    prob = pred["prob"] if pred["direction"] == "BUY" else 1 - pred["prob"]
    return (
        f"*{SYMBOL} Status* _{now}_\n\n"
        f"Price   : `${sig.entry:,.2f}`\n"
        f"Signal  : `{pred['direction']}` ({prob*100:.1f}%)\n"
        f"Filter  : `{getattr(sig,'score',0)}/7`\n"
        f"Regime  : `{getattr(sig,'regime','?')}`\n"
        f"RSI     : `{getattr(sig,'rsi',0):.1f}`\n"
        f"F&G     : `{sig.fg_value:.0f}`\n"
        f"News    : `{news.get('signal','NEUTRAL')}`\n\n"
        f"Alert?  : `{'YES' if getattr(sig,'passed',False) else 'NO — score '+str(getattr(sig,'score',0))+'/'+str(MIN_SCORE)}`"
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
            if not _chat_id: _chat_id = cid; log.info(f"Chat ID: {_chat_id}")
            cmd = text.split()[0].lower()

            if cmd == "/start":
                send_msg(
                    f"✅ *BTC AI Bot v5.1*\n\n"
                    f"Pipeline: ML → Filter({MIN_SCORE}/7) → Debate({MIN_DEBATE_SCORE}/10) → Alert\n"
                    f"Debate: {'ACTIVE' if _debate else 'Add ANTHROPIC_API_KEY'}\n"
                    f"Max trades: {MAX_TRADES_PER_DAY}/day\n\n"
                    f"/status /debate /news /last /help", cid)

            elif cmd == "/status":
                try:
                    sig, pred, df_feat = get_prediction()
                    news = get_news()
                    send_msg(format_status(sig, pred, news), cid)
                except Exception as e:
                    send_msg(f"❌ {e}", cid)

            elif cmd == "/debate":
                if not _debate:
                    send_msg("❌ Set ANTHROPIC_API_KEY in config.env", cid); continue
                try:
                    from ai.debate_panel import build_snapshot_from_features
                    send_msg("🤔 Running debate (30-60 sec)...", cid)
                    sig, pred, df_feat = get_prediction()
                    news   = get_news()
                    snap   = build_snapshot_from_features(df_feat, pred, sig, news)
                    result = _debate.debate(snap)
                    emoji  = "✅" if result["passed"] else "❌"
                    send_msg(
                        f"*Debate Result* {emoji}\n\n"
                        f"Verdict : `{result['verdict']}`\n"
                        f"Score   : `{result['score']}/10`\n"
                        f"Reason  : {result['primary_reason']}\n"
                        f"Risk    : {result['key_risk']}\n\n"
                        f"Bull={result.get('bull_score',0)} Bear={result.get('bear_score',0)} Risk={result.get('risk_score',0)}", cid)
                except Exception as e:
                    send_msg(f"❌ Debate error: {e}", cid)

            elif cmd == "/news":
                news = get_news(force=True)
                reasons = "\n".join(f"• {r}" for r in news.get("reasons",[]))
                send_msg(
                    f"*BTC News*\n\nSignal: `{news.get('signal','?')}`\n"
                    f"6h: `{news.get('sent_6h',0):.2f}` | F&G: `{news.get('fg',50)}`\n\n"
                    f"{reasons or 'No specific drivers'}", cid)

            elif cmd == "/last":
                if _last_signal:
                    sig, pred, debate, narrative, news = _last_signal
                    send_msg(format_signal(sig, pred, debate, narrative, news), cid)
                else:
                    send_msg("No signal sent yet.", cid)

            elif cmd == "/help":
                send_msg(
                    "*Commands*\n/start /status /debate /news /last /help", cid)
    except Exception as e:
        log.warning(f"Poll error: {e}")


def run():
    global _last_signal, _today_count, _today_date, _chat_id

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _today_date:
        _today_date = today; _today_count = 0

    if _today_count >= MAX_TRADES_PER_DAY:
        log.info(f"Max trades/day ({MAX_TRADES_PER_DAY}) reached"); return

    log.info("=" * 50)
    log.info("Getting prediction...")
    try:
        sig, pred, df_feat = get_prediction()
        news = get_news()

        log.info(
            f"ML: {pred['direction']} prob={pred['prob']:.3f} | "
            f"Filter: {getattr(sig,'score',0)}/{MIN_SCORE} passed={getattr(sig,'passed',False)} | "
            f"Regime: {getattr(sig,'regime','?')}"
        )

        if not getattr(sig, "passed", False):
            log.info(f"Filter failed: score {getattr(sig,'score',0)}/{MIN_SCORE}")
            return

        # Debate panel
        debate_result = {}
        if _debate and ANTHROPIC_KEY:
            try:
                from ai.debate_panel import build_snapshot_from_features
                snap         = build_snapshot_from_features(df_feat, pred, sig, news)
                debate_result = _debate.debate(snap)
                log.info(f"Debate: {debate_result['verdict']} score={debate_result['score']}/10")
                if not debate_result.get("passed", False):
                    log.info(f"Debate failed: {debate_result['score']}/{MIN_DEBATE_SCORE}")
                    return
            except Exception as e:
                log.warning(f"Debate error: {e}")

        # Narrative
        narrative_result = {}
        if _narrative and ANTHROPIC_KEY:
            try:
                from data.news import fetch_cryptopanic
                df_n = fetch_cryptopanic("BTC", pages=1)
                headlines = df_n["title"].tolist()[:15] if not df_n.empty else []
                narrative_result = _narrative.analyze(headlines, {"fg_value": sig.fg_value})
                delta, reason = _narrative.get_narrative_score(narrative_result, pred["direction"])
                if delta < -1:
                    log.info(f"Narrative conflict — skip"); return
            except Exception as e:
                log.warning(f"Narrative error: {e}")

        # Dedup
        sig_hash = hashlib.md5(
            f"{pred['direction']}:{debate_result.get('score',0)}:{round(pred['entry'],-2)}".encode()
        ).hexdigest()[:8]
        if sig_hash in _sent_hashes:
            log.info("Duplicate — skip"); return

        msg = format_signal(sig, pred, debate_result, narrative_result, news)
        if send_msg(msg):
            _last_signal   = (sig, pred, debate_result, narrative_result, news)
            _today_count  += 1
            _sent_hashes.add(sig_hash)
            if len(_sent_hashes) > 100: _sent_hashes.pop()
            log.info(f"✅ Signal sent! {pred['direction']} trades_today={_today_count}")

    except Exception as e:
        log.error(f"Run error: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    log.info("BTC AI Bot v5.1 starting...")
    log.info(f"  Symbol   : {SYMBOL} | TF: {BASE_TF}")
    log.info(f"  Filter   : score>={MIN_SCORE}/7 | BUY>={PROB_BUY_THR}")
    log.info(f"  Debate   : {'ACTIVE' if ANTHROPIC_KEY else 'INACTIVE (add ANTHROPIC_API_KEY)'}")
    log.info(f"  Interval : {CHECK_INTERVAL} min | Max {MAX_TRADES_PER_DAY}/day")

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN missing!"); sys.exit(1)
    if not load_model(): sys.exit(1)
    _init_ai()

    run()
    schedule.every(CHECK_INTERVAL).minutes.do(run)
    schedule.every(1).minutes.do(poll_commands)
    log.info(f"Scheduler running — every {CHECK_INTERVAL} min")
    while True:
        schedule.run_pending()
        time.sleep(15)
