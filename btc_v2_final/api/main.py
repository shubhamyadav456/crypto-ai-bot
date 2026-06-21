# api/main.py
import os
import sys

# ── Path setup FIRST — IDE aur runtime both.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Load .env explicitly ────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "config.env"))

# ── Now all imports will resolve ────────────────────────────────
import logging
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config.settings import (
    SYMBOL, BASE_TF, MODEL_PATH, BINANCE_SPOT,
    MIN_PROB_BUY, MAX_PROB_SELL,
    ATR_SL_MULT, ATR_TP_MULT,
    MIN_RR, RISK_PCT,
    API_HOST, API_PORT,
)
from data.fetcher import build_raw_dataset
from features.engineer import build_features
from risk.manager import calc_sl_tp, position_size, filter_signal
from storage.db import init_db, save_signal, get_stats, auto_close_pending

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_artifact = None


def _load_model():
    global _artifact
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH} — run trainer first!")
        _artifact = None
        return
    try:
        _artifact = joblib.load(MODEL_PATH)
        meta = _artifact.get("meta", {})
        log.info(
            f"Model loaded | AUC={meta.get('test_auc','?')} | "
            f"threshold={_artifact.get('threshold', 0.5)} | "
            f"version={meta.get('version','?')}"
        )
    except Exception as e:
        log.error(f"Model load failed: {e}")
        _artifact = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_model()
    yield


app = FastAPI(title="BTC AI Trading API v4", version="4.0.0", lifespan=lifespan)


def _run_inference(df_feat: pd.DataFrame) -> dict:
    art       = _artifact
    sc        = art["scaler"]
    xgb       = art["xgb"]
    rf        = art["rf"]
    cal       = art["calibrator"]
    all_feats = art.get("all_features", art.get("features", []))
    sel_idx   = art.get("sel_idx")
    thr       = art.get("threshold", 0.5)

    latest = df_feat.iloc[-1]
    row_all = np.array([[
        float(latest[f]) if f in df_feat.columns else 0.0
        for f in all_feats
    ]])

    row_sc    = sc.transform(row_all)
    row_model = row_sc[:, sel_idx] if sel_idx is not None else row_sc

    p_xgb = float(xgb.predict_proba(row_model)[:, 1][0])
    p_rf  = float(rf.predict_proba(row_model)[:, 1][0])
    raw   = (p_xgb + p_rf) / 2
    prob  = float(cal.transform([raw])[0])

    direction = "BUY" if prob >= thr else "SELL"
    cp = prob if direction == "BUY" else 1 - prob

    if   cp >= 0.72: confidence = "HIGH"
    elif cp >= 0.62: confidence = "MEDIUM"
    elif cp >= 0.55: confidence = "LOW"
    else:            confidence = "SKIP"

    return {
        "prob_up":    round(prob, 4),
        "prob_dn":    round(1 - prob, 4),
        "direction":  direction,
        "confidence": confidence,
        "model_meta": art.get("meta", {}),
    }


def _get_price(symbol: str = SYMBOL) -> float:
    r = requests.get(f"{BINANCE_SPOT}/ticker/price",
                     params={"symbol": symbol}, timeout=5)
    r.raise_for_status()
    return float(r.json()["price"])


@app.get("/status")
async def status():
    binance_ok = True
    try:
        _get_price()
    except Exception:
        binance_ok = False

    art_info = None
    if _artifact:
        all_feats = _artifact.get("all_features", _artifact.get("features", []))
        art_info  = {
            "n_features": len(all_feats),
            "version":    _artifact.get("meta", {}).get("version"),
            "test_auc":   _artifact.get("meta", {}).get("test_auc"),
            "threshold":  _artifact.get("threshold", 0.5),
        }

    return {
        "status":       "ok" if (_artifact and binance_ok) else "degraded",
        "model_loaded": _artifact is not None,
        "artifact":     art_info,
        "binance":      binance_ok,
        "ts":           datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
async def predict(symbol: str = SYMBOL,
                  interval: str = BASE_TF,
                  capital: float = 10_000.0):

    if _artifact is None:
        raise HTTPException(503, "Model not loaded — run trainer first!")

    try:
        df_raw  = build_raw_dataset(symbol, interval, total=300)
        df_feat = build_features(df_raw, include_market_data=True)

        if df_feat.empty:
            raise HTTPException(500, "Feature build failed")

        pred             = _run_inference(df_feat)
        pred["symbol"]   = symbol
        pred["interval"] = interval

        price = float(df_feat["close"].iloc[-1])
        atr_v = float(df_feat["atr_14"].iloc[-1] if "atr_14" in df_feat.columns else price * 0.01)
        mkt_r = int(df_feat["market_regime"].iloc[-1] if "market_regime" in df_feat.columns else 0)

        risk = calc_sl_tp(price, atr_v, pred["direction"])
        pred.update(risk)
        pos  = position_size(capital, risk["entry"], risk["sl"])

        tf_align = {
            "score":         2 if mkt_r != 0 else 1,
            "total":         3,
            "htf_direction": pred["direction"] if mkt_r == 1 else ("SELL" if mkt_r == -1 else ""),
            "aligned":       mkt_r != 0,
        }

        filt = filter_signal(pred, mkt_r, tf_align)

        result = {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "symbol":        symbol,
            "interval":      interval,
            "price":         price,
            "direction":     pred["direction"],
            "prob_up":       pred["prob_up"],
            "prob_dn":       pred["prob_dn"],
            "confidence":    pred["confidence"],
            "entry":         risk["entry"],
            "sl":            risk["sl"],
            "tp":            risk["tp"],
            "rr":            risk["rr"],
            "stop_pct":      risk["stop_pct"],
            "atr":           risk["atr"],
            "market_regime": mkt_r,
            "tf_alignment":  tf_align,
            "signal_filter": filt,
            "should_alert":  filt["pass"],
            "quality":       filt.get("quality", "C"),
            "position":      pos,
            "model_meta":    pred.get("model_meta", {}),
        }

        if filt["pass"]:
            sid = save_signal(pred, risk, mkt_r, tf_align["score"], filt["quality"])
            result["signal_id"] = sid

        auto_close_pending({symbol: price})
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        log.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@app.get("/stats")
async def stats(last_n: int = 20):
    return JSONResponse(get_stats(last_n))


@app.post("/reload")
async def reload_model():
    _load_model()
    return {"status": "reloaded", "loaded": _artifact is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)