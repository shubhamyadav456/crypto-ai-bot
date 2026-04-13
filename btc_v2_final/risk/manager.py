# risk/manager.py
"""
Risk Management + Signal Filter
- ATR-based SL/TP
- Dynamic position sizing (fixed fractional)
- Signal quality gate (prob + regime + MTF + R:R)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MIN_PROB_BUY, MAX_PROB_SELL,
    ATR_SL_MULT, ATR_TP_MULT,
    MIN_RR, RISK_PCT,
)


# ── SL / TP ────────────────────────────────────────────────────
def calc_sl_tp(price: float, atr: float, direction: str,
               sl_mult: float = ATR_SL_MULT,
               tp_mult: float = ATR_TP_MULT) -> dict:
    if direction == "BUY":
        sl = price - sl_mult * atr
        tp = price + tp_mult * atr
    else:
        sl = price + sl_mult * atr
        tp = price - tp_mult * atr

    rr       = round(tp_mult / sl_mult, 2)
    stop_pct = round(abs(price - sl) / price * 100, 3)

    return {
        "entry":    round(price, 2),
        "sl":       round(sl, 2),
        "tp":       round(tp, 2),
        "rr":       rr,
        "stop_pct": stop_pct,
        "atr":      round(atr, 2),
    }


# ── Position sizing ────────────────────────────────────────────
def position_size(capital: float, entry: float,
                  sl: float, risk_pct: float = RISK_PCT) -> dict:
    risk_per_unit = abs(entry - sl)
    if risk_per_unit < 1e-9:
        return {"size": 0, "risk_amount": 0, "position_value": 0, "leverage_implied": 0}
    risk_amount = capital * risk_pct
    size  = risk_amount / risk_per_unit
    value = size * entry
    return {
        "size":             round(size, 6),
        "risk_amount":      round(risk_amount, 2),
        "position_value":   round(value, 2),
        "leverage_implied": round(value / capital, 2),
    }


# ── Signal quality gate ────────────────────────────────────────
def filter_signal(pred: dict,
                  market_regime: int = 0,
                  tf_alignment: dict = None) -> dict:
    prob_up    = pred.get("prob_up", 0.5)
    prob_dn    = 1 - prob_up
    direction  = pred.get("direction", "")
    confidence = pred.get("confidence", "SKIP")
    rr         = pred.get("rr", 0)

    fails = []

    # Gate 1: probability
    if direction == "BUY"  and prob_up < MIN_PROB_BUY:
        fails.append(f"prob_up {prob_up:.2f} < {MIN_PROB_BUY}")
    if direction == "SELL" and prob_dn < (1 - MAX_PROB_SELL):
        fails.append(f"prob_dn {prob_dn:.2f} < {1-MAX_PROB_SELL:.2f}")

    # Gate 2: confidence
    if confidence in ("SKIP", "LOW"):
        fails.append(f"confidence={confidence}")

    # Gate 3: trending market
    if market_regime == 0:
        fails.append("market is sideways")

    # Gate 4: MTF alignment
    if tf_alignment:
        score   = tf_alignment.get("score", 0)
        total   = tf_alignment.get("total", 3)
        htf_dir = tf_alignment.get("htf_direction", "")
        if score < 2:
            fails.append(f"MTF weak ({score}/{total})")
        if htf_dir and htf_dir != direction:
            fails.append(f"HTF conflict: {htf_dir} vs {direction}")

    # Gate 5: minimum R:R
    if rr > 0 and rr < MIN_RR:
        fails.append(f"R:R {rr} < {MIN_RR}")

    if fails:
        return {"pass": False, "reason": " | ".join(fails), "quality": "SKIP"}

    if confidence == "HIGH" and prob_up >= 0.70:
        quality = "A"
    elif confidence in ("HIGH", "MEDIUM"):
        quality = "B"
    else:
        quality = "C"

    return {"pass": True, "reason": "All gates passed", "quality": quality}


# ── MTF alignment checker ──────────────────────────────────────
def check_mtf_alignment(predictions: dict) -> dict:
    """
    predictions = {"1h": {"direction": "BUY", "prob_up": 0.68}, ...}
    """
    dirs = [v["direction"] for v in predictions.values() if "direction" in v]
    if not dirs:
        return {"score": 0, "total": 0, "htf_direction": "", "aligned": False}

    htf_dir = (predictions.get("4h") or predictions.get("1h") or {}).get("direction", dirs[0])
    matching = sum(1 for d in dirs if d == htf_dir)

    return {
        "score":         matching,
        "total":         len(dirs),
        "htf_direction": htf_dir,
        "aligned":       matching >= 2,
        "pct":           round(matching / len(dirs) * 100),
    }
