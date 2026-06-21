# features/signal_filter.py — v2 (fixed confidence attribute)
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SignalScore:
    direction:  str
    prob:       float
    score:      int
    max_score:  int   = 7
    passed:     bool  = False
    filters:    dict  = field(default_factory=dict)
    entry:      float = 0.0
    sl:         float = 0.0
    tp:         float = 0.0
    rr:         float = 0.0
    atr:        float = 0.0
    rsi:        float = 0.0
    regime:     str   = ""
    fg_value:   float = 50.0
    confidence: str   = "LOW"
    timestamp:  Optional[str] = None


def filter_probability(prob, direction, buy_thr=0.70, sell_thr=0.30):
    if direction == "BUY":
        ok = prob >= buy_thr
        return ok, f"prob={prob:.3f} {'>='+str(buy_thr) if ok else '<'+str(buy_thr)}"
    ok = prob <= sell_thr
    return ok, f"prob={prob:.3f} {'<='+str(sell_thr) if ok else '>'+str(sell_thr)}"

def filter_trend(ema50, ema200, direction, price=None):
    bullish = ema50 > ema200
    gap = (ema50 - ema200) / (ema200 + 1e-9) * 100
    if direction == "BUY":
        ok = bullish
        return ok, f"EMA50 {'>' if bullish else '<'} EMA200 (gap={gap:+.2f}%)"
    ok = not bullish
    return ok, f"EMA50 {'<' if not bullish else '>'} EMA200 (gap={gap:+.2f}%)"

def filter_volatility(atr, price, min_atr_pct=0.01):
    atr_pct = atr / (price + 1e-9)
    ok = atr_pct >= min_atr_pct
    return ok, f"ATR%={atr_pct*100:.2f}%"

def filter_rsi_momentum(rsi, direction,
                         buy_min=45, buy_max=70,
                         sell_min=30, sell_max=55):
    if direction == "BUY":
        ok = buy_min <= rsi <= buy_max
        return ok, f"RSI={rsi:.1f} {'in' if ok else 'outside'} [{buy_min},{buy_max}]"
    ok = sell_min <= rsi <= sell_max
    return ok, f"RSI={rsi:.1f} {'in' if ok else 'outside'} [{sell_min},{sell_max}]"

def filter_bb_squeeze_breakout(price, bb_upper, bb_lower,
                                bb_width, bb_width_ma, direction):
    squeezed   = bb_width < bb_width_ma
    near_upper = price > (bb_lower + 0.6 * (bb_upper - bb_lower))
    near_lower = price < (bb_lower + 0.4 * (bb_upper - bb_lower))
    if direction == "BUY":
        ok = squeezed or near_upper
        return ok, f"squeeze={'yes' if squeezed else 'no'}, near_upper={'yes' if near_upper else 'no'}"
    ok = squeezed or near_lower
    return ok, f"squeeze={'yes' if squeezed else 'no'}, near_lower={'yes' if near_lower else 'no'}"

def filter_fear_greed(fg_value, direction,
                       extreme_fear=20, extreme_greed=80):
    if direction == "BUY":
        ok = fg_value <= extreme_greed
        return ok, f"F&G={fg_value:.0f} ({'OK' if ok else 'EXTREME GREED'})"
    ok = fg_value >= extreme_fear
    return ok, f"F&G={fg_value:.0f} ({'OK' if ok else 'EXTREME FEAR'})"

def filter_volume(vol_ratio, min_ratio=0.8):
    ok = vol_ratio >= min_ratio
    return ok, f"vol_ratio={vol_ratio:.2f}"


def score_signal(direction, prob, price,
                 ema50, ema200, atr, rsi,
                 bb_upper, bb_lower, bb_width, bb_width_ma,
                 fg_value, vol_ratio,
                 prob_buy_thr=0.70, prob_sell_thr=0.30,
                 min_atr_pct=0.01, min_score=5,
                 sl_atr_mult=1.5, tp_atr_mult=2.5,
                 timestamp=None) -> SignalScore:

    filters = {}
    filters["1_probability"] = filter_probability(prob, direction, prob_buy_thr, prob_sell_thr)
    filters["2_trend"]       = filter_trend(ema50, ema200, direction, price)
    filters["3_volatility"]  = filter_volatility(atr, price, min_atr_pct)
    filters["4_rsi"]         = filter_rsi_momentum(rsi, direction)
    filters["5_bb"]          = filter_bb_squeeze_breakout(price, bb_upper, bb_lower, bb_width, bb_width_ma, direction)
    filters["6_fg"]          = filter_fear_greed(fg_value, direction)
    filters["7_volume"]      = filter_volume(vol_ratio)

    score  = sum(1 for ok, _ in filters.values() if ok)
    passed = (score >= min_score) and filters["1_probability"][0]

    if direction == "BUY":
        sl = round(price - sl_atr_mult * atr, 2)
        tp = round(price + tp_atr_mult * atr, 2)
    else:
        sl = round(price + sl_atr_mult * atr, 2)
        tp = round(price - tp_atr_mult * atr, 2)
    rr = round(tp_atr_mult / sl_atr_mult, 1)

    cp = prob if direction == "BUY" else 1 - prob
    if   cp >= 0.72: confidence = "HIGH"
    elif cp >= 0.62: confidence = "MEDIUM"
    elif cp >= 0.55: confidence = "LOW"
    else:            confidence = "SKIP"

    regime = "Uptrend" if ema50 > ema200 else "Downtrend"

    return SignalScore(
        direction  = direction,
        prob       = round(prob, 4),
        score      = score,
        passed     = passed,
        filters    = filters,
        entry      = round(price, 2),
        sl         = sl,
        tp         = tp,
        rr         = rr,
        atr        = round(atr, 2),
        rsi        = round(rsi, 1),
        regime     = regime,
        fg_value   = fg_value,
        confidence = confidence,
        timestamp  = timestamp,
    )


def select_best_n_per_day(scored_signals, n=2):
    passed = [s for s in scored_signals if s.passed]
    if not passed: return []
    passed.sort(key=lambda s: (s.score, s.prob), reverse=True)
    return passed[:n]
