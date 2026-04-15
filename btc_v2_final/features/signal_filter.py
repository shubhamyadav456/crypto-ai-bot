# features/signal_filter.py — High Confidence Signal Filter v1.0
"""
Scoring system — minimum 4/7 required to generate signal.

Filters:
  1. Probability     >= 0.70 BUY  /  <= 0.30 SELL
  2. Trend           EMA50 vs EMA200
  3. Volatility      ATR% > 1%
  4. Momentum        RSI confirmation
  5. BB Squeeze      Breakout detection
  6. Fear/Greed      Avoid extreme zones
  7. Volume          Above average confirmation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalScore:
    direction:   str            # BUY / SELL
    prob:        float
    score:       int            # 0-7
    max_score:   int = 7
    passed:      bool = False
    filters:     dict = field(default_factory=dict)   # name -> (passed, detail)
    entry:       float = 0.0
    sl:          float = 0.0
    tp:          float = 0.0
    rr:          float = 0.0
    atr:         float = 0.0
    rsi:         float = 0.0
    regime:      str   = ""
    fg_value:    float = 50.0
    timestamp:   Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"Direction : {self.direction}",
            f"Score     : {self.score}/{self.max_score}  {'PASS' if self.passed else 'FAIL'}",
            f"Prob      : {self.prob*100:.1f}%",
            f"Entry     : ${self.entry:,.2f}",
            f"SL        : ${self.sl:,.2f}",
            f"TP        : ${self.tp:,.2f}",
            f"R:R       : 1:{self.rr}",
            f"ATR%      : {self.atr/self.entry*100:.2f}%" if self.entry > 0 else "",
            f"RSI       : {self.rsi:.1f}",
            f"Regime    : {self.regime}",
            f"F&G       : {self.fg_value:.0f}",
            "",
            "Filter breakdown:",
        ]
        for name, (ok, detail) in self.filters.items():
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark} {name:<20} {detail}")
        return "\n".join(l for l in lines if l is not None)


# ── Individual filter functions ────────────────────────────────

def filter_probability(prob: float, direction: str,
                        buy_thr: float = 0.70,
                        sell_thr: float = 0.30) -> tuple:
    """Filter 1 — strict probability gate."""
    if direction == "BUY":
        ok = prob >= buy_thr
        return ok, f"prob={prob:.3f} >= {buy_thr}" if ok else f"prob={prob:.3f} < {buy_thr}"
    else:
        ok = prob <= sell_thr
        return ok, f"prob={prob:.3f} <= {sell_thr}" if ok else f"prob={prob:.3f} > {sell_thr}"


def filter_trend(ema50: float, ema200: float, direction: str,
                 price: float = None) -> tuple:
    """Filter 2 — EMA50 vs EMA200 trend alignment."""
    bullish = ema50 > ema200
    if direction == "BUY":
        ok = bullish
        gap = (ema50 - ema200) / ema200 * 100
        return ok, f"EMA50 {'>' if bullish else '<'} EMA200 (gap={gap:+.2f}%)"
    else:
        ok = not bullish
        gap = (ema200 - ema50) / ema200 * 100
        return ok, f"EMA50 {'<' if not bullish else '>'} EMA200 (gap={gap:+.2f}%)"


def filter_volatility(atr: float, price: float,
                       min_atr_pct: float = 0.01) -> tuple:
    """Filter 3 — minimum ATR% to avoid choppy markets."""
    atr_pct = atr / price
    ok = atr_pct >= min_atr_pct
    return ok, f"ATR%={atr_pct*100:.2f}% {'>='+str(min_atr_pct*100)+'%' if ok else '<'+str(min_atr_pct*100)+'%'}"


def filter_rsi_momentum(rsi: float, direction: str,
                         buy_min: float = 45, buy_max: float = 70,
                         sell_min: float = 30, sell_max: float = 55) -> tuple:
    """
    Filter 4 — RSI momentum confirmation.
    BUY : RSI 45-70 (momentum building, not overbought)
    SELL: RSI 30-55 (bearish momentum, not oversold)
    """
    if direction == "BUY":
        ok = buy_min <= rsi <= buy_max
        return ok, f"RSI={rsi:.1f} in [{buy_min},{buy_max}]" if ok else f"RSI={rsi:.1f} outside [{buy_min},{buy_max}]"
    else:
        ok = sell_min <= rsi <= sell_max
        return ok, f"RSI={rsi:.1f} in [{sell_min},{sell_max}]" if ok else f"RSI={rsi:.1f} outside [{sell_min},{sell_max}]"


def filter_bb_squeeze_breakout(price: float, bb_upper: float,
                                bb_lower: float, bb_width: float,
                                bb_width_ma: float, direction: str) -> tuple:
    """
    Filter 5 — Bollinger Band squeeze breakout.
    Squeeze: current BB width < 20-period average width
    Breakout: price breaking out of squeeze in signal direction
    """
    squeezed = bb_width < bb_width_ma
    near_upper = (price > (bb_lower + 0.6 * (bb_upper - bb_lower)))
    near_lower = (price < (bb_lower + 0.4 * (bb_upper - bb_lower)))

    if direction == "BUY":
        ok = squeezed or near_upper
        detail = f"squeeze={'yes' if squeezed else 'no'}, near_upper={'yes' if near_upper else 'no'}"
        return ok, detail
    else:
        ok = squeezed or near_lower
        detail = f"squeeze={'yes' if squeezed else 'no'}, near_lower={'yes' if near_lower else 'no'}"
        return ok, detail


def filter_fear_greed(fg_value: float, direction: str,
                       extreme_fear: float = 20,
                       extreme_greed: float = 80) -> tuple:
    """
    Filter 6 — Fear & Greed zone.
    Avoid: BUY in extreme greed (>80), SELL in extreme fear (<20)
    Contrarian: extreme fear = BUY opportunity (extra bonus point)
    """
    if direction == "BUY":
        ok = fg_value <= extreme_greed
        return ok, f"F&G={fg_value:.0f} ({'OK' if ok else 'EXTREME GREED — avoid'})"
    else:
        ok = fg_value >= extreme_fear
        return ok, f"F&G={fg_value:.0f} ({'OK' if ok else 'EXTREME FEAR — avoid'})"


def filter_volume(vol_ratio: float, min_ratio: float = 0.8) -> tuple:
    """Filter 7 — volume above threshold (not dead market)."""
    ok = vol_ratio >= min_ratio
    return ok, f"vol_ratio={vol_ratio:.2f} {'>='+str(min_ratio) if ok else '<'+str(min_ratio)}"


# ── Main scoring function ──────────────────────────────────────

def score_signal(
    direction:    str,
    prob:         float,
    price:        float,
    ema50:        float,
    ema200:       float,
    atr:          float,
    rsi:          float,
    bb_upper:     float,
    bb_lower:     float,
    bb_width:     float,
    bb_width_ma:  float,
    fg_value:     float,
    vol_ratio:    float,
    # Thresholds
    prob_buy_thr:  float = 0.70,
    prob_sell_thr: float = 0.30,
    min_atr_pct:   float = 0.01,
    min_score:     int   = 4,
    # SL/TP
    sl_atr_mult:   float = 1.0,
    tp_atr_mult:   float = 2.0,
    timestamp:     str   = None,
) -> SignalScore:

    filters = {}

    # Run all filters
    filters["1_probability"]  = filter_probability(prob, direction, prob_buy_thr, prob_sell_thr)
    filters["2_trend"]        = filter_trend(ema50, ema200, direction, price)
    filters["3_volatility"]   = filter_volatility(atr, price, min_atr_pct)
    filters["4_rsi_momentum"] = filter_rsi_momentum(rsi, direction)
    filters["5_bb_squeeze"]   = filter_bb_squeeze_breakout(price, bb_upper, bb_lower, bb_width, bb_width_ma, direction)
    filters["6_fear_greed"]   = filter_fear_greed(fg_value, direction)
    filters["7_volume"]       = filter_volume(vol_ratio)

    # Count score
    score  = sum(1 for ok, _ in filters.values() if ok)
    passed = (score >= min_score) and filters["1_probability"][0]  # prob always required

    # SL/TP
    if direction == "BUY":
        sl = round(price - sl_atr_mult * atr, 2)
        tp = round(price + tp_atr_mult * atr, 2)
    else:
        sl = round(price + sl_atr_mult * atr, 2)
        tp = round(price - tp_atr_mult * atr, 2)
    rr = round(tp_atr_mult / sl_atr_mult, 1)

    regime_map = {True: "Uptrend", False: "Downtrend"}
    regime     = regime_map[ema50 > ema200]

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
        timestamp  = timestamp,
    )


# ── Daily best trade selector ──────────────────────────────────

def select_daily_best(scored_signals: list) -> Optional[SignalScore]:
    """
    From a list of SignalScore objects (same day),
    return the single best signal by score, then prob.
    Max 1 trade per day.
    """
    passed = [s for s in scored_signals if s.passed]
    if not passed:
        return None
    # Sort by score desc, then prob desc
    passed.sort(key=lambda s: (s.score, s.prob), reverse=True)
    return passed[0]


def select_best_n_per_day(scored_signals: list, n: int = 2) -> list:
    """Return top N signals per day (for 1-2 trades/day config)."""
    passed = [s for s in scored_signals if s.passed]
    if not passed:
        return []
    passed.sort(key=lambda s: (s.score, s.prob), reverse=True)
    return passed[:n]