# features/engineer.py  — v2 (news + OI + MTF features)
import numpy as np
import pandas as pd


# ── Indicator helpers ──────────────────────────────────────────
def _ema(s, p):  return s.ewm(span=p, adjust=False).mean()

def _rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _macd(close, f=12, s=26, sig=9):
    ml = _ema(close, f) - _ema(close, s)
    sl = _ema(ml, sig)
    return ml, sl, ml - sl

def _bb(close, p=20, mult=2):
    mid = close.rolling(p).mean()
    std = close.rolling(p).std()
    up  = mid + mult * std
    lo  = mid - mult * std
    pct = (close - lo) / (up - lo + 1e-9)
    wid = (up - lo) / (mid + 1e-9)
    return up, mid, lo, pct, wid

def _atr(high, low, close, p=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

def _vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)

def _regime(close, atr_s, fast=20, slow=50, mult=0.5):
    ef   = _ema(close, fast)
    es   = _ema(close, slow)
    diff = ef - es
    zone = atr_s * mult
    r    = pd.Series(0, index=close.index)
    r[diff >  zone] = 1
    r[diff < -zone] = -1
    return r


# ── Base feature builder ───────────────────────────────────────
def build_features(df: pd.DataFrame,
                   include_market_data: bool = True) -> pd.DataFrame:
    """Build all features. ALL shifted by 1 — zero lookahead."""
    d = df.copy()

    e8   = _ema(d["close"], 8)
    e13  = _ema(d["close"], 13)
    e21  = _ema(d["close"], 21)
    e50  = _ema(d["close"], 50)
    e200 = _ema(d["close"], 200)
    r14  = _rsi(d["close"], 14)
    r7   = _rsi(d["close"], 7)
    r21  = _rsi(d["close"], 21)
    ml, ms, mh = _macd(d["close"])
    bu, bm, bl, bp, bw = _bb(d["close"])
    at14 = _atr(d["high"], d["low"], d["close"], 14)
    at7  = _atr(d["high"], d["low"], d["close"], 7)
    vw   = _vwap(d["high"], d["low"], d["close"], d["volume"])
    vm5  = d["volume"].rolling(5).mean()
    vm20 = d["volume"].rolling(20).mean()
    vm50 = d["volume"].rolling(50).mean()
    vr   = _regime(d["close"], at14)
    vol_pct = at14.rolling(100).rank(pct=True)

    # ── EMA / trend ───────────────────────────────────────────
    d["ema_21"]           = e21.shift(1)
    d["ema_50"]           = e50.shift(1)
    d["ema_200"]          = e200.shift(1)
    d["price_vs_ema21"]   = ((d["close"] - e21) / e21).shift(1)
    d["price_vs_ema200"]  = ((d["close"] - e200) / e200).shift(1)
    d["ema_trend"]        = (e21 > e50).astype(int).shift(1)
    d["ema_8_13_gap"]     = ((e8 - e13) / d["close"]).shift(1)
    d["ema_13_34_gap"]    = ((e13 - _ema(d["close"], 34)) / d["close"]).shift(1)

    # ── RSI ───────────────────────────────────────────────────
    d["rsi_14"]         = r14.shift(1)
    d["rsi_7"]          = r7.shift(1)
    d["rsi_21"]         = r21.shift(1)
    d["rsi_change"]     = r14.diff().shift(1)
    d["rsi_slope"]      = r7.diff(3).shift(1)
    d["rsi_div"]        = (r7 - r21).shift(1)
    d["rsi_overbought"] = (r14 > 70).astype(int).shift(1)
    d["rsi_oversold"]   = (r14 < 30).astype(int).shift(1)

    # ── MACD ──────────────────────────────────────────────────
    d["macd_hist"]        = mh.shift(1)
    d["macd_hist_change"] = mh.diff().shift(1)
    d["macd_cross"]       = (ml > ms).astype(int).shift(1)

    # ── Bollinger ─────────────────────────────────────────────
    d["bb_pct"]    = bp.shift(1)
    d["bb_width"]  = bw.shift(1)
    d["bb_squeeze"]= (bw < bw.rolling(20).mean()).astype(int).shift(1)

    # ── ATR / volatility ──────────────────────────────────────
    d["atr_14"]      = at14.shift(1)
    d["atr_pct"]     = (at14 / d["close"]).shift(1)
    d["atr_ratio"]   = (at7 / (at14 + 1e-9)).shift(1)   # compression
    d["atr_slope"]   = at14.diff(5).shift(1)
    d["vol_regime"]  = vol_pct.shift(1)
    d["market_regime"] = vr.shift(1)
    d["is_trending"] = (vr.abs() > 0).astype(int).shift(1)

    # ── VWAP / volume ─────────────────────────────────────────
    d["price_above_vwap"] = (d["close"] > vw).astype(int).shift(1)
    d["vol_ratio"]        = (d["volume"] / vm20).shift(1)
    d["vol_ratio_5"]      = (d["volume"] / (vm5 + 1e-9)).shift(1)
    d["vol_ratio_50"]     = (d["volume"] / (vm50 + 1e-9)).shift(1)
    d["vol_surge"]        = (d["vol_ratio"] > 2.0).astype(int).shift(1)
    d["vol_trend"]        = (vm5 / (vm20 + 1e-9)).shift(1)

    # ── Momentum ──────────────────────────────────────────────
    d["mom_1"]   = d["close"].pct_change(1).shift(1)
    d["mom_3"]   = d["close"].pct_change(3).shift(1)
    d["mom_5"]   = d["close"].pct_change(5).shift(1)
    d["mom_15"]  = d["close"].pct_change(15).shift(1)
    d["mom_24"]  = d["close"].pct_change(24).shift(1)
    d["mom_48"]  = d["close"].pct_change(48).shift(1)
    d["mom_168"] = d["close"].pct_change(168).shift(1)

    # Price acceleration
    ret1 = d["close"].pct_change(1)
    d["accel_1"] = ret1.diff(1).shift(1)
    d["accel_3"] = ret1.rolling(3).mean().diff(3).shift(1)

    # ── Candle shape ──────────────────────────────────────────
    pc = d["close"].shift(1); po = d["open"].shift(1)
    ph = d["high"].shift(1);  pl = d["low"].shift(1)
    bt  = pd.concat([pc, po], axis=1).max(axis=1)
    bb2 = pd.concat([pc, po], axis=1).min(axis=1)
    range_ = (ph - pl).replace(0, np.nan)
    body   = (pc - po).abs()
    d["body_pct"]    = (body / (pc + 1e-9))
    d["body_ratio"]  = (body / range_).fillna(0.5)
    d["upper_wick"]  = (ph - bt)  / (pc + 1e-9)
    d["lower_wick"]  = (bb2 - pl) / (pc + 1e-9)
    d["wick_ratio"]  = ((ph - pl - body) / range_).fillna(0)
    d["is_bull"]     = (pc > po).astype(int)
    # Consecutive candles
    bull = (d["close"] > d["open"]).astype(int)
    d["consec_bull"] = bull.rolling(3).sum().shift(1)
    d["consec_bear"] = (1 - bull).rolling(3).sum().shift(1)

    # ── Support / resistance ──────────────────────────────────
    rh20 = d["high"].rolling(20).max()
    rl20 = d["low"].rolling(20).min()
    rh5  = d["high"].rolling(5).max()
    rl5  = d["low"].rolling(5).min()
    d["near_high_20"] = ((rh20 - d["close"]) / d["close"]).shift(1)
    d["near_low_20"]  = ((d["close"] - rl20) / d["close"]).shift(1)
    d["near_high_5"]  = ((rh5 - d["close"]) / d["close"]).shift(1)
    d["near_low_5"]   = ((d["close"] - rl5) / d["close"]).shift(1)

    # ── Market data (volume delta, funding, OI, F&G, news) ───
    if include_market_data:
        # Volume delta
        for col, default in [
            ("vol_delta_pct",   0.0),
            ("cum_delta_10",    0.0),
            ("taker_buy_ratio", 0.5),
        ]:
            src = df.get(col, pd.Series(default, index=df.index))
            d[col] = src.shift(1)

        # Funding rate
        for col, default in [
            ("funding_rate",          0.0),
            ("funding_rate_change",   0.0),
            ("funding_rate_momentum", 0.0),
        ]:
            src = df.get(col, pd.Series(default, index=df.index))
            d[col] = src.shift(1)

        # Open Interest (real data)
        for col, default in [
            ("oi_change",    0.0),
            ("oi_val_change",0.0),
            ("oi_vs_ma5",    0.0),
        ]:
            src = df.get(col, pd.Series(default, index=df.index))
            d[col] = src.shift(1)

        # Fear & Greed (enhanced)
        for col, default in [
            ("fg_value",  50.0),
            ("fg_zone",    0.0),
            ("fg_change",  0.0),
        ]:
            src = df.get(col, pd.Series(default, index=df.index))
            d[col] = src.shift(1)

        # News sentiment (from news.py, may be 0 during training)
        for col, default in [
            ("news_sentiment", 0.0),
            ("news_sent_6h",   0.0),
            ("news_sent_24h",  0.0),
            ("news_count_6h",  0.0),
            ("news_momentum",  0.0),
        ]:
            src = df.get(col, pd.Series(default, index=df.index))
            d[col] = src.shift(1)

    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


# ── Feature lists ──────────────────────────────────────────────
BASE_FEATURES = [
    # EMA
    "ema_21","ema_50","ema_200",
    "price_vs_ema21","price_vs_ema200","ema_trend",
    "ema_8_13_gap","ema_13_34_gap",
    # RSI
    "rsi_14","rsi_7","rsi_21","rsi_change","rsi_slope","rsi_div",
    "rsi_overbought","rsi_oversold",
    # MACD
    "macd_hist","macd_hist_change","macd_cross",
    # Bollinger
    "bb_pct","bb_width","bb_squeeze",
    # ATR / regime
    "atr_14","atr_pct","atr_ratio","atr_slope",
    "vol_regime","market_regime","is_trending",
    # Volume
    "price_above_vwap","vol_ratio","vol_ratio_5","vol_ratio_50",
    "vol_surge","vol_trend",
    # Momentum
    "mom_1","mom_3","mom_5","mom_15","mom_24","mom_48","mom_168",
    "accel_1","accel_3",
    # Candle
    "body_pct","body_ratio","upper_wick","lower_wick","wick_ratio",
    "is_bull","consec_bull","consec_bear",
    # Support/resistance
    "near_high_20","near_low_20","near_high_5","near_low_5",
]

MARKET_FEATURES = [
    "vol_delta_pct","cum_delta_10","taker_buy_ratio",
    "funding_rate","funding_rate_change","funding_rate_momentum",
    "oi_change","oi_val_change","oi_vs_ma5",
    "fg_value","fg_zone","fg_change",
]

NEWS_FEATURES = [
    "news_sentiment","news_sent_6h","news_sent_24h",
    "news_count_6h","news_momentum",
]

ALL_FEATURES = BASE_FEATURES + MARKET_FEATURES + NEWS_FEATURES


# ── Label generators ───────────────────────────────────────────
def make_labels_triple_barrier(df, horizon=3, sl_mult=1.0, tp_mult=1.5):
    at     = _atr(df["high"], df["low"], df["close"], 14).values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)
    labels = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        entry = closes[i]; a = at[i]
        tp = entry + tp_mult * a
        sl = entry - sl_mult * a
        for j in range(i+1, min(i+horizon+1, n)):
            if highs[j] >= tp: labels[i] = 1; break
            if lows[j]  <= sl: labels[i] = 0; break
    return pd.Series(labels, index=df.index)


def make_labels_volatility_adjusted(df, horizon=3, mult=0.4):
    at  = _atr(df["high"], df["low"], df["close"], 14)
    thr = mult * at / df["close"]
    fut = df["close"].shift(-horizon)
    pct = (fut - df["close"]) / df["close"]
    return (pct > thr).astype(int)


def make_labels_simple(df, horizon=3, threshold=0.003):
    fut = df["close"].shift(-horizon)
    pct = (fut - df["close"]) / df["close"]
    return (pct > threshold).astype(int)


def select_features(model, feature_list, top_n=30):
    imp = pd.Series(model.feature_importances_,
                    index=feature_list).sort_values(ascending=False)
    sel = imp.head(top_n).index.tolist()
    print(f"  Feature selection: {len(feature_list)} → {len(sel)}")
    return sel
