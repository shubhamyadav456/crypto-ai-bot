# data/fetcher.py  — v2 (cleaned + OI + MTF)
"""
Improvements:
  - Outlier candles removed (price gap >5%, volume >15x median)
  - Duplicate timestamps removed
  - Open Interest from Binance Futures (real data)
  - Multi-timeframe support (15m, 1h, 4h)
  - Bid/ask spread from ticker
  - Only essential columns kept
"""
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BINANCE_SPOT, BINANCE_FUTURES

# ── Session ────────────────────────────────────────────────────
def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=4, backoff_factor=0.6,
              status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

SESSION = _session()

_RAW_COLS = ["open_time","open","high","low","close","volume",
             "close_time","quote_vol","trades",
             "taker_buy_base","taker_buy_quote","ignore"]
_NUM_COLS  = ["open","high","low","close","volume","taker_buy_base"]
_KEEP_COLS = ["open","high","low","close","volume","taker_buy_base"]


# ── Parse raw klines ───────────────────────────────────────────
def _parse(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=_RAW_COLS)
    for c in _NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]   # remove duplicates
    return df[_KEEP_COLS]


# ── Data cleaning ──────────────────────────────────────────────
def clean_ohlcv(df: pd.DataFrame,
                max_price_gap: float = 0.05,
                max_vol_mult:  float = 15.0) -> pd.DataFrame:
    """
    Remove outlier candles:
      - Price gap > max_price_gap (5%) between consecutive closes
      - Volume spike > max_vol_mult x rolling median (15x)
      - Any NaN in OHLCV
    """
    df = df.copy()
    initial = len(df)

    # Drop NaN
    df.dropna(subset=["open","high","low","close","volume"], inplace=True)

    # Price gap filter
    price_gap = df["close"].pct_change().abs()
    df = df[price_gap <= max_price_gap]

    # Volume spike filter
    vol_med = df["volume"].rolling(48, min_periods=10).median()
    vol_ratio = df["volume"] / (vol_med + 1e-9)
    df = df[vol_ratio <= max_vol_mult]

    # OHLC sanity: high >= low, high >= close, low <= close
    df = df[(df["high"] >= df["low"]) &
            (df["high"] >= df["close"]) &
            (df["low"]  <= df["close"]) &
            (df["volume"] > 0)]

    removed = initial - len(df)
    if removed > 0:
        print(f"  [CLEAN] Removed {removed} outlier candles ({removed/initial*100:.1f}%)")

    return df


# ── Paginated klines fetch ─────────────────────────────────────
def fetch_klines(symbol: str, interval: str,
                 total: int = 1000) -> pd.DataFrame:
    all_rows = []
    end_ms   = None
    while len(all_rows) < total:
        need   = min(1000, total - len(all_rows))
        params = {"symbol": symbol, "interval": interval, "limit": need}
        if end_ms:
            params["endTime"] = end_ms
        r = SESSION.get(f"{BINANCE_SPOT}/klines", params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows = batch + all_rows
        end_ms   = batch[0][0] - 1
        time.sleep(0.2)
    df = _parse(all_rows)
    df = clean_ohlcv(df)
    print(f"  [FETCH] {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── 2-year fetch ───────────────────────────────────────────────
def fetch_2years(symbol: str = "BTCUSDT",
                 interval: str = "1h") -> pd.DataFrame:
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=730)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    all_rows = []
    cur      = start_ms

    print(f"  [FETCH] 2yr: {start_dt.date()} → {end_dt.date()}")
    while cur < end_ms:
        r = SESSION.get(f"{BINANCE_SPOT}/klines", params={
            "symbol": symbol, "interval": interval,
            "startTime": cur, "limit": 1000,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        cur = batch[-1][0] + 1
        time.sleep(0.2)

    df = _parse(all_rows)
    df = clean_ohlcv(df)
    print(f"  [FETCH] Total: {len(df)} candles (after cleaning)")
    return df


# ── Volume delta ───────────────────────────────────────────────
def add_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["buy_vol"]        = df["taker_buy_base"]
    df["sell_vol"]       = df["volume"] - df["taker_buy_base"]
    df["vol_delta"]      = df["buy_vol"] - df["sell_vol"]
    df["vol_delta_pct"]  = df["vol_delta"] / (df["volume"] + 1e-9)
    df["cum_delta_10"]   = df["vol_delta"].rolling(10).sum()
    df["taker_buy_ratio"]= df["taker_buy_base"] / (df["volume"] + 1e-9)
    return df


# ── Funding rate ───────────────────────────────────────────────
def fetch_funding_rate(symbol: str, limit: int = 1000) -> pd.Series:
    try:
        r = SESSION.get(f"{BINANCE_FUTURES}/fundingRate",
                        params={"symbol": symbol, "limit": limit},
                        timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["ts"]           = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df.set_index("ts", inplace=True)
        series = df["funding_rate"].resample("1h").last().ffill()
        print(f"  [FETCH] Funding rate: {len(series)} points")
        return series
    except Exception as e:
        print(f"  [FETCH] Funding rate WARN: {e}")
        return pd.Series(dtype=float)


# ── Open Interest (real Binance Futures data) ──────────────────
def fetch_open_interest_history(symbol: str,
                                 period: str = "1h",
                                 limit: int = 500) -> pd.DataFrame:
    """
    Fetch real Open Interest history from Binance Futures.
    Returns DataFrame with oi (sumOpenInterest) and oi_val columns.
    """
    try:
        r = SESSION.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"]     = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi"]     = df["sumOpenInterest"].astype(float)
        df["oi_val"] = df["sumOpenInterestValue"].astype(float)
        df.set_index("ts", inplace=True)
        df.sort_index(inplace=True)

        # Derived features
        df["oi_change"]     = df["oi"].pct_change(1)
        df["oi_val_change"] = df["oi_val"].pct_change(1)
        df["oi_ma5"]        = df["oi"].rolling(5).mean()
        df["oi_vs_ma5"]     = (df["oi"] - df["oi_ma5"]) / (df["oi_ma5"] + 1e-9)
        print(f"  [FETCH] Open Interest: {len(df)} points")
        return df[["oi","oi_val","oi_change","oi_val_change","oi_vs_ma5"]]
    except Exception as e:
        print(f"  [FETCH] OI WARN: {e}")
        return pd.DataFrame()


# ── Fear & Greed ───────────────────────────────────────────────
def fetch_fear_greed(limit: int = 730) -> pd.DataFrame:
    try:
        r = SESSION.get("https://api.alternative.me/fng/",
                        params={"limit": limit, "format": "json"},
                        timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["data"])
        df["ts"]       = pd.to_datetime(df["timestamp"].astype(int),
                                         unit="s", utc=True).dt.normalize()
        df["fg_value"] = df["value"].astype(float)
        # Extra: zone encoding
        def fg_zone(v):
            if v <= 25:  return -2   # extreme fear
            if v <= 40:  return -1   # fear
            if v <= 60:  return  0   # neutral
            if v <= 75:  return  1   # greed
            return 2                  # extreme greed
        df["fg_zone"]   = df["fg_value"].apply(fg_zone)
        df["fg_change"] = df["fg_value"].diff(-1)   # change from yesterday
        df.set_index("ts", inplace=True)
        print(f"  [FETCH] Fear & Greed: {len(df)} days")
        return df[["fg_value","fg_zone","fg_change"]].sort_index()
    except Exception as e:
        print(f"  [FETCH] F&G WARN: {e}")
        return pd.DataFrame()


# ── Order book snapshot ────────────────────────────────────────
def fetch_orderbook_snapshot(symbol: str, depth: int = 20) -> dict:
    try:
        r = SESSION.get(f"{BINANCE_SPOT}/depth",
                        params={"symbol": symbol, "limit": depth},
                        timeout=8)
        r.raise_for_status()
        ob      = r.json()
        bids    = [(float(p), float(q)) for p, q in ob["bids"]]
        asks    = [(float(p), float(q)) for p, q in ob["asks"]]
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        total   = bid_vol + ask_vol + 1e-9
        # Weighted mid price
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread   = (best_ask - best_bid) / (best_bid + 1e-9)
        return {
            "ob_imbalance":  (bid_vol - ask_vol) / total,
            "bid_ask_ratio":  bid_vol / (ask_vol + 1e-9),
            "spread_pct":     spread,
        }
    except Exception:
        return {"ob_imbalance": 0.0, "bid_ask_ratio": 1.0, "spread_pct": 0.0}


# ── Multi-timeframe fetch ──────────────────────────────────────
def fetch_higher_tf(symbol: str, interval: str = "4h",
                    total: int = 500) -> pd.DataFrame:
    """Fetch higher timeframe OHLCV for MTF features."""
    try:
        df = fetch_klines(symbol, interval, total)
        # Rename columns with tf prefix
        df.columns = [f"{interval}_{c}" for c in df.columns]
        return df
    except Exception as e:
        print(f"  [FETCH] HTF {interval} WARN: {e}")
        return pd.DataFrame()


# ── Full dataset builder ───────────────────────────────────────
def build_raw_dataset(symbol: str, interval: str,
                      total: int = 1000,
                      use_2years: bool = False,
                      include_oi: bool = True) -> pd.DataFrame:
    print(f"\n  [DATA] Fetching OHLCV ({symbol} {interval})...")
    df = fetch_2years(symbol, interval) if use_2years \
         else fetch_klines(symbol, interval, total)

    print("  [DATA] Volume delta...")
    df = add_volume_delta(df)

    print("  [DATA] Funding rate...")
    fr = fetch_funding_rate(symbol)
    if not fr.empty:
        fr_aligned = fr.reindex(df.index, method="ffill").fillna(0)
        df["funding_rate"]          = fr_aligned
        df["funding_rate_change"]   = fr_aligned.diff()
        df["funding_rate_momentum"] = fr_aligned.rolling(8).mean()
    else:
        df["funding_rate"]          = 0.0
        df["funding_rate_change"]   = 0.0
        df["funding_rate_momentum"] = 0.0

    print("  [DATA] Open Interest...")
    if include_oi:
        oi_limit = 1000 if use_2years else min(total, 500)
        oi_df = fetch_open_interest_history(symbol, period="1h", limit=oi_limit)
        if not oi_df.empty:
            for col in oi_df.columns:
                df[col] = oi_df[col].reindex(df.index, method="ffill").fillna(0)
        else:
            for col in ["oi","oi_val","oi_change","oi_val_change","oi_vs_ma5"]:
                df[col] = 0.0
    else:
        for col in ["oi","oi_val","oi_change","oi_val_change","oi_vs_ma5"]:
            df[col] = 0.0

    print("  [DATA] Fear & Greed...")
    fg = fetch_fear_greed()
    if not fg.empty:
        fg_h = fg.resample("1h").last().ffill()
        for col in ["fg_value","fg_zone","fg_change"]:
            if col in fg_h.columns:
                df[col] = fg_h[col].reindex(df.index, method="ffill").fillna(
                    50 if col == "fg_value" else 0)
    else:
        df["fg_value"] = 50.0
        df["fg_zone"]  = 0.0
        df["fg_change"]= 0.0

    df.dropna(subset=["open","high","low","close","volume"], inplace=True)
    print(f"  [DATA] Ready: {len(df)} rows | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df
