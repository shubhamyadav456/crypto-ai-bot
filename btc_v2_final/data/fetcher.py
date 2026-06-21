# data/fetcher.py — v3 (CoinGecko market_chart — proper hourly data)
import time, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BINANCE_FUTURES

def _session():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=1.0, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

SESSION = _session()
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def fetch_klines_coingecko(symbol="BTCUSDT", interval="1h", total=500):
    """
    CoinGecko market_chart gives proper hourly OHLCV.
    For hourly: max 90 days per call.
    We fetch in chunks if needed.
    """
    # Calculate days needed
    days_needed = max(90, (total // 24) + 10)
    days_needed = min(days_needed, 365)

    print(f"  [FETCH] CoinGecko market_chart: {days_needed} days...")

    all_prices  = []
    all_volumes = []

    # Fetch in 90-day chunks for hourly data
    chunks = []
    end_ts = int(datetime.utcnow().timestamp())
    remaining = days_needed

    while remaining > 0:
        chunk_days = min(90, remaining)
        start_ts   = end_ts - chunk_days * 86400
        chunks.append((start_ts, end_ts))
        end_ts    = start_ts
        remaining -= chunk_days

    chunks.reverse()  # oldest first

    for start_ts, end_ts_chunk in chunks:
        try:
            r = SESSION.get(
                f"{COINGECKO_BASE}/coins/bitcoin/market_chart/range",
                params={
                    "vs_currency": "usd",
                    "from": start_ts,
                    "to":   end_ts_chunk,
                },
                timeout=30
            )
            r.raise_for_status()
            data = r.json()

            prices  = data.get("prices", [])
            volumes = data.get("total_volumes", [])

            all_prices.extend(prices)
            all_volumes.extend(volumes)
            time.sleep(1.5)  # CoinGecko rate limit

        except Exception as e:
            print(f"  [FETCH] CoinGecko chunk error: {e}")
            time.sleep(2)
            continue

    if not all_prices:
        raise Exception("CoinGecko returned no data")

    # Build price dataframe
    pdf = pd.DataFrame(all_prices, columns=["ts", "close"])
    pdf["ts"] = pd.to_datetime(pdf["ts"], unit="ms", utc=True)
    pdf.set_index("ts", inplace=True)
    pdf.sort_index(inplace=True)
    pdf = pdf[~pdf.index.duplicated(keep="last")]

    # Build volume dataframe
    vdf = pd.DataFrame(all_volumes, columns=["ts", "volume"])
    vdf["ts"] = pd.to_datetime(vdf["ts"], unit="ms", utc=True)
    vdf.set_index("ts", inplace=True)
    vdf.sort_index(inplace=True)
    vdf = vdf[~vdf.index.duplicated(keep="last")]

    # Resample to hourly
    price_h  = pdf["close"].resample("1h").last().dropna()
    volume_h = vdf["volume"].resample("1h").sum()

    df = pd.DataFrame(index=price_h.index)
    df["close"]  = price_h
    df["volume"] = volume_h.reindex(price_h.index).fillna(0)

    # Estimate OHLC from close prices
    df["open"]  = df["close"].shift(1).fillna(df["close"])
    df["high"]  = df[["open","close"]].max(axis=1) * 1.002
    df["low"]   = df[["open","close"]].min(axis=1) * 0.998

    df["taker_buy_base"] = df["volume"] * 0.52
    df = df[["open","high","low","close","volume","taker_buy_base"]]
    df.dropna(inplace=True)

    # Keep last `total` rows
    df = df.tail(total)
    print(f"  [FETCH] CoinGecko: {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_2years_coingecko(symbol="BTCUSDT", interval="1h"):
    """Fetch 2 years of hourly data from CoinGecko."""
    print(f"  [FETCH] CoinGecko 2yr fetch...")
    return fetch_klines_coingecko(symbol, interval, total=17520)  # 2yr * 365 * 24


def clean_ohlcv(df):
    df = df.copy(); n = len(df)
    df.dropna(subset=["open","high","low","close","volume"], inplace=True)
    df = df[(df["high"] >= df["low"]) & (df["volume"] >= 0)]
    removed = n - len(df)
    if removed > 0:
        print(f"  [CLEAN] Removed {removed} bad candles")
    return df


def add_volume_delta(df):
    df = df.copy()
    df["buy_vol"]         = df["taker_buy_base"]
    df["sell_vol"]        = df["volume"] - df["taker_buy_base"]
    df["vol_delta"]       = df["buy_vol"] - df["sell_vol"]
    df["vol_delta_pct"]   = df["vol_delta"] / (df["volume"] + 1e-9)
    df["cum_delta_10"]    = df["vol_delta"].rolling(10).sum()
    df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)
    return df


def fetch_funding_rate(symbol, limit=1000):
    try:
        r = SESSION.get(f"{BINANCE_FUTURES}/fundingRate",
                        params={"symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df.set_index("ts", inplace=True)
        s = df["funding_rate"].resample("1h").last().ffill()
        print(f"  [FETCH] Funding rate: {len(s)} points")
        return s
    except Exception as e:
        print(f"  [FETCH] Funding rate WARN: {e}")
        return pd.Series(dtype=float)


def fetch_open_interest_history(symbol, period="1h", limit=500):
    try:
        r = SESSION.get("https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol": symbol, "period": period, "limit": limit},
                        timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi"] = df["sumOpenInterest"].astype(float)
        df["oi_val"] = df["sumOpenInterestValue"].astype(float)
        df.set_index("ts", inplace=True); df.sort_index(inplace=True)
        df["oi_change"]     = df["oi"].pct_change(1)
        df["oi_val_change"] = df["oi_val"].pct_change(1)
        df["oi_ma5"]        = df["oi"].rolling(5).mean()
        df["oi_vs_ma5"]     = (df["oi"] - df["oi_ma5"]) / (df["oi_ma5"] + 1e-9)
        print(f"  [FETCH] Open Interest: {len(df)} points")
        return df[["oi","oi_val","oi_change","oi_val_change","oi_vs_ma5"]]
    except Exception as e:
        print(f"  [FETCH] OI WARN: {e}")
        return pd.DataFrame()


def fetch_fear_greed(limit=730):
    try:
        r = SESSION.get("https://api.alternative.me/fng/",
                        params={"limit": limit, "format": "json"}, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["data"])
        df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.normalize()
        df["fg_value"] = df["value"].astype(float)
        def fg_zone(v):
            if v<=25: return -2
            if v<=40: return -1
            if v<=60: return 0
            if v<=75: return 1
            return 2
        df["fg_zone"]   = df["fg_value"].apply(fg_zone)
        df["fg_change"] = df["fg_value"].diff(-1)
        df.set_index("ts", inplace=True)
        print(f"  [FETCH] Fear & Greed: {len(df)} days")
        return df[["fg_value","fg_zone","fg_change"]].sort_index()
    except Exception as e:
        print(f"  [FETCH] F&G WARN: {e}")
        return pd.DataFrame()


def build_raw_dataset(symbol, interval, total=500,
                      use_2years=False, include_oi=True):
    print(f"\n  [DATA] Fetching OHLCV ({symbol} {interval})...")

    if use_2years:
        df = fetch_2years_coingecko(symbol, interval)
    else:
        # Need at least 250 rows for EMA200 + features
        actual_total = max(total, 300)
        df = fetch_klines_coingecko(symbol, interval, actual_total)

    df = clean_ohlcv(df)
    print(f"  [DATA] Raw rows: {len(df)}")

    print("  [DATA] Volume delta...")
    df = add_volume_delta(df)

    print("  [DATA] Funding rate...")
    fr = fetch_funding_rate(symbol)
    if not fr.empty:
        fra = fr.reindex(df.index, method="ffill").fillna(0)
        df["funding_rate"]          = fra
        df["funding_rate_change"]   = fra.diff()
        df["funding_rate_momentum"] = fra.rolling(8).mean()
    else:
        df["funding_rate"] = df["funding_rate_change"] = df["funding_rate_momentum"] = 0.0

    print("  [DATA] Open Interest...")
    if include_oi:
        oi_df = fetch_open_interest_history(symbol, period="1h", limit=500)
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
        fgh = fg.resample("1h").last().ffill()
        for col in ["fg_value","fg_zone","fg_change"]:
            if col in fgh.columns:
                df[col] = fgh[col].reindex(df.index, method="ffill").fillna(
                    50 if col=="fg_value" else 0)
    else:
        df["fg_value"] = 50.0
        df["fg_zone"]  = 0.0
        df["fg_change"]= 0.0

    df.dropna(subset=["open","high","low","close","volume"], inplace=True)
    print(f"  [DATA] Ready: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
    return df
