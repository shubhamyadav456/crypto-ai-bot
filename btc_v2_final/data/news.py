# data/news.py
"""
Crypto News Sentiment Fetcher
==============================
Sources:
  1. CryptoPanic API (free tier — no key needed for basic)
  2. Binance announcements RSS (listings, delistings)

Sentiment scoring:
  +2  = very bullish (liked, important + positive)
  +1  = bullish
   0  = neutral
  -1  = bearish
  -2  = very bearish (panic, FUD)

Usage:
  from data.news import fetch_news_sentiment, get_news_features
  features = get_news_features("BTC")
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── HTTP Session ───────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "BTC-AI-Bot/4.3"})


# ── CryptoPanic ────────────────────────────────────────────────
CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts/"

def fetch_cryptopanic(currency: str = "BTC",
                      pages: int = 3,
                      api_key: str = None) -> pd.DataFrame:
    """
    Fetch news from CryptoPanic.
    Free tier works without API key (rate limited).
    Get free key at: https://cryptopanic.com/developers/api/
    Set CRYPTOPANIC_API_KEY in config.env for more requests.
    """
    all_posts = []
    url       = CRYPTOPANIC_BASE

    # Validate API key — must be alphanumeric, 20-64 chars
    import re as _re
    clean_key = (api_key or "").strip()
    if clean_key and not _re.match(r"^[a-zA-Z0-9]{20,64}$", clean_key):
        print(f"  [NEWS] Invalid API key format — using free tier")
        clean_key = None

    for page in range(1, pages + 1):
        params = {
            "currencies": currency,
            "public":     "true",
            "kind":       "news",
        }
        if clean_key:
            params["auth_token"] = clean_key

        try:
            r = _SESSION.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            posts = data.get("results", [])
            if not posts:
                break
            all_posts.extend(posts)

            # Pagination
            next_url = data.get("next")
            if not next_url:
                break
            url = next_url
            time.sleep(0.5)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  [NEWS] CryptoPanic rate limited — using cached")
            else:
                print(f"  [NEWS] CryptoPanic error: {e}")
            break
        except Exception as e:
            print(f"  [NEWS] CryptoPanic fetch error: {e}")
            break

    if not all_posts:
        return pd.DataFrame()

    rows = []
    for p in all_posts:
        # Parse timestamp
        try:
            ts = pd.to_datetime(p.get("published_at") or p.get("created_at"),
                                 utc=True)
        except Exception:
            continue

        title  = (p.get("title") or "").lower()
        source = (p.get("source", {}).get("domain") or "")

        # Votes / sentiment from CryptoPanic
        votes  = p.get("votes", {})
        liked     = int(votes.get("liked",    0))
        disliked  = int(votes.get("disliked", 0))
        important = int(votes.get("important",0))
        negative  = int(votes.get("negative", 0))
        saved     = int(votes.get("saved",    0))
        lol       = int(votes.get("lol",      0))

        # Compute raw sentiment score
        raw_score = (liked + important + saved * 0.5) - (disliked + negative + lol * 0.3)

        # Keyword boost/penalty
        bullish_kw = ["rally","surge","pump","breakout","ath","bull",
                      "buy","accumulate","adoption","etf","halving",
                      "upgrade","partnership","institutional","recovery"]
        bearish_kw = ["crash","dump","bear","sell","hack","scam","ban",
                      "regulation","sec","lawsuit","liquidation","panic",
                      "fear","fraud","arrest","warning","exploit","rug"]

        kw_score = sum(2 for kw in bullish_kw if kw in title) - \
                   sum(2 for kw in bearish_kw if kw in title)

        total_score = raw_score + kw_score

        # CryptoPanic kind/filter
        kind   = p.get("kind", "news")
        filter_= p.get("filter") or ""
        if filter_ == "rising":     total_score += 1
        if filter_ == "hot":        total_score += 1
        if filter_ == "bullish":    total_score += 2
        if filter_ == "bearish":    total_score -= 2
        if filter_ == "important":  total_score += 1

        rows.append({
            "ts":        ts,
            "title":     p.get("title", ""),
            "source":    source,
            "score":     total_score,
            "liked":     liked,
            "disliked":  disliked,
            "important": important,
            "kind":      kind,
            "filter":    filter_,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── Alternative.me Fear & Greed (already in fetcher, here for standalone) ──
def fetch_fear_greed_current() -> dict:
    try:
        r = _SESSION.get("https://api.alternative.me/fng/?limit=1",
                         timeout=8)
        r.raise_for_status()
        d = r.json()["data"][0]
        return {
            "value":       int(d["value"]),
            "class":       d["value_classification"],
            "ts":          datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc),
        }
    except Exception:
        return {"value": 50, "class": "Neutral", "ts": datetime.now(timezone.utc)}


# ── Aggregate news into hourly sentiment series ────────────────
def news_to_hourly(df_news: pd.DataFrame,
                   start: pd.Timestamp = None,
                   end:   pd.Timestamp = None) -> pd.DataFrame:
    """
    Convert raw news posts to hourly sentiment features.
    Returns DataFrame with hourly index aligned to OHLCV data.
    """
    if df_news.empty:
        return pd.DataFrame()

    # Resample to 1h — sum of scores, count of articles
    df_h = df_news["score"].resample("1h").agg(["sum","count"])
    df_h.columns = ["news_score_sum", "news_count"]

    # Normalize score by count
    df_h["news_sentiment"] = df_h["news_score_sum"] / (df_h["news_count"] + 1e-9)

    # Rolling aggregations (look-back windows)
    df_h["news_sent_6h"]  = df_h["news_sentiment"].rolling(6,  min_periods=1).mean()
    df_h["news_sent_24h"] = df_h["news_sentiment"].rolling(24, min_periods=1).mean()
    df_h["news_count_6h"] = df_h["news_count"].rolling(6,  min_periods=1).sum()
    df_h["news_vol_6h"]   = df_h["news_sentiment"].rolling(6,  min_periods=1).std().fillna(0)

    # News momentum: 6h vs 24h
    df_h["news_momentum"] = df_h["news_sent_6h"] - df_h["news_sent_24h"]

    # Clip extremes
    for col in ["news_sentiment","news_sent_6h","news_sent_24h","news_momentum"]:
        df_h[col] = df_h[col].clip(-5, 5)

    return df_h[[
        "news_sentiment","news_sent_6h","news_sent_24h",
        "news_count_6h","news_vol_6h","news_momentum"
    ]]


# ── Main function for bot/trainer ──────────────────────────────
def get_news_features(currency: str = "BTC",
                      api_key: str = None) -> dict:
    """
    Fetch latest news and return current sentiment features.
    Used by telegram_bot.py for real-time signal enrichment.

    Returns dict of feature name → value (last hour).
    """
    print(f"  [NEWS] Fetching {currency} news...")
    df_news = fetch_cryptopanic(currency, pages=2, api_key=api_key)

    if df_news.empty:
        print(f"  [NEWS] No news fetched — using neutral")
        return {
            "news_sentiment":  0.0,
            "news_sent_6h":    0.0,
            "news_sent_24h":   0.0,
            "news_count_6h":   0.0,
            "news_vol_6h":     0.0,
            "news_momentum":   0.0,
        }

    df_h = news_to_hourly(df_news)
    if df_h.empty:
        return {k: 0.0 for k in ["news_sentiment","news_sent_6h",
                                   "news_sent_24h","news_count_6h",
                                   "news_vol_6h","news_momentum"]}

    latest = df_h.iloc[-1]
    features = latest.to_dict()

    # Also add Fear & Greed current
    fg = fetch_fear_greed_current()
    features["fg_current"]  = fg["value"]
    features["fg_class"]    = fg["class"]

    print(f"  [NEWS] Sentiment 6h={features['news_sent_6h']:.2f} | "
          f"24h={features['news_sent_24h']:.2f} | "
          f"Articles 6h={int(features['news_count_6h'])} | "
          f"F&G={fg['value']} ({fg['class']})")

    return features


def get_news_signal(features: dict) -> dict:
    """
    Convert news features to a simple signal for bot alert.
    Returns: signal (BULLISH/BEARISH/NEUTRAL), strength (0-1), reason.
    """
    sent_6h  = features.get("news_sent_6h",  0)
    sent_24h = features.get("news_sent_24h", 0)
    momentum = features.get("news_momentum", 0)
    fg       = features.get("fg_current", 50)
    count    = features.get("news_count_6h", 0)

    score = 0
    reasons = []

    # Sentiment scoring
    if sent_6h > 1.5:
        score += 2; reasons.append(f"Bullish news 6h ({sent_6h:.1f})")
    elif sent_6h > 0.5:
        score += 1; reasons.append(f"Mild bullish news ({sent_6h:.1f})")
    elif sent_6h < -1.5:
        score -= 2; reasons.append(f"Bearish news 6h ({sent_6h:.1f})")
    elif sent_6h < -0.5:
        score -= 1; reasons.append(f"Mild bearish news ({sent_6h:.1f})")

    # Momentum
    if momentum > 1.0:
        score += 1; reasons.append("News turning bullish")
    elif momentum < -1.0:
        score -= 1; reasons.append("News turning bearish")

    # Fear & Greed
    if fg <= 20:
        score += 1; reasons.append(f"Extreme fear (F&G={fg}) — contrarian buy")
    elif fg >= 80:
        score -= 1; reasons.append(f"Extreme greed (F&G={fg}) — caution")

    # Low news volume = uncertain
    if count < 2:
        reasons.append("Low news volume — uncertain")

    # Final signal
    if score >= 2:
        signal = "BULLISH"
    elif score <= -2:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    strength = min(abs(score) / 4.0, 1.0)

    return {
        "signal":   signal,
        "strength": round(strength, 2),
        "score":    score,
        "reasons":  reasons,
        "sent_6h":  round(sent_6h, 2),
        "sent_24h": round(sent_24h, 2),
        "fg":       fg,
    }


NEWS_FEATURE_COLS = [
    "news_sentiment", "news_sent_6h", "news_sent_24h",
    "news_count_6h",  "news_vol_6h",  "news_momentum",
]


if __name__ == "__main__":
    print("Testing news fetcher...")
    features = get_news_features("BTC")
    signal   = get_news_signal(features)
    print(f"\nNews Signal: {signal['signal']} (strength={signal['strength']})")
    print(f"Reasons: {'; '.join(signal['reasons'])}")
    print(f"F&G: {signal['fg']}")
