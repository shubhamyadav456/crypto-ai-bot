# data/news.py — v2.1 (fixed CryptoPanic API)
import time, re, requests, logging
import pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "BTC-AI-Bot/5.0"})

# CryptoPanic v2 API (free, no key needed for public posts)
CRYPTOPANIC_BASE = "https://cryptopanic.com/api/free/v2/posts/"


def fetch_cryptopanic(currency: str = "BTC",
                      pages: int = 2,
                      api_key: str = None) -> pd.DataFrame:
    all_posts = []

    # Validate key — must be 20-64 alphanumeric chars
    clean_key = (api_key or "").strip()
    if clean_key and not re.match(r"^[a-zA-Z0-9]{20,64}$", clean_key):
        log.info("  [NEWS] Invalid API key format — using free tier")
        clean_key = None

    url = CRYPTOPANIC_BASE
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
            data  = r.json()
            posts = data.get("results", [])
            if not posts:
                break
            all_posts.extend(posts)
            next_url = data.get("next")
            if not next_url:
                break
            url = next_url
            time.sleep(0.5)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                log.warning("  [NEWS] CryptoPanic rate limited")
            else:
                log.warning(f"  [NEWS] CryptoPanic HTTP error: {e}")
            break
        except Exception as e:
            log.warning(f"  [NEWS] CryptoPanic error: {e}")
            break

    if not all_posts:
        return pd.DataFrame()

    rows = []
    bullish_kw = ["rally","surge","pump","breakout","ath","bull","buy",
                  "accumulate","adoption","etf","halving","upgrade",
                  "partnership","institutional","recovery","inflow"]
    bearish_kw = ["crash","dump","bear","sell","hack","scam","ban",
                  "regulation","sec","lawsuit","liquidation","panic",
                  "fear","fraud","arrest","warning","exploit","rug","outflow"]

    for p in all_posts:
        try:
            ts = pd.to_datetime(
                p.get("published_at") or p.get("created_at"), utc=True)
        except Exception:
            continue

        title  = (p.get("title") or "").lower()
        votes  = p.get("votes", {})
        liked     = int(votes.get("liked",    0))
        disliked  = int(votes.get("disliked", 0))
        important = int(votes.get("important",0))
        negative  = int(votes.get("negative", 0))

        raw_score = (liked + important) - (disliked + negative)
        kw_score  = (sum(2 for kw in bullish_kw if kw in title) -
                     sum(2 for kw in bearish_kw if kw in title))

        flt = p.get("filter") or ""
        if flt == "bullish":  kw_score += 2
        elif flt == "bearish": kw_score -= 2

        rows.append({
            "ts":    ts,
            "title": p.get("title", ""),
            "score": raw_score + kw_score,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


def news_to_hourly(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame()
    df_h = df_news["score"].resample("1h").agg(["sum","count"])
    df_h.columns = ["news_score_sum","news_count"]
    df_h["news_sentiment"] = df_h["news_score_sum"] / (df_h["news_count"] + 1e-9)
    df_h["news_sent_6h"]   = df_h["news_sentiment"].rolling(6,  min_periods=1).mean()
    df_h["news_sent_24h"]  = df_h["news_sentiment"].rolling(24, min_periods=1).mean()
    df_h["news_count_6h"]  = df_h["news_count"].rolling(6,  min_periods=1).sum()
    df_h["news_vol_6h"]    = df_h["news_sentiment"].rolling(6,  min_periods=1).std().fillna(0)
    df_h["news_momentum"]  = df_h["news_sent_6h"] - df_h["news_sent_24h"]
    for col in ["news_sentiment","news_sent_6h","news_sent_24h","news_momentum"]:
        df_h[col] = df_h[col].clip(-5, 5)
    return df_h[["news_sentiment","news_sent_6h","news_sent_24h",
                  "news_count_6h","news_vol_6h","news_momentum"]]


def fetch_fear_greed_current() -> dict:
    try:
        r = _SESSION.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        r.raise_for_status()
        d = r.json()["data"][0]
        return {
            "value": int(d["value"]),
            "class": d["value_classification"],
            "ts":    datetime.now(timezone.utc),
        }
    except Exception:
        return {"value": 50, "class": "Neutral", "ts": datetime.now(timezone.utc)}


def get_news_features(currency: str = "BTC", api_key: str = None) -> dict:
    print(f"  [NEWS] Fetching {currency} news...")
    df_news = fetch_cryptopanic(currency, pages=2, api_key=api_key)

    default = {
        "news_sentiment": 0.0, "news_sent_6h": 0.0,
        "news_sent_24h":  0.0, "news_count_6h": 0.0,
        "news_vol_6h":    0.0, "news_momentum": 0.0,
        "top_headline":   "",  "recent_headlines": [],
    }

    if df_news.empty:
        print("  [NEWS] No news — using neutral")
        return default

    df_h = news_to_hourly(df_news)
    if df_h.empty:
        return default

    latest   = df_h.iloc[-1].to_dict()
    fg       = fetch_fear_greed_current()
    headlines = df_news["title"].tolist()[:10]

    latest.update({
        "fg_current":       fg["value"],
        "fg_class":         fg["class"],
        "top_headline":     headlines[0] if headlines else "",
        "recent_headlines": headlines,
    })

    print(f"  [NEWS] Sentiment 6h={latest.get('news_sent_6h',0):.2f} | "
          f"24h={latest.get('news_sent_24h',0):.2f} | "
          f"Articles={int(latest.get('news_count_6h',0))} | "
          f"F&G={fg['value']} ({fg['class']})")
    return latest


def get_news_signal(features: dict) -> dict:
    sent_6h  = features.get("news_sent_6h",  0)
    momentum = features.get("news_momentum", 0)
    fg       = features.get("fg_current", features.get("fg_value", 50))
    count    = features.get("news_count_6h", 0)

    score, reasons = 0, []

    if   sent_6h >  1.5: score += 2; reasons.append(f"Strong bullish news ({sent_6h:.1f})")
    elif sent_6h >  0.5: score += 1; reasons.append(f"Mild bullish news ({sent_6h:.1f})")
    elif sent_6h < -1.5: score -= 2; reasons.append(f"Strong bearish news ({sent_6h:.1f})")
    elif sent_6h < -0.5: score -= 1; reasons.append(f"Mild bearish news ({sent_6h:.1f})")

    if   momentum >  1.0: score += 1; reasons.append("News turning bullish")
    elif momentum < -1.0: score -= 1; reasons.append("News turning bearish")

    if   fg <= 20: score += 1; reasons.append(f"Extreme fear F&G={fg} — contrarian buy")
    elif fg >= 80: score -= 1; reasons.append(f"Extreme greed F&G={fg} — caution")

    if count < 2: reasons.append("Low news volume — uncertain")

    signal   = "BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "NEUTRAL"
    strength = min(abs(score) / 4.0, 1.0)

    return {
        "signal":   signal,
        "strength": round(strength, 2),
        "score":    score,
        "reasons":  reasons,
        "sent_6h":  round(sent_6h, 2),
        "sent_24h": round(features.get("news_sent_24h", 0), 2),
        "fg":       fg,
        "fg_class": features.get("fg_class", "Neutral"),
    }
