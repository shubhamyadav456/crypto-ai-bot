# ai/debate_panel.py — Claude Multi-Agent Debate Panel v1.0
"""
3 specialized Claude agents debate a trade signal.
  Bull Agent  → strongest BUY case
  Bear Agent  → strongest SELL/SKIP case
  Risk Agent  → objective risk assessment
  Judge Agent → final verdict (score 1-10)

Only score >= 7 passes to execution.

Usage:
    from ai.debate_panel import DebatePanel, build_snapshot_from_features
    panel  = DebatePanel()
    result = panel.debate(market_snapshot)
"""

import os, sys, json, time, logging, re
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "config.env"))

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"
DEBATE_TIMEOUT    = 45


# ── Single Claude call ─────────────────────────────────────────
def _call_claude(system_prompt: str, user_message: str,
                 max_tokens: int = 500) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY missing in config.env")

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_message}],
    }
    r = requests.post(ANTHROPIC_API_URL, headers=headers,
                      json=payload, timeout=DEBATE_TIMEOUT)
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


# ── Market snapshot formatter ──────────────────────────────────
def format_snapshot(s: dict) -> str:
    ema21, ema50, ema200 = s.get("ema_21",0), s.get("ema_50",0), s.get("ema_200",0)
    if ema21 > ema50 > ema200:
        trend_str = "STRONG UPTREND (EMA21>50>200)"
    elif ema21 < ema50 < ema200:
        trend_str = "STRONG DOWNTREND (EMA21<50<200)"
    else:
        trend_str = "MIXED / TRANSITIONING"

    return f"""
=== BTCUSDT MARKET SNAPSHOT ===

PRICE: ${s.get('price',0):,.2f}  |  24h Change: {s.get('change_24h',0):+.2f}%
ATR(14): ${s.get('atr',0):,.0f}  ({s.get('atr_pct',0):.2f}% of price)
Regime: {s.get('regime','Unknown')}

MOMENTUM:
  RSI(14)={s.get('rsi',50):.1f}  RSI(7)={s.get('rsi_7',50):.1f}
  MACD Hist={s.get('macd_hist',0):.4f}
  Mom 1h={s.get('mom_1',0)*100:+.3f}%  Mom 24h={s.get('mom_24',0)*100:+.3f}%

TREND: {trend_str}
  EMA21=${s.get('ema_21',0):,.0f}  EMA50=${s.get('ema_50',0):,.0f}  EMA200=${s.get('ema_200',0):,.0f}

VOLUME & STRUCTURE:
  Volume ratio={s.get('vol_ratio',1):.2f}x 20d avg
  BB position={s.get('bb_pct',0.5)*100:.0f}%  Squeeze={'YES' if s.get('bb_squeeze') else 'NO'}
  Distance from 20d High={s.get('near_high_20',0)*100:.2f}%
  Distance from 20d Low={s.get('near_low_20',0)*100:.2f}%

SENTIMENT & POSITIONING:
  Fear & Greed={s.get('fg_value',50):.0f}/100 ({s.get('fg_zone_label','Neutral')})
  Funding Rate={s.get('funding_rate',0)*100:+.4f}%
  OI Change={s.get('oi_change',0)*100:+.2f}%

NEWS:
  Signal={s.get('news_signal','NEUTRAL')}
  Sentiment 6h={s.get('news_sent_6h',0):+.2f}  24h={s.get('news_sent_24h',0):+.2f}
  Headlines: {s.get('news_headlines','None')}

ML MODEL:
  Direction={s.get('direction','N/A')}  Prob={s.get('prob',0.5)*100:.1f}%
  Confidence={s.get('confidence','LOW')}  Filter Score={s.get('signal_score',0)}/7

PROPOSED TRADE:
  Entry=${s.get('entry',0):,.2f}  SL=${s.get('sl',0):,.2f}  TP=${s.get('tp',0):,.2f}
  R:R = 1:{s.get('rr',2)}
""".strip()


# ── Agent system prompts ───────────────────────────────────────
BULL_SYSTEM = """You are an aggressive but disciplined BULL trader specializing in Bitcoin.
Make the STRONGEST case for taking a LONG/BUY position using the given data.
Focus on: momentum confluence, trend alignment, accumulation signals, volume confirmation, news catalysts.
Be specific — cite actual numbers. Be concise (under 180 words).
End your response with exactly: BULL_SCORE: X/10"""

BEAR_SYSTEM = """You are a skeptical BEAR analyst specializing in Bitcoin risk assessment.
Find EVERY reason why this trade is dangerous or why SELL/SKIP is the right call.
Focus on: overextension signs, resistance levels, weak volume, macro risks, sentiment extremes.
Be specific — cite actual numbers. Be concise (under 180 words).
End your response with exactly: BEAR_SCORE: X/10 (higher = more bearish)"""

RISK_SYSTEM = """You are a strict RISK MANAGER for a professional crypto trading desk.
Assess the RISK of this trade objectively — not bullish or bearish, just risk-focused.
Evaluate: stop loss placement validity, funding rate risk, liquidity conditions,
news event risk, regime risk, position sizing adequacy, max adverse excursion.
Be specific — cite actual numbers. Be concise (under 180 words).
End your response with exactly: RISK_SCORE: X/10 (10=very high risk, 1=very low risk)"""

JUDGE_SYSTEM = """You are the HEAD TRADER making the final call after hearing from 3 analysts.
Synthesize the Bull case, Bear case, and Risk assessment into ONE final verdict.

Respond ONLY with valid JSON — no other text before or after:
{
  "verdict": "BUY" or "SELL" or "SKIP",
  "score": <integer 1-10>,
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "primary_reason": "<one sentence explaining the verdict>",
  "key_risk": "<one sentence — biggest risk if trade is taken>",
  "bull_weight": <float 0.0-1.0>,
  "bear_weight": <float 0.0-1.0>
}

Scoring guide:
  8-10 = strong conviction → TRADE
  5-7  = uncertain → SKIP
  1-4  = strong against → SKIP
Only BUY or SELL with score >= 7 should ever be executed."""


# ── Debate Panel ───────────────────────────────────────────────
class DebatePanel:

    def __init__(self, min_judge_score: int = 7):
        self.min_judge_score = min_judge_score

    # ── Pre-filter: don't waste API calls on weak signals ──────
    def quick_check(self, snapshot: dict) -> tuple:
        prob   = snapshot.get("prob", 0.5)
        score  = snapshot.get("signal_score", 0)
        regime = snapshot.get("regime", "Sideways")
        conf   = snapshot.get("confidence", "LOW")

        if prob < 0.62 and prob > 0.38:
            return False, f"Prob {prob:.2f} too weak for debate"
        if score < 3:
            return False, f"Signal score {score}/7 too low"
        if regime == "Sideways" and conf in ("LOW", "SKIP"):
            return False, "Sideways market + low confidence"
        return True, "Pre-check passed"

    # ── Full debate ────────────────────────────────────────────
    def debate(self, snapshot: dict) -> dict:
        result = {
            "passed":         False,
            "verdict":        "SKIP",
            "score":          0,
            "confidence":     "LOW",
            "primary_reason": "",
            "key_risk":       "",
            "bull_case":      "",
            "bear_case":      "",
            "risk_case":      "",
            "bull_score":     0,
            "bear_score":     0,
            "risk_score":     0,
            "error":          None,
        }

        # Pre-check
        ok, reason = self.quick_check(snapshot)
        if not ok:
            result["primary_reason"] = reason
            log.info(f"[DEBATE] Skipped: {reason}")
            return result

        market_text = format_snapshot(snapshot)
        direction   = snapshot.get("direction", "?")
        log.info(f"[DEBATE] Starting 4-agent debate — signal={direction} "
                 f"prob={snapshot.get('prob',0):.2f} score={snapshot.get('signal_score',0)}/7")

        try:
            # Agent 1 — Bull
            log.info("[DEBATE] Bull agent...")
            bull_case = _call_claude(
                BULL_SYSTEM,
                f"Market data:\n\n{market_text}\n\nMake your BULL case.",
                max_tokens=400
            )
            result["bull_case"]  = bull_case
            result["bull_score"] = self._extract_score(bull_case, "BULL_SCORE")
            time.sleep(0.3)

            # Agent 2 — Bear
            log.info("[DEBATE] Bear agent...")
            bear_case = _call_claude(
                BEAR_SYSTEM,
                f"Market data:\n\n{market_text}\n\nMake your BEAR case against this trade.",
                max_tokens=400
            )
            result["bear_case"]  = bear_case
            result["bear_score"] = self._extract_score(bear_case, "BEAR_SCORE")
            time.sleep(0.3)

            # Agent 3 — Risk
            log.info("[DEBATE] Risk agent...")
            risk_case = _call_claude(
                RISK_SYSTEM,
                f"Market data:\n\n{market_text}\n\nAssess the risk of this trade.",
                max_tokens=400
            )
            result["risk_case"]  = risk_case
            result["risk_score"] = self._extract_score(risk_case, "RISK_SCORE")
            time.sleep(0.3)

            # Agent 4 — Judge
            log.info("[DEBATE] Judge making final decision...")
            judge_input = (
                f"Market data:\n{market_text}\n\n"
                f"BULL ANALYST:\n{bull_case}\n\n"
                f"BEAR ANALYST:\n{bear_case}\n\n"
                f"RISK MANAGER:\n{risk_case}\n\n"
                f"Give your final JSON verdict."
            )
            judge_raw = _call_claude(JUDGE_SYSTEM, judge_input, max_tokens=350)
            verdict   = self._parse_judge(judge_raw)
            result.update(verdict)

            # Pass/fail
            result["passed"] = (
                result["score"] >= self.min_judge_score and
                result["verdict"] in ("BUY", "SELL")
            )

            log.info(
                f"[DEBATE] RESULT: {result['verdict']} "
                f"score={result['score']}/10 "
                f"passed={result['passed']} | "
                f"Bull={result['bull_score']} "
                f"Bear={result['bear_score']} "
                f"Risk={result['risk_score']}"
            )

        except Exception as e:
            log.error(f"[DEBATE] Error: {e}")
            result["error"] = str(e)

        return result

    # ── Helpers ────────────────────────────────────────────────
    def _extract_score(self, text: str, label: str) -> int:
        match = re.search(rf"{label}:\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else 5

    def _parse_judge(self, raw: str) -> dict:
        try:
            s = raw.find("{"); e = raw.rfind("}") + 1
            if s >= 0 and e > s:
                d = json.loads(raw[s:e])
                return {
                    "verdict":        str(d.get("verdict","SKIP")).upper(),
                    "score":          max(0, min(10, int(d.get("score", 0)))),
                    "confidence":     str(d.get("confidence","LOW")).upper(),
                    "primary_reason": str(d.get("primary_reason","")),
                    "key_risk":       str(d.get("key_risk","")),
                }
        except Exception:
            pass
        # Fallback
        score = int(m.group(1)) if (m := re.search(r'"score"\s*:\s*(\d+)', raw)) else 0
        verdict = ("BUY"  if '"verdict": "BUY"'  in raw else
                   "SELL" if '"verdict": "SELL"' in raw else "SKIP")
        return {
            "verdict": verdict, "score": score,
            "confidence": "LOW",
            "primary_reason": "Parse fallback",
            "key_risk": "Check logs for raw response",
        }


# ── Snapshot builder (connects to existing pipeline) ──────────
def build_snapshot_from_features(df_feat, pred: dict,
                                  sig_score, news: dict) -> dict:
    last = df_feat.iloc[-1]
    def g(col, default=0.0):
        return float(last[col]) if col in df_feat.columns else default

    fg_val = g("fg_value", 50)
    fg_labels = [(20,"Extreme Fear"),(40,"Fear"),(60,"Neutral"),(80,"Greed"),(101,"Extreme Greed")]
    fg_label  = next(l for v, l in fg_labels if fg_val < v)

    return {
        "price":         g("close"),
        "change_24h":    g("mom_24") * 100,
        "atr":           g("atr_14"),
        "atr_pct":       g("atr_pct") * 100,
        "regime":        {1:"Uptrend",-1:"Downtrend",0:"Sideways"}.get(int(g("market_regime")),"Unknown"),
        "rsi":           g("rsi_14"),
        "rsi_7":         g("rsi_7"),
        "macd_hist":     g("macd_hist"),
        "mom_1":         g("mom_1"),
        "mom_24":        g("mom_24"),
        "ema_21":        g("ema_21"),
        "ema_50":        g("ema_50"),
        "ema_200":       g("ema_200"),
        "vol_ratio":     g("vol_ratio"),
        "bb_pct":        g("bb_pct"),
        "bb_squeeze":    bool(g("bb_squeeze")),
        "near_high_20":  g("near_high_20"),
        "near_low_20":   g("near_low_20"),
        "fg_value":      fg_val,
        "fg_zone_label": fg_label,
        "funding_rate":  g("funding_rate"),
        "oi_change":     g("oi_change"),
        "news_signal":   news.get("signal","NEUTRAL"),
        "news_sent_6h":  news.get("sent_6h", 0),
        "news_sent_24h": news.get("sent_24h", 0),
        "news_headlines":news.get("top_headline",""),
        "direction":     pred.get("direction",""),
        "prob":          pred.get("prob", 0.5),
        "confidence":    pred.get("confidence","LOW"),
        "signal_score":  getattr(sig_score, "score", 0) if sig_score else 0,
        "entry":         pred.get("entry", g("close")),
        "sl":            pred.get("sl", 0),
        "tp":            pred.get("tp", 0),
        "rr":            pred.get("rr", 2.0),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Test snapshot
    snap = {
        "price":84500,"change_24h":1.8,"atr":1100,"atr_pct":1.3,
        "regime":"Uptrend","rsi":56,"rsi_7":59,"macd_hist":80,
        "mom_1":0.005,"mom_24":0.018,
        "ema_21":83800,"ema_50":81200,"ema_200":75000,
        "vol_ratio":1.3,"bb_pct":0.6,"bb_squeeze":False,
        "near_high_20":0.01,"near_low_20":0.04,
        "fg_value":55,"fg_zone_label":"Neutral",
        "funding_rate":0.0002,"oi_change":0.008,
        "news_signal":"BULLISH","news_sent_6h":1.5,"news_sent_24h":1.0,
        "news_headlines":"BTC ETF inflows hit record $500M",
        "direction":"BUY","prob":0.72,"confidence":"HIGH","signal_score":5,
        "entry":84500,"sl":83200,"tp":87100,"rr":2.0,
    }

    panel  = DebatePanel(min_judge_score=7)
    result = panel.debate(snap)

    print(f"\n{'='*55}")
    print(f"  DEBATE PANEL RESULT")
    print(f"{'='*55}")
    print(f"  Verdict    : {result['verdict']}")
    print(f"  Score      : {result['score']}/10")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Passed     : {result['passed']}")
    print(f"  Reason     : {result['primary_reason']}")
    print(f"  Key Risk   : {result['key_risk']}")
    print(f"  Bull Score : {result['bull_score']}/10")
    print(f"  Bear Score : {result['bear_score']}/10")
    print(f"  Risk Score : {result['risk_score']}/10")
    if result['error']:
        print(f"  Error      : {result['error']}")
    print(f"{'='*55}\n")
