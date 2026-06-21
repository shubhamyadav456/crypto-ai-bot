
"""
Claude analyzes current market narrative — what story is driving price.
Narrative often moves BEFORE price. This gives early directional bias.

Usage:
    from ai.narrative_engine import NarrativeEngine
    engine = NarrativeEngine()
    result = engine.analyze(headlines, market_context)
"""

import os, sys, json, logging, re
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "config.env"))

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"


def _call_claude(system: str, user: str, max_tokens: int = 400) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY missing in config.env")
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    r = requests.post(ANTHROPIC_API_URL, headers=headers, json={
        "model": CLAUDE_MODEL, "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }, timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


NARRATIVE_SYSTEM = """You are a crypto market narrative analyst.
Your job: identify the DOMINANT market narrative driving BTC price right now.

Respond ONLY in this exact JSON format:
{
  "narrative_type": "BULLISH_NARRATIVE" or "BEARISH_NARRATIVE" or "CONFUSED" or "TRANSITIONING",
  "theme": "<2-5 word theme, e.g. 'ETF inflows + institutional buying'>",
  "strength": <integer 1-10>,
  "trade_bias": "BUY" or "SELL" or "NEUTRAL",
  "catalysts": ["<catalyst 1>", "<catalyst 2>"],
  "risks": ["<risk 1>", "<risk 2>"],
  "time_horizon": "hours" or "days" or "weeks",
  "summary": "<one sentence summary of current narrative>"
}

Strength guide: 8-10=very strong, 5-7=moderate, 1-4=weak/noisy"""


class NarrativeEngine:

    def analyze(self, headlines: list, market_context: dict = None) -> dict:
        """
        Analyze current market narrative.

        headlines: list of recent news headline strings
        market_context: optional dict with price, fg_value, etc.
        """
        default = {
            "narrative_type": "CONFUSED",
            "theme":          "No clear narrative",
            "strength":       3,
            "trade_bias":     "NEUTRAL",
            "catalysts":      [],
            "risks":          [],
            "time_horizon":   "hours",
            "summary":        "Insufficient data for narrative analysis",
            "error":          None,
        }

        if not headlines:
            log.warning("[NARRATIVE] No headlines provided")
            return default

        ctx = ""
        if market_context:
            ctx = (
                f"\nMarket context: Price=${market_context.get('price',0):,.0f} "
                f"| F&G={market_context.get('fg_value',50):.0f} "
                f"| 24h change={market_context.get('change_24h',0):+.1f}% "
                f"| Regime={market_context.get('regime','Unknown')}"
            )

        headlines_text = "\n".join(f"- {h}" for h in headlines[:20])
        user_msg = (
            f"Recent BTC/crypto headlines (last 24h):\n{headlines_text}"
            f"{ctx}\n\nWhat is the dominant market narrative right now?"
        )

        try:
            raw    = _call_claude(NARRATIVE_SYSTEM, user_msg)
            s = raw.find("{"); e = raw.rfind("}") + 1
            if s >= 0 and e > s:
                data = json.loads(raw[s:e])
                result = {**default, **data, "error": None}
                log.info(
                    f"[NARRATIVE] {result['narrative_type']} | "
                    f"theme='{result['theme']}' | "
                    f"strength={result['strength']}/10 | "
                    f"bias={result['trade_bias']}"
                )
                return result
        except Exception as e:
            log.error(f"[NARRATIVE] Error: {e}")
            default["error"] = str(e)

        return default

    def get_narrative_score(self, narrative: dict,
                             signal_direction: str) -> tuple:
        """
        Returns (score_delta, reason) to add/subtract from signal score.
        Positive = narrative supports trade
        Negative = narrative conflicts
        """
        ntype  = narrative.get("narrative_type","CONFUSED")
        bias   = narrative.get("trade_bias","NEUTRAL")
        strength = narrative.get("strength", 3)

        if ntype == "CONFUSED" or bias == "NEUTRAL":
            return 0, "Narrative unclear — neutral"

        aligned = (
            (signal_direction == "BUY"  and bias == "BUY")  or
            (signal_direction == "SELL" and bias == "SELL")
        )

        if aligned:
            boost = +1 if strength >= 7 else 0
            return boost, f"Narrative aligned: {narrative.get('theme','')}"
        else:
            penalty = -2 if strength >= 7 else -1
            return penalty, f"Narrative conflict: {narrative.get('theme','')} vs {signal_direction}"
