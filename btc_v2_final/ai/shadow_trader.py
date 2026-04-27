# ai/shadow_trader.py — Shadow Trader + Weekly Claude Review v1.0
"""
Parallel paper-trade system with weekly Claude-powered critique.
Every real signal also creates a shadow paper trade.
Claude reviews losses weekly and suggests filter improvements.

Usage:
    from ai.shadow_trader import ShadowTrader
    shadow = ShadowTrader()
    shadow.open_trade(signal_data)
    shadow.update_prices({"BTCUSDT": 85000})
    report = shadow.weekly_review()
"""

import os, sys, json, logging, sqlite3
import requests
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "config.env"))

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"

_db = os.getenv("DB_PATH", "storage/trades.db")
SHADOW_DB = _db if os.path.isabs(_db) else os.path.join(_ROOT, _db)


# ── DB setup ───────────────────────────────────────────────────
def init_shadow_db(db_path: str = SHADOW_DB):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS shadow_trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        open_ts     TEXT NOT NULL,
        close_ts    TEXT,
        symbol      TEXT DEFAULT 'BTCUSDT',
        direction   TEXT NOT NULL,
        entry       REAL NOT NULL,
        sl          REAL NOT NULL,
        tp          REAL NOT NULL,
        exit_price  REAL,
        exit_reason TEXT DEFAULT 'open',
        pnl_pct     REAL,
        prob        REAL,
        signal_score INTEGER,
        debate_score INTEGER,
        narrative   TEXT,
        features_json TEXT
    );
    CREATE TABLE IF NOT EXISTS claude_reviews (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        period     TEXT,
        n_trades   INTEGER,
        win_rate   REAL,
        suggestions TEXT,
        raw_response TEXT
    );
    """)
    conn.commit()
    conn.close()


class ShadowTrader:

    def __init__(self, db_path: str = SHADOW_DB):
        self.db_path = db_path
        init_shadow_db(db_path)

    # ── Open a shadow trade ────────────────────────────────────
    def open_trade(self, signal: dict) -> int:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.execute("""
            INSERT INTO shadow_trades
            (open_ts, symbol, direction, entry, sl, tp,
             prob, signal_score, debate_score, narrative, features_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            signal.get("symbol", "BTCUSDT"),
            signal["direction"],
            signal["entry"],
            signal["sl"],
            signal["tp"],
            signal.get("prob", 0),
            signal.get("signal_score", 0),
            signal.get("debate_score", 0),
            signal.get("narrative_theme", ""),
            json.dumps(signal.get("features", {})),
        ))
        sid = cur.lastrowid
        conn.commit(); conn.close()
        log.info(f"[SHADOW] Opened paper trade #{sid} {signal['direction']} @ {signal['entry']}")
        return sid

    # ── Update open trades with current price ─────────────────
    def update_prices(self, prices: dict):
        conn   = sqlite3.connect(self.db_path)
        trades = conn.execute(
            "SELECT id, symbol, direction, entry, sl, tp FROM shadow_trades WHERE exit_reason='open'"
        ).fetchall()
        conn.close()

        for sid, sym, direction, entry, sl, tp in trades:
            price = prices.get(sym)
            if price is None:
                continue
            if direction == "BUY":
                if price >= tp:
                    self._close(sid, price, "TP")
                elif price <= sl:
                    self._close(sid, price, "SL")
            else:
                if price <= tp:
                    self._close(sid, price, "TP")
                elif price >= sl:
                    self._close(sid, price, "SL")

    def _close(self, sid: int, exit_price: float, reason: str):
        conn = sqlite3.connect(self.db_path)
        row  = conn.execute(
            "SELECT direction, entry FROM shadow_trades WHERE id=?", (sid,)
        ).fetchone()
        if not row:
            conn.close(); return
        direction, entry = row
        pnl_pct = ((exit_price - entry) / entry) if direction == "BUY" \
                  else ((entry - exit_price) / entry)
        conn.execute("""
            UPDATE shadow_trades
            SET close_ts=?, exit_price=?, exit_reason=?, pnl_pct=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), exit_price, reason, pnl_pct, sid))
        conn.commit(); conn.close()
        log.info(f"[SHADOW] Closed #{sid} via {reason} pnl={pnl_pct*100:+.2f}%")

    # ── Weekly Claude review ───────────────────────────────────
    def weekly_review(self, last_n: int = 20) -> dict:
        """
        Claude reviews last N closed paper trades.
        Returns suggestions for filter improvements.
        """
        conn   = sqlite3.connect(self.db_path)
        trades = conn.execute("""
            SELECT direction, entry, exit_price, exit_reason, pnl_pct,
                   prob, signal_score, debate_score, narrative, open_ts
            FROM shadow_trades
            WHERE exit_reason != 'open'
            ORDER BY close_ts DESC
            LIMIT ?
        """, (last_n,)).fetchall()
        conn.close()

        if not trades:
            return {"suggestions": [], "summary": "No closed trades yet"}

        wins   = sum(1 for t in trades if t[4] and t[4] > 0)
        losses = len(trades) - wins
        win_rate = wins / len(trades)

        # Format trades for Claude
        trades_text = ""
        for i, t in enumerate(trades, 1):
            direction, entry, exit_p, reason, pnl, prob, score, debate, narrative, ts = t
            pnl_str = f"{pnl*100:+.2f}%" if pnl else "N/A"
            trades_text += (
                f"\nTrade {i}: {direction} | Entry=${entry:,.0f} | "
                f"Exit={reason} @ ${exit_p:,.0f} | PnL={pnl_str} | "
                f"Prob={prob:.2f} | Score={score}/7 | Debate={debate}/10 | "
                f"Narrative={narrative or 'N/A'}\n"
            )

        review_system = """You are a quantitative trading analyst reviewing paper trades.
Your job: find patterns in losses and suggest SPECIFIC filter improvements.
Be concrete — suggest exact threshold changes, not vague advice.

Respond in JSON:
{
  "loss_patterns": ["<pattern 1>", "<pattern 2>"],
  "suggestions": [
    {"filter": "<filter name>", "change": "<specific change>", "reason": "<why>"},
    ...
  ],
  "winning_patterns": ["<pattern 1>"],
  "summary": "<2 sentence overall assessment>"
}"""

        user_msg = (
            f"Last {len(trades)} paper trades:\n{trades_text}\n\n"
            f"Win rate: {win_rate*100:.1f}% ({wins}W/{losses}L)\n\n"
            f"What patterns do you see in the losses? "
            f"Suggest specific filter improvements."
        )

        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                return {"suggestions": [], "summary": "No API key"}

            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            r = requests.post(ANTHROPIC_API_URL, headers=headers, json={
                "model": CLAUDE_MODEL, "max_tokens": 600,
                "system": review_system,
                "messages": [{"role": "user", "content": user_msg}],
            }, timeout=30)
            r.raise_for_status()
            raw  = r.json()["content"][0]["text"].strip()
            s = raw.find("{"); e = raw.rfind("}") + 1
            data = json.loads(raw[s:e]) if s >= 0 and e > s else {}

            # Save review to DB
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO claude_reviews
                (ts, period, n_trades, win_rate, suggestions, raw_response)
                VALUES (?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                f"last_{last_n}",
                len(trades), win_rate,
                json.dumps(data.get("suggestions", [])),
                raw,
            ))
            conn.commit(); conn.close()

            log.info(f"[SHADOW] Review complete — {len(data.get('suggestions',[]))} suggestions")
            return data

        except Exception as e:
            log.error(f"[SHADOW] Review error: {e}")
            return {"suggestions": [], "summary": f"Error: {e}"}

    # ── Stats ──────────────────────────────────────────────────
    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT exit_reason, pnl_pct, direction, prob, signal_score
            FROM shadow_trades WHERE exit_reason != 'open'
        """).fetchall()
        open_count = conn.execute(
            "SELECT COUNT(*) FROM shadow_trades WHERE exit_reason='open'"
        ).fetchone()[0]
        conn.close()

        if not rows:
            return {"total": 0, "open": open_count}

        wins = [r[1] for r in rows if r[1] and r[1] > 0]
        losses = [r[1] for r in rows if r[1] and r[1] <= 0]

        return {
            "total":       len(rows),
            "open":        open_count,
            "wins":        len(wins),
            "losses":      len(losses),
            "win_rate":    round(len(wins)/len(rows)*100, 1),
            "avg_win":     round(sum(wins)/len(wins)*100, 2) if wins else 0,
            "avg_loss":    round(sum(losses)/len(losses)*100, 2) if losses else 0,
            "by_exit":     {r[0]: sum(1 for x in rows if x[0]==r[0])
                            for r in rows},
        }
