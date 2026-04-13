# storage/db.py
"""
Signal storage + performance tracking (SQLite).
"""
import sqlite3
import json
import os
from datetime import datetime
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS signals (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT    NOT NULL,
        symbol        TEXT    NOT NULL,
        interval      TEXT    NOT NULL,
        direction     TEXT    NOT NULL,
        prob_up       REAL    NOT NULL,
        confidence    TEXT    NOT NULL,
        quality       TEXT    DEFAULT 'B',
        entry         REAL    NOT NULL,
        sl            REAL    NOT NULL,
        tp            REAL    NOT NULL,
        rr            REAL    NOT NULL,
        atr           REAL,
        market_regime INTEGER DEFAULT 0,
        mtf_score     INTEGER DEFAULT 0,
        outcome       TEXT    DEFAULT 'pending',
        exit_price    REAL,
        exit_ts       TEXT,
        pnl_pct       REAL,
        extra_json    TEXT
    );

    CREATE TABLE IF NOT EXISTS model_versions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        version    TEXT,
        cv_auc     REAL,
        test_auc   REAL,
        n_samples  INTEGER,
        notes      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_signals_ts      ON signals(ts);
    CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome);
    """)
    conn.commit()
    conn.close()


def save_signal(pred: dict, risk: dict,
                market_regime: int = 0,
                mtf_score: int = 0,
                quality: str = "B",
                db_path: str = DB_PATH) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur  = conn.execute("""
        INSERT INTO signals
        (ts, symbol, interval, direction, prob_up, confidence,
         quality, entry, sl, tp, rr, atr,
         market_regime, mtf_score, outcome, extra_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)
    """, (
        datetime.utcnow().isoformat(),
        pred.get("symbol", "BTCUSDT"),
        pred.get("interval", "1h"),
        pred["direction"],
        pred["prob_up"],
        pred["confidence"],
        quality,
        risk["entry"],
        risk["sl"],
        risk["tp"],
        risk["rr"],
        risk.get("atr", 0),
        market_regime,
        mtf_score,
        json.dumps({
            "prob_dn":   pred.get("prob_dn"),
            "model_ver": pred.get("model_meta", {}).get("version", ""),
        }),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_outcome(signal_id: int, outcome: str,
                   exit_price: float, pnl_pct: float,
                   db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        UPDATE signals
        SET outcome=?, exit_price=?, exit_ts=?, pnl_pct=?
        WHERE id=?
    """, (outcome, exit_price, datetime.utcnow().isoformat(), pnl_pct, signal_id))
    conn.commit()
    conn.close()


def auto_close_pending(current_prices: dict, db_path: str = DB_PATH):
    """Mark pending signals as win/loss if SL or TP hit."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT id, symbol, direction, entry, sl, tp
        FROM signals WHERE outcome='pending'
    """).fetchall()
    conn.close()

    for sid, sym, direct, entry, sl, tp in rows:
        price = current_prices.get(sym)
        if price is None:
            continue
        if direct == "BUY":
            if price >= tp:
                update_outcome(sid, "win",  price, (tp - entry) / entry, db_path)
            elif price <= sl:
                update_outcome(sid, "loss", price, (sl - entry) / entry, db_path)
        else:
            if price <= tp:
                update_outcome(sid, "win",  price, (entry - tp) / entry, db_path)
            elif price >= sl:
                update_outcome(sid, "loss", price, (entry - sl) / entry, db_path)


def get_stats(last_n: int = 20, db_path: str = DB_PATH) -> dict:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT * FROM signals
        WHERE outcome != 'pending'
        ORDER BY ts DESC LIMIT ?
    """, conn, params=(last_n,))
    conn.close()

    if df.empty:
        return {"total": 0, "message": "No closed trades yet"}

    wins  = (df["outcome"] == "win").sum()
    total = len(df)
    wr    = wins / total if total > 0 else 0

    avg_win  = df[df["outcome"] == "win"]["pnl_pct"].mean()  or 0
    avg_loss = df[df["outcome"] == "loss"]["pnl_pct"].mean() or 0
    expect   = wr * avg_win + (1 - wr) * avg_loss

    return {
        "total":         total,
        "wins":          int(wins),
        "losses":        int(total - wins),
        "win_rate_pct":  round(wr * 100, 1),
        "avg_win_pct":   round(avg_win * 100, 3),
        "avg_loss_pct":  round(avg_loss * 100, 3),
        "expectancy":    round(expect * 100, 3),
        "recent_trades": df[["ts","direction","entry","outcome","pnl_pct"]].head(10).to_dict("records"),
    }


def should_retrain(min_new_signals: int = 50, db_path: str = DB_PATH) -> bool:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    last_retrain = conn.execute("SELECT MAX(ts) FROM model_versions").fetchone()[0]

    if last_retrain is None:
        count = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome != 'pending'"
        ).fetchone()[0]
    else:
        count = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE outcome != 'pending' AND ts > ?",
            (last_retrain,)
        ).fetchone()[0]
    conn.close()
    return count >= min_new_signals


def log_model_version(meta: dict, db_path: str = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO model_versions (ts, version, cv_auc, test_auc, n_samples, notes)
        VALUES (?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        meta.get("version", "v4"),
        meta.get("cv_auc"),
        meta.get("test_auc"),
        meta.get("n_samples"),
        json.dumps(meta),
    ))
    conn.commit()
    conn.close()
