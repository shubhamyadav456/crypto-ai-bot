# backtest/engine.py — v2
"""
Realistic Backtest Engine
==========================
Run:
    python backtest/engine.py
    python backtest/engine.py --total 5000
    python backtest/engine.py --use-2years
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.fetcher import build_raw_dataset
from features.engineer import build_features, _atr
from config.settings import (
    SYMBOL, BASE_TF, MODEL_PATH,
    ATR_SL_MULT, ATR_TP_MULT, RISK_PCT, MIN_RR
)


def run_backtest(symbol=SYMBOL, interval=BASE_TF,
                 total=3000, use_2years=False,
                 model_path=MODEL_PATH,
                 capital=10_000.0,
                 sl_mult=ATR_SL_MULT,
                 tp_mult=ATR_TP_MULT,
                 risk_pct=RISK_PCT,
                 slippage_pct=0.0005,
                 fee_pct=0.0005,
                 max_hold=24,
                 verbose=True):

    print(f"\n{'='*58}")
    print(f"  BACKTEST — {symbol} {interval}")
    print(f"  Capital: ${capital:,.0f} | SL:{sl_mult}x ATR | TP:{tp_mult}x ATR")
    print(f"  Slippage: {slippage_pct*100:.2f}% | Fee: {fee_pct*100:.2f}%")
    print(f"{'='*58}")

    # Load model
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found: {model_path}")
        return {}
    art     = joblib.load(model_path)
    sc      = art["scaler"]
    xgb     = art["xgb"]
    rf      = art["rf"]
    cal     = art["calibrator"]
    feats   = art.get("all_features", art.get("features", []))
    sel_idx = art.get("sel_idx")
    thr     = art.get("threshold", 0.5)
    meta    = art.get("meta", {})
    print(f"  Model: {meta.get('version','?')} | AUC={meta.get('test_auc','?')} | threshold={thr}")

    # Fetch data
    print(f"\n  Fetching data...")
    df_raw  = build_raw_dataset(symbol, interval, total, use_2years)
    df_feat = build_features(df_raw, include_market_data=True)

    # Fill missing features
    for f in feats:
        if f not in df_feat.columns:
            df_feat[f] = 0.0

    print(f"  Rows: {len(df_feat)} | {df_feat.index[0] if hasattr(df_feat.index, '__len__') else ''}")

    # Precompute all probabilities
    X = df_feat[feats].fillna(0).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc   = sc.transform(X)
    X_sel  = X_sc[:, sel_idx] if sel_idx is not None else X_sc
    raw_p  = (xgb.predict_proba(X_sel)[:,1] + rf.predict_proba(X_sel)[:,1]) / 2
    probs  = cal.transform(raw_p)

    # Get OHLCV aligned to feat rows
    n_raw  = len(df_raw)
    offset = n_raw - len(df_feat)
    closes = df_raw["close"].values[offset:]
    highs  = df_raw["high"].values[offset:]
    lows   = df_raw["low"].values[offset:]

    n = len(df_feat)

    # ── Simulate trades ────────────────────────────────────────
    trades     = []
    equity     = [capital]
    cap        = capital
    open_trade = None   # {entry_bar, entry_px, direction, sl, tp, size}

    for i in range(1, n):
        price = closes[i]
        hi    = highs[i]
        lo    = lows[i]

        # Check open trade
        if open_trade is not None:
            t = open_trade
            hit_tp = hit_sl = False

            if t["dir"] == "BUY":
                hit_tp = hi >= t["tp"]
                hit_sl = lo <= t["sl"]
            else:
                hit_tp = lo <= t["tp"]
                hit_sl = hi >= t["sl"]

            timed_out = (i - t["bar"]) >= max_hold

            if hit_tp or hit_sl or timed_out:
                if hit_tp:
                    exit_px = t["tp"]; reason = "TP"
                elif hit_sl:
                    exit_px = t["sl"]; reason = "SL"
                else:
                    exit_px = price;   reason = "TIMEOUT"

                # Apply slippage on exit
                if t["dir"] == "BUY":
                    exit_px *= (1 - slippage_pct)
                    gross = t["size"] * (exit_px - t["entry"])
                else:
                    exit_px *= (1 + slippage_pct)
                    gross = t["size"] * (t["entry"] - exit_px)

                fees = (t["size"] * t["entry"] + t["size"] * exit_px) * fee_pct
                pnl  = gross - fees
                cap += pnl

                trades.append({
                    "entry_bar": t["bar"], "exit_bar": i,
                    "direction": t["dir"],
                    "entry_px":  round(t["entry"], 2),
                    "exit_px":   round(exit_px, 2),
                    "exit_reason": reason,
                    "pnl":       round(pnl, 2),
                    "pnl_pct":   round(pnl / (t["entry"] * t["size"] + 1e-9) * 100, 3),
                    "hold_bars": i - t["bar"],
                })
                open_trade = None

        equity.append(round(cap, 2))

        # New entry
        if open_trade is not None:
            continue

        prob = float(probs[i])
        if prob < thr and prob > (1 - thr):
            continue

        direction = "BUY" if prob >= thr else "SELL"

        # ATR from features
        atr_v = float(df_feat["atr_14"].iloc[i]) if "atr_14" in df_feat.columns else price * 0.015

        entry_px = price * (1 + slippage_pct) if direction == "BUY" else price * (1 - slippage_pct)

        if direction == "BUY":
            sl_px = entry_px - sl_mult * atr_v
            tp_px = entry_px + tp_mult * atr_v
        else:
            sl_px = entry_px + sl_mult * atr_v
            tp_px = entry_px - tp_mult * atr_v

        rr = tp_mult / sl_mult
        if rr < MIN_RR:
            continue

        risk_per_unit = abs(entry_px - sl_px)
        if risk_per_unit < 1e-9:
            continue
        size = (cap * risk_pct) / risk_per_unit

        open_trade = {
            "bar": i, "entry": entry_px,
            "dir": direction, "sl": sl_px, "tp": tp_px, "size": size,
        }

    # ── Metrics ────────────────────────────────────────────────
    if not trades:
        print("  No trades executed.")
        return {}

    df_t = pd.DataFrame(trades)
    wins   = (df_t["pnl"] > 0).sum()
    losses = (df_t["pnl"] <= 0).sum()
    total_t = len(df_t)
    wr      = wins / total_t

    eq   = np.array(equity)
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe   = float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(8760)
    roll_max = np.maximum.accumulate(eq)
    dd       = (eq - roll_max) / (roll_max + 1e-9)
    max_dd   = float(dd.min()) * 100

    total_return = (cap - capital) / capital * 100

    by_dir  = df_t.groupby("direction")["pnl"].agg(["count","sum","mean"])
    by_exit = df_t.groupby("exit_reason")["pnl"].agg(["count","sum"])

    avg_hold = df_t["hold_bars"].mean()
    avg_win  = df_t[df_t["pnl"] > 0]["pnl"].mean()
    avg_loss = df_t[df_t["pnl"] <= 0]["pnl"].mean()
    expect   = wr * avg_win + (1 - wr) * (avg_loss or 0)

    print(f"\n  {'─'*50}")
    print(f"  RESULTS")
    print(f"  {'─'*50}")
    print(f"  Total trades    : {total_t}")
    print(f"  Win rate        : {wr*100:.1f}%  ({int(wins)}W / {int(losses)}L)")
    print(f"  Total return    : {total_return:+.2f}%")
    print(f"  Final capital   : ${cap:,.2f}")
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print(f"  Sharpe ratio    : {sharpe:.3f}")
    print(f"  Avg win         : ${avg_win:.2f}")
    print(f"  Avg loss        : ${avg_loss:.2f}")
    print(f"  Expectancy/trade: ${expect:.2f}")
    print(f"  Avg hold (bars) : {avg_hold:.1f}h")
    print(f"\n  By direction:")
    print(by_dir.to_string())
    print(f"\n  By exit reason:")
    print(by_exit.to_string())

    # Monthly breakdown
    if "entry_bar" in df_t.columns and len(df_t) > 0:
        print(f"\n  {'─'*50}")
        print(f"  MONTHLY BREAKDOWN")
        print(f"  {'─'*50}")
        df_t["month"] = pd.to_datetime(
            df_raw.index[df_t["entry_bar"].values + offset],
            errors="coerce"
        ).to_period("M")
        monthly = df_t.groupby("month").agg(
            trades=("pnl","count"),
            pnl=("pnl","sum"),
            wr=("pnl", lambda x: (x>0).mean()*100)
        )
        for m, row in monthly.iterrows():
            bar = "+" * int(max(0, row["pnl"]/50)) if row["pnl"] > 0 else "-" * int(max(0, -row["pnl"]/50))
            print(f"  {m}  trades={int(row['trades']):>3}  "
                  f"pnl=${row['pnl']:>+8.2f}  wr={row['wr']:.0f}%  {bar[:20]}")

    print(f"\n{'='*58}\n")

    return {
        "total_trades":   total_t,
        "win_rate":       round(wr, 3),
        "total_return":   round(total_return, 2),
        "final_capital":  round(cap, 2),
        "max_drawdown":   round(max_dd, 2),
        "sharpe":         round(sharpe, 3),
        "expectancy":     round(expect, 2),
        "avg_hold_bars":  round(avg_hold, 1),
        "trades_df":      df_t,
        "equity_curve":   equity,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BTC AI Backtest")
    ap.add_argument("--symbol",     default=SYMBOL)
    ap.add_argument("--interval",   default=BASE_TF)
    ap.add_argument("--total",      type=int, default=3000)
    ap.add_argument("--use-2years", action="store_true")
    ap.add_argument("--capital",    type=float, default=10000)
    ap.add_argument("--sl-mult",    type=float, default=ATR_SL_MULT)
    ap.add_argument("--tp-mult",    type=float, default=ATR_TP_MULT)
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    run_backtest(
        symbol     = args.symbol,
        interval   = args.interval,
        total      = args.total,
        use_2years = args.use_2years,
        capital    = args.capital,
        sl_mult    = args.sl_mult,
        tp_mult    = args.tp_mult,
        model_path = args.model_path,
    )