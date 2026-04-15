# backtest/engine.py — v3 (High Confidence Strategy)
"""
Backtest with scoring system integration.

Run:
    python backtest/engine.py
    python backtest/engine.py --use-2years
    python backtest/engine.py --min-score 5
    python backtest/engine.py --max-trades-per-day 1
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "config.env"))

from data.fetcher import build_raw_dataset
from features.engineer import build_features, _atr, _ema, _rsi, _bb
from features.signal_filter import score_signal, select_best_n_per_day
from config.settings import (
    SYMBOL, BASE_TF, MODEL_PATH,
    ATR_SL_MULT, ATR_TP_MULT, RISK_PCT
)


def run_backtest(
    symbol           = SYMBOL,
    interval         = BASE_TF,
    total            = 3000,
    use_2years       = False,
    model_path       = MODEL_PATH,
    capital          = 10_000.0,
    sl_mult          = 1.0,
    tp_mult          = 2.0,
    risk_pct         = RISK_PCT,
    slippage_pct     = 0.0005,
    fee_pct          = 0.0005,
    max_hold         = 24,
    min_score        = 4,
    max_trades_day   = 2,
    prob_buy_thr     = 0.70,
    prob_sell_thr    = 0.30,
):
    print(f"\n{'='*60}")
    print(f"  HIGH CONFIDENCE BACKTEST — {symbol} {interval}")
    print(f"  Capital: ${capital:,.0f} | SL:{sl_mult}x | TP:{tp_mult}x | R:R 1:{tp_mult/sl_mult:.1f}")
    print(f"  Min score: {min_score}/7 | Max trades/day: {max_trades_day}")
    print(f"  Prob: BUY>={prob_buy_thr} | SELL<={prob_sell_thr}")
    print(f"{'='*60}")

    # Load model
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found — run trainer first")
        return {}

    art     = joblib.load(model_path)
    sc      = art["scaler"]
    xgb     = art["xgb"]
    rf      = art["rf"]
    cal     = art["calibrator"]
    feats   = art.get("all_features", art.get("features", []))
    sel_idx = art.get("sel_idx")
    meta    = art.get("meta", {})
    print(f"  Model: {meta.get('version','?')} | CV AUC={meta.get('cv_auc','?')} | Test AUC={meta.get('test_auc','?')}")

    # Fetch + features
    print(f"\n  Fetching data...")
    df_raw  = build_raw_dataset(symbol, interval, total, use_2years)
    df_feat = build_features(df_raw, include_market_data=True)

    for f in feats:
        if f not in df_feat.columns:
            df_feat[f] = 0.0

    n_raw  = len(df_raw)
    offset = n_raw - len(df_feat)

    # Align raw OHLCV to feat rows
    closes  = df_raw["close"].values[offset:]
    highs   = df_raw["high"].values[offset:]
    lows    = df_raw["low"].values[offset:]
    n       = len(df_feat)

    # Compute probabilities for all rows
    X      = df_feat[feats].fillna(0).values.astype(np.float32)
    X      = np.nan_to_num(X)
    X_sc   = sc.transform(X)
    X_sel  = X_sc[:, sel_idx] if sel_idx is not None else X_sc
    raw_p  = (xgb.predict_proba(X_sel)[:,1] + rf.predict_proba(X_sel)[:,1]) / 2
    probs  = cal.transform(raw_p)

    # Precompute indicators for scoring
    raw_ri = df_raw.reset_index(drop=True)
    close_s = raw_ri["close"].iloc[offset:].reset_index(drop=True)
    high_s  = raw_ri["high"].iloc[offset:].reset_index(drop=True)
    low_s   = raw_ri["low"].iloc[offset:].reset_index(drop=True)
    vol_s   = raw_ri["volume"].iloc[offset:].reset_index(drop=True)

    e50     = _ema(close_s, 50).values
    e200    = _ema(close_s, 200).values
    rsi14   = _rsi(close_s, 14).values
    at14    = _atr(high_s, low_s, close_s, 14).values
    bu, bm, bl, bp, bw = _bb(close_s, 20, 2)
    bw_v    = bw.values
    bw_ma   = bw.rolling(20).mean().values
    bu_v    = bu.values
    bl_v    = bl.values
    vm20    = vol_s.rolling(20).mean().values
    vol_r   = (vol_s.values / (vm20 + 1e-9))

    fg_col  = df_feat.get("fg_value", pd.Series(50.0, index=df_feat.index)).values

    # Get timestamps
    try:
        raw_idx = df_raw.index[offset:]
        ts_arr  = pd.to_datetime(raw_idx)
    except Exception:
        ts_arr  = pd.RangeIndex(n)

    print(f"  Rows: {n} | scoring with {min_score}/7 threshold")

    # ── Score every bar & group by day ────────────────────────
    daily_candidates = {}   # date_str -> list of SignalScore

    for i in range(200, n):   # skip warmup (need EMA200)
        prob = float(probs[i])
        direction = "BUY" if prob >= prob_buy_thr else ("SELL" if prob <= prob_sell_thr else None)
        if direction is None:
            continue

        price = float(closes[i])
        if price <= 0:
            continue

        try:
            ts  = ts_arr[i]
            day = str(ts.date())
        except Exception:
            day = str(i // 24)

        sig = score_signal(
            direction    = direction,
            prob         = prob,
            price        = price,
            ema50        = float(e50[i]) if not np.isnan(e50[i]) else price,
            ema200       = float(e200[i]) if not np.isnan(e200[i]) else price,
            atr          = float(at14[i]) if not np.isnan(at14[i]) else price * 0.015,
            rsi          = float(rsi14[i]) if not np.isnan(rsi14[i]) else 50,
            bb_upper     = float(bu_v[i]) if not np.isnan(bu_v[i]) else price * 1.02,
            bb_lower     = float(bl_v[i]) if not np.isnan(bl_v[i]) else price * 0.98,
            bb_width     = float(bw_v[i]) if not np.isnan(bw_v[i]) else 0.04,
            bb_width_ma  = float(bw_ma[i]) if not np.isnan(bw_ma[i]) else 0.04,
            fg_value     = float(fg_col[i]) if not np.isnan(fg_col[i]) else 50,
            vol_ratio    = float(vol_r[i]) if not np.isnan(vol_r[i]) else 1.0,
            prob_buy_thr = prob_buy_thr,
            prob_sell_thr= prob_sell_thr,
            min_score    = min_score,
            sl_atr_mult  = sl_mult,
            tp_atr_mult  = tp_mult,
            timestamp    = day,
        )
        sig._bar_idx = i   # store for simulation

        if day not in daily_candidates:
            daily_candidates[day] = []
        daily_candidates[day].append(sig)

    # Select best N per day
    selected_signals = []
    for day, sigs in sorted(daily_candidates.items()):
        best = select_best_n_per_day(sigs, n=max_trades_day)
        selected_signals.extend(best)

    print(f"  Candidate bars   : {sum(len(v) for v in daily_candidates.values())}")
    total_passed = sum(
        sum(1 for s in v if s.passed)
        for v in daily_candidates.values()
    )
    print(f"  Passed filter    : {total_passed}")
    print(f"  Selected signals : {len(selected_signals)}")

    if not selected_signals:
        print("\n  No signals passed. Try --min-score 3 or lower --prob-buy-thr")
        return {}

    # ── Simulate trades ────────────────────────────────────────
    trades     = []
    equity     = [capital]
    cap        = capital
    open_trade = None
    signal_idx = 0
    signal_bars = {s._bar_idx: s for s in selected_signals}

    for i in range(1, n):
        price = float(closes[i])
        hi    = float(highs[i])
        lo    = float(lows[i])

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
                exit_px = t["tp"] if hit_tp else (t["sl"] if hit_sl else price)
                reason  = "TP" if hit_tp else ("SL" if hit_sl else "TIMEOUT")

                slip = (1 - slippage_pct) if t["dir"] == "BUY" else (1 + slippage_pct)
                exit_px *= slip

                gross = t["size"] * (exit_px - t["entry"]) if t["dir"] == "BUY" \
                        else t["size"] * (t["entry"] - exit_px)
                fees  = (t["size"] * t["entry"] + t["size"] * exit_px) * fee_pct
                pnl   = gross - fees
                cap  += pnl

                trades.append({
                    "day":        t["day"],
                    "entry_bar":  t["bar"],
                    "exit_bar":   i,
                    "direction":  t["dir"],
                    "score":      t["score"],
                    "prob":       round(t["prob"], 3),
                    "entry_px":   round(t["entry"], 2),
                    "exit_px":    round(exit_px, 2),
                    "reason":     reason,
                    "pnl":        round(pnl, 2),
                    "pnl_pct":    round(pnl / (t["entry"] * t["size"] + 1e-9) * 100, 3),
                    "hold_bars":  i - t["bar"],
                    "capital":    round(cap, 2),
                })
                open_trade = None

        equity.append(round(cap, 2))

        # New entry from selected signals
        if open_trade is None and i in signal_bars:
            sig = signal_bars[i]
            atr_v = sig.atr
            risk_per_unit = abs(sig.entry - sig.sl)
            if risk_per_unit > 0:
                size = (cap * risk_pct) / risk_per_unit
                open_trade = {
                    "bar":   i,
                    "day":   sig.timestamp,
                    "dir":   sig.direction,
                    "entry": sig.entry,
                    "sl":    sig.sl,
                    "tp":    sig.tp,
                    "size":  size,
                    "score": sig.score,
                    "prob":  sig.prob,
                }

    if not trades:
        print("  No trades executed in simulation.")
        return {}

    # ── Metrics ────────────────────────────────────────────────
    df_t = pd.DataFrame(trades)
    wins    = (df_t["pnl"] > 0).sum()
    total_t = len(df_t)
    wr      = wins / total_t
    eq      = np.array(equity)
    rets    = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe  = float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(8760)
    roll_max = np.maximum.accumulate(eq)
    dd      = (eq - roll_max) / (roll_max + 1e-9)
    max_dd  = float(dd.min()) * 100
    total_return = (cap - capital) / capital * 100
    avg_win  = df_t[df_t["pnl"] > 0]["pnl"].mean()
    avg_loss = df_t[df_t["pnl"] <= 0]["pnl"].mean() if (df_t["pnl"] <= 0).any() else 0
    expect   = wr * avg_win + (1 - wr) * avg_loss

    # Score distribution
    score_dist = df_t["score"].value_counts().sort_index()
    score_wr   = df_t.groupby("score").apply(lambda x: (x["pnl"]>0).mean()*100)

    print(f"\n  {'─'*54}")
    print(f"  RESULTS")
    print(f"  {'─'*54}")
    print(f"  Total trades     : {total_t}  (~{total_t/2:.0f}/yr)")
    print(f"  Win rate         : {wr*100:.1f}%  ({int(wins)}W / {total_t-int(wins)}L)")
    print(f"  Total return     : {total_return:+.2f}%")
    print(f"  Final capital    : ${cap:,.2f}")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print(f"  Sharpe ratio     : {sharpe:.3f}")
    print(f"  Avg win          : ${avg_win:.2f}")
    print(f"  Avg loss         : ${avg_loss:.2f}")
    print(f"  Expectancy/trade : ${expect:.2f}")
    print(f"  Avg hold (bars)  : {df_t['hold_bars'].mean():.1f}h")

    print(f"\n  Score distribution (score -> count | win rate):")
    for s in sorted(score_dist.index):
        cnt = score_dist[s]
        w   = score_wr.get(s, 0)
        bar = "█" * int(cnt / max(score_dist) * 20)
        print(f"    Score {s}: {cnt:>4} trades | WR={w:.0f}%  {bar}")

    print(f"\n  By direction:")
    print(df_t.groupby("direction").agg(
        trades=("pnl","count"),
        win_rate=("pnl", lambda x: f"{(x>0).mean()*100:.1f}%"),
        total_pnl=("pnl","sum"),
        avg_pnl=("pnl","mean")
    ).to_string())

    print(f"\n  By exit reason:")
    print(df_t.groupby("reason").agg(
        count=("pnl","count"),
        total_pnl=("pnl","sum"),
        avg_pnl=("pnl","mean")
    ).to_string())

    # Monthly
    print(f"\n  {'─'*54}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"  {'─'*54}")
    df_t["month"] = pd.to_datetime(df_t["day"], errors="coerce").dt.to_period("M")
    monthly = df_t.groupby("month").agg(
        trades=("pnl","count"),
        pnl=("pnl","sum"),
        wr=("pnl", lambda x: (x>0).mean()*100)
    )
    for m, row in monthly.iterrows():
        sign = "+" if row["pnl"] >= 0 else ""
        bar  = "█" * int(abs(row["pnl"]) / 100) if abs(row["pnl"]) < 2000 else "█" * 20
        col  = "" if row["pnl"] >= 0 else ""
        print(f"  {m}  t={int(row['trades']):>3}  pnl=${row['pnl']:>+8.2f}  wr={row['wr']:.0f}%  {bar[:15]}")

    print(f"\n{'='*60}\n")

    return {
        "total_trades":  total_t,
        "win_rate":      round(wr, 3),
        "total_return":  round(total_return, 2),
        "final_capital": round(cap, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe":        round(sharpe, 3),
        "expectancy":    round(expect, 2),
        "trades_df":     df_t,
        "equity_curve":  equity,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",           default=SYMBOL)
    ap.add_argument("--interval",         default=BASE_TF)
    ap.add_argument("--total",            type=int,   default=3000)
    ap.add_argument("--use-2years",       action="store_true")
    ap.add_argument("--capital",          type=float, default=10000)
    ap.add_argument("--sl-mult",          type=float, default=1.0)
    ap.add_argument("--tp-mult",          type=float, default=2.0)
    ap.add_argument("--min-score",        type=int,   default=4)
    ap.add_argument("--max-trades-day",   type=int,   default=2)
    ap.add_argument("--prob-buy-thr",     type=float, default=0.70)
    ap.add_argument("--prob-sell-thr",    type=float, default=0.30)
    ap.add_argument("--model-path",       default=MODEL_PATH)
    args = ap.parse_args()

    run_backtest(
        symbol         = args.symbol,
        interval       = args.interval,
        total          = args.total,
        use_2years     = args.use_2years,
        capital        = args.capital,
        sl_mult        = args.sl_mult,
        tp_mult        = args.tp_mult,
        min_score      = args.min_score,
        max_trades_day = args.max_trades_day,
        prob_buy_thr   = args.prob_buy_thr,
        prob_sell_thr  = args.prob_sell_thr,
        model_path     = args.model_path,
    )