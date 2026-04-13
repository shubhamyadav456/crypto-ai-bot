# models/trainer.py — v4.4
import os, sys, argparse, joblib, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report, precision_recall_curve
from xgboost import XGBClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.fetcher import build_raw_dataset
from features.engineer import (
    build_features, make_labels_simple,
    make_labels_volatility_adjusted, make_labels_triple_barrier,
    ALL_FEATURES, select_features,
)
from config.settings import SYMBOL, BASE_TF, HORIZON, MODEL_PATH, THRESHOLD_PCT


def walk_forward_eval(X, y, xgb_p, rf_p, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, briers = [], []
    cv_p = {k: v for k, v in xgb_p.items() if k not in ("early_stopping_rounds","eval_metric")}
    cv_p["n_estimators"] = 300

    print(f"\n  {'Fold':<5} {'Train':>7} {'Test':>7} {'AUC':>8} {'Brier':>8} {'UP%':>8}")
    print("  " + "-"*47)
    for i, (tr, val) in enumerate(tscv.split(X)):
        Xtr, Xvl = X[tr], X[val]
        ytr, yvl = y[tr], y[val]
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xvl_sc = sc.transform(Xvl)
        spw = float((ytr==0).sum()) / max(float((ytr==1).sum()), 1)
        xgb = XGBClassifier(**{**cv_p, "scale_pos_weight": spw})
        xgb.fit(Xtr_sc, ytr)
        rf = RandomForestClassifier(**rf_p)
        rf.fit(Xtr_sc, ytr)
        prob = (xgb.predict_proba(Xvl_sc)[:,1] + rf.predict_proba(Xvl_sc)[:,1]) / 2
        auc = roc_auc_score(yvl, prob)
        brier = brier_score_loss(yvl, prob)
        aucs.append(auc); briers.append(brier)
        print(f"  {i+1:<5} {len(tr):>7} {len(val):>7} {auc:>8.4f} {brier:>8.4f} {yvl.mean()*100:>7.1f}%")
    print("  " + "-"*47)
    print(f"  {'Mean':<5} {'':>7} {'':>7} {np.mean(aucs):>8.4f} {np.mean(briers):>8.4f}")
    return {"mean_auc": round(float(np.mean(aucs)),4), "std_auc": round(float(np.std(aucs)),4),
            "mean_brier": round(float(np.mean(briers)),4)}


def find_best_threshold(probs, labels, min_precision=0.45, min_recall=0.20):
    prec, rec, thresholds = precision_recall_curve(labels, probs)
    best_thr, best_f1 = 0.50, 0.0
    for p, r, t in zip(prec, rec, thresholds):
        if p >= min_precision and r >= min_recall:
            f1 = 2*p*r/(p+r+1e-9)
            if f1 > best_f1:
                best_f1 = f1; best_thr = float(t)
    return round(best_thr, 3), round(best_f1, 4)


def train(symbol=SYMBOL, interval=BASE_TF, total=3000, use_2years=False,
          horizon=3, label_method="balanced_barrier", model_path=None):

    if model_path is None:
        model_path = MODEL_PATH

    print(f"\n{'='*58}")
    print(f"  BTC AI Trainer v4.4")
    print(f"  Symbol  : {symbol} | TF: {interval}")
    print(f"  Data    : {'2-year' if use_2years else str(total)+' candles'}")
    print(f"  Label   : {label_method} | Horizon: {horizon}h")
    print(f"  Output  : {model_path}")
    print(f"{'='*58}")

    # ── 1. Fetch ──────────────────────────────────────────────
    print("\n[1/6] Fetching data...")
    df_raw    = build_raw_dataset(symbol, interval, total, use_2years)
    df_raw_ri = df_raw.reset_index(drop=True)
    n_raw     = len(df_raw_ri)

    # ── 2. Labels — BALANCED ──────────────────────────────────
    print(f"\n[2/6] Making labels ({label_method})...")

    if label_method == "balanced_barrier":
        # Dynamic TP/SL based on ATR percentile — more UP labels
        from features.engineer import _atr
        at = _atr(df_raw_ri["high"], df_raw_ri["low"], df_raw_ri["close"], 14).values
        closes = df_raw_ri["close"].values
        highs  = df_raw_ri["high"].values
        lows   = df_raw_ri["low"].values
        n      = len(df_raw_ri)
        labels = np.zeros(n, dtype=int)
        # Use wider TP (2.0x) and tighter SL (0.8x) → more UP hits
        for i in range(n - horizon):
            entry = closes[i]; a = at[i]
            tp = entry + 2.0 * a   # wider TP
            sl = entry - 0.8 * a   # tighter SL
            for j in range(i+1, min(i+horizon+1, n)):
                if highs[j] >= tp:  labels[i] = 1; break
                if lows[j]  <= sl:  labels[i] = 0; break
        raw_labels = pd.Series(labels, index=df_raw_ri.index)

    elif label_method == "vol_adjusted":
        raw_labels = make_labels_volatility_adjusted(df_raw_ri, horizon=horizon, mult=0.3)

    elif label_method == "simple":
        raw_labels = make_labels_simple(df_raw_ri, horizon=horizon, threshold=0.002)

    else:  # triple_barrier default
        raw_labels = make_labels_triple_barrier(df_raw_ri, horizon=horizon,
                                                 sl_mult=1.0, tp_mult=1.5)

    label_arr = raw_labels.values

    # ── 3. Features ───────────────────────────────────────────
    print(f"\n[3/6] Building features...")
    df_feat = build_features(df_raw, include_market_data=True)
    n_feat  = len(df_feat)
    offset  = n_raw - n_feat

    keep          = n_feat - horizon
    df_feat_final = df_feat.iloc[:keep].copy()
    labels_aligned = label_arr[offset : offset + n_feat]
    labels_final   = labels_aligned[:keep].astype(np.int32)

    n      = len(df_feat_final)
    up_pct = labels_final.mean() * 100
    print(f"  Samples: {n} | UP={up_pct:.1f}% | DOWN={100-up_pct:.1f}%")

    if n < 500:
        raise ValueError(f"Only {n} samples — need 500+")
    if up_pct < 20:
        print(f"  WARNING: UP={up_pct:.1f}% still low — switching to vol_adjusted labels")
        raw_labels2 = make_labels_volatility_adjusted(df_raw_ri, horizon=horizon, mult=0.3)
        labels_final = raw_labels2.values[offset:offset+n_feat][:keep].astype(np.int32)
        up_pct = labels_final.mean() * 100
        print(f"  Auto-switched: UP={up_pct:.1f}%")

    # ── 4. Feature matrix — deduplicate ───────────────────────
    print(f"\n[4/6] Preparing features...")
    avail = [f for f in ALL_FEATURES if f in df_feat_final.columns]
    # Remove exact duplicate columns
    avail = list(dict.fromkeys(avail))
    X_raw = df_feat_final[avail].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove near-duplicate columns (correlation > 0.98)
    corr = np.corrcoef(X_raw.T)
    keep_idx = []
    removed_corr = []
    for i in range(len(avail)):
        if all(abs(corr[i,j]) < 0.98 for j in keep_idx):
            keep_idx.append(i)
        else:
            removed_corr.append(avail[i])
    if removed_corr:
        print(f"  Removed {len(removed_corr)} near-duplicate features: {removed_corr[:5]}...")
    avail = [avail[i] for i in keep_idx]
    X = X_raw[:, keep_idx]
    y = labels_final

    # 3-way split
    n_train = int(n * 0.70)
    n_cal   = int(n * 0.15)
    X_train, y_train = X[:n_train],              y[:n_train]
    X_cal,   y_cal   = X[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
    X_test,  y_test  = X[n_train+n_cal:],         y[n_train+n_cal:]
    print(f"  Features: {len(avail)} (deduped) | Train:{len(X_train)} Cal:{len(X_cal)} Test:{len(X_test)}")

    # ── 5. Walk-forward CV ────────────────────────────────────
    up_train = float((y_train==1).sum()) / len(y_train)
    # Tune scale_pos_weight for better UP recall
    # spw ~2.5 pushes model to predict more UP
    spw = min(float((y_train==0).sum()) / max(float((y_train==1).sum()),1), 4.0)

    xgb_params = dict(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.05, reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="auc", early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    rf_params = dict(
        n_estimators=400, max_depth=12,
        min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, random_state=42,
        class_weight="balanced",
    )

    print(f"\n[5/6] Walk-forward CV ({len(X_train)} rows)...")
    sc_cv = StandardScaler()
    X_tr_sc = sc_cv.fit_transform(X_train)
    val = walk_forward_eval(X_tr_sc, y_train, xgb_params, rf_params, n_splits=5)

    # ── 6. Final model ────────────────────────────────────────
    print(f"\n[6/6] Training final model...")
    sc = StandardScaler()
    sc.fit(X_train)
    Xtr_sc  = sc.transform(X_train)
    Xcal_sc = sc.transform(X_cal)
    Xte_sc  = sc.transform(X_test)

    spw_tr = min(float((y_train==0).sum()) / max(float((y_train==1).sum()),1), 4.0)

    # Feature selection
    xgb_sel = XGBClassifier(**{**xgb_params, "scale_pos_weight": spw_tr})
    xgb_sel.fit(Xtr_sc, y_train, eval_set=[(Xcal_sc, y_cal)], verbose=False)
    top_feats = select_features(xgb_sel, avail, top_n=min(35, len(avail)))
    sel_idx   = [avail.index(f) for f in top_feats]

    Xtr_sel  = Xtr_sc[:,  sel_idx]
    Xcal_sel = Xcal_sc[:, sel_idx]
    Xte_sel  = Xte_sc[:,  sel_idx]

    xgb_final = XGBClassifier(**{**xgb_params, "scale_pos_weight": spw_tr})
    xgb_final.fit(Xtr_sel, y_train, eval_set=[(Xcal_sel, y_cal)], verbose=False)

    # RF gets balanced weights to improve UP recall
    rf_final = RandomForestClassifier(**rf_params)
    Xtr_cal = np.vstack([Xtr_sel, Xcal_sel])
    y_tr_cal = np.concatenate([y_train, y_cal])
    rf_final.fit(Xtr_cal, y_tr_cal)

    # Calibrate on CAL
    raw_cal = (xgb_final.predict_proba(Xcal_sel)[:,1] +
               rf_final.predict_proba(Xcal_sel)[:,1]) / 2
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_cal, y_cal)

    # Optimize threshold — min_precision=0.45 to allow more UP signals
    cal_cal = cal.transform(raw_cal)
    best_thr, best_f1 = find_best_threshold(cal_cal, y_cal,
                                              min_precision=0.45,
                                              min_recall=0.25)
    print(f"  Threshold: {best_thr}  (F1={best_f1})")

    # Test metrics
    raw_test  = (xgb_final.predict_proba(Xte_sel)[:,1] +
                 rf_final.predict_proba(Xte_sel)[:,1]) / 2
    cal_test  = cal.transform(raw_test)
    test_auc  = round(float(roc_auc_score(y_test, cal_test)), 4)
    raw_auc   = round(float(roc_auc_score(y_test, raw_test)), 4)

    print(f"\n  {'='*44}")
    print(f"  CV  AUC : {val['mean_auc']} ± {val['std_auc']}")
    print(f"  Test AUC: {test_auc}")
    print(f"  {'='*44}")
    print(f"\n--- threshold=0.50 ---")
    print(classification_report(y_test, (cal_test>=0.50).astype(int), target_names=["DOWN","UP"]))
    print(f"--- threshold={best_thr} (optimized) ---")
    print(classification_report(y_test, (cal_test>=best_thr).astype(int), target_names=["DOWN","UP"]))

    print(f"\n  Top 10 features:")
    imp = pd.Series(xgb_final.feature_importances_, index=top_feats)
    for feat, score in imp.nlargest(10).items():
        print(f"    {feat:<28} {score:.4f}")

    # Save
    version = f"v4_{time.strftime('%Y%m%d_%H%M')}"
    artifact = {
        "scaler": sc, "xgb": xgb_final, "rf": rf_final,
        "calibrator": cal,
        "all_features": avail, "sel_features": top_feats, "sel_idx": sel_idx,
        "threshold": best_thr,
        "meta": {
            "version": version, "symbol": symbol, "interval": interval,
            "horizon": horizon, "label_method": label_method,
            "n_samples": n, "n_raw_candles": n_raw,
            "n_features": len(avail), "n_sel_features": len(top_feats),
            "cv_auc": val["mean_auc"], "cv_std": val["std_auc"],
            "test_auc": test_auc, "up_pct": round(up_pct,1),
            "best_threshold": best_thr,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)

    print(f"\n{'='*58}")
    print(f"  Saved   : {model_path}")
    print(f"  Version : {version}")
    print(f"  CV AUC  : {val['mean_auc']} ± {val['std_auc']}")
    print(f"  Test AUC: {test_auc} | Threshold: {best_thr}")
    print(f"  UP%     : {up_pct:.1f}%")
    print(f"{'='*58}\n")
    return artifact, val, {"test_auc": test_auc}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",     default=SYMBOL)
    ap.add_argument("--interval",   default=BASE_TF)
    ap.add_argument("--total",      type=int, default=3000)
    ap.add_argument("--use-2years", action="store_true")
    ap.add_argument("--horizon",    type=int, default=3)
    ap.add_argument("--label",      default="balanced_barrier",
                    choices=["balanced_barrier","vol_adjusted","simple","triple_barrier"])
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()
    train(symbol=args.symbol, interval=args.interval, total=args.total,
          use_2years=args.use_2years, horizon=args.horizon,
          label_method=args.label, model_path=args.model_path)