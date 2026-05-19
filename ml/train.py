"""
Train the XGBoost signal-quality classifier.

Usage:
  python3 ml/train.py --csv reports/signals_NIFTY_2026-02-01_2026-05-17.csv
  python3 ml/train.py --csv reports/signals_NIFTY_*.csv reports/signals_BANKNIFTY_*.csv

The trained model is saved to ml/models/xgboost_model.pkl.
Use ML_MIN_CONFIDENCE in .env to control the confidence threshold.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR  = "ml/models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")


def train(csv_paths: list[str]) -> None:
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("xgboost not installed. Run: pip install xgboost")

    from ml.data_prep import load_and_prepare, load_multiple, FEATURE_COLS

    if len(csv_paths) == 1:
        X, y = load_and_prepare(csv_paths[0])
    else:
        X, y = load_multiple(csv_paths)

    if len(y) < 30:
        raise SystemExit(
            f"Only {len(y)} tradeable signals found. "
            "Need at least 30 — run backtest over a longer period."
        )

    print()
    print("─" * 55)
    print("  ML TRAINING — XGBoost Signal Quality Classifier")
    print("─" * 55)
    print(f"  Samples   : {len(y)}")
    print(f"  Win rate  : {y.mean()*100:.1f}%")
    print(f"  Features  : {FEATURE_COLS}")
    print("─" * 55)

    # Class weights to handle imbalance
    scale = float((y == 0).sum()) / float((y == 1).sum()) if (y == 1).sum() > 0 else 1.0

    # Scale model complexity with dataset size.
    # Small datasets (<150 samples) need shallower trees + heavier regularisation
    # to avoid overfitting on a handful of splits.
    n = len(y)
    if n < 100:
        n_est, depth, mcw, alpha = 80,  3, 5, 0.5
    elif n < 300:
        n_est, depth, mcw, alpha = 150, 3, 3, 0.2
    else:
        n_est, depth, mcw, alpha = 300, 4, 2, 0.1

    model = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=mcw,
        reg_alpha=alpha,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
    )

    # Adaptive CV folds: never use more folds than we have minority-class samples.
    n_folds = min(5, max(3, n // 10))
    cv     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"  CV folds  : {n_folds}  (adaptive to sample size)")
    print(f"\n  Cross-val AUC : {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Per-fold AUC  : {[f'{s:.3f}' for s in scores]}")

    # Train on full dataset
    model.fit(X, y)

    # Feature importance
    importance = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Feature importance:")
    for feat, imp in importance:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<20} {imp:.3f}  {bar}")

    # In-sample report
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print("\n  In-sample classification report:")
    print(classification_report(y, y_pred, target_names=["LOSS", "WIN"]))
    print(f"  In-sample AUC : {roc_auc_score(y, y_prob):.3f}")

    # Save model + metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    import json as _json
    from datetime import datetime as _dt
    metadata = {
        "cv_auc":          round(float(scores.mean()), 4),
        "cv_auc_std":      round(float(scores.std()), 4),
        "n_samples":       int(len(y)),
        "n_features":      int(len(FEATURE_COLS)),
        "trained_at":      _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_win_rate":  round(float(y.mean()), 4),
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        _json.dump(metadata, f, indent=2)
    print(f"\n  Model saved → {MODEL_PATH}")
    print("─" * 55)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="Path(s) to signal CSV file(s) produced by run_backtest.py. "
             "Supports wildcards: reports/signals_*.csv",
    )
    args = parser.parse_args()

    # Expand any glob patterns
    paths: list[str] = []
    for pattern in args.csv:
        expanded = glob.glob(pattern)
        if not expanded:
            raise SystemExit(f"No files matched: {pattern}")
        paths.extend(expanded)

    train(paths)


if __name__ == "__main__":
    main()
