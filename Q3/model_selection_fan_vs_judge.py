"""Model comparison for predicting fan support and judge scores using multiple algorithms.
Generates cross-validated metrics and picks the best-performing model per target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Q3.fan_vote_regression import (
    LONG_PATH,
    LONG_PATH_V2,
    prepare_dataset,
    build_design_matrix,
)

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_PATH = REPORT_DIR / "model_selection_fan_vs_judge.json"


def zscore_group(series: pd.Series, group: pd.Series) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std()
        return (s - mean) / std if std else pd.Series(np.zeros(len(s)), index=s.index)

    return series.groupby(group).transform(_z)


def build_targets() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = prepare_dataset()

    # bring in judge scores from the long table
    data_path = LONG_PATH_V2 if LONG_PATH_V2.exists() else LONG_PATH
    long_df = pd.read_csv(data_path)
    if "active_week" in long_df.columns:
        mask = long_df["active_week"].astype(bool)
    else:
        mask = pd.Series(True, index=long_df.index)

    judge_table = (
        long_df.loc[mask, ["season", "week", "celebrity_name", "judge_score_total"]]
        .drop_duplicates()
    )

    df = df.merge(judge_table, on=["season", "week", "celebrity_name"], how="left")

    # judge score z per season to mirror fan_share_z scale
    df["judge_score_z"] = zscore_group(df["judge_score_total"], df["season"])

    # drop rows missing key fields
    df = df.dropna(subset=["fan_share_z", "celebrity_age_during_season", "partner_strength", "judge_score_z"])

    X, _ = build_design_matrix(df)
    y_fan = df["fan_share_z"]
    y_judge = df["judge_score_z"]
    groups = df["season"]
    return X, y_fan, y_judge, groups


def eval_model(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    r2_list: List[float] = []
    mae_list: List[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_list.append(r2_score(y_test, preds))
        mae_list.append(mean_absolute_error(y_test, preds))

    return {
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
    }


def main() -> None:
    X, y_fan, y_judge, groups = build_targets()

    models = {
        "ols": make_pipeline(StandardScaler(with_mean=False), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "lasso": make_pipeline(StandardScaler(with_mean=False), LassoCV(alphas=np.logspace(-3, 3, 13), max_iter=5000)),
        "elasticnet": make_pipeline(StandardScaler(with_mean=False), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=np.logspace(-3, 3, 13), max_iter=5000)),
        "random_forest": RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "gbrt": GradientBoostingRegressor(random_state=42),
        "hist_gbrt": HistGradientBoostingRegressor(random_state=42),
    }

    results: Dict[str, Dict[str, Dict[str, float]]] = {"fan_share_z": {}, "judge_score_z": {}}

    for name, model in models.items():
        results["fan_share_z"][name] = eval_model(model, X, y_fan, groups)
        results["judge_score_z"][name] = eval_model(model, X, y_judge, groups)

    # pick best by R^2
    def best_model(target: str) -> Tuple[str, Dict[str, float]]:
        items = results[target].items()
        best = max(items, key=lambda kv: kv[1]["r2_mean"])
        return best[0], best[1]

    best_fan_name, best_fan_metrics = best_model("fan_share_z")
    best_judge_name, best_judge_metrics = best_model("judge_score_z")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_seasons": int(groups.nunique()),
        "models": results,
        "best": {
            "fan_share_z": {"model": best_fan_name, **best_fan_metrics},
            "judge_score_z": {"model": best_judge_name, **best_judge_metrics},
        },
        "notes": [
            "GroupKFold by season to avoid leakage across seasons.",
            "Features follow fan_vote_regression design matrix (age, partner_strength, industry dummies, age×industry).",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Model selection complete. Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
