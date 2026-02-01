"""Target-encoded features + LightGBM vs baselines for fan and judge predictions.
GroupKFold by season to avoid leakage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Q3.fan_vote_regression import (
    LONG_PATH,
    LONG_PATH_V2,
    prepare_dataset,
)

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_PATH = REPORT_DIR / "model_selection_te_lgbm.json"


def zscore_group(series: pd.Series, group: pd.Series) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std()
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mean) / std

    return series.groupby(group).transform(_z)


def load_with_judges() -> pd.DataFrame:
    df = prepare_dataset()
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
    df["judge_score_z"] = zscore_group(df["judge_score_total"], df["season"])
    df = df.dropna(
        subset=["fan_share_z", "judge_score_z", "celebrity_age_during_season", "partner_strength", "celebrity_industry"]
    )
    return df


def target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str, target: str, smoothing: float = 10.0) -> Tuple[pd.Series, pd.Series]:
    global_mean = train[target].mean()
    stats = train.groupby(col)[target].agg(["mean", "count"])
    smoothing_weights = 1 / (1 + np.exp(-(stats["count"] - 1) / smoothing))
    enc_values = global_mean * (1 - smoothing_weights) + stats["mean"] * smoothing_weights
    enc_map = enc_values.to_dict()
    train_enc = train[col].map(enc_map).fillna(global_mean)
    test_enc = test[col].map(enc_map).fillna(global_mean)
    return train_enc, test_enc


def build_folds(df: pd.DataFrame, target: str, groups: pd.Series, n_splits: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    train_indices, test_indices = [], []
    for tr, te in gkf.split(df, df[target], groups):
        train_indices.append(tr)
        test_indices.append(te)
    return train_indices, test_indices


def encode_and_split(df: pd.DataFrame, target: str, groups: pd.Series) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.Series], List[pd.Series]]:
    train_idx_list, test_idx_list = build_folds(df, target, groups)
    X_train_folds: List[pd.DataFrame] = []
    X_test_folds: List[pd.DataFrame] = []
    y_train_folds: List[pd.Series] = []
    y_test_folds: List[pd.Series] = []

    for tr_idx, te_idx in zip(train_idx_list, test_idx_list):
        train_df = df.iloc[tr_idx].copy()
        test_df = df.iloc[te_idx].copy()

        te_train, te_test = target_encode(train_df, test_df, "celebrity_industry", target)
        train_df["industry_te"] = te_train
        test_df["industry_te"] = te_test

        # interaction: age × encoded industry
        train_df["age_x_te"] = train_df["celebrity_age_during_season"] * train_df["industry_te"]
        test_df["age_x_te"] = test_df["celebrity_age_during_season"] * test_df["industry_te"]

        feats = ["celebrity_age_during_season", "partner_strength", "industry_te", "age_x_te"]
        X_train_folds.append(train_df[feats])
        X_test_folds.append(test_df[feats])
        y_train_folds.append(train_df[target])
        y_test_folds.append(test_df[target])

    return X_train_folds, X_test_folds, y_train_folds, y_test_folds


def eval_models(df: pd.DataFrame, target: str, groups: pd.Series) -> Dict[str, Dict[str, float]]:
    X_tr_folds, X_te_folds, y_tr_folds, y_te_folds = encode_and_split(df, target, groups)

    models = {
        "ridge": make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "lasso": make_pipeline(StandardScaler(with_mean=False), LassoCV(alphas=np.logspace(-3, 3, 13), max_iter=5000)),
        "random_forest": RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "lightgbm": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        r2_list: List[float] = []
        mae_list: List[float] = []
        for X_tr, X_te, y_tr, y_te in zip(X_tr_folds, X_te_folds, y_tr_folds, y_te_folds):
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            r2_list.append(r2_score(y_te, preds))
            mae_list.append(mean_absolute_error(y_te, preds))
        results[name] = {
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
        }
    return results


def main() -> None:
    df = load_with_judges()
    groups = df["season"]

    fan_results = eval_models(df, target="fan_share_z", groups=groups)
    judge_results = eval_models(df, target="judge_score_z", groups=groups)

    def best_model(res: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
        items = res.items()
        best = max(items, key=lambda kv: kv[1]["r2_mean"])
        return best[0], best[1]

    best_fan_name, best_fan_metrics = best_model(fan_results)
    best_judge_name, best_judge_metrics = best_model(judge_results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": int(len(df)),
        "n_features": 4,
        "n_seasons": int(groups.nunique()),
        "models": {
            "fan_share_z": fan_results,
            "judge_score_z": judge_results,
        },
        "best": {
            "fan_share_z": {"model": best_fan_name, **best_fan_metrics},
            "judge_score_z": {"model": best_judge_name, **best_judge_metrics},
        },
        "notes": [
            "Target encoding on celebrity_industry; interaction age×industry_te included.",
            "GroupKFold by season to avoid leakage.",
            "LightGBM tuned conservatively for small feature space; feel free to retune.",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("TE+LightGBM selection complete. Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
