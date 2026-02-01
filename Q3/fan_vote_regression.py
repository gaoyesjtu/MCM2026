"""Regression analysis of fan vote drivers based on final_estimation.csv."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
ESTIMATE_PATH = ROOT / "data" / "processed" / "final_estimation.csv"
LONG_PATH = ROOT / "data" / "processed" / "dwts_processed_long.csv"
LONG_PATH_V2 = ROOT / "data" / "processed" / "dwts_processed_long_v2.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_PATH = REPORT_DIR / "fan_vote_driver_report.json"
IMPORTANCE_OLS_PATH = FIG_DIR / "fan_vote_feature_importance_ols.png"
IMPORTANCE_RF_PATH = FIG_DIR / "fan_vote_feature_importance_rf.png"
INDUSTRY_PATH = REPORT_DIR / "fan_vote_industry_effects.csv"

PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]


def set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        rc={
            "axes.edgecolor": "#2f2f2f",
            "grid.alpha": 0.3,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        },
    )


def load_reference_table() -> pd.DataFrame:
    data_path = LONG_PATH_V2 if LONG_PATH_V2.exists() else LONG_PATH
    long_df = pd.read_csv(data_path)
    cols = [
        "season",
        "celebrity_name",
        "ballroom_partner",
        "celebrity_age_during_season",
        "placement",
    ]
    reference = long_df[cols].drop_duplicates()
    return reference


def build_partner_strength(reference: pd.DataFrame) -> pd.DataFrame:
    partner_strength = (
        reference.dropna(subset=["ballroom_partner", "placement"])
        .groupby("ballroom_partner", as_index=False)
        .agg(partner_mean_placement=("placement", "mean"))
    )
    max_place = partner_strength["partner_mean_placement"].max()
    partner_strength["partner_strength"] = max_place + 1 - partner_strength[
        "partner_mean_placement"
    ]
    return partner_strength[["ballroom_partner", "partner_strength"]]


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def prepare_dataset() -> pd.DataFrame:
    estimates = pd.read_csv(ESTIMATE_PATH)
    if "normalized_fan_support" not in estimates.columns:
        if "predicted_fan_vote" not in estimates.columns:
            raise ValueError("final_estimation.csv 缺少 predicted_fan_vote 列")
        estimates["normalized_fan_support"] = (
            estimates.groupby(["season", "week"])["predicted_fan_vote"]
            .transform(lambda s: s / s.sum() if s.sum() else 0.0)
        )
    reference = load_reference_table()
    partner_strength = build_partner_strength(reference)

    merged = estimates.merge(
        reference,
        on=["season", "celebrity_name"],
        how="left",
    ).merge(
        partner_strength,
        on="ballroom_partner",
        how="left",
    )

    merged["fan_share_z"] = (
        merged.groupby("season")["normalized_fan_support"].transform(zscore)
    )
    return merged


def build_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df[["celebrity_age_during_season", "partner_strength"]].copy()
    industry = pd.get_dummies(df["celebrity_industry"], prefix="industry")
    interactions = industry.mul(
        df["celebrity_age_during_season"].fillna(0), axis=0
    )
    interactions.columns = [f"age_x_{col}" for col in industry.columns]
    features = pd.concat([base, industry, interactions], axis=1)
    return features, industry


def standardize_matrix(X: pd.DataFrame) -> pd.DataFrame:
    standardized = X.copy()
    for col in standardized.columns:
        mean = standardized[col].mean()
        std = standardized[col].std()
        if std == 0 or np.isnan(std):
            standardized[col] = 0.0
        else:
            standardized[col] = (standardized[col] - mean) / std
    return standardized


def fit_ols(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    X_np = np.column_stack([np.ones(len(X)), X.to_numpy()])
    y_np = y.to_numpy()
    coef, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
    names = ["intercept"] + list(X.columns)
    return {name: float(value) for name, value in zip(names, coef)}


def plot_feature_importance(importances: pd.Series, title: str, out_path: Path) -> None:
    top = importances.sort_values(ascending=True).tail(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=top.values,
        y=top.index,
        color=PALETTE[1],
        edgecolor="white",
        linewidth=0.5,
    )
    plt.title(title)
    plt.xlabel("Absolute coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    if not ESTIMATE_PATH.exists():
        raise FileNotFoundError("缺少 data/processed/final_estimation.csv")

    set_plot_style()
    df = prepare_dataset()

    df = df.dropna(subset=["fan_share_z", "celebrity_age_during_season", "partner_strength"])

    X, industry = build_design_matrix(df)
    X_std = standardize_matrix(X)
    y = df["fan_share_z"]

    coef = fit_ols(X_std, y)
    coef_series = pd.Series(coef).drop("intercept")
    importances = coef_series.abs()

    rf_model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf_model.fit(X.fillna(0.0), y)
    rf_importances = pd.Series(
        rf_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    industry_effects = (
        df.groupby("celebrity_industry")["fan_share_z"].mean().sort_values(ascending=False)
    )
    industry_effects.to_csv(INDUSTRY_PATH)

    plot_feature_importance(
        importances,
        "Feature importance (OLS | standardized coefficients)",
        IMPORTANCE_OLS_PATH,
    )
    plot_feature_importance(
        rf_importances,
        "Feature importance (Random Forest)",
        IMPORTANCE_RF_PATH,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "model": "OLS (standardized features)",
        "target": "fan_share_z (within-season z-score)",
        "coefficients": coef,
        "random_forest": {
            "n_estimators": int(rf_model.n_estimators),
            "feature_importances": rf_importances.head(15).to_dict(),
        },
        "top_positive_coefficients": coef_series.sort_values(ascending=False).head(5).to_dict(),
        "top_negative_coefficients": coef_series.sort_values().head(5).to_dict(),
        "top_interactions": coef_series[coef_series.index.str.startswith("age_x_")]
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .to_dict(),
        "industry_effects_top": industry_effects.head(5).to_dict(),
        "industry_effects_bottom": industry_effects.tail(5).to_dict(),
        "notes": [
            "Gender feature is not available in the dataset and was skipped.",
            "Partner strength is derived from average placement (higher is stronger).",
            "Interaction terms: age × industry are included in the OLS model.",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("Fan vote regression analysis complete.")
    print(f"Report: {REPORT_PATH}")
    print(f"OLS importance figure: {IMPORTANCE_OLS_PATH}")
    print(f"RF importance figure: {IMPORTANCE_RF_PATH}")


if __name__ == "__main__":
    main()
