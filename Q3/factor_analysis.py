"""Factor analysis for MCM 2026 Problem C (Dancing with the Stars).

Generates engineered features, runs comparative models, produces plots,
outputs metrics for LaTeX reporting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
ESTIMATE_PATH = DATA_DIR / "final_estimation.csv"
LONG_PATH = DATA_DIR / "dwts_processed_long_v2.csv"
LONG_PATH_FALLBACK = DATA_DIR / "dwts_processed_long.csv"
OUTPUT_FIG_DIR = ROOT / "outputs" / "figures"
OUTPUT_REPORT_DIR = ROOT / "outputs" / "reports"
METRIC_PATH = OUTPUT_REPORT_DIR / "factor_analysis_metrics.json"
PDP_DATA_PATH = OUTPUT_REPORT_DIR / "factor_analysis_pdp_age.csv"

FIG_R2 = OUTPUT_FIG_DIR / "impact_performance_vs_demographics.png"
FIG_PDP = OUTPUT_FIG_DIR / "age_partial_dependence.png"
FIG_HEATMAP = OUTPUT_FIG_DIR / "age_industry_heatmap.png"

PALETTE_CUSTOM = [
    "#1B4332",  # 墨绿 深基色
    "#2D6A4F",  # 青绿 次要
    "#40916C",  # 蓝绿 提亮
    "#52796F",  # 石板蓝 过渡
    "#6A8EAE",  # 灰紫蓝 冷色补充
    "#9D6B94",  # 薰衣草紫 点缀
]
sns.set_theme(style="whitegrid", palette=PALETTE_CUSTOM)


def load_long_table() -> pd.DataFrame:
    path = LONG_PATH if LONG_PATH.exists() else LONG_PATH_FALLBACK
    if not path.exists():
        raise FileNotFoundError("Missing dwts_processed_long CSV")
    cols = [
        "season",
        "celebrity_name",
        "ballroom_partner",
        "celebrity_age_during_season",
        "placement",
    ]
    return pd.read_csv(path)[cols].drop_duplicates()


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


def prepare_contestant_level() -> pd.DataFrame:
    if not ESTIMATE_PATH.exists():
        raise FileNotFoundError("Missing final_estimation.csv")
    estimates = pd.read_csv(ESTIMATE_PATH)
    if "harmonized_fan_percent" in estimates.columns:
        estimates["fan_share"] = estimates["harmonized_fan_percent"].astype(float)
    else:
        estimates["fan_share"] = (
            estimates.groupby(["season", "week"]) ["predicted_fan_vote"].transform(
                lambda s: s / s.sum() if s.sum() else 0.0
            )
        )

    estimates["judge_score_avg"] = estimates["total_judge_score"] / estimates["num_judges"].replace(0, np.nan)

    reference = load_long_table()
    partner_strength = build_partner_strength(reference)

    merged = estimates.merge(reference, on=["season", "celebrity_name"], how="left")
    merged = merged.merge(partner_strength, on="ballroom_partner", how="left")

    agg = (
        merged.groupby(["season", "celebrity_name"], as_index=False)
        .agg(
            fan_share=("fan_share", "mean"),
            average_judge_score=("judge_score_avg", "mean"),
            celebrity_industry=("celebrity_industry", "first"),
            celebrity_age_during_season=("celebrity_age_during_season", "first"),
            partner_strength=("partner_strength", "first"),
        )
    )

    agg["fan_share_z"] = agg.groupby("season")["fan_share"].transform(zscore)
    # top 50% fan favorite threshold per season
    agg["fan_share_median"] = agg.groupby("season")["fan_share"].transform("median")
    agg["is_fan_favorite"] = (agg["fan_share"] >= agg["fan_share_median"]).astype(int)
    agg = agg.drop(columns=["fan_share_median"])
    return agg.dropna(subset=["fan_share_z", "celebrity_age_during_season", "partner_strength", "average_judge_score"])


def build_preprocess(cat_cols: list, num_cols: list) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )


def run_regression_models(df: pd.DataFrame) -> Tuple[float, float, Pipeline]:
    groups = df["season"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    cat_cols = ["celebrity_industry"]
    num_a = ["celebrity_age_during_season", "partner_strength"]
    num_b = num_a + ["average_judge_score"]

    X_a = df[cat_cols + num_a]
    X_b = df[cat_cols + num_b]
    y = df["fan_share_z"]

    pre_a = build_preprocess(cat_cols, num_a)
    pre_b = build_preprocess(cat_cols, num_b)

    model_a = Pipeline([("pre", pre_a), ("reg", LinearRegression())])
    model_b = Pipeline([("pre", pre_b), ("reg", LinearRegression())])

    r2_a = cross_val_score(model_a, X_a, y, cv=gkf, groups=groups, scoring="r2")
    r2_b = cross_val_score(model_b, X_b, y, cv=gkf, groups=groups, scoring="r2")

    model_a.fit(X_a, y)
    model_b.fit(X_b, y)

    return float(r2_a.mean()), float(r2_b.mean()), model_b


def run_classifier(df: pd.DataFrame) -> Tuple[float, float, Pipeline]:
    groups = df["season"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    cat_cols = ["celebrity_industry"]
    num_cols = ["celebrity_age_during_season", "partner_strength"]

    X = df[cat_cols + num_cols]
    y = df["is_fan_favorite"]

    pre = build_preprocess(cat_cols, num_cols)
    clf = Pipeline(
        [
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    proba = cross_val_predict(
        clf,
        X,
        y,
        cv=gkf,
        groups=groups,
        method="predict_proba",
    )[:, 1]
    preds = (proba >= 0.5).astype(int)
    accuracy = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba)
    clf.fit(X, y)
    return float(accuracy), float(auc), clf


def plot_r2_bar(r2_a: float, r2_b: float) -> None:
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["Model A: Demographics", "Model B: +Judge Score"],
        y=[r2_a, r2_b],
        palette=[PALETTE_CUSTOM[1], PALETTE_CUSTOM[5]],
    )
    plt.ylabel("R² (GroupKFold by season)")
    plt.title("Impact of Performance vs. Demographics")
    plt.ylim(0, max(r2_b, r2_a, 0.05) + 0.05)
    for i, v in enumerate([r2_a, r2_b]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_R2, dpi=240)
    plt.close()


def plot_age_pdp(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    X = df[["celebrity_industry", "celebrity_age_during_season", "partner_strength", "average_judge_score"]]
    pd_result = partial_dependence(
        model,
        X,
        features=["celebrity_age_during_season"],  # type: ignore[arg-type]
        grid_resolution=40,
        kind="average",
    )
    ages = pd_result["grid_values"][0]
    preds = pd_result["average"][0]

    fig, ax = plt.subplots(figsize=(6, 4))
    line_color = PALETTE_CUSTOM[2]
    ax.plot(ages, preds, color=line_color, linewidth=2.2)
    ax.fill_between(ages, preds, color=line_color, alpha=0.16)
    ax.set_ylabel("Predicted fan_share_z")
    ax.set_xlabel("Celebrity age")
    ax.set_title("Partial dependence of age on fan support")
    plt.tight_layout()
    plt.savefig(FIG_PDP, dpi=240)
    plt.close(fig)

    pdp_df = pd.DataFrame({"age": ages, "pred": preds})
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdp_df.to_csv(PDP_DATA_PATH, index=False)
    return pdp_df


def plot_age_industry_heatmap(df: pd.DataFrame) -> None:
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    bins = [0, 25, 35, 45, 55, 100]
    labels = ["<25", "25-34", "35-44", "45-54", "55+"]
    df = df.copy()
    df["age_group"] = pd.cut(df["celebrity_age_during_season"], bins=bins, labels=labels, right=False)
    top_industries = df["celebrity_industry"].value_counts().head(5).index
    filtered = df[df["celebrity_industry"].isin(top_industries)]
    pivot = filtered.pivot_table(
        index="celebrity_industry",
        columns="age_group",
        values="fan_share_z",
        aggfunc="mean",
    )

    plt.figure(figsize=(7, 4))
    cmap = sns.color_palette(PALETTE_CUSTOM[:5], as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, cbar_kws={"label": "Avg fan z-score"})
    plt.title("Age × Industry interaction on fan support")
    plt.xlabel("Age group")
    plt.ylabel("Industry (Top 5)")
    plt.tight_layout()
    plt.savefig(FIG_HEATMAP, dpi=240)
    plt.close()


def main() -> None:
    df = prepare_contestant_level()

    r2_a, r2_b, reg_model_b = run_regression_models(df)
    acc, auc, clf = run_classifier(df)

    # feature importance for halo effect
    rf_reg = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    feats = ["celebrity_industry", "celebrity_age_during_season", "partner_strength", "average_judge_score"]
    pre = build_preprocess(["celebrity_industry"], feats[1:])
    rf_pipe = Pipeline([( "pre", pre), ("rf", rf_reg)])
    rf_pipe.fit(df[["celebrity_industry", "celebrity_age_during_season", "partner_strength", "average_judge_score"]], df["fan_share_z"])
    # approximate importance by mapping back: average importance over transformed columns for partner_strength
    ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["celebrity_industry"]))
    num_names = feats[1:]
    feature_names = cat_names + num_names
    importances = pd.Series(rf_pipe.named_steps["rf"].feature_importances_, index=feature_names)
    partner_importance = float(importances.get("partner_strength", np.nan))

    plot_r2_bar(r2_a, r2_b)
    pdp_df = plot_age_pdp(reg_model_b, df)
    plot_age_industry_heatmap(df)

    metrics: Dict[str, float] = {
        "n_records": int(len(df)),
        "model_a_r2": r2_a,
        "model_b_r2": r2_b,
        "r2_gap": r2_b - r2_a,
        "clf_accuracy": acc,
        "clf_auc": auc,
        "partner_strength_importance": partner_importance,
        "pdp_age_min": float(pdp_df["pred"].min()),
        "pdp_age_max": float(pdp_df["pred"].max()),
    }

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with METRIC_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Factor analysis complete. Metrics saved to", METRIC_PATH)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
