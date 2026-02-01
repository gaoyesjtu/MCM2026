"""Evaluate a proposed fair voting system against rank and percent methods."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
LONG_PATH = ROOT / "data" / "processed" / "dwts_processed_long.csv"
LONG_PATH_V2 = ROOT / "data" / "processed" / "dwts_processed_long_v2.csv"
ESTIMATES_PATH = ROOT / "data" / "processed" / "fan_vote_estimates.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_PATH = REPORT_DIR / "fair_method_comparison.json"
FIG_PATH = FIG_DIR / "fair_method_comparison.png"
HEATMAP_PATH = FIG_DIR / "fair_threshold_heatmap.png"

PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]


@dataclass
class MethodMetrics:
    consistency: float
    avg_fan_rank: float
    avg_judge_rank: float
    extreme_elimination_rate: float
    avg_agreement: float
    overall_score: float


@dataclass
class ConfigResult:
    gamma: float
    rescue_rank_threshold: float
    fan_gap_threshold: float
    metrics: Dict[str, MethodMetrics]


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


def build_weekly_table(long_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "week",
        "celebrity_name",
        "judge_score_total",
        "active_week",
        "result_status",
        "eliminated_week",
    ]
    for extra in ("elimination_count", "no_elimination_week"):
        if extra in long_df.columns:
            columns.append(extra)

    weekly = long_df[columns].drop_duplicates().copy()
    weekly = weekly.loc[weekly["active_week"]].copy()
    weekly["is_eliminated"] = (
        weekly["result_status"].eq("eliminated")
        & weekly["eliminated_week"].eq(weekly["week"])
    )
    weekly["elimination_count"] = weekly.get("elimination_count", 0)
    return weekly


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def judge_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(ascending=False, method="average").to_numpy()


def predict_elimination(
    week_df: pd.DataFrame,
    method: str,
    season: int,
    elimination_count: int,
    gamma: float,
    rescue_rank_threshold: float,
    fan_gap_threshold: float,
) -> tuple[np.ndarray, bool]:
    fan_share = week_df["fan_share_mean"].to_numpy(dtype=float)
    judge_scores = week_df["judge_score_total"].to_numpy(dtype=float)
    judge_share = judge_scores / judge_scores.sum() if judge_scores.sum() else np.zeros_like(judge_scores)

    judge_rank_vals = judge_rank(judge_scores)
    judge_rank_pct = (judge_rank_vals - 1) / max(len(judge_rank_vals) - 1, 1)
    fan_gap = float(fan_share.max() - fan_share.min())
    enable_rescue = (judge_rank_pct.max() >= rescue_rank_threshold) and (
        fan_gap >= fan_gap_threshold
    )

    if method == "percent":
        total_score = fan_share + judge_share
        if season >= 28 and elimination_count == 1 and enable_rescue:
            return np.argsort(total_score)[:2], True
        return np.argsort(total_score)[:elimination_count], False

    if method == "rank":
        total_rank = rank_desc(fan_share) + judge_rank_vals
        if season >= 28 and elimination_count == 1 and enable_rescue:
            return np.argsort(total_rank)[-2:], True
        return np.argsort(total_rank)[-elimination_count:], False

    if method == "fair_linear":
        total_score = 0.5 * fan_share + 0.5 * judge_share + gamma * np.minimum(
            fan_share, judge_share
        )
        if season >= 28 and elimination_count == 1 and enable_rescue:
            return np.argsort(total_score)[:2], True
        return np.argsort(total_score)[:elimination_count], False

    if method == "fair_nonlinear":
        agreement = 1.0 - np.abs(fan_share - judge_share)
        total_score = 0.5 * fan_share + 0.5 * judge_share + gamma * agreement**2
        if season >= 28 and elimination_count == 1 and enable_rescue:
            return np.argsort(total_score)[:2], True
        return np.argsort(total_score)[:elimination_count], False

    raise ValueError(f"Unknown method: {method}")


def evaluate_methods(
    merged: pd.DataFrame,
    gamma: float,
    rescue_rank_threshold: float,
    fan_gap_threshold: float,
) -> Dict[str, MethodMetrics]:
    methods = ["percent", "rank", "fair_linear", "fair_nonlinear"]
    records: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

    for (season, week), week_df in merged.groupby(["season", "week"], sort=True):
        elimination_count = int(week_df["elimination_count"].iloc[0])
        if elimination_count == 0:
            continue

        fan_rank = rank_desc(week_df["fan_share_mean"].to_numpy(dtype=float))
        judge_rank_vals = judge_rank(week_df["judge_score_total"].to_numpy(dtype=float))
        eliminated_indices = np.where(week_df["is_eliminated"].to_numpy())[0]
        if eliminated_indices.size == 0:
            continue

        for method in methods:
            predicted, rescue_mode = predict_elimination(
                week_df,
                method,
                int(season),
                elimination_count,
                gamma,
                rescue_rank_threshold,
                fan_gap_threshold,
            )
            if rescue_mode:
                consistent = bool(np.any(np.isin(eliminated_indices, predicted)))
            else:
                consistent = bool(
                    np.array_equal(np.sort(predicted), np.sort(eliminated_indices))
                )

            eval_indices = predicted
            elim_fan_rank = fan_rank[eval_indices].mean()
            elim_judge_rank = judge_rank_vals[eval_indices].mean()
            agreement = 1.0 - np.abs(
                week_df["fan_share_mean"].to_numpy(dtype=float)
                - week_df["judge_score_total"].to_numpy(dtype=float)
                / max(week_df["judge_score_total"].sum(), 1e-12)
            )
            elim_agreement = agreement[eval_indices].mean()
            extreme = float(
                np.mean(
                    (fan_rank[eval_indices] <= 2)
                    | (judge_rank_vals[eval_indices] <= 2)
                )
            )

            records[method].append(
                {
                    "consistency": float(consistent),
                    "fan_rank": float(elim_fan_rank),
                    "judge_rank": float(elim_judge_rank),
                    "agreement": float(elim_agreement),
                    "extreme": float(extreme),
                }
            )

    metrics: Dict[str, MethodMetrics] = {}
    raw = {}
    for method, rows in records.items():
        df = pd.DataFrame(rows)
        raw[method] = {
            "consistency": df["consistency"].mean(),
            "fan_rank": df["fan_rank"].mean(),
            "judge_rank": df["judge_rank"].mean(),
            "agreement": df["agreement"].mean(),
            "extreme": df["extreme"].mean(),
        }

    for key in ["consistency", "fan_rank", "judge_rank", "agreement", "extreme"]:
        values = np.array([raw[m][key] for m in methods], dtype=float)
        if key == "extreme":
            values = -values
        min_val = values.min()
        max_val = values.max()
        if np.isclose(max_val, min_val):
            scaled = np.ones_like(values)
        else:
            scaled = (values - min_val) / (max_val - min_val)
        for idx, method in enumerate(methods):
            raw[method][f"{key}_scaled"] = float(scaled[idx])

    for method in methods:
        overall = np.mean(
            [
                raw[method]["consistency_scaled"],
                raw[method]["fan_rank_scaled"],
                raw[method]["judge_rank_scaled"],
                raw[method]["agreement_scaled"],
                raw[method]["extreme_scaled"],
            ]
        )
        metrics[method] = MethodMetrics(
            consistency=float(raw[method]["consistency"]),
            avg_fan_rank=float(raw[method]["fan_rank"]),
            avg_judge_rank=float(raw[method]["judge_rank"]),
            extreme_elimination_rate=float(raw[method]["extreme"]),
            avg_agreement=float(raw[method]["agreement"]),
            overall_score=float(overall),
        )

    return metrics


def plot_metrics(metrics: Dict[str, MethodMetrics]) -> None:
    data = []
    for method, metric in metrics.items():
        data.append({"method": method, "metric": "Consistency", "value": metric.consistency})
        data.append({"method": method, "metric": "Avg fan rank", "value": metric.avg_fan_rank})
        data.append({"method": method, "metric": "Avg judge rank", "value": metric.avg_judge_rank})
        data.append({"method": method, "metric": "Elim agreement", "value": metric.avg_agreement})
        data.append({"method": method, "metric": "Extreme elimination rate", "value": metric.extreme_elimination_rate})
        data.append({"method": method, "metric": "Overall score", "value": metric.overall_score})

    plot_df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="method",
        palette=PALETTE[:4],
        edgecolor="white",
        linewidth=0.5,
    )
    plt.title("Method comparison with fairness metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=15)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=220)
    plt.close()


def plot_threshold_heatmap(
    search_log: Dict[str, Dict[str, float]],
    gammas: List[float],
    rescue_thresholds: List[float],
    fan_gap_thresholds: List[float],
) -> None:
    records: List[Dict[str, float]] = []
    for key, values in search_log.items():
        parts = dict(item.split("=") for item in key.split("|"))
        records.append(
            {
                "gamma": float(parts["g"]),
                "rescue": float(parts["r"]),
                "gap": float(parts["gap"]),
                "fair_linear": values["fair_linear"],
                "fair_nonlinear": values["fair_nonlinear"],
            }
        )

    df = pd.DataFrame(records)
    df = df[df["gamma"].isin(gammas)]
    df["threshold_pair"] = df.apply(
        lambda row: f"r={row['rescue']:.2f}, gap={row['gap']:.2f}", axis=1
    )
    pivot = df.pivot_table(
        index="threshold_pair",
        columns="gamma",
        values="fair_nonlinear",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        linewidths=0.3,
        cbar_kws={"label": "Overall score"},
    )
    plt.title("Threshold-performance heatmap (fair nonlinear)")
    plt.xlabel("Gamma")
    plt.ylabel("Rescue threshold & fan gap")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=220)
    plt.close()


def grid_search_configs(
    merged: pd.DataFrame,
    gammas: List[float],
    rescue_thresholds: List[float],
    fan_gap_thresholds: List[float],
) -> Tuple[ConfigResult, Dict[str, Dict[str, float]]]:
    best_config: ConfigResult | None = None
    best_score = -np.inf
    search_log: Dict[str, Dict[str, float]] = {}

    for gamma in gammas:
        for rescue_threshold in rescue_thresholds:
            for gap_threshold in fan_gap_thresholds:
                metrics = evaluate_methods(
                    merged, gamma, rescue_threshold, gap_threshold
                )
                key = f"g={gamma:.2f}|r={rescue_threshold:.2f}|gap={gap_threshold:.2f}"
                search_log[key] = {
                    "fair_linear": metrics["fair_linear"].overall_score,
                    "fair_nonlinear": metrics["fair_nonlinear"].overall_score,
                }
                current_best = max(
                    metrics["fair_linear"].overall_score,
                    metrics["fair_nonlinear"].overall_score,
                )
                if current_best > best_score:
                    best_score = current_best
                    best_config = ConfigResult(
                        gamma=gamma,
                        rescue_rank_threshold=rescue_threshold,
                        fan_gap_threshold=gap_threshold,
                        metrics=metrics,
                    )

    if best_config is None:
        raise RuntimeError("Grid search failed to produce any configuration.")

    return best_config, search_log


def main() -> None:
    data_path = LONG_PATH_V2 if LONG_PATH_V2.exists() else LONG_PATH
    long_df = pd.read_csv(data_path)
    weekly = build_weekly_table(long_df)

    if not ESTIMATES_PATH.exists():
        raise FileNotFoundError("Missing fan_vote_estimates.csv. Run vote_estimation.py first.")

    estimates = pd.read_csv(ESTIMATES_PATH)
    merged = weekly.merge(
        right=estimates,
        on=["season", "week", "celebrity_name"],
        how="inner",
    )

    set_plot_style()
    gamma_grid = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]
    rescue_thresholds = [0.7, 0.8, 0.9]
    fan_gap_thresholds = [0.05, 0.1, 0.2]

    best_config, search_log = grid_search_configs(
        merged, gamma_grid, rescue_thresholds, fan_gap_thresholds
    )
    gamma = best_config.gamma
    rescue_rank_threshold = best_config.rescue_rank_threshold
    fan_gap_threshold = best_config.fan_gap_threshold
    metrics = best_config.metrics

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "gamma": gamma,
        "gamma_grid": gamma_grid,
        "rescue_rank_threshold": rescue_rank_threshold,
        "fan_gap_threshold": fan_gap_threshold,
        "search_log": search_log,
        "metrics": {k: metrics[k].__dict__ for k in metrics},
        "notes": [
            "Fair linear: 0.5*fan + 0.5*judge + gamma*min(fan, judge).",
            "Fair nonlinear: 0.5*fan + 0.5*judge + gamma*sqrt(fan*judge).",
            "Rescue is enabled only when judge-rank percentile >= rescue_rank_threshold.",
            "Rescue also requires fan share gap >= fan_gap_threshold.",
            "Consistency treats season>=28 as Bottom Two inclusion when elimination_count=1.",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    plot_metrics(metrics)
    plot_threshold_heatmap(
        search_log, gamma_grid, rescue_thresholds, fan_gap_thresholds
    )

    print("Fair-system analysis complete.")
    print(f"Report: {REPORT_PATH}")
    print(f"Figure: {FIG_PATH}")
    print(f"Heatmap: {HEATMAP_PATH}")


if __name__ == "__main__":
    main()
