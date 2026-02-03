"""Model IV: Engagement-Centric scoring system simulation.
Simulates baseline Rank+Save vs proposed Competency-Gated Fan Dominance.
Generates metrics, plots, and LaTeX-ready numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union, cast
from math import exp

import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "final_estimation.csv"
OUTPUT_FIG_DIR = ROOT / "outputs" / "figures"
OUTPUT_REPORT_DIR = ROOT / "outputs" / "reports"
METRIC_PATH = OUTPUT_REPORT_DIR / "engagement_scoring_metrics.json"
FIG_REGRET = OUTPUT_FIG_DIR / "cumulative_viewer_regret.png"
FIG_SURVIVAL = OUTPUT_FIG_DIR / "fan_favorite_survival.png"

PALETTE = ["#1B4332", "#2D6A4F", "#40916C", "#52796F", "#6A8EAE", "#9D6B94"]
sns.set_theme(style="whitegrid", palette=PALETTE)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/processed/final_estimation.csv")
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df["fan_share"] = df["harmonized_fan_percent"].astype(float)
    df["judge_score"] = df["total_judge_score"].astype(float)
    df = df.sort_values(["season", "celebrity_name", "week"])
    df["prev_judge_score"] = (
        df.groupby(["season", "celebrity_name"], sort=False)["judge_score"].shift(1)
    )
    return df


def rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(ascending=False, method="min")


def simulate_mechanism_a(week_df: pd.DataFrame) -> Tuple[str, float]:
    df = week_df.copy()
    df["judge_rank"] = rank_desc(df["judge_score"])
    df["fan_rank"] = rank_desc(df["fan_share"])
    df["combined"] = df["judge_rank"] + df["fan_rank"]
    bottom_two = df.nlargest(2, "combined")
    # save rule: eliminate the one with lower judge_score; tie -> lower fan_share
    elim_row = bottom_two.sort_values(["judge_score", "fan_share"], ascending=[True, True]).iloc[0]
    return str(elim_row["celebrity_name"]), float(elim_row["fan_share"])


def simulate_mechanism_b(week_df: pd.DataFrame) -> Tuple[str, float]:
    df = week_df.copy()
    # immunity
    immune_idx = df["fan_share"].idxmax()
    non_immune = df[df.index != immune_idx].copy()
    avg_j = non_immune["judge_score"].mean()
    tau = 0.75 * avg_j
    non_immune["fan_eff"] = non_immune.apply(
        lambda r: r["fan_share"] * 0.5 if r["judge_score"] < tau else r["fan_share"], axis=1
    )
    elim_row = non_immune.sort_values(["fan_eff", "judge_score"], ascending=[True, True]).iloc[0]
    return str(elim_row["celebrity_name"]), float(elim_row["fan_share"])


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def calculate_score_v2(
    row: pd.Series,
    weekly_mean_J: float,
    k: float = 0.5,
    lambda_: float = 0.15,
    tau_mult: float = 0.8,
) -> float:
    tau = tau_mult * weekly_mean_J
    W = sigmoid(k * (float(row["judge_score"]) - tau))
    momentum = 0.0
    if pd.notna(row.get("prev_judge_score", float("nan"))):
        momentum = lambda_ * max(0.0, float(row["judge_score"]) - float(row["prev_judge_score"]))
    eff_fan = float(row["fan_share"]) * W
    return eff_fan + momentum


def simulate_mechanism_c(
    week_df: pd.DataFrame,
    k: float = 0.5,
    lambda_: float = 0.15,
    tau_mult: float = 0.8,
) -> Tuple[str, float]:
    df = week_df.copy()
    immune_idx = df["fan_share"].idxmax()
    non_immune = df[df.index != immune_idx].copy()
    weekly_mean_J = non_immune["judge_score"].mean()
    non_immune["total_score_v2"] = non_immune.apply(
        lambda r: calculate_score_v2(r, weekly_mean_J, k=k, lambda_=lambda_, tau_mult=tau_mult), axis=1
    )
    elim_row = non_immune.sort_values(["total_score_v2", "judge_score"], ascending=[True, True]).iloc[0]
    return str(elim_row["celebrity_name"]), float(elim_row["fan_share"])


def calculate_score_v3(
    row: pd.Series,
    weekly_mean_J: float,
    k: float = 0.2,
    lambda_: float = 0.1,
    tau_mult: float = 0.75,
) -> float:
    tau = tau_mult * weekly_mean_J
    j = float(row["judge_score"])
    if j >= tau:
        w_base = 1.0
    else:
        w_base = exp(-k * (tau - j))

    momentum = 0.0
    if pd.notna(row.get("prev_judge_score", float("nan"))):
        delta_j = max(0.0, j - float(row["prev_judge_score"]))
        momentum = lambda_ * delta_j

    w_final = min(1.0, w_base + momentum)
    eff_fan = float(row["fan_share"]) * w_final
    return eff_fan


def simulate_mechanism_d(
    week_df: pd.DataFrame,
    k: float = 0.2,
    lambda_: float = 0.1,
    tau_mult: float = 0.75,
) -> Tuple[str, float]:
    df = week_df.copy()
    immune_idx = df["fan_share"].idxmax()
    non_immune = df[df.index != immune_idx].copy()
    weekly_mean_J = non_immune["judge_score"].mean()
    non_immune["total_score_v3"] = non_immune.apply(
        lambda r: calculate_score_v3(r, weekly_mean_J, k=k, lambda_=lambda_, tau_mult=tau_mult), axis=1
    )
    elim_row = non_immune.sort_values(["total_score_v3", "judge_score"], ascending=[True, True]).iloc[0]
    return str(elim_row["celebrity_name"]), float(elim_row["fan_share"])


def iterate_weeks(df: pd.DataFrame):
    # keep only weeks with elimination recorded or plausible (>=2 contestants)
    grouped = df.groupby(["season", "week"], sort=True)
    weeks = []
    for (season, week), g in grouped:
        if len(g) < 2:
            continue
        if not (g.get("is_eliminated", pd.Series(dtype=bool)).any() or g.get("eliminated_week_num", pd.Series(dtype=float)).notna().any()):
            continue
        weeks.append(((season, week), g))
    return weeks


def compute_survival(favorite_set: set, eliminated_name: str, total_favorites: int, eliminated_fav: int) -> Tuple[int, int]:
    if eliminated_name in favorite_set:
        eliminated_fav += 1
    total_favorites += len(favorite_set)
    return total_favorites, eliminated_fav


def run_simulation(
    df: pd.DataFrame,
    k_v2: float = 0.5,
    lambda_v2: float = 0.15,
    tau_mult_v2: float = 0.8,
    k_v3: float = 0.2,
    lambda_v3: float = 0.1,
    tau_mult_v3: float = 0.75,
) -> Dict[str, Union[int, float, List[float]]]:
    weeks = iterate_weeks(df)
    cum_regret_a: List[float] = []
    cum_regret_b: List[float] = []
    cum_regret_c: List[float] = []
    cum_regret_d: List[float] = []
    total_fav_a = total_fav_b = total_fav_c = 0
    total_fav_d = 0
    elim_fav_a = elim_fav_b = elim_fav_c = 0
    elim_fav_d = 0
    reg_a = reg_b = reg_c = reg_d = 0.0

    for _, g_raw in weeks:
        g = g_raw.reset_index(drop=True)
        fav_cut = g["fan_share"].quantile(0.75)
        favorites = set(g.loc[g["fan_share"] >= fav_cut, "celebrity_name"])

        elim_a, regret_a = simulate_mechanism_a(g)
        elim_b, regret_b = simulate_mechanism_b(g)
        elim_c, regret_c = simulate_mechanism_c(g, k=k_v2, lambda_=lambda_v2, tau_mult=tau_mult_v2)
        elim_d, regret_d = simulate_mechanism_d(g, k=k_v3, lambda_=lambda_v3, tau_mult=tau_mult_v3)

        reg_a += regret_a
        reg_b += regret_b
        reg_c += regret_c
        reg_d += regret_d
        cum_regret_a.append(reg_a)
        cum_regret_b.append(reg_b)
        cum_regret_c.append(reg_c)
        cum_regret_d.append(reg_d)

        total_fav_a, elim_fav_a = compute_survival(favorites, elim_a, total_fav_a, elim_fav_a)
        total_fav_b, elim_fav_b = compute_survival(favorites, elim_b, total_fav_b, elim_fav_b)
        total_fav_c, elim_fav_c = compute_survival(favorites, elim_c, total_fav_c, elim_fav_c)
        total_fav_d, elim_fav_d = compute_survival(favorites, elim_d, total_fav_d, elim_fav_d)

    survival_a = 1.0 - (elim_fav_a / total_fav_a if total_fav_a else 0.0)
    survival_b = 1.0 - (elim_fav_b / total_fav_b if total_fav_b else 0.0)
    survival_c = 1.0 - (elim_fav_c / total_fav_c if total_fav_c else 0.0)
    survival_d = 1.0 - (elim_fav_d / total_fav_d if total_fav_d else 0.0)

    return {
        "weeks": len(weeks),
        "cum_regret_a": reg_a,
        "cum_regret_b": reg_b,
        "cum_regret_c": reg_c,
        "cum_regret_d": reg_d,
        "survival_a": survival_a,
        "survival_b": survival_b,
        "survival_c": survival_c,
        "survival_d": survival_d,
        "cum_regret_series_a": cum_regret_a,
        "cum_regret_series_b": cum_regret_b,
        "cum_regret_series_c": cum_regret_c,
        "cum_regret_series_d": cum_regret_d,
    }


def plot_regret(series_a, series_b, series_c, series_d):
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 4.6))
    x = range(1, len(series_a) + 1)
    plt.plot(x, series_a, label="Baseline", color="#1f77b4", linewidth=1.9)
    plt.plot(x, series_b, label="V1 Hard Gate", color="#ff7f0e", linewidth=2.1)
    plt.plot(x, series_c, label="V2 Soft+Momentum", color="#2ca02c", linewidth=2.0, alpha=0.85)
    plt.plot(x, series_d, label="V3 Safe-Harbor", color="#8c564b", linewidth=2.2)
    plt.xlabel("Cumulative Weeks")
    plt.ylabel("Cumulative Viewer Regret")
    plt.title("Cumulative Viewer Regret Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_REGRET, dpi=260)
    plt.close()


def plot_survival(survival_a: float, survival_b: float, survival_c: float, survival_d: float):
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4))
    labels = ["Baseline", "V1", "V2", "V3"]
    vals = [survival_a, survival_b, survival_c, survival_d]
    sns.barplot(x=labels, y=vals, palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b"])
    plt.ylim(0, 1)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Survival Rate (Top 25% Fan Favorites)")
    plt.title("Fan Favorite Survival Comparison")
    plt.tight_layout()
    plt.savefig(FIG_SURVIVAL, dpi=260)
    plt.close()


def main() -> None:
    df = load_data()
    metrics = run_simulation(df)

    plot_regret(
        metrics["cum_regret_series_a"],
        metrics["cum_regret_series_b"],
        metrics["cum_regret_series_c"],
        metrics["cum_regret_series_d"],
    )
    survival_a = cast(float, metrics["survival_a"])
    survival_b = cast(float, metrics["survival_b"])
    survival_c = cast(float, metrics["survival_c"])
    survival_d = cast(float, metrics["survival_d"])
    plot_survival(survival_a, survival_b, survival_c, survival_d)

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with METRIC_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Simulation complete.")
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("series_a") and not k.endswith("series_b")}, indent=2))


if __name__ == "__main__":
    main()
