"""Анализ расхода топлива, проверка гипотез и оценка распределений."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

TARGET = "FUELCONSUMPTION_COMB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Статистический анализ комбинированного расхода топлива."
    )
    parser.add_argument("csv", type=Path, help="Путь к CSV-файлу Fuel Consumption")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Каталог результатов (по умолчанию: results)",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = (
        frame.columns.str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "_", regex=False)
    )
    aliases = {
        "FUELCONSUMPTIONCOMB": TARGET,
        "FUEL_CONSUMPTION_COMB": TARGET,
        "ENGINE_SIZE": "ENGINESIZE",
        "FUEL_TYPE": "FUELTYPE",
        "VEHICLE_CLASS": "VEHICLECLASS",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
    if TARGET not in frame.columns:
        raise ValueError(f"В CSV не найден обязательный столбец {TARGET}")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame.dropna(subset=[TARGET]).copy()


def descriptive_statistics(values: pd.Series) -> dict[str, float | int]:
    mode = values.mode()
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return {
        "count": int(values.count()),
        "missing": int(values.isna().sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": mean,
        "median": float(values.median()),
        "mode": float(mode.iloc[0]) if not mode.empty else float("nan"),
        "variance": float(values.var(ddof=1)),
        "standard_deviation": std,
        "coefficient_of_variation_percent": 100 * std / mean,
        "skewness": float(values.skew()),
        "excess_kurtosis": float(values.kurt()),
    }


def fit_distributions(values: np.ndarray) -> dict[str, dict[str, object]]:
    candidates = {
        "normal": stats.norm,
        "gamma": stats.gamma,
        "exponential": stats.expon,
        "lognormal": stats.lognorm,
    }
    results: dict[str, dict[str, object]] = {}
    for name, distribution in candidates.items():
        params = (
            distribution.fit(values, floc=0)
            if name in {"gamma", "lognormal"}
            else distribution.fit(values)
        )
        log_likelihood = float(np.sum(distribution.logpdf(values, *params)))
        results[name] = {
            "parameters": [float(value) for value in params],
            "aic": 2 * len(params) - 2 * log_likelihood,
        }
    return results


def confidence_intervals(values: np.ndarray) -> dict[str, list[float]]:
    n = len(values)
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    std = float(np.sqrt(variance))
    t_value = stats.t.ppf(0.975, n - 1)
    chi_low = stats.chi2.ppf(0.975, n - 1)
    chi_high = stats.chi2.ppf(0.025, n - 1)
    return {
        "mean_95_percent": [
            mean - t_value * std / np.sqrt(n),
            mean + t_value * std / np.sqrt(n),
        ],
        "variance_95_percent": [
            (n - 1) * variance / chi_low,
            (n - 1) * variance / chi_high,
        ],
    }


def regression_summary(frame: pd.DataFrame) -> dict[str, object] | None:
    predictors = [name for name in ("ENGINESIZE", "CYLINDERS") if name in frame]
    if len(predictors) != 2:
        return None
    clean = frame[predictors + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    model = LinearRegression().fit(clean[predictors], clean[TARGET])
    prediction = model.predict(clean[predictors])
    return {
        "predictors": predictors,
        "coefficients": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "r_squared": float(r2_score(clean[TARGET], prediction)),
    }


def hypothesis_tests(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    if "FUELTYPE" in frame:
        groups = {
            str(name): group[TARGET].dropna().to_numpy()
            for name, group in frame.groupby("FUELTYPE")
            if len(group) >= 2
        }
        if "Z" in groups and "D" in groups:
            statistic, p_value = stats.ttest_ind(groups["Z"], groups["D"], equal_var=False)
            result["welch_t_test_gasoline_Z_vs_diesel_D"] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
    if "VEHICLECLASS" in frame:
        labels = frame["VEHICLECLASS"].astype(str)
        compact = frame.loc[labels.str.contains("COMPACT", case=False, na=False), TARGET]
        suv = frame.loc[labels.str.contains("SUV", case=False, na=False), TARGET]
        if len(compact) >= 2 and len(suv) >= 2:
            statistic, p_value = stats.mannwhitneyu(compact, suv, alternative="two-sided")
            result["mann_whitney_compact_vs_suv"] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
    return result


def save_plots(values: np.ndarray, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.histplot(values, bins="sturges", stat="density", kde=True, ax=axes[0], color="#2878B5")
    axes[0].set(title="Распределение расхода топлива", xlabel="л/100 км", ylabel="Плотность")

    sorted_values = np.sort(values)
    ecdf = np.arange(1, len(values) + 1) / len(values)
    axes[1].step(sorted_values, ecdf, where="post", color="#2E8B57")
    axes[1].set(title="Эмпирическая функция распределения", xlabel="л/100 км", ylabel="F(x)")

    gamma_params = stats.gamma.fit(values, floc=0)
    stats.probplot(values, dist=stats.gamma, sparams=gamma_params, plot=axes[2])
    axes[2].set_title("Q-Q график гамма-распределения")

    figure.tight_layout()
    figure.savefig(output / "distribution_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_data(args.csv)
    values = frame[TARGET].to_numpy(dtype=float)

    shapiro_values = values if len(values) <= 5_000 else values[:5_000]
    shapiro = stats.shapiro(shapiro_values)
    summary = {
        "target": TARGET,
        "descriptive_statistics": descriptive_statistics(frame[TARGET]),
        "confidence_intervals": confidence_intervals(values),
        "distribution_fits": fit_distributions(values),
        "shapiro_wilk": {
            "statistic": float(shapiro.statistic),
            "p_value": float(shapiro.pvalue),
        },
        "ols_regression": regression_summary(frame),
        "hypothesis_tests": hypothesis_tests(frame),
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary["descriptive_statistics"]]).to_csv(
        args.output / "descriptive_statistics.csv", index=False
    )
    save_plots(values, args.output)
    print(f"Готово. Результаты сохранены в {args.output.resolve()}")


if __name__ == "__main__":
    main()
