"""Part 1: Data acquisition, cleaning, and exploratory data analysis.

Dataset: Gapminder country-year indicators (public data distributed with Plotly Express).
The script loads the committed CSV, completes all Part 1 tasks, saves figures, and writes
cleaned_data.csv for later capstone parts.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ------------------------------- Configuration -------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "gapminder_raw.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CLEAN_DATA_PATH = PROJECT_DIR / "cleaned_data.csv"

TARGET_COLUMN = "lifeExp"
CATEGORICAL_COLUMNS = ["country", "continent", "iso_alpha", "iso_num"]
IQR_COLUMNS = ["pop", "gdpPercap"]
GROUP_CATEGORICAL_COLUMN = "continent"
GROUP_NUMERIC_COLUMN = "lifeExp"


# ------------------------------- Helper functions -----------------------------
def save_current_figure(filename: str) -> None:
    """Save the active Matplotlib figure consistently and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def make_null_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the required null count and null percentage table."""
    null_count = frame.isnull().sum()
    null_percentage = (frame.isnull().sum() / frame.shape[0]) * 100
    return pd.DataFrame(
        {
            "null_count": null_count,
            "null_percentage": null_percentage.round(2),
        }
    )


def strongest_correlation_pair(correlation_matrix: pd.DataFrame) -> tuple[str, str, float]:
    """Find the non-diagonal Pearson pair with the highest absolute correlation."""
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    stacked = upper_triangle.stack()
    pair = stacked.abs().idxmax()
    return pair[0], pair[1], float(correlation_matrix.loc[pair[0], pair[1]])


def make_spearman_difference_table(
    pearson_matrix: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Create a one-row-per-pair table sorted by |Spearman - Pearson|."""
    rows: list[dict[str, float | str]] = []
    for column_a, column_b in combinations(pearson_matrix.columns, 2):
        pearson_value = float(pearson_matrix.loc[column_a, column_b])
        spearman_value = float(spearman_matrix.loc[column_a, column_b])
        rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "pearson": pearson_value,
                "spearman": spearman_value,
                "abs_difference": abs(spearman_value - pearson_value),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_difference", ascending=False).reset_index(drop=True)


# -------------------------------- Main analysis --------------------------------
def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    sns.set_theme(context="notebook")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Task 1: load the CSV and inspect it.
    df = pd.read_csv(RAW_DATA_PATH)
    print("\n" + "=" * 88)
    print("TASK 1 — DATASET PREVIEW")
    print("=" * 88)
    print("First five rows:\n", df.head())
    print("\nInferred data types:\n", df.dtypes)
    print("\nDataFrame shape:", df.shape)

    # Keep an untouched copy so Task 9 can compare mean and median before any imputation.
    df_before_any_imputation = df.copy(deep=True)

    # Task 2: null value analysis and median fill for numeric columns below 20% nulls.
    print("\n" + "=" * 88)
    print("TASK 2 — NULL VALUE ANALYSIS")
    print("=" * 88)
    null_summary_before = make_null_summary(df)
    print("Null count and percentage by column:\n", null_summary_before)

    columns_over_twenty_percent = null_summary_before.index[
        null_summary_before["null_percentage"] > 20
    ].tolist()
    print("\nColumns exceeding a 20% null rate:", columns_over_twenty_percent or "None")

    numeric_columns_before_dtype_correction = df.select_dtypes(include="number").columns.tolist()
    numeric_columns_below_twenty = [
        column
        for column in numeric_columns_before_dtype_correction
        if null_summary_before.loc[column, "null_percentage"] < 20
    ]

    median_fill_report: list[dict[str, float | int | str]] = []
    for column in numeric_columns_below_twenty:
        missing_before = int(df[column].isnull().sum())
        # Required median-imputation form for numeric columns below the 20% threshold.
        df[column] = df[column].fillna(df[column].median())
        median_fill_report.append(
            {
                "column": column,
                "missing_values_filled": missing_before,
                "median_used": float(df[column].median()),
            }
        )
    print("\nMedian-imputation report for eligible numeric columns:\n", pd.DataFrame(median_fill_report))

    # Task 3: duplicate detection and removal.
    print("\n" + "=" * 88)
    print("TASK 3 — DUPLICATE DETECTION AND REMOVAL")
    print("=" * 88)
    duplicates_before = int(df.duplicated().sum())
    print("Duplicate rows before removal:", duplicates_before)
    null_summary_before_deduplication = make_null_summary(df)
    rows_before_deduplication = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    rows_removed = rows_before_deduplication - df.shape[0]
    print("Rows removed:", rows_removed)

    null_summary_after_deduplication = make_null_summary(df)
    null_rate_comparison = null_summary_before_deduplication[["null_percentage"]].rename(
        columns={"null_percentage": "before_duplicate_removal"}
    ).join(
        null_summary_after_deduplication[["null_percentage"]].rename(
            columns={"null_percentage": "after_duplicate_removal"}
        )
    )
    null_rate_comparison["changed"] = (
        null_rate_comparison["before_duplicate_removal"]
        != null_rate_comparison["after_duplicate_removal"]
    )
    print("\nNull-percentage comparison before and after duplicate removal:\n", null_rate_comparison)
    print(
        "Any null percentage changed after duplicate removal:",
        bool(null_rate_comparison["changed"].any()),
    )

    # Task 4: correct semantically inappropriate inferred types and report memory usage.
    print("\n" + "=" * 88)
    print("TASK 4 — DATA TYPE CORRECTION AND MEMORY USAGE")
    print("=" * 88)
    memory_before = int(df.memory_usage(deep=True).sum())
    print("Data types before correction:\n", df.dtypes)
    print("Memory usage before correction (bytes):", memory_before)

    # iso_num is an identifier, not a quantity. Its integer dtype is semantically misleading.
    # country, continent, and iso_alpha are repeated text labels and are stored efficiently as categories.
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("category")

    memory_after = int(df.memory_usage(deep=True).sum())
    print("\nData types after correction:\n", df.dtypes)
    print("Memory usage after correction (bytes):", memory_after)
    print("Memory change (bytes):", memory_after - memory_before)
    print("Memory reduction (%):", round((memory_before - memory_after) / memory_before * 100, 2))

    # Identify purely numeric columns after dtype correction.
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    numeric_df = df[numeric_columns]

    # Task 5: descriptive statistics and skewness.
    print("\n" + "=" * 88)
    print("TASK 5 — DESCRIPTIVE STATISTICS AND SKEWNESS")
    print("=" * 88)
    print("Descriptive statistics for numeric columns:\n", numeric_df.describe())
    skewness = numeric_df.apply(lambda series: series.skew()).sort_values(key=lambda series: series.abs(), ascending=False)
    print("\nSkewness by numeric column, ranked by absolute magnitude:\n", skewness)
    most_skewed_column = str(skewness.index[0])
    print("\nColumn with highest absolute skewness:", most_skewed_column)
    print("Skewness value:", round(float(skewness.iloc[0]), 6))

    # Task 6: IQR outlier detection for two numeric columns.
    print("\n" + "=" * 88)
    print("TASK 6 — IQR OUTLIER DETECTION")
    print("=" * 88)
    iqr_rows: list[dict[str, float | int | str]] = []
    for column in IQR_COLUMNS:
        q1 = float(df[column].quantile(0.25))
        q3 = float(df[column].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((df[column] < lower_bound) | (df[column] > upper_bound)).sum())
        iqr_rows.append(
            {
                "column": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
            }
        )
    iqr_results = pd.DataFrame(iqr_rows)
    print(iqr_results.to_string(index=False))
    print(
        "\nDecision: retain all IQR-flagged rows. Large populations and high GDP-per-capita values "
        "are valid country observations rather than data-entry errors. Part 2 can consider logarithmic "
        "transforms or robust models instead of deleting countries."
    )

    # Task 7: all five required visualizations.
    print("\n" + "=" * 88)
    print("TASK 7 — VISUALIZATIONS")
    print("=" * 88)

    # 7a: line plot of mean life expectancy sorted by the time column.
    mean_life_expectancy_by_year = df.groupby("year", observed=True)[TARGET_COLUMN].mean().sort_index()
    plt.figure(figsize=(9, 5))
    plt.plot(mean_life_expectancy_by_year.index, mean_life_expectancy_by_year.values, marker="o")
    plt.title("Mean Life Expectancy by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean life expectancy (years)")
    save_current_figure("01_line_mean_life_expectancy_by_year.png")

    # 7b: bar chart of mean target by continent.
    mean_life_expectancy_by_continent = (
        df.groupby("continent", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 5))
    plt.bar(mean_life_expectancy_by_continent.index.astype(str), mean_life_expectancy_by_continent.values)
    plt.title("Mean Life Expectancy by Continent")
    plt.xlabel("Continent")
    plt.ylabel("Mean life expectancy (years)")
    plt.xticks(rotation=25)
    save_current_figure("02_bar_mean_life_expectancy_by_continent.png")

    # 7c: histogram of the most skewed numeric column.
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x=most_skewed_column, bins=20, color="C0")
    plt.title(f"Distribution of Most Skewed Variable: {most_skewed_column}")
    plt.xlabel(most_skewed_column)
    plt.ylabel("Count")
    save_current_figure("03_histogram_most_skewed_variable.png")

    # 7d: scatter plot of two expected-to-be-correlated numeric columns.
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="gdpPercap", y=TARGET_COLUMN)
    plt.title("GDP per Capita vs Life Expectancy")
    plt.xlabel("GDP per capita")
    plt.ylabel("Life expectancy (years)")
    save_current_figure("04_scatter_gdp_per_capita_vs_life_expectancy.png")

    # 7e: box plot of target split by a categorical variable.
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="continent", y=TARGET_COLUMN)
    plt.title("Life Expectancy Distribution by Continent")
    plt.xlabel("Continent")
    plt.ylabel("Life expectancy (years)")
    plt.xticks(rotation=25)
    save_current_figure("05_boxplot_life_expectancy_by_continent.png")
    print("Saved five required plots in:", OUTPUT_DIR)

    # Task 8: Pearson correlation matrix and heatmap.
    print("\n" + "=" * 88)
    print("TASK 8 — PEARSON CORRELATION HEATMAP")
    print("=" * 88)
    pearson_corr = numeric_df.corr()
    print("Pearson correlation matrix:\n", pearson_corr)
    corr_a, corr_b, corr_value = strongest_correlation_pair(pearson_corr)
    print(
        f"\nHighest absolute non-diagonal Pearson correlation: {corr_a} and {corr_b} "
        f"(r = {corr_value:.4f})"
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", square=True)
    plt.title("Pearson Correlation Heatmap for Numeric Variables")
    save_current_figure("06_pearson_correlation_heatmap.png")

    # Task 9a: compare mean and median for the two most skewed numerical columns.
    print("\n" + "=" * 88)
    print("TASK 9A — IMPUTATION STRATEGY COMPARISON")
    print("=" * 88)
    two_most_skewed_columns = skewness.abs().nlargest(2).index.tolist()
    imputation_rows: list[dict[str, float | int | str]] = []
    for column in two_most_skewed_columns:
        column_mean = float(df_before_any_imputation[column].mean())
        column_median = float(df_before_any_imputation[column].median())
        direction = "positive" if skewness.loc[column] > 0 else "negative"
        chosen_statistic = "median" if skewness.loc[column] != 0 else "mean"
        value_for_fill = column_median if chosen_statistic == "median" else column_mean
        missing_before = int(df_before_any_imputation[column].isnull().sum())
        # Apply the selected strategy even when this source happens to have no remaining nulls.
        df[column] = df[column].fillna(value_for_fill)
        missing_after = int(df[column].isnull().sum())
        imputation_rows.append(
            {
                "column": column,
                "skewness": float(skewness.loc[column]),
                "skew_direction": direction,
                "mean_before_imputation": column_mean,
                "median_before_imputation": column_median,
                "chosen_statistic": chosen_statistic,
                "nulls_before": missing_before,
                "nulls_after": missing_after,
            }
        )
    imputation_comparison = pd.DataFrame(imputation_rows)
    print(imputation_comparison.to_string(index=False))
    print(
        "\nNull check after selected imputation:\n",
        df[two_most_skewed_columns].isnull().sum(),
    )

    # Task 9b: Spearman matrix and the three largest Pearson/Spearman differences.
    print("\n" + "=" * 88)
    print("TASK 9B — SPEARMAN RANK CORRELATION")
    print("=" * 88)
    spearman_corr = numeric_df.corr(method="spearman")
    spearman_difference_table = make_spearman_difference_table(pearson_corr, spearman_corr)
    top_three_spearman_differences = spearman_difference_table.head(3)
    print("Pearson correlation matrix:\n", pearson_corr)
    print("\nSpearman correlation matrix:\n", spearman_corr)
    print(
        "\nThree column pairs with the largest |Spearman - Pearson| difference:\n",
        top_three_spearman_differences.to_string(index=False),
    )

    # Task 9c: grouped aggregation.
    print("\n" + "=" * 88)
    print("TASK 9C — GROUPED AGGREGATION")
    print("=" * 88)
    grouped_aggregation = (
        df.groupby(GROUP_CATEGORICAL_COLUMN, observed=True)[GROUP_NUMERIC_COLUMN]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print(grouped_aggregation)
    highest_mean_group = str(grouped_aggregation["mean"].idxmax())
    highest_std_group = str(grouped_aggregation["std"].idxmax())
    highest_mean = float(grouped_aggregation["mean"].max())
    lowest_mean = float(grouped_aggregation["mean"].min())
    mean_ratio = highest_mean / lowest_mean
    print("\nGroup with highest mean:", highest_mean_group)
    print("Group with highest standard deviation:", highest_std_group)
    print("Highest group mean / lowest group mean:", round(mean_ratio, 4))

    # Task 10: save the clean dataset.
    print("\n" + "=" * 88)
    print("TASK 10 — SAVE CLEAN DATASET")
    print("=" * 88)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("Saved clean dataset to:", CLEAN_DATA_PATH)
    print("Final shape:", df.shape)
    print("Final total null values:", int(df.isnull().sum().sum()))


if __name__ == "__main__":
    main()
