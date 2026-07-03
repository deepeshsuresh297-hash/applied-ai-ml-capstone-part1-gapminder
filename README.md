# Part 1 — Data Acquisition, Cleaning, and Exploratory Data Analysis

## Project overview

This project analyses a public country-year **Gapminder** dataset containing development indicators for 142 countries from 1952 to 2007. The analytical target is **`lifeExp`** (life expectancy in years). This is a suitable dataset for the capstone because it has 1,704 rows, eight columns, several numeric variables, and repeated categorical variables.

The raw CSV is committed at `data/raw/gapminder_raw.csv`, so no external download is needed to run the analysis. The source values are the public Gapminder country-year indicators distributed in Plotly Express's sample dataset.

### Variables

| Column | Description | Initial dtype |
|---|---|---|
| `country` | Country name | object |
| `continent` | Continent classification | object |
| `year` | Observation year | int64 |
| `lifeExp` | Life expectancy in years; chosen numeric target | float64 |
| `pop` | Country population | int64 |
| `gdpPercap` | GDP per capita | float64 |
| `iso_alpha` | Three-letter country identifier | object |
| `iso_num` | Numeric country identifier | int64 |

## Repository contents

```text
part1_gapminder/
├── data/
│   └── raw/
│       └── gapminder_raw.csv
├── outputs/
│   ├── 01_line_mean_life_expectancy_by_year.png
│   ├── 02_bar_mean_life_expectancy_by_continent.png
│   ├── 03_histogram_most_skewed_variable.png
│   ├── 04_scatter_gdp_per_capita_vs_life_expectancy.png
│   ├── 05_boxplot_life_expectancy_by_continent.png
│   └── 06_pearson_correlation_heatmap.png
├── cleaned_data.csv
├── part1_eda.py
├── requirements.txt
└── README.md
```

## Installation and execution

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run the analysis from this folder:

```bash
pip install -r requirements.txt
python part1_eda.py
```

The script runs from top to bottom, prints all required tables and results to the terminal, writes the six plot images to `outputs/`, and creates `cleaned_data.csv` in the folder root.

## Data inspection and cleaning

### 1. Initial inspection

The CSV loads into a pandas DataFrame with **1,704 rows and 8 columns**. The script prints the first five records, inferred data types, and DataFrame shape.

### 2. Null-value analysis and median strategy

The null-count and null-percentage table showed **zero missing values in every column**. Therefore, no columns exceed the 20% null threshold and no observed values required replacement.

The code still applies the required median-imputation logic to every eligible numeric column using `fillna(df[column].median())`. This makes the pipeline ready for a future client extract that contains a small amount of missing numeric data. Median is preferred over mean for skewed variables because it is less affected by extreme observations. This is especially important for `pop` and `gdpPercap`, which are strongly right-skewed.

### 3. Duplicate detection and removal

The dataset contains **0 exact duplicate rows**, so `drop_duplicates()` removes **0 rows**. Since no rows were removed, the null percentage of every column remained unchanged.

### 4. Data-type correction and memory usage

`iso_num` was initially inferred as `int64`; however, it is a nominal country identifier, not a numeric measurement. It was therefore converted to `category`. The repeated string fields `country`, `continent`, and `iso_alpha` were also converted from `object` to `category`.

| Measure | Value |
|---|---:|
| Memory before conversion | 348,180 bytes |
| Memory after conversion | 96,144 bytes |
| Reduction | 252,036 bytes (72.39%) |

Using categories saves memory while also preventing an identifier from being interpreted as an ordinal or continuous quantity.

## Descriptive statistics and skewness

The script prints `describe()` for all numeric variables and calculates skewness for each one.

| Numeric column | Skewness | Interpretation |
|---|---:|---|
| `pop` | 8.3402 | Extremely positively skewed |
| `gdpPercap` | 3.8503 | Strongly positively skewed |
| `lifeExp` | -0.2527 | Slight negative skew |
| `year` | 0.0000 | Symmetric because each year has the same number of country records |

The highest absolute skewness belongs to **`pop`**. Its strong positive skew means that a small number of very populous countries pull the mean far above the median. This is why the median is more representative than the mean when imputing a missing population value. Positive skew means a long right tail; negative skew would instead mean a long left tail, where unusually small observations pull the mean downward.

## IQR outlier analysis

IQR outlier detection was performed for `pop` and `gdpPercap`.

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | IQR-flagged rows |
|---|---:|---:|---:|---:|---:|---:|
| `pop` | 2,793,664 | 19,585,216 | 16,791,552 | -22,393,668 | 44,772,548 | 208 |
| `gdpPercap` | 1,202.06 | 9,325.46 | 8,123.40 | -10,983.04 | 21,510.57 | 143 |

I retained these observations. Countries with very high population or GDP per capita are plausible, informative real-world cases rather than automatically invalid entries. Removing them would discard meaningful global variation and may bias later modelling. In Part 2, I will consider a log transformation for `pop` and `gdpPercap`, along with models that are less sensitive to skewed distributions, instead of capping or deleting valid countries.

## Visualisations

All figures are created by `part1_eda.py` and saved with `plt.savefig()`.

1. **Line plot — `01_line_mean_life_expectancy_by_year.png`**: Mean global life expectancy rises steadily over time. This describes a broad improvement across the observed country-year records, although it does not prove that time itself caused the increase.
2. **Bar chart — `02_bar_mean_life_expectancy_by_continent.png`**: Oceania has the highest mean life expectancy (74.33 years), followed by Europe (71.90 years). Africa has the lowest mean (48.87 years).
3. **Histogram — `03_histogram_most_skewed_variable.png`**: `pop` has a highly right-skewed distribution. Most observations are relatively small compared with a few extremely populous countries.
4. **Scatter plot — `04_scatter_gdp_per_capita_vs_life_expectancy.png`**: The plot shows a positive relationship: higher GDP per capita generally corresponds to higher life expectancy. The points are not arranged on a straight line, and gains in life expectancy appear to flatten at high GDP values.
5. **Box plot — `05_boxplot_life_expectancy_by_continent.png`**: Life expectancy differs visibly across continents. Oceania and Europe have high medians, Africa has the lowest distribution, and Asia shows a broad spread of country outcomes.
6. **Correlation heatmap — `06_pearson_correlation_heatmap.png`**: The heatmap summarises linear relationships among numeric variables.

## Pearson correlation heatmap

The strongest non-diagonal Pearson relationship is between **`lifeExp`** and **`gdpPercap`** at **r = 0.5837**, a moderate positive linear association.

This relationship must not be interpreted as proof that GDP per capita directly causes life expectancy. A plausible alternative explanation is that factors such as healthcare access, education, sanitation, nutrition, political stability, and historical regional differences affect both national income and population health. GDP per capita may also act as a proxy for several of these underlying factors.

## Imputation comparison for the two most skewed columns

The following values were calculated from an untouched copy of the raw data before the imputation step.

| Column | Mean | Median | Chosen statistic | Reason |
|---|---:|---:|---|---|
| `pop` | 29,601,210 | 7,023,596 | Median | Extreme high-population countries pull the mean upward. |
| `gdpPercap` | 7,215.33 | 3,531.85 | Median | Extreme high-income observations pull the mean upward. |

Both variables are positively skewed, so median is the more representative central measure. The script uses `fillna()` with the selected statistic and confirms with `isnull().sum()` that `pop` and `gdpPercap` have zero remaining nulls. In this dataset they began with zero nulls, so no real source value was changed.

## Spearman rank correlation

Spearman correlation was calculated for all numeric columns and compared with Pearson correlation. The three largest absolute differences are shown below.

| Pair | Pearson | Spearman | Absolute difference | Interpretation |
|---|---:|---:|---:|---|
| `lifeExp` and `gdpPercap` | 0.5837 | 0.8265 | 0.2428 | Strong monotonic but non-linear relationship; life-expectancy gains appear to level off as GDP rises. |
| `year` and `pop` | 0.0823 | 0.2198 | 0.1375 | Weak positive rank pattern that is not well represented by a straight-line relationship. |
| `lifeExp` and `pop` | 0.0650 | 0.1806 | 0.1157 | Weak monotonic association, but population alone does not have a useful proportional linear relationship with life expectancy. |

For Part 2, I will use **Spearman correlation as the primary exploratory feature-screening guide for heavily skewed numeric variables**, because it is rank-based and less distorted by extreme values or non-linear monotonic patterns. I will not rely on correlation alone: model validation and domain reasoning will determine the final feature set. For any linear model, I will also consider log-transforming `pop` and `gdpPercap` and then re-checking linear relationships.

## Grouped aggregation

The grouped aggregation uses `continent` and `lifeExp`.

| Continent | Mean life expectancy | Standard deviation | Count |
|---|---:|---:|---:|
| Oceania | 74.33 | 3.80 | 24 |
| Europe | 71.90 | 5.43 | 360 |
| Americas | 64.66 | 9.35 | 300 |
| Asia | 60.06 | 11.86 | 396 |
| Africa | 48.87 | 9.15 | 624 |

- **Highest mean:** Oceania (74.33 years)
- **Highest standard deviation:** Asia (11.86 years)
- **Ratio of highest to lowest group mean:** 74.33 / 48.87 = **1.521**

The 1.521 ratio indicates that continent carries meaningful predictive signal, so it should be tested as a categorical feature in Part 2. However, Asia's high within-group standard deviation shows that continent alone is insufficient for accurate predictions: countries in the same continent can have very different outcomes. Country, year, GDP per capita, and potentially transformed population should therefore be considered together.

## Final output

`cleaned_data.csv` is created with 1,704 rows and 8 columns. It contains zero missing values and is ready to be used in Parts 2 and 3.
