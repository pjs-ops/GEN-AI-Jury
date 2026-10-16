# Retail Demand Forecasting & Inventory Decision Support System

An enterprise-grade, comparative retail sales forecasting system built on the Walmart weekly sales dataset. Designed for academic and college jury presentations, this project features rigorous non-leaking feature engineering, evaluation across **eight distinct model families**, multi-step recursive future projections, and inventory replenishment decision support.

---

## 📋 Table of Contents
- [1. Project Architecture](#1-project-architecture)
- [2. Dataset Overview](#2-dataset-overview)
- [3. Installation & Quick Start](#3-installation--quick-start)
- [4. The 8 Forecasting Models](#4-the-8-forecasting-models)
- [5. Evaluation Methodology](#5-evaluation-methodology)
- [6. Empirical Benchmark Results](#6-empirical-benchmark-results)
- [7. Inventory Decision Support](#7-inventory-decision-support)
- [8. Automated Testing & Verification](#8-automated-testing--verification)
- [9. College Jury Presentation Guide](#9-college-jury-presentation-guide)
- [10. Limitations & Future Roadmap](#10-limitations--future-roadmap)

---

## 1. Project Architecture

The codebase follows a modular production architecture where the Jupyter Notebook and the interactive Streamlit Dashboard import from the same shared core engine:

```text
c:\Users\jadha\Desktop\GEN AI\
├── data/                                 # Datasets (fallback search supported)
│   ├── train.csv                         # Historical weekly sales (421,570 rows)
│   ├── features.csv                      # Store-date attributes, markdowns, CPI, fuel price
│   ├── stores.csv                        # Store type and square footage size
│   └── test.csv                          # Unlabeled future competition dates
├── notebooks/
│   └── retail_demand_forecasting.ipynb   # Executable 15-section analytical notebook
├── src/                                  # Shared Core Forecasting Engine
│   ├── __init__.py                       # Package initializer
│   ├── data_loader.py                    # Multi-path search, joins, and diagnostics
│   ├── preprocessing.py                  # Gap handling, returns, and markdown imputation
│   ├── feature_engineering.py            # Non-leaking lags, shifted rolling stats, harmonics
│   ├── forecasting_models.py             # Unified object-oriented interface for 8 models
│   ├── evaluation.py                     # Holdout, rolling CV, metrics, and ranking
│   ├── forecasting.py                    # Multi-step recursive forecasting with 95% bounds
│   ├── inventory.py                      # Reorder point, safety stock, and order quantities
│   └── visualization.py                  # Publication Plotly and Matplotlib charting
├── tests/                                # Automated Test Suites (pytest)
│   ├── test_forecasting.py               # 14 unit and integration tests
│   └── test_evaluation_fixes.py          # 6 focused index and warmup tests
├── app.py                                # 6-page interactive Streamlit dashboard
├── requirements.txt                      # Minimum package dependencies
├── Gen_AI_Jury_1 (1).ipynb               # Preserved original baseline notebook
├── Gen_AI_Jury_1 (1)_backup.ipynb        # Verified backup of original notebook
└── README.md                             # Comprehensive technical documentation
```

---

## 2. Dataset Overview

- **Source:** Walmart Store Sales Dataset.
- **Coverage:** February 5, 2010 to October 26, 2012 (143 weekly observations for Store 1, Department 1).
- **Core Variables:**
  - `Weekly_Sales`: Total monetary sales revenue ($) for the department in the given week.
  - `IsHoliday`: Binary indicator of whether the week contains a major retail holiday (Super Bowl, Labor Day, Thanksgiving, Christmas).
  - `MarkDown1` – `MarkDown5`: Promotional price markdowns (introduced by Walmart in **November 2011**).
  - `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`: Store-regional macroeconomic indicators.
  - `Type`, `Size`: Store classification and physical square footage.

---

## 3. Installation & Quick Start

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### Setup Instructions
1. Open PowerShell or Command Prompt in the repository folder:
   ```powershell
   cd "c:\Users\jadha\Desktop\GEN AI"
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the automated test suite:
   ```powershell
   python -m pytest tests/ -v
   ```

4. Launch the interactive Streamlit Dashboard:
   ```powershell
   streamlit run app.py
   ```
   *The dashboard will automatically open in your browser at `http://localhost:8501`.*

5. Open and run the Jupyter Notebook:
   ```powershell
   jupyter notebook notebooks/retail_demand_forecasting.ipynb
   ```

---

## 4. The 8 Forecasting Models

| # | Model Class | Family | Description |
| :-: | :--- | :---: | :--- |
| 1 | **`NaiveForecaster`** | Baseline | Predicts the most recent observed sales value ($y_t = y_{t-1}$). |
| 2 | **`SeasonalNaiveForecaster`** | Baseline | Predicts sales from exactly 52 weeks prior ($y_t = y_{t-52}$). |
| 3 | **`HoltWintersForecaster`** | Statistical | Triple Exponential Smoothing modeling level, additive trend, and 52-week annual cycles. |
| 4 | **`SARIMAForecaster`** | Statistical | State-space autoregressive integrated moving average ($ARIMA(1, 1, 0)$). |
| 5 | **`LagLinearRegressionForecaster`** | Baseline ML | Multi-lag linear regression using Lags 1–4 (replicates original notebook baseline). |
| 6 | **`ImprovedSeasonalRegressionForecaster`**| ML | Linear model combining Lags 1–4, Lag 52, trigonometric week harmonics, and holiday flags. |
| 7 | **`RandomForestForecaster`** | ML | 100-tree ensemble modeling non-linear interactions across lags, rolling stats, and external drivers. |
| 8 | **`GradientBoostingForecaster`** | ML | Sequentially boosted decision trees optimized to minimize residuals during demand spikes. |

---

## 5. Evaluation Methodology

### Chronological Holdout (80:20 Split)
- Strict chronological order is maintained (no random shuffling).
- The first 80% (72 weeks) form the training set; the final 20% (19 weeks) are reserved for out-of-sample holdout testing.

### Rolling-Origin Cross-Validation (Expanding Window)
- Uses multiple chronological forecast origins across 4 expanding folds.
- In each fold, models train only on past observations, predict the subsequent 4-week horizon ($H = 4$), and step forward.
- Prevents overfitting and measures how reliably a model performs across different business quarters.

### Evaluation Metrics
- **MAE ($)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **RMSE ($)**: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **MAPE (%)**: $\frac{100\%}{N} \sum \frac{|y_i - \hat{y}_i|}{|y_i| + \epsilon}$ (includes zero-division protection)
- **WAPE (%)**: $\frac{\sum |y_i - \hat{y}_i|}{\sum |y_i|} \times 100\%$ (*Primary Selection Metric; weights errors by sales volume to prevent small-denominator distortions*).

---

## 6. Empirical Benchmark Results (Store 1, Dept 1)

All 8 models were trained and evaluated on identical test slices (Store 1, Dept 1):

### Rolling-Origin Cross-Validation (4 Folds $\times$ 4-Week Horizon)
| Rank | Model | Type | Folds | Obs | MAPE (%) | MAE ($) | RMSE ($) | WAPE (%) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Improved Seasonal Regression** | ML | 4 / 4 | 16 | **4.20%** | **$809.83** | **$977.52** | **4.22%** | **Valid (Winner)** |
| **2** | **Random Forest** | ML | 4 / 4 | 16 | 5.24% | $1,055.89 | $1,372.19 | **5.35%** | Valid |
| **3** | **Gradient Boosting** | ML | 4 / 4 | 16 | 6.11% | $1,316.25 | $1,846.30 | **6.38%** | Valid |
| **4** | **Seasonal Naive ($t-52$)** | Baseline | 4 / 4 | 16 | 6.67% | $1,280.76 | $1,488.76 | **6.80%** | Valid |
| **5** | **SARIMA** | Statistical | 4 / 4 | 16 | 11.14% | $2,337.63 | $2,478.72 | **11.28%** | Valid |
| **6** | **Naive (Last Value)** | Baseline | 4 / 4 | 16 | 11.17% | $2,343.41 | $2,484.40 | **11.31%** | Valid |
| **7** | **Holt-Winters Exponential Smoothing** | Statistical | 4 / 4 | 16 | 11.41% | $2,433.12 | $2,610.71 | **11.58%** | Valid |
| **8** | **Lag Linear Regression (Baseline)** | Baseline | 4 / 4 | 16 | 15.61% | $2,766.68 | $2,937.28 | **15.64%** | Valid |

> **Key Finding:** Incorporating annual seasonal recurrence (`Lag_52`) and calendar harmonics reduced forecasting error from **15.64% WAPE** (baseline) down to **4.22% WAPE**—a **>73% error reduction**!

---

## 7. Inventory Decision Support

The inventory module translates demand forecasts into continuous $(s, S)$ or periodic $(R, S)$ replenishment policies:

### Mathematical Equations
1. **Lead-Time Demand**:
   $$D_L = \text{Forecast} \times L$$
2. **Safety Stock**:
   $$SS = Z \times \sigma_D \times \sqrt{L + R}$$
3. **Reorder Point (ROP)**:
   $$ROP = D_L + SS$$
4. **Order-Up-To Target ($S$)**:
   $$S = D_{L+R} + SS$$
5. **Replenishment Order Quantity ($Q$)**:
   $$Q = \max\left(0, S - (\text{On-Hand} + \text{On-Order} - \text{Backorders})\right)$$

### Scenarios
- **Monetary Demand Planning Scenario ($):** Formulates replenishment budgets and safety working capital directly in dollars.
- **Demonstration Unit Scenario (pieces):** Illustrative unit calculation using an assumed Average Selling Price (ASP) to convert sales revenue into synthetic physical unit equivalents.

---

## 8. Automated Testing & Verification

The project includes 2 test suites containing 20 automated tests:

```powershell
python -m pytest tests/ -v
```

### Verified Test Cases:
- `test_data_loading_and_joins`: Ensures dataset merges preserve row counts and schema integrity.
- `test_chronological_sorting_and_keys`: Confirms duplicate keys are absent and dates are strictly monotonic.
- `test_rolling_feature_strictly_prevents_lookahead_leakage`: Proves rolling features are computed on `shift(1)` to eliminate lookahead bias.
- `test_metrics_safe_with_zero_actuals`: Verifies zero-division protection in MAPE and WAPE.
- `test_all_eight_models_holdout_and_ranking`: Confirms all 8 models execute cleanly without errors.
- `test_rolling_origin_cross_validation`: Validates multi-fold rolling validation across origins.
- `test_future_forecast_generation_and_dates`: Verifies future dates step forward at exact 7-day intervals without future-actual leakage.
- `test_inventory_policy_calculations_consistent_units`: Confirms inventory non-negativity and unit consistency.

---

## 9. College Jury Presentation Guide

When presenting this project to evaluators:

1. **Why the original baseline was insufficient:**
   - The original notebook used a 4-lag linear regression that achieved a **15.64% WAPE**. Explain that 4-week lags only see recent noise and completely miss Thanksgiving/Christmas spikes.
2. **Why Improved Seasonal Regression and Random Forest win:**
   - Incorporating `Lag_52` (annual cycle), cyclical sine/cosine week harmonics, and holiday flags drops the error to **4.22% WAPE** ($809 MAE).
   - Random Forest captures non-linear external interactions (temperature, fuel prices, and store markdowns) achieving **5.35% WAPE**.
3. **Preventing Data Leakage:**
   - Point out that rolling averages are shifted by 1 week (`shift(1)`) so the target value of week $t$ is never used to compute the features for week $t$.
   - Point out that future multi-step forecasts are generated recursively (predicted sales at $t+1$ becomes the input lag for $t+2$).
4. **Data Nuances:**
   - Clarify that Walmart `Weekly_Sales` represents dollar revenue, not unit piece counts.
   - Explain that MarkDowns were unrecorded before November 2011.

---

## 10. Limitations & Future Roadmap

- **Granularity:** The dataset provides store-department level weekly revenue. Sub-SKU item-level demand cannot be inferred without barcode-level transactional logs.
- **Festival Transferability:** US holidays (Thanksgiving, Christmas) do not directly capture international festival shocks (e.g., Diwali). In production, a local promotional calendar must be integrated.
- **Future Enhancements:** Deep learning state-space models (e.g., Temporal Fusion Transformers) could be integrated when hundreds of departments are modeled simultaneously.

---

## 👥 Authors & Academic Context
- **Project:** Retail Demand Forecasting & Inventory Optimization System
- **Frameworks:** Python, Scikit-Learn, Statsmodels, Streamlit, Plotly, Pytest
- **Status:** Complete, Verified, and Presentation Ready.
"# GEN-AI-Jury" 
