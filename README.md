# 🛒 Walmart Weekly Sales Forecasting Using ML & Time Series

> **End-to-end forecasting system** across 45 stores and 143 weeks — combining causal inference, machine learning, and time series modeling to drive automated inventory and staffing decisions.

---

## 📌 Project Overview

Retail forecasting is hard. Demand fluctuates with holidays, markdowns, economic shifts, and store-specific factors that simple trend models can't capture. This project builds a rigorous, production-style forecasting pipeline on Walmart's weekly sales data — from causal driver validation all the way to out-of-sample deployment.

**The result:** Random Forest (OOS RMSE ≈ $120K) and SARIMAX (OOS RMSE ≈ $84K) — both within ~8% of the $1.05M average weekly sales, validated through proper held-out testing.

---

## 🎯 Business Problem

Walmart store managers need reliable weekly sales forecasts to:
- Optimize **inventory levels** and reduce overstock/stockout costs
- Plan **staffing schedules** ahead of demand peaks
- Anticipate the impact of **holidays, markdowns, and macro conditions** on store performance

---

## 🗂️ Dataset

| Feature | Detail |
|---|---|
| Source | Walmart Store Sales (Kaggle) |
| Observations | 6,435 weekly records |
| Stores | 45 stores |
| Time Span | 143 weeks |
| Key Variables | Weekly Sales, Store, Date, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment, Markdown 1–5 |

---

## 🔬 Methodology

### 1. 🧹 Data Cleaning & EDA
- Handled missing markdowns, parsed dates, engineered time features (week, month, year, holiday proximity)
- Explored sales distributions across stores, holiday vs. non-holiday weeks, and seasonal patterns

### 2. 📐 Causal Driver Validation
- **Propensity Score Matching (PSM):** Validated the causal effect of holiday weeks on sales by constructing matched control groups — ruling out confounding from store size or location
- **Lasso Regression:** Used L1 regularization to identify and confirm the most predictive demand drivers, reducing feature noise before model training

### 3. 🤖 Model Benchmarking (5 Models, 10-Fold CV)
All models were rigorously evaluated using **10-fold cross-validation** before out-of-sample testing:

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | Feature selection |
| **Random Forest** ✅ | Best ML model |
| **SARIMAX** ✅ | Best time series model |

### 4. 📈 Final Model Performance (Out-of-Sample)

| Model | OOS RMSE | % of Avg Weekly Sales |
|---|---|---|
| Random Forest | ~$120,000 | ~11.4% |
| SARIMAX | ~$84,000 | ~8.0% |

> SARIMAX captures seasonality and temporal autocorrelation, making it the strongest performer for store-level weekly forecasting.

---

## 📁 Repository Structure

```
├── Data_files/          # Raw and processed datasets
├── Docs/                # Project documentation and reports
├── Pytorch_code/        # Model training notebooks and scripts
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Languages | Python, R |
| ML Libraries | Scikit-learn, XGBoost, Statsmodels |
| Time Series | SARIMAX, seasonal decomposition |
| Causal Inference | PSM, Lasso feature selection |
| Validation | 10-Fold Cross Validation, OOS holdout testing |
| Visualization | Matplotlib, Seaborn |

---

## 💡 Key Takeaways

- **PSM and Lasso aren't just academic exercises** — they materially improved model performance by ensuring only validated, causal features entered the pipeline
- **SARIMAX outperformed Random Forest** on OOS RMSE despite simpler structure — temporal autocorrelation in weekly retail data is too strong to ignore
- **10-fold CV prevented overfitting** and gave reliable model selection before the final holdout test
- The pipeline is designed to be **reusable**: swap in new store data and re-run forecasts with minimal changes

---

## 👤 Author

**Salman Khan Shafi**
MS Business Analytics — Duke Fuqua '26
[LinkedIn](https://linkedin.com/in/salmankhanshafi) • [GitHub](https://github.com/salmanshafi9898)
