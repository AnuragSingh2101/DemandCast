# DemandCast - Interview Cheat Sheet (2 Pages)

---

## 🎯 Project One-Liner
> **"End-to-end demand forecasting platform: from raw supply-chain transactions to model comparison with statistical significance testing, deployed as a Streamlit dashboard."**

---

## 📊 Problem Statement
**Business Context**: Retail/wholesale supply chain needs to predict weekly demand by product category & market to plan inventory.

**Three Questions**:
1. **Descriptive**: What happened to historical demand? → EDA, Pareto, seasonality
2. **Predictive**: What will demand do next? → Baselines, ETS, XGBoost with backtesting
3. **Prescriptive**: What actions follow? → Dashboard for planners (forecast + intervals)

---

## 📦 Data (Evidence-Based)
| Aspect | Detail |
|--------|--------|
| **Source** | DataCo SMART SUPPLY CHAIN (Mendeley, CC BY 4.0) |
| **Rows** | 180,519 transaction lines |
| **Date Range** | 2015-01 to 2018-01 (3 years, 162 weeks) |
| **Target** | Weekly `order_item_quantity` (W-MON) |
| **Categories** | 51 (7 drive 80% of demand, Gini=0.80) |
| **Markets** | 5 (Africa, Europe, LATAM, Pacific Asia, USCA) |
| **Key Limitation** | 2014 & 2018 are partial years; no promotions/holidays |

**PII Dropped**: Customer email, name, address, zipcode, passwords, product names/URLs

---

## 🔧 Pipeline (No Leakage)
```
Raw CSV → Cleaning → Validation → DuckDB Aggregation → Features → Rolling-Origin Backtest
```
- **Chronological splits only** (no random splits)
- **Lag features**: 1,2,4,8,12,26,52 weeks (shifted, no future data)
- **Rolling stats**: 4,8,12,26 weeks (window ends at t-1)
- **Leakage tests**: Perturb future → assert predictions unchanged

---

## 📈 Models Tested

| Model | Type | Avg MAE | Notes |
|-------|------|---------|-------|
| **Moving Avg (4)** | Baseline | **40.1** ⭐ | Winner - simple, robust |
| Naive | Baseline | 45.4 | Benchmark |
| ETS (Auto) | Statistical | 45.4 | Fell back to naive (no seasonality) |
| Seasonal Naive | Baseline | 89.9 | Poor - weak annual seasonality |
| XGBoost | ML | N/A | Disabled: feature mismatch |

**Honest Result**: Simple moving average beats everything. No complex model won.

---

## 📏 Evaluation (Rigorous)
| Metric | Purpose |
|--------|---------|
| **MAE** | Primary metric (interpretable, handles zeros) |
| **RMSE** | Penalizes large errors |
| **WAPE** | Weighted %, scale-invariant |
| **sMAPE** | Symmetric %, bounded |
| **MASE** | Scaled by in-sample naive error (comparable across series) |
| **Diebold-Mariano** | Statistical significance of model differences |
| **Rolling-Origin** | 4-week horizon, 4-week step, 52-week min train |

**Key Finding**: Moving Avg (4) significantly beats ETS on 2/5 categories (p<0.05). On 3 categories, no significant difference.

---

## 🎨 Dashboard (3 Pages)
| Page | Features |
|------|----------|
| **Overview** | KPIs, overall trend, category Pareto, market share |
| **Forecasting** | Category selector, 4/8/12-week horizon, 4 models, ~95% intervals |
| **Model Comparison** | Metrics table, win counts, DM test results, model cards |

**Live Demo**: `streamlit run app/Home.py`

---

## ⚠️ Limitations (Own Them)
1. **Simple baseline won** - no complex model justified
2. **ETS fell back to naive** - data lacks clear trend/seasonality
3. **XGBoost disabled** - feature engineering mismatch (time constraint)
4. **2014/2018 partial years** - only 3 clean years
5. **No external drivers** - no promotions, holidays, weather, econ indicators
5. **5 categories tested** - not all 51 (time constraint)
6. **No capacity planning** - shipping days ≠ supplier lead time

---

## 💬 Interview Talking Points

### "Why did simple moving average win?"
> "The demand series has moderate noise (CV=0.29) and weak seasonality. ETS auto-selected naive because AIC favored simpler models. Moving average (4) smooths noise better than single-point naive. Complex models overfit without clear signal."

### "How did you prevent leakage?"
> "1) Chronological splits only. 2) All lag/rolling features shifted to end at t-1. 3) Leakage test: perturb future actuals by 100x → predictions unchanged. 4) No target encoding using future folds."

### "What's MASE and why use it?"
> "Mean Absolute Scaled Error. Scales MAE by in-sample naive MAE. MASE<1 = better than naive. Comparable across series with different scales. We used it alongside MAE for cross-category comparison."

### "How did you validate models?"
> "Rolling-origin backtest: 52-week min train, 4-week horizon, 4-week step. Produces multiple train/test splits across time. Aggregated metrics + per-horizon metrics. DM test for statistical significance."

### "What would you do with more time?"
> "1) Fix XGBoost feature mismatch. 2) Add holiday/promotion features. 3) Try hierarchical forecasting (category→market). 4) Test on all 51 categories. 5) Add conformal prediction intervals for XGBoost."

---

## 🛠 Tech Stack
- **Data**: Pandas, DuckDB, PyArrow
- **Models**: Statsmodels (ETS), XGBoost, custom baselines
- **Eval**: Custom metrics, SciPy (DM test), NumPy
- **Dashboard**: Streamlit, Plotly
- **Infra**: uv, pytest, ruff, joblib

---

## 🔗 Links
- **GitHub**: [your-repo]
- **Live Demo**: `streamlit run app/Home.py`
- **Data**: https://data.mendeley.com/datasets/8gx2fvg2k6/5
- **Citation**: Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019)

---

*Print this double-sided for interview prep.*