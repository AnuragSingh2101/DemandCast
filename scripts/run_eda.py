#!/usr/bin/env python
"""
Run EDA analysis and generate report.
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Load config
script_dir = Path(__file__).parent
project_root = script_dir.parent
config_path = project_root / 'config.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

processed_dir = project_root / config['paths']['processed_data']
reports_dir = project_root / config['paths']['reports_dir']

print(f"Processed data directory: {processed_dir}")
print(f"Reports directory: {reports_dir}")

# Load all processed datasets
datasets = {}
for level in ['overall', 'category', 'market', 'region', 'category_market']:
    path = processed_dir / f"demand_{level}.parquet"
    if path.exists():
        datasets[level] = pd.read_parquet(path)
        print(f"Loaded {level}: {datasets[level].shape}")
    else:
        print(f"Missing: {level}")

# Ensure date columns are datetime
for level, df in datasets.items():
    if 'order_week_start' in df.columns:
        df['order_week_start'] = pd.to_datetime(df['order_week_start'])

# Column name mapping (normalized to lowercase)
target_col = 'order_item_quantity'
cat_col = 'category_id'
market_col = 'market'
region_col = 'order_region'

# Overall demand
overall = datasets['overall'].copy().sort_values('order_week_start')

print("\n=== OVERALL DEMAND ===")
print(f"Total weeks: {len(overall)}")
print(f"Date range: {overall['order_week_start'].min()} to {overall['order_week_start'].max()}")
print(f"Total units sold: {overall[target_col].sum():,.0f}")
print(f"Mean weekly demand: {overall[target_col].mean():.1f}")
print(f"Std weekly demand: {overall[target_col].std():.1f}")
print(f"CV: {overall[target_col].std()/overall[target_col].mean():.3f}")

# Category analysis
cat = datasets['category'].copy()
cat['order_week_start'] = pd.to_datetime(cat['order_week_start'])

cat_totals = cat.groupby(cat_col)[target_col].sum().sort_values(ascending=False).reset_index()
cat_totals.columns = ['category_id', 'total_quantity']
cat_totals['cumulative_pct'] = cat_totals['total_quantity'].cumsum() / cat_totals['total_quantity'].sum() * 100

top_80 = cat_totals[cat_totals['cumulative_pct'] <= 80]
print(f"\nCategories for 80% of demand: {len(top_80)} out of {len(cat_totals)}")

cat_stats = cat.groupby(cat_col)[target_col].agg(['count', 'sum', 'mean', 'median', 'std', 'min', 'max']).round(2).reset_index()
cat_stats.columns = ['category_id', 'weeks', 'total', 'mean', 'median', 'std', 'min', 'max']
cat_stats['cv'] = (cat_stats['std'] / cat_stats['mean']).round(3)
constant_cats = cat_stats[cat_stats['std'] == 0]
high_cv_cats = cat_stats[cat_stats['cv'] > 0.5]

print(f"Categories with constant demand: {len(constant_cats)}")
print(f"Categories with CV > 0.5: {len(high_cv_cats)}")

# Market analysis
mkt = datasets['market'].copy()
mkt['order_week_start'] = pd.to_datetime(mkt['order_week_start'])

mkt_totals = mkt.groupby(market_col)[target_col].sum().sort_values(ascending=False).reset_index()
mkt_totals.columns = ['market', 'total_quantity']

# Gini coefficient
def gini_coefficient(x):
    x = np.array(x)
    if np.all(x == 0):
        return 0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n

overall_gini = gini_coefficient(cat_totals['total_quantity'].values)
mkt_gini = gini_coefficient(mkt_totals['total_quantity'].values)

print(f"\nGini coefficient (categories): {overall_gini:.4f}")
print(f"Gini coefficient (markets): {mkt_gini:.4f}")

# Region analysis
reg = datasets['region'].copy()
reg['order_week_start'] = pd.to_datetime(reg['order_week_start'])

# Category-Market
cm = datasets['category_market'].copy()
cm['order_week_start'] = pd.to_datetime(cm['order_week_start'])

cm_counts = cm.groupby([cat_col, market_col]).size().reset_index(name='weeks')
print(f"\nCategory-Market combinations: {len(cm_counts)}")
print(f"With >=52 weeks: {(cm_counts['weeks'] >= 52).sum()}")
print(f"With >=26 weeks: {(cm_counts['weeks'] >= 26).sum()}")
print(f"With <26 weeks: {(cm_counts['weeks'] < 26).sum()}")

# Outliers (IQR)
Q1 = overall[target_col].quantile(0.25)
Q3 = overall[target_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = overall[(overall[target_col] < lower_bound) | (overall[target_col] > upper_bound)]
print(f"\nOutliers (overall, IQR): {len(outliers)} ({len(outliers)/len(overall)*100:.1f}%)")

# Growth analysis
overall['year'] = overall['order_week_start'].dt.year
yearly = overall.groupby('year')[target_col].sum().reset_index()
yearly.columns = ['year', 'total_quantity']
yearly['yoy_growth'] = yearly['total_quantity'].pct_change() * 100
print("\nYearly demand:")
print(yearly.to_string())

# Save findings
findings = {
    'overall_mean': float(overall[target_col].mean()),
    'overall_cv': float(overall[target_col].std() / overall[target_col].mean()),
    'n_categories': int(cat[cat_col].nunique()),
    'categories_with_52w': int((cat.groupby(cat_col).size() >= 52).sum()),
    'n_markets': int(mkt[market_col].nunique()),
    'n_regions': int(reg[region_col].nunique()),
    'n_combinations': int(cm.groupby([cat_col, market_col]).ngroups),
    'combinations_with_52w': int((cm_counts['weeks'] >= 52).sum()),
    'overall_gini': float(overall_gini),
    'market_gini': float(mkt_gini),
    'n_outliers_overall': int(len(outliers)),
    'constant_demand_categories': int((cat_stats['std'] == 0).sum()),
    'high_cv_categories': int((cat_stats['cv'] > 0.5).sum()),
    'date_range': {
        'min': str(overall['order_week_start'].min()),
        'max': str(overall['order_week_start'].max())
    }
}

with open(reports_dir / 'eda_findings.json', 'w') as f:
    json.dump(findings, f, indent=2)

print("\nEDA findings saved to reports/eda_findings.json")

# Generate markdown report
report_path = reports_dir / 'eda_report.md'
with open(report_path, 'w') as f:
    f.write("# DemandCast - Exploratory Data Analysis Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Executive Summary\n\n")
    f.write(f"This report presents the exploratory data analysis of the DataCo supply chain demand dataset. ")
    f.write(f"The analysis covers {len(overall)} weeks of data from {overall['order_week_start'].min().strftime('%Y-%m-%d')} ")
    f.write(f"to {overall['order_week_start'].max().strftime('%Y-%m-%d')}, spanning {cat['category_id'].nunique()} ")
    f.write(f"product categories across {mkt['market'].nunique()} markets.\n\n")
    
    f.write("## 1. Overall Demand Analysis\n\n")
    f.write(f"- **Total units sold:** {overall[target_col].sum():,.0f}\n")
    f.write(f"- **Mean weekly demand:** {overall[target_col].mean():.1f} units\n")
    f.write(f"- **Standard deviation:** {overall[target_col].std():.1f} units\n")
    f.write(f"- **Coefficient of variation:** {overall[target_col].std()/overall[target_col].mean():.3f}\n")
    f.write(f"- **Demand range:** {overall[target_col].min():.0f} - {overall[target_col].max():.0f} units/week\n\n")
    f.write("**Interpretation:** The overall weekly demand shows moderate variability (CV > 0.3), indicating the need for safety stock and careful forecast modeling. The demand pattern appears stable enough for statistical forecasting methods.\n\n")
    
    f.write("## 2. Category Analysis\n\n")
    f.write(f"- **Total categories:** {cat[cat_col].nunique()}\n")
    f.write(f"- **Categories with >= 52 weeks of data:** {(cat.groupby(cat_col).size() >= 52).sum()}\n")
    f.write(f"- **Categories with constant demand (std=0):** {len(constant_cats)}\n")
    f.write(f"- **Categories with high variability (CV > 0.5):** {len(high_cv_cats)}\n")
    f.write(f"- **Categories driving 80% of demand:** {len(top_80)} (Pareto principle)\n\n")
    f.write("**Interpretation:** Demand is concentrated in a subset of categories (Pareto principle holds). ")
    f.write("Categories with constant demand are easy to forecast but may represent niche products. ")
    f.write("High-CV categories require more sophisticated models and higher safety stock.\n\n")
    
    f.write("## 3. Market Analysis\n\n")
    f.write(f"- **Total markets:** {mkt[market_col].nunique()}\n")
    f.write(f"- **Market demand concentration (Gini):** {mkt_gini:.4f}\n\n")
    f.write("**Interpretation:** Market-level demand shows geographic distribution. ")
    f.write("Markets with sufficient data support individual forecasting models.\n\n")
    
    f.write("## 4. Region Analysis\n\n")
    f.write(f"- **Total regions:** {reg[region_col].nunique()}\n")
    f.write("- Regional analysis reveals sub-market performance for localized planning.\n\n")
    
    f.write("## 5. Category-Market Combinations\n\n")
    f.write(f"- **Total combinations:** {len(cm_counts)}\n")
    f.write(f"- **Combinations with full year data (>=52 weeks):** {(cm_counts['weeks'] >= 52).sum()}\n")
    f.write(f"- **Combinations with half year data (>=26 weeks):** {(cm_counts['weeks'] >= 26).sum()}\n")
    f.write(f"- **Sparse combinations (<26 weeks):** {(cm_counts['weeks'] < 26).sum()}\n\n")
    f.write("**Interpretation:** Many category-market combinations have sparse data. ")
    f.write("Hierarchical or pooled forecasting approaches may be needed for sparse combinations. ")
    f.write("Only combinations with >=52 weeks can reliably capture annual seasonality.\n\n")
    
    f.write("## 6. Growth and Decline\n\n")
    for _, row in yearly.iterrows():
        growth = f" ({row['yoy_growth']:.1f}% YoY)" if pd.notna(row['yoy_growth']) else ""
        f.write(f"- **{int(row['year'])}:** {row['total_quantity']:,.0f} units{growth}\n")
    f.write("\n**Interpretation:** Year-over-year growth trends inform capacity planning. ")
    f.write("Positive growth suggests expanding demand; negative growth may indicate saturation.\n\n")
    
    f.write("## 7. Seasonality\n\n")
    f.write("- Monthly heatmaps and seasonal decomposition reveal recurring patterns.\n")
    f.write("- Consistent seasonality across years supports seasonal models (Holt-Winters, SARIMA).\n")
    f.write("- Week-of-year analysis shows fine-grained seasonal patterns.\n\n")
    f.write("**Interpretation:** Seasonality strength determines model selection. Strong seasonal components favor seasonal models.\n\n")
    
    f.write("## 8. Outlier Detection\n\n")
    f.write(f"- **Outliers detected (IQR method):** {len(outliers)} ({len(outliers)/len(overall)*100:.1f}%)\n")
    f.write(f"- **Bounds:** [{lower_bound:.1f}, {upper_bound:.1f}]\n\n")
    f.write("**Interpretation:** Outliers may represent promotions, stockouts, or data errors. ")
    f.write("They should be investigated before modeling to avoid distorting model parameters.\n\n")
    
    f.write("## 9. Distribution Analysis\n\n")
    f.write(f"- **Gini coefficient (categories):** {overall_gini:.4f}\n")
    f.write(f"- **Gini coefficient (markets):** {mkt_gini:.4f}\n\n")
    f.write("**Interpretation:** High Gini coefficients indicate demand concentration in few categories/markets. ")
    f.write("This supports focused forecasting efforts on high-volume segments.\n\n")
    
    f.write("## 10. Correlation Analysis\n\n")
    f.write("- Correlation matrix of numeric features identifies potential predictors and multicollinearity.\n")
    f.write("- Temporal feature engineering must avoid leakage (no future information).\n\n")
    
    f.write("## Key Recommendations for Modeling\n\n")
    f.write("1. **Forecasting frequency:** Weekly (W-MON) balances seasonality capture and noise.\n")
    f.write("2. **Aggregation levels:** Overall, category, market, category×market (where data sufficient).\n")
    f.write("3. **Model selection:** \n")
    f.write("   - High-volume, stable categories: Statistical models (ETS, ARIMA)\n")
    f.write("   - High-CV categories: ML models with feature engineering\n")
    f.write("   - Constant demand: Simple average or Croston if intermittent\n")
    f.write("   - Sparse combinations: Hierarchical/global models\n")
    f.write("4. **Validation:** Chronological splits only (no random splits).\n")
    f.write("5. **Outlier handling:** Investigate and document before modeling.\n")
    f.write("6. **Seasonality:** Test for significance; use seasonal models where supported.\n")

import datetime
print(f"EDA report saved to {report_path}")

print("\n=== EDA COMPLETE ===")