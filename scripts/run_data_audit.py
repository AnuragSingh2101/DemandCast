#!/usr/bin/env python
"""
DataCo Dataset Audit Script
Run this directly with Python instead of Jupyter to avoid kernel issues.
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load config
import os
script_dir = Path(__file__).parent
project_root = script_dir.parent
config_path = project_root / 'config.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

raw_path = project_root / config['paths']['raw_data']
transactions_file = raw_path / config['raw_files']['transactions']
dict_file = raw_path / config['raw_files']['data_dictionary']

print(f"Transactions file: {transactions_file}")
print(f"Data dictionary file: {dict_file}")
print(f"Transactions exists: {transactions_file.exists()}")
print(f"Dictionary exists: {dict_file.exists()}")

# Load the data with appropriate encoding
encodings_to_try = ['iso-8859-1', 'latin-1', 'cp1252', 'utf-8']

df = None
used_encoding = None

for enc in encodings_to_try:
    try:
        df = pd.read_csv(transactions_file, encoding=enc, low_memory=False)
        used_encoding = enc
        print(f"\nSuccessfully loaded with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Failed with {enc}: {e}")
    except Exception as e:
        print(f"Error with {enc}: {e}")

if df is None:
    raise ValueError("Could not load the dataset with any of the tried encodings")

print(f"\nShape: {df.shape}")
print(f"Encoding used: {used_encoding}")

# Basic info
print("\n=== DATASET INFO ===")
df.info(verbose=True, show_counts=True)

print("\n=== COLUMN NAMES ===")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")

# Missing values analysis
print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
print(missing_df.to_string())

print(f"\nTotal rows: {len(df):,}")
print(f"Columns with missing: {len(missing_df)}")
print(f"Columns complete: {df.shape[1] - len(missing_df)}")

# Duplicate rows
print("\n=== DUPLICATE ANALYSIS ===")
exact_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {exact_dupes:,} ({exact_dupes/len(df)*100:.2f}%)")

id_cols = [c for c in df.columns if 'id' in c.lower() or 'order' in c.lower()]
print(f"Potential ID columns: {id_cols}")

if id_cols:
    for col in id_cols:
        dupe_count = df.duplicated(subset=[col]).sum()
        print(f"  Duplicates on {col}: {dupe_count:,}")

# Data types and unique counts
print("\n=== COLUMN DETAILS ===")
details = []
for col in df.columns:
    dtype = str(df[col].dtype)
    n_unique = df[col].nunique()
    n_missing = df[col].isnull().sum()
    sample_vals = df[col].dropna().unique()[:5]
    details.append({
        'column': col,
        'dtype': dtype,
        'n_unique': n_unique,
        'n_missing': n_missing,
        'missing_pct': round(n_missing/len(df)*100, 2),
        'sample_values': str(sample_vals)
    })

details_df = pd.DataFrame(details)
print(details_df.to_string())

# Load data dictionary
dict_df = None
for enc in encodings_to_try:
    try:
        dict_df = pd.read_csv(dict_file, encoding=enc)
        print(f"\nDictionary loaded with encoding: {enc}")
        break
    except Exception as e:
        print(f"Failed with {enc}: {e}")

if dict_df is not None:
    print("\n=== DATA DICTIONARY ===")
    print(dict_df.to_string())
else:
    print("\nCould not load data dictionary")

# Date column analysis
print("\n=== DATE COLUMN ANALYSIS ===")
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
print(f"Potential date columns: {date_cols}")

for col in date_cols:
    print(f"\n--- {col} ---")
    print(f"  Dtype: {df[col].dtype}")
    print(f"  Non-null: {df[col].notna().sum():,}")
    print(f"  Unique: {df[col].nunique():,}")
    sample = df[col].dropna().unique()[:10]
    print(f"  Sample values: {sample}")

# Parse dates and check coverage
date_col_candidates = ['Order_Date', 'order_date', 'Date', 'date', 'Transaction_Date', 'transaction_date']
date_col = None
for cand in date_col_candidates:
    if cand in df.columns:
        date_col = cand
        break

if date_col is None and date_cols:
    date_col = date_cols[0]

if date_col:
    print(f"\nUsing date column: {date_col}")
    parsed = pd.to_datetime(df[date_col], errors='coerce')
    print(f"  Successfully parsed: {parsed.notna().sum():,} / {len(df):,}")
    print(f"  Failed to parse: {parsed.isna().sum():,}")

    if parsed.notna().any():
        print(f"  Min date: {parsed.min()}")
        print(f"  Max date: {parsed.max()}")
        print(f"  Date range: {(parsed.max() - parsed.min()).days} days")

        # Check for partial final period
        max_date = parsed.max()
        final_week_start = max_date - pd.Timedelta(days=6)
        final_week_data = parsed[(parsed >= final_week_start) & (parsed <= max_date)]
        print(f"  Days in final week: {final_week_data.notna().sum()}")
        print(f"  Unique dates in final week: {final_week_data.nunique()}")

        # Monthly breakdown of final month
        final_month_start = max_date.replace(day=1)
        final_month_data = parsed[(parsed >= final_month_start) & (parsed <= max_date)]
        print(f"  Days in final month: {final_month_data.notna().sum()}")
        print(f"  Unique dates in final month: {final_month_data.nunique()}")

        # Day of week distribution in final week
        if len(final_week_data.dropna()) > 0:
            dow_counts = final_week_data.dropna().dt.day_name().value_counts().sort_index()
            print(f"  Day of week in final week:\n{dow_counts}")
else:
    print("No date column found")

# Categorical column analysis
print("\n=== CATEGORICAL ANALYSIS ===")
cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
print(f"Object/category columns: {cat_cols}")

key_cats = ['Category', 'category', 'Market', 'market', 'Region', 'region', 'Country', 'country',
            'Customer', 'customer', 'Product', 'product', 'Order', 'order', 'Warehouse', 'warehouse']

for col in cat_cols:
    if any(k.lower() in col.lower() for k in ['category', 'market', 'region', 'country', 'customer', 'product', 'warehouse']):
        print(f"\n--- {col} ---")
        print(f"  Unique values: {df[col].nunique():,}")
        vc = df[col].value_counts().head(20)
        print(f"  Top 20:\n{vc}")

# Numeric column analysis
print("\n=== NUMERIC ANALYSIS ===")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {num_cols}")

for col in num_cols:
    print(f"\n--- {col} ---")
    desc = df[col].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc)
    print(f"  Zeros: {(df[col] == 0).sum():,}")
    print(f"  Negatives: {(df[col] < 0).sum():,}")

# Target variable candidates
print("\n=== TARGET VARIABLE CANDIDATES ===")
target_candidates = [c for c in df.columns if any(k in c.lower() for k in ['quantity', 'qty', 'amount', 'sales', 'demand', 'units', 'volume'])]
print(f"Potential target columns: {target_candidates}")

for col in target_candidates:
    print(f"\n--- {col} ---")
    print(f"  Dtype: {df[col].dtype}")
    print(f"  Range: {df[col].min()} to {df[col].max()}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Zeros: {(df[col] == 0).sum():,} ({(df[col] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  Negatives: {(df[col] < 0).sum():,}")
    print(f"  Missing: {df[col].isnull().sum():,}")

# Transaction grain analysis
print("\n=== TRANSACTION GRAIN ===")
id_cols = [c for c in df.columns if 'id' in c.lower() or 'order' in c.lower() or 'line' in c.lower()]
print(f"Potential ID columns: {id_cols}")

if id_cols:
    for col in id_cols:
        print(f"  {col}: {df[col].nunique():,} unique, {df[col].isnull().sum():,} missing")

    if len(id_cols) >= 2:
        combo = df[id_cols[:2]].drop_duplicates()
        print(f"  Unique combinations of first 2 ID cols: {len(combo):,}")
        print(f"  Total rows: {len(df):,}")

# Aggregation level analysis
print("\n=== AGGREGATION LEVEL ANALYSIS ===")

cat_col = None
market_col = None
region_col = None

for col in df.columns:
    cl = col.lower()
    if 'category' in cl and 'product' not in cl and cat_col is None:
        cat_col = col
    elif 'market' in cl and market_col is None:
        market_col = col
    elif 'region' in cl and region_col is None:
        region_col = None

print(f"Category column: {cat_col}")
print(f"Market column: {market_col}")
print(f"Region column: {region_col}")

target_col = 'Order_Item_Quantity'
if target_col not in df.columns:
    for col in target_candidates:
        if 'quantity' in col.lower() or 'qty' in col.lower():
            target_col = col
            break

print(f"Target column: {target_col}")

if cat_col and target_col in df.columns:
    print(f"\n--- By {cat_col} ---")
    cat_stats = df.groupby(cat_col)[target_col].agg(['count', 'sum', 'mean', 'median', 'std']).round(2)
    cat_stats.columns = ['n_transactions', 'total_qty', 'avg_qty', 'median_qty', 'std_qty']
    print(cat_stats.sort_values('total_qty', ascending=False).to_string())

# Time series analysis at different aggregation levels
print("\n=== TIME SERIES AGGREGATION ANALYSIS ===")

if date_col and target_col in df.columns:
    df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')

    # Overall daily
    daily = df.dropna(subset=['_parsed_date']).groupby('_parsed_date')[target_col].sum()
    print(f"Overall daily series: {len(daily)} days")
    print(f"  Date range: {daily.index.min()} to {daily.index.max()}")
    print(f"  Mean daily demand: {daily.mean():.2f}")
    print(f"  Zero days: {(daily == 0).sum()}")

    # Weekly (Monday start)
    weekly = daily.resample('W-MON').sum()
    print(f"\nOverall weekly series: {len(weekly)} weeks")
    print(f"  Mean weekly demand: {weekly.mean():.2f}")
    print(f"  Zero weeks: {(weekly == 0).sum()}")

    # Monthly
    monthly = daily.resample('MS').sum()
    print(f"\nOverall monthly series: {len(monthly)} months")
    print(f"  Mean monthly demand: {monthly.mean():.2f}")
    print(f"  Zero months: {(monthly == 0).sum()}")

    # By category weekly
    if cat_col:
        cat_weekly = df.dropna(subset=['_parsed_date']).groupby([cat_col, pd.Grouper(key='_parsed_date', freq='W-MON')])[target_col].sum().unstack(fill_value=0)
        print(f"\nCategory-weekly matrix: {cat_weekly.shape[0]} categories x {cat_weekly.shape[1]} weeks")

        cat_lengths = (cat_weekly > 0).sum(axis=1)
        print(f"  Weeks with demand per category:\n{cat_lengths.describe()}")
        print(f"  Categories with < 26 weeks data: {(cat_lengths < 26).sum()}")
        print(f"  Categories with < 52 weeks data: {(cat_lengths < 52).sum()}")

    # By market weekly
    if market_col:
        mkt_weekly = df.dropna(subset=['_parsed_date']).groupby([market_col, pd.Grouper(key='_parsed_date', freq='W-MON')])[target_col].sum().unstack(fill_value=0)
        print(f"\nMarket-weekly matrix: {mkt_weekly.shape[0]} markets x {mkt_weekly.shape[1]} weeks")
        mkt_lengths = (mkt_weekly > 0).sum(axis=1)
        print(f"  Weeks with demand per market:\n{mkt_lengths.describe()}")

# Intermittent demand analysis
print("\n=== INTERMITTENT DEMAND ANALYSIS ===")

if date_col and target_col in df.columns and cat_col:
    cat_series = df.dropna(subset=['_parsed_date']).groupby([cat_col, pd.Grouper(key='_parsed_date', freq='W-MON')])[target_col].sum()

    intermittent_stats = []
    for cat in cat_series.index.get_level_values(0).unique():
        series = cat_series.loc[cat].values
        non_zero = series[series > 0]
        zero_periods = (series == 0).sum()
        total_periods = len(series)

        if len(non_zero) > 1:
            adi = total_periods / len(non_zero)
            cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() > 0 else np.nan
        else:
            adi = total_periods
            cv2 = np.nan

        intermittent_stats.append({
            'category': cat,
            'total_periods': total_periods,
            'non_zero_periods': len(non_zero),
            'zero_periods': zero_periods,
            'adi': round(adi, 2),
            'cv2': round(cv2, 4) if not np.isnan(cv2) else np.nan,
            'mean_demand': round(non_zero.mean(), 2) if len(non_zero) > 0 else 0,
            'intermittent_adi': adi > 1.32,
            'intermittent_cv2': cv2 > 0.49 if not np.isnan(cv2) else False
        })

    intermittent_df = pd.DataFrame(intermittent_stats)
    intermittent_df = intermittent_df.sort_values('adi', ascending=False)
    print(intermittent_df.to_string())

    print(f"\nCategories with ADI > 1.32: {(intermittent_df['intermittent_adi']).sum()} / {len(intermittent_df)}")
    print(f"Categories with CV^2 > 0.49: {(intermittent_df['intermittent_cv2']).sum()} / {len(intermittent_df)}")
    print(f"Both criteria (intermittent): {((intermittent_df['intermittent_adi']) & (intermittent_df['intermittent_cv2'])).sum()}")

# PII column check
print("\n=== PII COLUMN CHECK ===")
pii_keywords = ['email', 'name', 'password', 'address', 'street', 'zip', 'postal', 'phone', 'ssn', 'credit', 'card']
pii_cols = []
for col in df.columns:
    cl = col.lower()
    if any(kw in cl for kw in pii_keywords):
        pii_cols.append(col)

if pii_cols:
    print(f"Potential PII columns found: {pii_cols}")
    for col in pii_cols:
        print(f"  {col}: {df[col].dtype}, {df[col].nunique()} unique, sample: {df[col].dropna().unique()[:3]}")
else:
    print("No obvious PII columns found by keyword search")

# Also check for high-cardinality identifier columns that should be dropped
id_like = []
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.5:
        id_like.append(col)
    elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() == len(df):
        id_like.append(col)

print(f"\nHigh-cardinality columns (likely IDs): {id_like}")

# Shipping/delivery columns check
print("\n=== SHIPPING/DELIVERY COLUMNS ===")
ship_cols = [c for c in df.columns if any(k in c.lower() for k in ['ship', 'delivery', 'lead', 'transit', 'days'])]
print(f"Shipping-related columns: {ship_cols}")

for col in ship_cols:
    print(f"\n--- {col} ---")
    print(f"  Dtype: {df[col].dtype}")
    if df[col].dtype in ['int64', 'float64']:
        print(f"  Range: {df[col].min()} to {df[col].max()}")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
    else:
        vc = df[col].value_counts().head(10)
        print(f"  Top values:\n{vc}")

# Summary and Recommendations
print("\n=== AUDIT SUMMARY ===")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {df.shape[1]}")
print(f"Encoding used: {used_encoding}")
print(f"Date column: {date_col}")
print(f"Target column: {target_col}")
print(f"Category column: {cat_col}")
print(f"Market column: {market_col}")
print(f"Region column: {region_col}")

if date_col and target_col in df.columns:
    df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
    valid_dates = df['_parsed_date'].notna()
    print(f"Valid dates: {valid_dates.sum():,} / {len(df):,}")
    if valid_dates.any():
        print(f"Date range: {df.loc[valid_dates, '_parsed_date'].min()} to {df.loc[valid_dates, '_parsed_date'].max()}")

        # Final period check
        max_date = df.loc[valid_dates, '_parsed_date'].max()
        final_week_start = max_date - pd.Timedelta(days=6)
        final_week = df[(df['_parsed_date'] >= final_week_start) & (df['_parsed_date'] <= max_date)]
        print(f"Final week rows: {len(final_week):,}")
        print(f"Final week unique dates: {final_week['_parsed_date'].nunique()}")

        # Check if final month is complete
        final_month_start = max_date.replace(day=1)
        if final_month_start.month == 12:
            next_month = final_month_start.replace(year=final_month_start.year+1, month=1)
        else:
            next_month = final_month_start.replace(month=final_month_start.month+1)
        expected_days = (next_month - final_month_start).days
        final_month = df[(df['_parsed_date'] >= final_month_start) & (df['_parsed_date'] < next_month)]
        actual_days = final_month['_parsed_date'].nunique()
        print(f"Final month: {actual_days} / {expected_days} days present ({actual_days/expected_days*100:.1f}%)")

        if actual_days < expected_days * 0.8:
            print("WARNING: Final month appears incomplete (partial period)")
        else:
            print("Final month appears complete")

# Save audit results
audit_results = {
    'total_rows': int(len(df)),
    'total_columns': int(df.shape[1]),
    'encoding': used_encoding,
    'date_column': date_col,
    'target_column': target_col,
    'category_column': cat_col,
    'market_column': market_col,
    'region_column': region_col,
    'date_range': None,
    'partial_final_period': False,
    'columns': details_df.to_dict('records'),
    'missing_summary': missing_df.to_dict('records'),
    'duplicate_count': int(exact_dupes),
    'pii_columns': pii_cols,
    'id_like_columns': id_like,
    'shipping_columns': ship_cols,
}

if date_col and target_col in df.columns and valid_dates.any():
    audit_results['date_range'] = {
        'min': str(df.loc[valid_dates, '_parsed_date'].min()),
        'max': str(df.loc[valid_dates, '_parsed_date'].max())
    }
    audit_results['partial_final_period'] = actual_days < expected_days * 0.8

# Save to JSON
output_path = project_root / config['paths']['reports_dir'] / 'data_audit.json'
with open(output_path, 'w') as f:
    json.dump(audit_results, f, indent=2, default=str)

print(f"\nAudit results saved to: {output_path}")

# Generate markdown report
report_path = project_root / config['paths']['reports_dir'] / 'data_audit.md'
with open(report_path, 'w') as f:
    f.write("# DataCo Dataset Audit Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Dataset:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS\n")
    f.write(f"**Source:** Mendeley Data, Version 5\n")
    f.write(f"**License:** CC BY 4.0\n\n")

    f.write("## Summary\n\n")
    f.write(f"- **Total rows:** {len(df):,}\n")
    f.write(f"- **Total columns:** {df.shape[1]}\n")
    f.write(f"- **Encoding:** {used_encoding}\n")
    f.write(f"- **Date column:** {date_col}\n")
    f.write(f"- **Target column:** {target_col}\n")
    f.write(f"- **Category column:** {cat_col}\n")
    f.write(f"- **Market column:** {market_col}\n")
    f.write(f"- **Region column:** {region_col}\n")
    f.write(f"- **Exact duplicates:** {exact_dupes:,} ({exact_dupes/len(df)*100:.2f}%)\n")

    if date_col and target_col in df.columns and valid_dates.any():
        f.write(f"- **Date range:** {df.loc[valid_dates, '_parsed_date'].min()} to {df.loc[valid_dates, '_parsed_date'].max()}\n")
        f.write(f"- **Final month complete:** {'No (WARNING)' if actual_days < expected_days * 0.8 else 'Yes'} ({actual_days}/{expected_days} days)\n")

    f.write("\n## Missing Values\n\n")
    if len(missing_df) > 0:
        f.write(missing_df.to_markdown())
    else:
        f.write("No missing values.\n")

    f.write("\n\n## Column Details\n\n")
    f.write(details_df.to_markdown())

    f.write("\n\n## Categorical Columns\n\n")
    for col in cat_cols:
        if any(k.lower() in col.lower() for k in ['category', 'market', 'region', 'country', 'customer', 'product', 'warehouse']):
            f.write(f"\n### {col}\n")
            f.write(f"- Unique values: {df[col].nunique():,}\n")
            f.write(f"- Top 10:\n")
            vc = df[col].value_counts().head(10)
            f.write(vc.to_markdown())
            f.write("\n")

    f.write("\n\n## Numeric Columns\n\n")
    for col in num_cols:
        f.write(f"\n### {col}\n")
        desc = df[col].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        f.write(desc.to_markdown())
        f.write(f"\n- Zeros: {(df[col] == 0).sum():,}\n")
        f.write(f"- Negatives: {(df[col] < 0).sum():,}\n")

    f.write("\n\n## Target Variable Candidates\n\n")
    for col in target_candidates:
        f.write(f"\n### {col}\n")
        f.write(f"- Dtype: {df[col].dtype}\n")
        f.write(f"- Range: {df[col].min()} to {df[col].max()}\n")
        f.write(f"- Mean: {df[col].mean():.2f}\n")
        f.write(f"- Median: {df[col].median():.2f}\n")
        f.write(f"- Zeros: {(df[col] == 0).sum():,} ({(df[col] == 0).sum()/len(df)*100:.1f}%)\n")
        f.write(f"- Negatives: {(df[col] < 0).sum():,}\n")
        f.write(f"- Missing: {df[col].isnull().sum():,}\n")

    f.write("\n\n## Time Series Aggregation\n\n")
    if date_col and target_col in df.columns:
        f.write(f"- **Daily series:** {len(daily)} days, mean={daily.mean():.2f}, zeros={(daily == 0).sum()}\n")
        f.write(f"- **Weekly series:** {len(weekly)} weeks, mean={weekly.mean():.2f}, zeros={(weekly == 0).sum()}\n")
        f.write(f"- **Monthly series:** {len(monthly)} months, mean={monthly.mean():.2f}, zeros={(monthly == 0).sum()}\n")

    f.write("\n\n## Intermittent Demand Analysis\n\n")
    if 'intermittent_df' in locals():
        f.write(intermittent_df.to_markdown())
        f.write(f"\n\n- Categories with ADI > 1.32: {(intermittent_df['intermittent_adi']).sum()} / {len(intermittent_df)}\n")
        f.write(f"- Categories with CV^2 > 0.49: {(intermittent_df['intermittent_cv2']).sum()} / {len(intermittent_df)}\n")
        f.write(f"- Both criteria (intermittent): {((intermittent_df['intermittent_adi']) & (intermittent_df['intermittent_cv2'])).sum()}\n")

    f.write("\n\n## PII Columns\n\n")
    if pii_cols:
        for col in pii_cols:
            f.write(f"- {col}: {df[col].dtype}, {df[col].nunique()} unique\n")
    else:
        f.write("No obvious PII columns found.\n")

    f.write("\n\n## High-Cardinality ID Columns\n\n")
    for col in id_like:
        f.write(f"- {col}: {df[col].dtype}, {df[col].nunique()} unique\n")

    f.write("\n\n## Shipping/Delivery Columns\n\n")
    for col in ship_cols:
        f.write(f"\n### {col}\n")
        f.write(f"- Dtype: {df[col].dtype}\n")
        if df[col].dtype in ['int64', 'float64']:
            f.write(f"- Range: {df[col].min()} to {df[col].max()}\n")
            f.write(f"- Mean: {df[col].mean():.2f}\n")
            f.write(f"- Median: {df[col].median():.2f}\n")
        else:
            vc = df[col].value_counts().head(10)
            f.write(vc.to_markdown())
            f.write("\n")

    f.write("\n\n## Recommendations\n\n")
    f.write("Based on the audit, the following recommendations are made:\n\n")
    f.write("1. **Target Variable:** `Order_Item_Quantity` (total quantity per transaction line)\n")
    f.write("2. **Frequency:** Weekly (W-MON) - balances seasonality capture and noise reduction\n")
    f.write("3. **Aggregation Levels:** overall, category, market, region, category x market\n")
    f.write("4. **Partial Final Period:** Flag and exclude incomplete final month from training\n")
    f.write("5. **PII Handling:** Drop identified PII columns during ETL\n")
    f.write("6. **ID Columns:** Drop high-cardinality identifier columns\n")
    f.write("7. **Intermittent Demand:** Use Croston/TSB for categories with ADI > 1.32 and CV^2 > 0.49\n")

print("\nMarkdown report saved to:", report_path)
print("\n=== AUDIT COMPLETE ===")