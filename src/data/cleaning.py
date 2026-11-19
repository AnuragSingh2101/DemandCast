"""
Data Cleaning Module for DemandCast.

Fixes dtypes, parses timestamps, handles invalid records,
investigates missing values, removes exact duplicate rows,
drops PII, and validates numeric ranges.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Columns to drop (PII and high-cardinality identifiers)
PII_COLUMNS = [
    'customer_email',
    'customer_fname',
    'customer_lname',
    'customer_password',
    'customer_street',
    'customer_zipcode',
    'category_name',
    'department_name',
    'order_item_cardprod_id',
    'order_zipcode',
    'product_card_id',
    'product_name',
    'product_description',  # 100% missing
    'product_status',       # All zeros
]

# Additional ID columns that are not useful for forecasting
ID_COLUMNS = [
    'order_item_id',  # Unique per row (180,519 unique)
    'benefit_per_order',  # Derived, not a predictor
    'sales_per_customer',  # Derived, not a predictor
    'order_profit_per_order',  # Derived
    'order_item_total',  # Derived from price * qty * (1-discount)
    'order_item_profit_ratio',  # Derived
    'latitude',  # Location info, not needed for demand forecasting
    'longitude',
    'order_city',
    'order_country',
    'order_state',
    'customer_city',
    'customer_country',
    'customer_id',
    'order_customer_id',  # Same as customer_id
    'department_id',
]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns to datetime.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with parsed date columns
    """
    df = df.copy()
    
    # Main order date column
    date_cols = {
        'order_date_dateorders': 'order_date',
        'shipping_date_dateorders': 'shipping_date',
    }
    
    for orig, new in date_cols.items():
        if orig in df.columns:
            # Parse with flexible format (DataCo uses M/D/YYYY H:MM)
            parsed = pd.to_datetime(df[orig], format='mixed', errors='coerce')
            failed = parsed.isna().sum()
            if failed > 0:
                logger.warning(f"{orig}: {failed:,} rows failed to parse")
            df[new] = parsed
            logger.info(f"Parsed {orig} -> {new}: {parsed.notna().sum():,} valid dates")
    
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types for known columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with corrected dtypes
    """
    df = df.copy()
    
    # Integer columns (should be int but may be float due to NaN)
    int_columns = [
        'order_item_quantity',
        'days_for_shipping_real',
        'days_for_shipment_scheduled',
        'late_delivery_risk',
        'category_id',
        'product_category_id',
        'product_status',
    ]
    
    for col in int_columns:
        if col in df.columns:
            # Only convert if no NaN (or fill NaN first)
            if df[col].isna().sum() == 0:
                df[col] = df[col].astype('int64')
            else:
                df[col] = df[col].astype('Int64')  # Nullable integer
            logger.debug(f"Fixed dtype for {col}: {df[col].dtype}")
    
    # Float columns
    float_columns = [
        'order_item_discount',
        'order_item_discount_rate',
        'order_item_product_price',
        'sales',
        'order_item_total',
    ]
    
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Categorical columns
    cat_columns = [
        'type',
        'delivery_status',
        'market',
        'order_region',
        'order_status',
        'shipping_mode',
        'customer_segment',
    ]
    
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on column semantics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Order Zipcode: 86% missing - drop column (handled in drop_columns)
    # Product Description: 100% missing - drop column
    # Product Status: All zeros - drop column
    
    # For numeric columns with few missing, we could impute
    # But in DataCo, missing seems structural
    
    logger.info(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows only.
    Keeps legitimate separate transaction lines.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with exact duplicates removed
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} exact duplicate rows")
    else:
        logger.info("No exact duplicate rows found")
    return df


def drop_pii_and_ids(df: pd.DataFrame, drop_ids: bool = True) -> pd.DataFrame:
    """
    Drop PII columns and optionally high-cardinality ID columns.
    
    Args:
        df: Input DataFrame
        drop_ids: Whether to drop ID columns in addition to PII
    
    Returns:
        DataFrame with PII/ID columns removed
    """
    df = df.copy()
    
    columns_to_drop = PII_COLUMNS.copy()
    if drop_ids:
        columns_to_drop.extend(ID_COLUMNS)
    
    # Only drop columns that exist
    existing_to_drop = [c for c in columns_to_drop if c in df.columns]
    missing_cols = [c for c in columns_to_drop if c not in df.columns]
    
    if missing_cols:
        logger.debug(f"Columns not found (already dropped or renamed): {missing_cols}")
    
    df = df.drop(columns=existing_to_drop, errors='ignore')
    logger.info(f"Dropped {len(existing_to_drop)} PII/ID columns: {existing_to_drop}")
    
    return df


def validate_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate numeric columns are within expected ranges.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame (unchanged, but logs warnings for anomalies)
    """
    checks = {
        'order_item_quantity': (1, 5),  # Known range from audit
        'days_for_shipping_real': (0, 6),
        'days_for_shipment_scheduled': (0, 4),
        'late_delivery_risk': (0, 1),
        'order_item_discount_rate': (0, 1),  # 0-100% as decimal
        'order_item_product_price': (0, 2000),
        'sales': (0, 2000),
    }
    
    for col, (min_val, max_val) in checks.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                logger.warning(f"{col}: {len(out_of_range)} rows outside range [{min_val}, {max_val}]")
                # Log examples
                logger.debug(f"  Examples: {out_of_range[col].head().tolist()}")
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features useful for downstream processing.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with additional columns
    """
    df = df.copy()
    
    # Extract date components from order_date
    if 'order_date' in df.columns:
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.month
        df['order_week'] = df['order_date'].dt.isocalendar().week
        df['order_day_of_week'] = df['order_date'].dt.dayofweek  # Mon=0
        df['order_day_of_year'] = df['order_date'].dt.dayofyear
        df['order_quarter'] = df['order_date'].dt.quarter
        # Compute Monday of each week (floor to previous Monday)
        # dayofweek: Mon=0, Tue=1, ..., Sun=6
        df['order_week_start'] = df['order_date'] - pd.to_timedelta(df['order_date'].dt.dayofweek, unit='D')
        df['order_week_start'] = df['order_week_start'].dt.normalize()  # Remove time component
    
    # Shipping delay (real - scheduled)
    if 'days_for_shipping_real' in df.columns and 'days_for_shipment_scheduled' in df.columns:
        df['shipping_delay'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']
    
    # Is late delivery
    if 'late_delivery_risk' in df.columns:
        df['is_late'] = df['late_delivery_risk'].astype(bool)
    
    return df


def clean_data(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Main cleaning pipeline.
    
    Args:
        df: Raw transactions DataFrame
        config: Optional config dict
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = load_config()
    
    logger.info(f"Starting cleaning pipeline on {len(df):,} rows")
    
    # Parse dates
    df = parse_dates(df)
    
    # Fix dtypes
    df = fix_dtypes(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove exact duplicates
    df = remove_duplicates(df)
    
    # Drop PII and ID columns
    df = drop_pii_and_ids(df, drop_ids=True)
    
    # Validate numeric ranges
    df = validate_numeric_ranges(df)
    
    # Add derived features
    df = add_derived_features(df)
    
    logger.info(f"Cleaning complete: {len(df):,} rows x {df.shape[1]} columns")
    
    return df


def save_interim(df: pd.DataFrame, config: dict, filename: str = "cleaned_transactions.parquet") -> Path:
    """Save cleaned data to interim directory."""
    project_root = Path(__file__).parent.parent.parent
    interim_dir = project_root / config['paths']['interim_data']
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved interim data to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingestion import load_raw_data
    transactions, _ = load_raw_data()
    cleaned = clean_data(transactions)
    print(f"\nCleaned shape: {cleaned.shape}")
    print(f"Columns: {list(cleaned.columns)}")
    print(f"\nDtypes:\n{cleaned.dtypes}")