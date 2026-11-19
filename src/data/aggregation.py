"""
Data Aggregation Module for DemandCast.

Uses DuckDB to efficiently aggregate demand data at multiple levels:
- Overall
- Category
- Market
- Category x Market
- Region (if available)

Builds continuous date indexes and separates true zeros from missing periods.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import duckdb
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_processed_path(config: dict, filename: str) -> Path:
    """Get path in processed data directory."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / config['paths']['processed_data'] / filename


def get_interim_path(config: dict, filename: str) -> Path:
    """Get path in interim data directory."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / config['paths']['interim_data'] / filename


def build_continuous_index(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq: str = 'W-MON'
) -> pd.DatetimeIndex:
    """
    Build a continuous date index for time series.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        freq: Pandas frequency string (default: weekly Monday)
    
    Returns:
        Continuous DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def aggregate_with_duckdb(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = 'order_week_start',
    target_col: str = 'order_item_quantity',
    freq: str = 'W-MON'
) -> pd.DataFrame:
    """
    Aggregate demand using DuckDB for performance.
    
    Args:
        df: Cleaned transactions DataFrame
        group_cols: Columns to group by (e.g., ['category_id'], ['market'])
        date_col: Date column for time grouping
        target_col: Target column to sum
        freq: Frequency for resampling (used for continuous index)
    
    Returns:
        DataFrame with columns: group_cols + [date_col, target_col]
    """
    # Register DataFrame with DuckDB
    con = duckdb.connect()
    con.register('transactions', df)
    
    # Build group by clause
    group_str = ', '.join(group_cols + [date_col]) if group_cols else date_col
    
    query = f"""
    SELECT {group_str}, SUM({target_col}) as {target_col}
    FROM transactions
    WHERE {date_col} IS NOT NULL
    GROUP BY {group_str}
    ORDER BY {group_str}
    """
    
    result = con.execute(query).df()
    con.close()
    
    logger.info(f"Aggregated by {group_cols}: {len(result)} rows")
    return result


def create_continuous_series(
    agg_df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = 'order_week_start',
    target_col: str = 'order_item_quantity',
    freq: str = 'W-MON'
) -> pd.DataFrame:
    """
    Create continuous time series for each group, filling missing periods with 0.
    
    This distinguishes:
    - True zeros: periods with transactions but zero demand (should not happen for quantity)
    - Missing periods: no transactions recorded (could be true zero or missing data)
    
    Args:
        agg_df: Aggregated DataFrame from aggregate_with_duckdb
        group_cols: Grouping columns
        date_col: Date column
        target_col: Target column
        freq: Frequency for continuous index
    
    Returns:
        DataFrame with continuous date index per group
    """
    if len(agg_df) == 0:
        return pd.DataFrame()
    
    # Get global date range
    min_date = agg_df[date_col].min()
    max_date = agg_df[date_col].max()
    
    # Build continuous index with proper name
    full_index = build_continuous_index(min_date, max_date, freq)
    full_index.name = date_col
    
    if not group_cols:
        # Overall series - simple reindex
        series = agg_df.set_index(date_col)[target_col]
        series = series.reindex(full_index, fill_value=0)
        result = pd.DataFrame({target_col: series})
        result.index.name = date_col
        result = result.reset_index()
    elif len(group_cols) == 1:
        # Single group column - use pivot table
        group_col = group_cols[0]
        pivot = agg_df.pivot_table(
            index=date_col,
            columns=group_col,
            values=target_col,
            fill_value=0
        )
        
        # Ensure pivot index has the correct name
        pivot.index.name = date_col
        
        # Reindex to full date range
        pivot = pivot.reindex(full_index, fill_value=0)
        pivot.index.name = date_col
        
        # Melt back to long format
        result = pivot.reset_index().melt(
            id_vars=date_col,
            var_name=group_col,
            value_name=target_col
        )
    else:
        # Multiple group columns - create a combined key but preserve original dtypes
        agg_df = agg_df.copy()
        
        # Store original dtypes for group columns
        group_dtypes = {col: agg_df[col].dtype for col in group_cols}
        
        # Create combined key for pivoting
        agg_df['_group_key'] = agg_df[group_cols].astype(str).agg('|'.join, axis=1)
        
        pivot = agg_df.pivot_table(
            index=date_col,
            columns='_group_key',
            values=target_col,
            fill_value=0
        )
        
        pivot.index.name = date_col
        pivot = pivot.reindex(full_index, fill_value=0)
        pivot.index.name = date_col
        
        # Melt
        result = pivot.reset_index().melt(
            id_vars=date_col,
            var_name='_group_key',
            value_name=target_col
        )
        
        # Split the combined key back into original columns and restore dtypes
        split_cols = result['_group_key'].str.split('|', expand=True)
        split_cols.columns = group_cols
        for col in group_cols:
            if col in group_dtypes:
                try:
                    split_cols[col] = split_cols[col].astype(group_dtypes[col])
                except Exception:
                    pass  # Keep as string if conversion fails
        
        result = pd.concat([result.drop(columns=['_group_key']), split_cols], axis=1)
    
    logger.info(f"Created continuous series: {len(result)} rows, "
                f"{len(full_index)} periods x {agg_df[group_cols].drop_duplicates().shape[0] if group_cols else 1} groups")
    
    return result


def flag_partial_final_period(
    df: pd.DataFrame,
    date_col: str = 'order_week_start',
    threshold_days: int = 7
) -> pd.DataFrame:
    """
    Flag or drop the final period if it appears incomplete.
    
    Args:
        df: DataFrame with date column
        date_col: Date column name
        threshold_days: Minimum days in final period to consider complete
    
    Returns:
        DataFrame with 'is_partial_final' column
    """
    df = df.copy()
    
    if date_col not in df.columns:
        return df
    
    max_date = df[date_col].max()
    # For weekly data, check if the final week has full 7 days of source data
    # Since we're already aggregated to weekly, check the source data
    # For now, just flag the final period
    df['is_final_period'] = df[date_col] == max_date
    df['is_partial_final'] = False  # Will be set by pipeline based on source data check
    
    return df


def separate_zeros_from_missing(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = 'order_week_start',
    target_col: str = 'order_item_quantity',
    source_df: Optional[pd.DataFrame] = None,
    source_date_col: str = 'order_date'
) -> pd.DataFrame:
    """
    Separate true zero demand from missing source data periods.
    
    True zero: transactions exist but quantity=0 (unlikely for this dataset)
    Missing source: no transactions recorded for that period/group
    
    Args:
        df: Continuous series DataFrame
        group_cols: Grouping columns
        date_col: Date column
        target_col: Target column
        source_df: Original transaction data (to check if period has any records)
        source_date_col: Date column in source data
    
    Returns:
        DataFrame with 'demand_type' column: 'observed', 'true_zero', 'missing_source'
    """
    df = df.copy()
    df['demand_type'] = 'observed'  # Default
    
    # Mark explicit zeros as true_zero
    df.loc[df[target_col] == 0, 'demand_type'] = 'true_zero'
    
    # If we have source data, check which periods have NO transactions at all
    if source_df is not None and len(source_df) > 0:
        # Aggregate source to same level to find periods with any transactions
        if group_cols:
            source_agg = source_df.groupby(group_cols + [source_date_col]).size().reset_index(name='txn_count')
            source_agg[date_col] = source_agg[source_date_col].dt.to_period('W-MON').dt.start_time
            source_periods = source_agg.groupby(group_cols + [date_col]).size().reset_index()
            source_periods['has_data'] = True
            
            # Merge to find missing source periods
            merge_cols = group_cols + [date_col]
            df = df.merge(source_periods[merge_cols + ['has_data']], on=merge_cols, how='left')
            df['has_data'] = df['has_data'].fillna(False)
            
            # Periods with demand=0 but has_data=True -> true_zero (already set)
            # Periods with demand=0 and has_data=False -> missing_source
            df.loc[(df[target_col] == 0) & (~df['has_data']), 'demand_type'] = 'missing_source'
            
            df = df.drop(columns=['has_data'])
    
    return df


def aggregate_all_levels(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    source_df: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate demand at all configured levels.
    
    Args:
        df: Cleaned transactions DataFrame (must have order_week_start)
        config: Configuration dict
        source_df: Original transactions for missing source detection
    
    Returns:
        Dictionary mapping level name to aggregated DataFrame
    """
    if config is None:
        config = load_config()
    
    freq = config.get('frequency', 'W-MON')
    target_col = config['target']['column']
    date_col = 'order_week_start'
    levels = config.get('aggregation_levels', [])
    
    results = {}
    
    # 1. Overall
    if 'overall' in levels:
        logger.info("Aggregating overall demand...")
        overall = aggregate_with_duckdb(df, [], date_col, target_col, freq)
        overall = create_continuous_series(overall, [], date_col, target_col, freq)
        overall = flag_partial_final_period(overall, date_col)
        overall = separate_zeros_from_missing(overall, [], date_col, target_col, source_df, 'order_date')
        results['overall'] = overall
    
    # 2. Category
    if 'category' in levels and 'category_id' in df.columns:
        logger.info("Aggregating by category...")
        cat = aggregate_with_duckdb(df, ['category_id'], date_col, target_col, freq)
        cat = create_continuous_series(cat, ['category_id'], date_col, target_col, freq)
        cat = flag_partial_final_period(cat, date_col)
        cat = separate_zeros_from_missing(cat, ['category_id'], date_col, target_col, source_df, 'order_date')
        results['category'] = cat
    
    # 3. Market
    if 'market' in levels and 'market' in df.columns:
        logger.info("Aggregating by market...")
        mkt = aggregate_with_duckdb(df, ['market'], date_col, target_col, freq)
        mkt = create_continuous_series(mkt, ['market'], date_col, target_col, freq)
        mkt = flag_partial_final_period(mkt, date_col)
        mkt = separate_zeros_from_missing(mkt, ['market'], date_col, target_col, source_df, 'order_date')
        results['market'] = mkt
    
    # 4. Region (if available)
    if 'region' in levels and 'order_region' in df.columns:
        logger.info("Aggregating by region...")
        reg = aggregate_with_duckdb(df, ['order_region'], date_col, target_col, freq)
        reg = create_continuous_series(reg, ['order_region'], date_col, target_col, freq)
        reg = flag_partial_final_period(reg, date_col)
        reg = separate_zeros_from_missing(reg, ['order_region'], date_col, target_col, source_df, 'order_date')
        results['region'] = reg
    
    # 5. Category x Market
    if 'category_market' in levels and 'category_id' in df.columns and 'market' in df.columns:
        logger.info("Aggregating by category x market...")
        cm = aggregate_with_duckdb(df, ['category_id', 'market'], date_col, target_col, freq)
        cm = create_continuous_series(cm, ['category_id', 'market'], date_col, target_col, freq)
        cm = flag_partial_final_period(cm, date_col)
        cm = separate_zeros_from_missing(cm, ['category_id', 'market'], date_col, target_col, source_df, 'order_date')
        results['category_market'] = cm
    
    return results


def save_aggregated_results(
    results: Dict[str, pd.DataFrame],
    config: dict
) -> Dict[str, Path]:
    """Save all aggregated results to processed directory."""
    saved_paths = {}
    
    for level, df in results.items():
        filename = f"demand_{level}.parquet"
        path = get_processed_path(config, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        saved_paths[level] = path
        logger.info(f"Saved {level}: {len(df)} rows to {path}")
    
    return saved_paths


def run_aggregation_pipeline(
    cleaned_path: Optional[Path] = None,
    config: Optional[dict] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run the full aggregation pipeline.
    
    Args:
        cleaned_path: Path to cleaned transactions parquet
        config: Configuration dict
    
    Returns:
        Dictionary of aggregated DataFrames
    """
    if config is None:
        config = load_config()
    
    if cleaned_path is None:
        cleaned_path = get_interim_path(config, "cleaned_transactions.parquet")
    
    logger.info(f"Loading cleaned data from {cleaned_path}")
    df = pd.read_parquet(cleaned_path)
    
    # Keep source for missing source detection
    source_df = df.copy()
    
    # Ensure order_week_start exists
    if 'order_week_start' not in df.columns:
        if 'order_date' in df.columns:
            df['order_week_start'] = df['order_date'].dt.to_period('W-MON').dt.start_time
        else:
            raise ValueError("No order_date or order_week_start column found")
    
    # Run aggregation
    results = aggregate_all_levels(df, config, source_df)
    
    # Save
    saved_paths = save_aggregated_results(results, config)
    logger.info(f"Aggregation complete. Saved to: {saved_paths}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingestion import load_raw_data
    from cleaning import clean_data, save_interim
    transactions, _ = load_raw_data()
    cleaned = clean_data(transactions)
    interim_path = save_interim(cleaned, load_config())
    results = run_aggregation_pipeline(interim_path)
    for level, df in results.items():
        print(f"\n{level}: {df.shape}")
        print(df.head())