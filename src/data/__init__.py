"""
Data Engineering Package for DemandCast.

Modules:
- ingestion: Load raw data with correct encoding
- cleaning: Fix dtypes, parse dates, handle missing, drop PII
- validation: Data quality checks
- aggregation: DuckDB aggregation at multiple levels
- pipeline: Main ETL entry point
"""

from .ingestion import load_raw_data, load_transactions, load_data_dictionary, normalize_column_names
from .cleaning import clean_data, parse_dates, fix_dtypes, drop_pii_and_ids
from .validation import validate_data, DataValidator
from .aggregation import run_aggregation_pipeline, aggregate_all_levels
from .pipeline import run_pipeline

__all__ = [
    'load_raw_data',
    'load_transactions',
    'load_data_dictionary',
    'normalize_column_names',
    'clean_data',
    'parse_dates',
    'fix_dtypes',
    'drop_pii_and_ids',
    'validate_data',
    'DataValidator',
    'run_aggregation_pipeline',
    'aggregate_all_levels',
    'run_pipeline',
]