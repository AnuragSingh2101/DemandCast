"""
Data Ingestion Module for DemandCast.

Loads raw CSV data with correct encoding, normalizes column names,
and performs initial validation.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Go up from src/data/ to project root
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_raw_file_paths(config: dict) -> tuple[Path, Path]:
    """Get paths to raw data files."""
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / config['paths']['raw_data']
    transactions_file = raw_dir / config['raw_files']['transactions']
    dict_file = raw_dir / config['raw_files']['data_dictionary']
    return transactions_file, dict_file


def load_transactions(
    file_path: Path,
    encoding: str = 'iso-8859-1',
    **kwargs
) -> pd.DataFrame:
    """
    Load transactions CSV with specified encoding.
    
    Args:
        file_path: Path to the CSV file
        encoding: Character encoding (default: iso-8859-1 for DataCo)
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame with raw transaction data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If encoding is incorrect
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {file_path}")
    
    logger.info(f"Loading transactions from {file_path} with encoding={encoding}")
    
    # Default read_csv parameters
    defaults = {
        'low_memory': False,
        'parse_dates': False,  # We'll parse dates manually after cleaning
    }
    defaults.update(kwargs)
    
    df = pd.read_csv(file_path, encoding=encoding, **defaults)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def load_data_dictionary(
    file_path: Path,
    encoding: str = 'iso-8859-1'
) -> pd.DataFrame:
    """
    Load data dictionary CSV.
    
    Args:
        file_path: Path to the dictionary CSV file
        encoding: Character encoding
    
    Returns:
        DataFrame with field descriptions
    """
    if not file_path.exists():
        logger.warning(f"Data dictionary not found: {file_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading data dictionary from {file_path}")
    df = pd.read_csv(file_path, encoding=encoding)
    logger.info(f"Loaded dictionary with {len(df)} fields")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to snake_case for consistency.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    
    # Create mapping from original to normalized names
    def normalize(name: str) -> str:
        # Replace spaces and special chars with underscore
        name = name.strip()
        name = name.replace(' ', '_')
        name = name.replace('(', '').replace(')', '')
        name = name.replace('/', '_')
        name = name.replace('-', '_')
        name = name.replace('.', '_')
        # Handle multiple underscores
        while '__' in name:
            name = name.replace('__', '_')
        name = name.strip('_')
        return name.lower()
    
    mapping = {col: normalize(col) for col in df.columns}
    df = df.rename(columns=mapping)
    
    logger.info(f"Normalized {len(mapping)} column names")
    logger.debug(f"Column mapping: {mapping}")
    
    return df


def inspect_data(df: pd.DataFrame, name: str = "dataset") -> dict:
    """
    Perform basic inspection of the DataFrame.
    
    Args:
        df: DataFrame to inspect
        name: Name for logging
    
    Returns:
        Dictionary with inspection results
    """
    info = {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_counts': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    
    logger.info(f"{name} inspection: {info['shape'][0]:,} rows x {info['shape'][1]} cols, "
                f"{info['duplicate_rows']} duplicates, {info['memory_usage_mb']} MB")
    
    return info


def load_raw_data(config: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point to load all raw data.
    
    Args:
        config: Optional config dict (loads from file if not provided)
    
    Returns:
        Tuple of (transactions_df, dictionary_df)
    """
    if config is None:
        config = load_config()
    
    transactions_file, dict_file = get_raw_file_paths(config)
    
    # Load transactions
    transactions = load_transactions(transactions_file)
    
    # Load dictionary
    dictionary = load_data_dictionary(dict_file)
    
    # Normalize column names
    transactions = normalize_column_names(transactions)
    if len(dictionary) > 0:
        dictionary = normalize_column_names(dictionary)
    
    # Inspect
    inspect_data(transactions, "transactions")
    if len(dictionary) > 0:
        inspect_data(dictionary, "data_dictionary")
    
    return transactions, dictionary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    transactions, dictionary = load_raw_data()
    print(f"\nTransactions shape: {transactions.shape}")
    print(f"Columns: {list(transactions.columns)}")
    if len(dictionary) > 0:
        print(f"\nDictionary shape: {dictionary.shape}")
        print(dictionary.head())