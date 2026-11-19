"""
Unit tests for the ingestion module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data.ingestion import (
    load_transactions,
    load_data_dictionary,
    normalize_column_names,
    inspect_data,
    load_raw_data,
)


class TestNormalizeColumnNames:
    """Tests for column name normalization."""
    
    def test_basic_normalization(self):
        df = pd.DataFrame({
            'Order Date': [1, 2],
            'Customer Name': ['a', 'b'],
            'Order-ID': [10, 20],
        })
        result = normalize_column_names(df)
        assert list(result.columns) == ['order_date', 'customer_name', 'order_id']
    
    def test_special_characters(self):
        df = pd.DataFrame({
            'Order (Date)': [1],
            'Customer/Name': ['a'],
            'Price.USD': [10.0],
        })
        result = normalize_column_names(df)
        assert list(result.columns) == ['order_date', 'customer_name', 'price_usd']
    
    def test_multiple_underscores(self):
        df = pd.DataFrame({
            'Order__Date': [1],
            'Customer___Name': ['a'],
        })
        result = normalize_column_names(df)
        assert list(result.columns) == ['order_date', 'customer_name']
    
    def test_leading_trailing_spaces(self):
        df = pd.DataFrame({
            '  Order Date  ': [1],
        })
        result = normalize_column_names(df)
        assert list(result.columns) == ['order_date']
    
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = normalize_column_names(df)
        assert len(result.columns) == 0


class TestInspectData:
    """Tests for data inspection."""
    
    def test_basic_inspection(self):
        df = pd.DataFrame({
            'a': range(3000),  # Unique values
            'b': [f'x{i}' for i in range(3000)],  # Unique values
        })
        info = inspect_data(df, "test")
        assert info['shape'] == (3000, 2)
        assert info['columns'] == ['a', 'b']
        assert info['duplicate_rows'] == 0
        assert info['memory_usage_mb'] > 0
    
    def test_with_missing_values(self):
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': ['x', 'y', np.nan],
        })
        info = inspect_data(df, "test")
        assert info['missing_counts']['a'] == 1
        assert info['missing_counts']['b'] == 1
        assert info['missing_pct']['a'] == 33.33
    
    def test_with_duplicates(self):
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': ['x', 'x', 'y'],
        })
        info = inspect_data(df, "test")
        assert info['duplicate_rows'] == 1


class TestLoadTransactions:
    """Tests for loading transactions CSV."""
    
    def test_load_valid_csv(self):
        import tempfile
        import os
        # Create a test CSV file
        csv_content = """col1,col2,col3
1,a,10.5
2,b,20.3
3,c,30.1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            file_path = Path(f.name)
        
        try:
            df = load_transactions(file_path)
            assert len(df) == 3
            assert list(df.columns) == ['col1', 'col2', 'col3']
        finally:
            os.unlink(file_path)
    
    def test_load_with_different_encoding(self):
        import tempfile
        import os
        # Create a test CSV with special characters
        csv_content = "col1,col2\n1,café\n2,naïve\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            file_path = Path(f.name)
        
        try:
            df = load_transactions(file_path, encoding='utf-8')
            assert len(df) == 2
            assert df['col2'].iloc[0] == 'café'
        finally:
            os.unlink(file_path)
    
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_transactions(Path("/nonexistent/file.csv"))


class TestLoadDataDictionary:
    """Tests for loading data dictionary."""
    
    def test_load_valid_dictionary(self):
        import tempfile
        import os
        csv_content = """field,description
order_id,Unique order identifier
customer_id,Customer identifier
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            file_path = Path(f.name)
        
        try:
            df = load_data_dictionary(file_path)
            assert len(df) == 2
            assert 'field' in df.columns
            assert 'description' in df.columns
        finally:
            os.unlink(file_path)
    
    def test_missing_file_returns_empty(self):
        df = load_data_dictionary(Path("/nonexistent/dict.csv"))
        assert len(df) == 0


class TestLoadRawData:
    """Tests for the main load_raw_data function."""
    
    def test_load_both_files(self):
        # This is a placeholder test - actual integration testing
        # would require mocking the config loading
        assert True  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])