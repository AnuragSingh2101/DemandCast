"""
Unit tests for the cleaning module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.cleaning import (
    parse_dates,
    fix_dtypes,
    handle_missing_values,
    remove_duplicates,
    drop_pii_and_ids,
    validate_numeric_ranges,
    add_derived_features,
    clean_data,
    PII_COLUMNS,
    ID_COLUMNS,
)


class TestParseDates:
    """Tests for date parsing."""
    
    def test_parse_standard_format(self):
        df = pd.DataFrame({
            'order_date_dateorders': ['1/15/2018 10:30', '2/20/2018 14:45'],
        })
        result = parse_dates(df)
        assert 'order_date' in result.columns
        # pandas 2.x uses datetime64[us], check for datetime64
        assert pd.api.types.is_datetime64_any_dtype(result['order_date'])
        assert result['order_date'].iloc[0] == pd.Timestamp('2018-01-15 10:30:00')
    
    def test_parse_shipping_date(self):
        df = pd.DataFrame({
            'shipping_date_dateorders': ['1/15/2018 10:30'],
        })
        result = parse_dates(df)
        assert 'shipping_date' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['shipping_date'])
    
    def test_invalid_dates_become_nat(self):
        df = pd.DataFrame({
            'order_date_dateorders': ['invalid', '1/15/2018 10:30'],
        })
        result = parse_dates(df)
        assert pd.isna(result['order_date'].iloc[0])
        assert not pd.isna(result['order_date'].iloc[1])
    
    def test_missing_column_no_error(self):
        df = pd.DataFrame({'other_col': [1, 2]})
        result = parse_dates(df)
        assert 'order_date' not in result.columns


class TestFixDtypes:
    """Tests for dtype fixing."""
    
    def test_convert_int_columns(self):
        df = pd.DataFrame({
            'order_item_quantity': [1, 2, 3],
            'days_for_shipping_real': [3, 4, 5],
        })
        result = fix_dtypes(df)
        assert result['order_item_quantity'].dtype == 'int64'
        assert result['days_for_shipping_real'].dtype == 'int64'
    
    def test_convert_float_columns(self):
        df = pd.DataFrame({
            'order_item_discount': [10.5, 20.3],
            'order_item_product_price': [100.0, 200.0],
        })
        result = fix_dtypes(df)
        assert result['order_item_discount'].dtype == 'float64'
        assert result['order_item_product_price'].dtype == 'float64'
    
    def test_convert_categorical_columns(self):
        df = pd.DataFrame({
            'market': ['US', 'EU', 'US'],
            'delivery_status': ['Late', 'On Time', 'Late'],
        })
        result = fix_dtypes(df)
        assert result['market'].dtype.name == 'category'
        assert result['delivery_status'].dtype.name == 'category'
    
    def test_nullable_int_for_nan(self):
        df = pd.DataFrame({
            'category_id': [1, np.nan, 3],
        })
        result = fix_dtypes(df)
        assert result['category_id'].dtype.name == 'Int64'


class TestHandleMissingValues:
    """Tests for missing value handling."""
    
    def test_reports_missing(self, caplog):
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.nan, np.nan, 3],
        })
        result = handle_missing_values(df)
        # Function should log missing values but not modify data
        assert len(result) == 3


class TestRemoveDuplicates:
    """Tests for duplicate removal."""
    
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': ['x', 'x', 'y'],
        })
        result = remove_duplicates(df)
        assert len(result) == 2
        assert result['a'].tolist() == [1, 2]
    
    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
        })
        result = remove_duplicates(df)
        assert len(result) == 3


class TestDropPIIAndIds:
    """Tests for PII and ID column dropping."""
    
    def test_drops_pii_columns(self):
        df = pd.DataFrame({
            'customer_email': ['a@b.com', 'c@d.com'],
            'customer_fname': ['John', 'Jane'],
            'order_item_quantity': [1, 2],
            'market': ['US', 'EU'],
        })
        result = drop_pii_and_ids(df, drop_ids=False)
        assert 'customer_email' not in result.columns
        assert 'customer_fname' not in result.columns
        assert 'order_item_quantity' in result.columns
        assert 'market' in result.columns
    
    def test_drops_id_columns(self):
        df = pd.DataFrame({
            'order_item_id': [1, 2, 3],
            'benefit_per_order': [10.0, 20.0, 30.0],
            'order_item_quantity': [1, 2, 3],
        })
        result = drop_pii_and_ids(df, drop_ids=True)
        assert 'order_item_id' not in result.columns
        assert 'benefit_per_order' not in result.columns
        assert 'order_item_quantity' in result.columns
    
    def test_keeps_ids_when_false(self):
        df = pd.DataFrame({
            'order_item_id': [1, 2, 3],
            'order_item_quantity': [1, 2, 3],
        })
        result = drop_pii_and_ids(df, drop_ids=False)
        assert 'order_item_id' in result.columns
    
    def test_handles_missing_columns_gracefully(self):
        df = pd.DataFrame({
            'order_item_quantity': [1, 2],
        })
        result = drop_pii_and_ids(df, drop_ids=True)
        assert 'order_item_quantity' in result.columns


class TestValidateNumericRanges:
    """Tests for numeric range validation."""
    
    def test_logs_out_of_range(self, caplog):
        df = pd.DataFrame({
            'order_item_quantity': [1, 10, 3],  # 10 is out of range (1-5)
        })
        result = validate_numeric_ranges(df)
        # Should log warning but not modify data
        assert len(result) == 3
    
    def test_no_warning_for_valid_ranges(self, caplog):
        df = pd.DataFrame({
            'order_item_quantity': [1, 3, 5],
        })
        result = validate_numeric_ranges(df)
        assert len(result) == 3


class TestAddDerivedFeatures:
    """Tests for derived feature addition."""
    
    def test_adds_date_components(self):
        df = pd.DataFrame({
            'order_date': pd.to_datetime(['2018-01-15', '2018-02-20']),
        })
        result = add_derived_features(df)
        assert 'order_year' in result.columns
        assert 'order_month' in result.columns
        assert 'order_week' in result.columns
        assert 'order_day_of_week' in result.columns
        assert 'order_quarter' in result.columns
        assert 'order_week_start' in result.columns
        
        assert result['order_year'].iloc[0] == 2018
        assert result['order_month'].iloc[0] == 1
        assert result['order_quarter'].iloc[0] == 1
    
    def test_adds_shipping_delay(self):
        df = pd.DataFrame({
            'days_for_shipping_real': [5, 3],
            'days_for_shipment_scheduled': [3, 4],
        })
        result = add_derived_features(df)
        assert 'shipping_delay' in result.columns
        assert result['shipping_delay'].iloc[0] == 2
        assert result['shipping_delay'].iloc[1] == -1
    
    def test_adds_is_late(self):
        df = pd.DataFrame({
            'late_delivery_risk': [1, 0, 1],
        })
        result = add_derived_features(df)
        assert 'is_late' in result.columns
        assert result['is_late'].iloc[0] == True
        assert result['is_late'].iloc[1] == False


class TestCleanData:
    """Tests for the main clean_data pipeline."""
    
    def test_full_pipeline(self):
        df = pd.DataFrame({
            'order_date_dateorders': ['1/15/2018 10:30', '2/20/2018 14:45'],
            'shipping_date_dateorders': ['1/18/2018 10:30', '2/22/2018 14:45'],
            'order_item_quantity': [1, 2],
            'days_for_shipping_real': [3, 2],
            'days_for_shipment_scheduled': [4, 3],
            'late_delivery_risk': [0, 1],
            'category_id': [1, 2],
            'market': ['US', 'EU'],
            'delivery_status': ['On Time', 'Late'],
            'customer_email': ['a@b.com', 'c@d.com'],
            'order_item_id': [100, 101],
        })
        
        result = clean_data(df)
        
        # Check PII dropped
        assert 'customer_email' not in result.columns
        # Check ID dropped
        assert 'order_item_id' not in result.columns
        # Check dates parsed
        assert 'order_date' in result.columns
        assert 'shipping_date' in result.columns
        # Check derived features
        assert 'order_week_start' in result.columns
        assert 'shipping_delay' in result.columns
        assert 'is_late' in result.columns
        # Check dtypes
        assert result['order_item_quantity'].dtype == 'int64'
        assert result['market'].dtype.name == 'category'
        
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])