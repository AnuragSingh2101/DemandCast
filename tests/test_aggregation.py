"""
Unit tests for the aggregation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.aggregation import (
    build_continuous_index,
    create_continuous_series,
    flag_partial_final_period,
    separate_zeros_from_missing,
)


class TestBuildContinuousIndex:
    """Tests for building continuous date index."""
    
    def test_weekly_index(self):
        start = pd.Timestamp('2018-01-01')
        end = pd.Timestamp('2018-01-31')
        index = build_continuous_index(start, end, 'W-MON')
        
        assert len(index) == 5  # 5 Mondays in Jan 2018
        assert index[0] == pd.Timestamp('2018-01-01')
        assert index[-1] == pd.Timestamp('2018-01-29')
        assert index.freq == 'W-MON'
    
    def test_daily_index(self):
        start = pd.Timestamp('2018-01-01')
        end = pd.Timestamp('2018-01-05')
        index = build_continuous_index(start, end, 'D')
        
        assert len(index) == 5
        assert index.freq == 'D'
    
    def test_monthly_index(self):
        start = pd.Timestamp('2018-01-01')
        end = pd.Timestamp('2018-12-31')
        index = build_continuous_index(start, end, 'MS')
        
        assert len(index) == 12
        assert index.freq == 'MS'


class TestCreateContinuousSeries:
    """Tests for creating continuous time series."""
    
    def test_overall_series(self):
        # Create sample data with 5 data points over 9 weeks
        dates = pd.date_range('2018-01-01', periods=9, freq='W-MON')
        overall = pd.DataFrame({
            'order_week_start': dates[[0, 2, 4, 6, 8]],
            'order_item_quantity': [10, 20, 15, 25, 30],
        })
        result = create_continuous_series(overall, [], 'order_week_start', 'order_item_quantity', 'W-MON')
        
        assert len(result) == 9  # All 9 weeks in range
        assert result['order_item_quantity'].sum() == 100
        assert (result['order_item_quantity'] >= 0).all()
    
    def test_category_series(self):
        dates = pd.date_range('2018-01-01', periods=9, freq='W-MON')
        df = pd.DataFrame({
            'order_week_start': dates[[0, 2, 4, 6, 8]],
            'category_id': [1, 1, 1, 1, 1],
            'order_item_quantity': [10, 20, 15, 25, 30],
        })
        
        result = create_continuous_series(
            df, 
            ['category_id'], 
            'order_week_start', 
            'order_item_quantity', 
            'W-MON'
        )
        
        assert len(result) == 9  # 9 weeks x 1 category
        assert 'category_id' in result.columns
        assert result['category_id'].iloc[0] == 1
    
    def test_multiple_categories(self):
        dates = pd.date_range('2018-01-01', periods=4, freq='W-MON')
        df = pd.DataFrame({
            'order_week_start': dates.tolist() * 2,
            'category_id': [1, 1, 1, 1, 2, 2, 2, 2],
            'order_item_quantity': [10, 20, 15, 25, 5, 10, 8, 12],
        })
        
        result = create_continuous_series(df, ['category_id'], 'order_week_start', 'order_item_quantity', 'W-MON')
        
        assert len(result) == 8  # 4 weeks x 2 categories
        assert set(result['category_id'].unique()) == {1, 2}
    
    def test_multi_group_columns(self):
        """Test with multiple group columns (category + market)."""
        dates = pd.date_range('2018-01-01', periods=4, freq='W-MON')
        df = pd.DataFrame({
            'order_week_start': dates.tolist() * 4,
            'category_id': [1, 1, 1, 1] * 2 + [2, 2, 2, 2] * 2,
            'market': ['US', 'US', 'US', 'US', 'EU', 'EU', 'EU', 'EU'] * 2,
            'order_item_quantity': range(16),
        })
        
        result = create_continuous_series(df, ['category_id', 'market'], 'order_week_start', 'order_item_quantity', 'W-MON')
        
        assert len(result) == 16  # 4 weeks x 4 groups
        assert 'category_id' in result.columns
        assert 'market' in result.columns
        assert result['category_id'].dtype == 'int64'
        # StringDtype is used in pandas 2.x, check for string type
        assert pd.api.types.is_string_dtype(result['market']) or result['market'].dtype == 'object'


class TestFlagPartialFinalPeriod:
    """Tests for flagging partial final period."""
    
    def test_flags_final_period(self):
        dates = pd.date_range('2018-01-01', periods=4, freq='W-MON')
        df = pd.DataFrame({
            'order_week_start': dates,
            'order_item_quantity': [10, 20, 15, 25],
        })
        
        result = flag_partial_final_period(df)
        
        assert 'is_final_period' in result.columns
        assert 'is_partial_final' in result.columns
        assert result['is_final_period'].sum() == 1
        assert result['is_final_period'].iloc[-1] == True
        assert result['is_partial_final'].all() == False  # Default to False


class TestSeparateZerosFromMissing:
    """Tests for separating true zeros from missing source data."""
    
    def test_marks_true_zeros(self):
        df = pd.DataFrame({
            'order_week_start': pd.date_range('2018-01-01', periods=4, freq='W-MON'),
            'category_id': [1, 1, 1, 1],
            'order_item_quantity': [10, 0, 15, 0],
        })
        
        result = separate_zeros_from_missing(
            df, 
            ['category_id'], 
            'order_week_start', 
            'order_item_quantity'
        )
        
        assert 'demand_type' in result.columns
        assert result.loc[1, 'demand_type'] == 'true_zero'  # index 1 has 0
        assert result.loc[3, 'demand_type'] == 'true_zero'  # index 3 has 0
        assert result.loc[0, 'demand_type'] == 'observed'
        assert result.loc[2, 'demand_type'] == 'observed'
    
    def test_marks_missing_source_with_source_df(self):
        # Continuous series has a period with 0 demand
        df = pd.DataFrame({
            'order_week_start': pd.date_range('2018-01-01', periods=4, freq='W-MON'),
            'category_id': [1, 1, 1, 1],
            'order_item_quantity': [10, 0, 15, 0],
        })
        
        # Source data only has transactions for weeks 0, 2 (not 1, 3)
        source_df = pd.DataFrame({
            'order_date': pd.to_datetime(['2018-01-01', '2018-01-15']),
            'category_id': [1, 1],
            'order_item_quantity': [10, 15],
        })
        
        result = separate_zeros_from_missing(
            df, 
            ['category_id'], 
            'order_week_start', 
            'order_item_quantity',
            source_df=source_df,
            source_date_col='order_date'
        )
        
        # Week 1 (index 1): demand=0, no source data -> missing_source
        # Week 3 (index 3): demand=0, no source data -> missing_source
        assert result.loc[1, 'demand_type'] == 'missing_source'
        assert result.loc[3, 'demand_type'] == 'missing_source'
    
    def test_observed_when_source_exists(self):
        df = pd.DataFrame({
            'order_week_start': pd.date_range('2018-01-01', periods=2, freq='W-MON'),
            'category_id': [1, 1],
            'order_item_quantity': [10, 5],
        })
        
        source_df = pd.DataFrame({
            'order_date': pd.to_datetime(['2018-01-01', '2018-01-08']),
            'category_id': [1, 1],
            'order_item_quantity': [10, 5],
        })
        
        result = separate_zeros_from_missing(
            df, 
            ['category_id'], 
            'order_week_start', 
            'order_item_quantity',
            source_df=source_df,
            source_date_col='order_date'
        )
        
        assert result.loc[0, 'demand_type'] == 'observed'
        assert result.loc[1, 'demand_type'] == 'observed'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])