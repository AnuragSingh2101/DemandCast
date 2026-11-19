"""
Unit tests for the validation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.validation import DataValidator, validate_data


class TestDataValidator:
    """Tests for the DataValidator class."""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'data_quality': {
                'max_missing_pct': 0.05,
                'max_duplicate_pct': 0.01,
                'min_observations_per_series': 26,
            },
            'aggregation_levels': ['overall', 'category', 'market', 'category_market'],
            'target': {'column': 'order_item_quantity'},
            'frequency': 'W-MON',
        }
    
    @pytest.fixture
    def clean_df(self):
        """Create a clean test DataFrame."""
        dates = pd.date_range('2018-01-01', periods=100, freq='W-MON')
        np.random.seed(42)
        return pd.DataFrame({
            'order_week_start': dates,
            'category_id': np.random.randint(1, 5, 100),
            'market': np.random.choice(['US', 'EU', 'APAC'], 100),
            'order_item_quantity': np.random.randint(1, 6, 100),
            'order_date': pd.date_range('2018-01-01', periods=100, freq='D'),
        })
    
    def test_check_schema_passes(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_schema(clean_df, list(clean_df.columns))
        assert result['passed'] is True
        assert result['missing_columns'] == []
    
    def test_check_schema_fails_missing(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_schema(clean_df, ['order_week_start', 'missing_col'])
        assert result['passed'] is False
        assert 'missing_col' in result['missing_columns']
    
    def test_check_missing_values_passes(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_missing_values(clean_df)
        assert result['passed'] is True
        assert len(result['high_missing_columns']) == 0
    
    def test_check_missing_values_fails(self, sample_config):
        df = pd.DataFrame({
            'a': [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            'b': [1, 2, 3, 4, 5],
        })
        validator = DataValidator(sample_config)
        result = validator.check_missing_values(df)
        assert result['passed'] is False
        assert 'a' in result['high_missing_columns']
    
    def test_check_duplicates_passes(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_duplicates(clean_df)
        assert result['passed'] == True
        assert result['duplicate_count'] == 0
    
    def test_check_duplicates_fails(self, sample_config):
        df = pd.DataFrame({
            'a': [1, 1, 2, 3],  # 25% duplicates
            'b': ['x', 'x', 'y', 'z'],
        })
        validator = DataValidator(sample_config)
        result = validator.check_duplicates(df)
        assert result['passed'] == False
        assert result['duplicate_count'] == 1
    
    def test_check_date_continuity_passes(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_date_continuity(clean_df)
        assert result['passed'] is True
        assert result['coverage_pct'] == 100.0
    
    def test_check_date_continuity_fails_gaps(self, sample_config):
        # Create data with gaps
        dates = pd.date_range('2018-01-01', periods=10, freq='W-MON')
        # Remove some dates to create gaps
        dates = dates[[0, 1, 3, 4, 6, 7, 8, 9]]
        df = pd.DataFrame({
            'order_date': dates,
            'order_item_quantity': [1] * 8,
        })
        validator = DataValidator(sample_config)
        result = validator.check_date_continuity(df, 'order_date')
        assert result['passed'] is False
        assert result['coverage_pct'] < 95
    
    def test_check_target_distribution(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_target_distribution(clean_df)
        assert result['passed'] is True
        assert result['count'] == 100
        assert result['zero_count'] == 0
        assert result['negative_count'] == 0
    
    def test_check_target_distribution_with_zeros(self, sample_config):
        df = pd.DataFrame({
            'order_item_quantity': [0, 1, 2, 0, 3],
        })
        validator = DataValidator(sample_config)
        result = validator.check_target_distribution(df)
        assert result['zero_count'] == 2
    
    def test_check_aggregation_levels(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_aggregation_levels(clean_df)
        assert 'category' in result
        assert 'market' in result
        assert 'category_market' in result
    
    def test_check_pii_removed_passes(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        result = validator.check_pii_removed(clean_df)
        assert result['passed'] is True
        assert result['pii_columns_found'] == []
    
    def test_check_pii_removed_fails(self, sample_config):
        df = pd.DataFrame({
            'customer_email': ['a@b.com', 'c@d.com'],
            'order_item_quantity': [1, 2],
        })
        validator = DataValidator(sample_config)
        result = validator.check_pii_removed(df)
        assert result['passed'] is False
        assert 'customer_email' in result['pii_columns_found']
    
    def test_run_all_checks(self, sample_config, clean_df):
        validator = DataValidator(sample_config)
        summary = validator.run_all_checks(clean_df)
        
        assert 'timestamp' in summary
        assert 'total_rows' in summary
        assert 'total_columns' in summary
        assert 'overall_passed' in summary
        assert 'checks' in summary
        assert len(summary['checks']) > 0


class TestValidateData:
    """Tests for the validate_data convenience function."""
    
    def test_validate_data_returns_summary(self):
        df = pd.DataFrame({
            'order_week_start': pd.date_range('2018-01-01', periods=10, freq='W-MON'),
            'category_id': [1, 2] * 5,
            'market': ['US', 'EU'] * 5,
            'order_item_quantity': np.random.randint(1, 6, 10),
            'order_date': pd.date_range('2018-01-01', periods=10, freq='D'),
        })
        
        config = {
            'paths': {
                'reports_dir': 'reports',
            },
            'data_quality': {
                'max_missing_pct': 0.05,
                'max_duplicate_pct': 0.01,
                'min_observations_per_series': 5,
            },
            'aggregation_levels': ['overall', 'category', 'market'],
            'target': {'column': 'order_item_quantity'},
            'frequency': 'W-MON',
        }
        
        summary = validate_data(df, config)
        assert 'overall_passed' in summary
        assert 'checks' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])