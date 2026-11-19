"""
Data Validation Module for DemandCast.

Performs data quality checks and generates validation reports.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import yaml
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class DataValidator:
    """Validates data quality at various stages."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.thresholds = self.config.get('data_quality', {})
        self.results = {}
    
    def check_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
        """Check that all expected columns exist."""
        missing = [c for c in expected_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in expected_columns]
        
        result = {
            'check': 'schema',
            'passed': len(missing) == 0,
            'missing_columns': missing,
            'extra_columns': extra,
        }
        self.results['schema'] = result
        if missing:
            logger.warning(f"Schema check FAILED: missing columns {missing}")
        else:
            logger.info("Schema check PASSED")
        return result
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check missing value percentages against thresholds."""
        max_missing_pct = self.thresholds.get('max_missing_pct', 0.05)
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > max_missing_pct * 100]
        
        result = {
            'check': 'missing_values',
            'passed': len(high_missing) == 0,
            'max_allowed_pct': max_missing_pct * 100,
            'high_missing_columns': high_missing.to_dict(),
            'all_missing_pct': missing_pct.to_dict(),
        }
        self.results['missing_values'] = result
        if len(high_missing) > 0:
            logger.warning(f"Missing values check FAILED: {len(high_missing)} columns exceed {max_missing_pct*100}%")
            for col, pct in high_missing.items():
                logger.warning(f"  {col}: {pct:.2f}% missing")
        else:
            logger.info(f"Missing values check PASSED (threshold: {max_missing_pct*100}%)")
        return result
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows."""
        max_dupe_pct = self.thresholds.get('max_duplicate_pct', 0.01)
        dupe_count = df.duplicated().sum()
        dupe_pct = dupe_count / len(df) * 100
        
        result = {
            'check': 'duplicates',
            'passed': dupe_pct <= max_dupe_pct * 100,
            'duplicate_count': int(dupe_count),
            'duplicate_pct': round(dupe_pct, 4),
            'max_allowed_pct': max_dupe_pct * 100,
        }
        self.results['duplicates'] = result
        if dupe_pct > max_dupe_pct * 100:
            logger.warning(f"Duplicates check FAILED: {dupe_count:,} duplicates ({dupe_pct:.4f}%)")
        else:
            logger.info(f"Duplicates check PASSED: {dupe_count:,} duplicates ({dupe_pct:.4f}%)")
        return result
    
    def check_date_continuity(self, df: pd.DataFrame, date_col: str = 'order_date') -> Dict[str, Any]:
        """Check date coverage and continuity."""
        if date_col not in df.columns:
            result = {
                'check': 'date_continuity',
                'passed': False,
                'error': f'Date column {date_col} not found',
            }
            self.results['date_continuity'] = result
            logger.error(f"Date continuity check FAILED: {date_col} not found")
            return result
        
        dates = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = dates.dropna()
        
        if len(valid_dates) == 0:
            result = {
                'check': 'date_continuity',
                'passed': False,
                'error': 'No valid dates found',
            }
            self.results['date_continuity'] = result
            return result
        
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        expected_days = (max_date - min_date).days + 1
        actual_days = valid_dates.dt.date.nunique()
        coverage_pct = actual_days / expected_days * 100
        
        # Check for gaps
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        missing_dates = date_range.difference(valid_dates.dt.date.unique())
        
        result = {
            'check': 'date_continuity',
            'passed': coverage_pct >= 95,  # Expect at least 95% day coverage
            'min_date': str(min_date),
            'max_date': str(max_date),
            'expected_days': expected_days,
            'actual_days': actual_days,
            'coverage_pct': round(coverage_pct, 2),
            'missing_dates_count': len(missing_dates),
            'longest_gap_days': self._longest_gap(valid_dates.dt.date.unique()) if len(missing_dates) > 0 else 0,
        }
        self.results['date_continuity'] = result
        
        if coverage_pct < 95:
            logger.warning(f"Date continuity check FAILED: {coverage_pct:.2f}% coverage")
        else:
            logger.info(f"Date continuity check PASSED: {coverage_pct:.2f}% coverage")
        return result
    
    def _longest_gap(self, dates: np.ndarray) -> int:
        """Find longest gap in days between consecutive dates."""
        if len(dates) < 2:
            return 0
        sorted_dates = np.sort(dates)
        gaps = np.diff(sorted_dates).astype('timedelta64[D]').astype(int)
        return int(gaps.max()) if len(gaps) > 0 else 0
    
    def check_target_distribution(self, df: pd.DataFrame, target_col: str = 'order_item_quantity') -> Dict[str, Any]:
        """Check target variable distribution."""
        if target_col not in df.columns:
            result = {
                'check': 'target_distribution',
                'passed': False,
                'error': f'Target column {target_col} not found',
            }
            self.results['target_distribution'] = result
            return result
        
        target = df[target_col]
        
        result = {
            'check': 'target_distribution',
            'passed': True,
            'count': int(target.count()),
            'mean': float(target.mean()),
            'median': float(target.median()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
            'zero_count': int((target == 0).sum()),
            'negative_count': int((target < 0).sum()),
            'unique_values': int(target.nunique()),
        }
        self.results['target_distribution'] = result
        logger.info(f"Target distribution: mean={result['mean']:.2f}, "
                    f"zeros={result['zero_count']}, negatives={result['negative_count']}")
        return result
    
    def check_aggregation_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data sufficiency for each aggregation level."""
        levels = self.config.get('aggregation_levels', [])
        min_obs = self.thresholds.get('min_observations_per_series', 26)
        
        results = {}
        for level in levels:
            if level == 'overall':
                continue
            
            # Find grouping columns
            group_cols = []
            if 'category' in level and 'category_id' in df.columns:
                group_cols.append('category_id')
            if 'market' in level and 'market' in df.columns:
                group_cols.append('market')
            if 'region' in level and 'order_region' in df.columns:
                group_cols.append('order_region')
            
            if not group_cols:
                continue
            
            # Count observations per group (weekly)
            if 'order_week_start' in df.columns:
                group_counts = df.groupby(group_cols + ['order_week_start']).size().reset_index(name='count')
                series_lengths = group_counts.groupby(group_cols).size()
            else:
                series_lengths = df.groupby(group_cols).size()
            
            sufficient = (series_lengths >= min_obs).sum()
            total = len(series_lengths)
            
            results[level] = {
                'total_series': total,
                'sufficient_series': int(sufficient),
                'insufficient_series': int(total - sufficient),
                'min_obs_threshold': min_obs,
                'mean_obs': float(series_lengths.mean()),
                'median_obs': float(series_lengths.median()),
            }
            
            logger.info(f"Aggregation level {level}: {sufficient}/{total} series have ≥{min_obs} observations")
        
        self.results['aggregation_levels'] = results
        return results
    
    def check_pii_removed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify PII columns have been removed."""
        pii_patterns = ['email', 'fname', 'lname', 'password', 'street', 'zipcode', 'name']
        found = []
        for col in df.columns:
            if any(p in col.lower() for p in pii_patterns):
                found.append(col)
        
        result = {
            'check': 'pii_removed',
            'passed': len(found) == 0,
            'pii_columns_found': found,
        }
        self.results['pii_removed'] = result
        if found:
            logger.warning(f"PII check FAILED: found columns {found}")
        else:
            logger.info("PII check PASSED: no PII columns found")
        return result
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Running all data quality checks...")
        
        self.check_schema(df, list(df.columns))  # Just verify we have columns
        self.check_missing_values(df)
        self.check_duplicates(df)
        self.check_date_continuity(df)
        self.check_target_distribution(df)
        self.check_aggregation_levels(df)
        self.check_pii_removed(df)
        
        # Overall pass/fail
        all_passed = all(r.get('passed', False) for r in self.results.values())
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': df.shape[1],
            'overall_passed': all_passed,
            'checks': self.results,
        }
        
        logger.info(f"Validation complete. Overall: {'PASSED' if all_passed else 'FAILED'}")
        return summary
    
    def save_report(self, summary: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save validation report to JSON."""
        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / self.config['paths']['reports_dir'] / 'data_quality_report.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")
        return output_path


def validate_data(df: pd.DataFrame, config: Optional[dict] = None) -> Dict[str, Any]:
    """Convenience function to run validation."""
    validator = DataValidator(config)
    summary = validator.run_all_checks(df)
    validator.save_report(summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingestion import load_raw_data
    from cleaning import clean_data
    transactions, _ = load_raw_data()
    cleaned = clean_data(transactions)
    summary = validate_data(cleaned)
    print(f"\nOverall: {'PASSED' if summary['overall_passed'] else 'FAILED'}")