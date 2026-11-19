"""
Main ETL Pipeline for DemandCast.

Entry point: python -m src.data.pipeline

Orchestrates the full data engineering workflow:
1. Ingestion - Load raw CSV with correct encoding
2. Cleaning - Fix dtypes, parse dates, handle missing, drop PII
3. Validation - Data quality checks
4. Aggregation - DuckDB aggregation at multiple levels
5. Save processed Parquet outputs
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.ingestion import load_raw_data, normalize_column_names
from src.data.cleaning import clean_data, save_interim
from src.data.validation import validate_data
from src.data.aggregation import run_aggregation_pipeline, save_aggregated_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(config: Optional[dict] = None, skip_validation: bool = False) -> dict:
    """
    Run the complete ETL pipeline.
    
    Args:
        config: Configuration dict (loads from file if None)
        skip_validation: Skip data validation step (for testing)
    
    Returns:
        Dictionary with pipeline results and paths
    """
    if config is None:
        config = load_config()
    
    logger.info("=" * 60)
    logger.info("Starting DemandCast ETL Pipeline")
    logger.info("=" * 60)
    
    results = {
        'steps': {},
        'paths': {},
        'stats': {}
    }
    
    try:
        # Step 1: Ingestion
        logger.info("\n[1/5] INGESTION: Loading raw data...")
        transactions, dictionary = load_raw_data(config)
        results['steps']['ingestion'] = {
            'status': 'success',
            'rows': len(transactions),
            'columns': transactions.shape[1],
        }
        results['stats']['raw_rows'] = len(transactions)
        results['stats']['raw_columns'] = transactions.shape[1]
        
        # Step 2: Cleaning
        logger.info("\n[2/5] CLEANING: Fixing dtypes, parsing dates, dropping PII...")
        cleaned = clean_data(transactions, config)
        interim_path = save_interim(cleaned, config)
        results['steps']['cleaning'] = {
            'status': 'success',
            'rows': len(cleaned),
            'columns': cleaned.shape[1],
            'interim_path': str(interim_path),
        }
        results['stats']['cleaned_rows'] = len(cleaned)
        results['stats']['cleaned_columns'] = cleaned.shape[1]
        results['paths']['interim'] = str(interim_path)
        
        # Step 3: Validation
        if not skip_validation:
            logger.info("\n[3/5] VALIDATION: Running data quality checks...")
            validation_summary = validate_data(cleaned, config)
            results['steps']['validation'] = {
                'status': 'success' if validation_summary['overall_passed'] else 'failed',
                'overall_passed': validation_summary['overall_passed'],
                'checks': validation_summary['checks'],
            }
            if not validation_summary['overall_passed']:
                logger.warning("Validation checks failed - review data_quality_report.json")
        else:
            logger.info("\n[3/5] VALIDATION: Skipped")
            results['steps']['validation'] = {'status': 'skipped'}
        
        # Step 4: Aggregation
        logger.info("\n[4/5] AGGREGATION: Building demand series at all levels...")
        aggregated = run_aggregation_pipeline(interim_path, config)
        results['steps']['aggregation'] = {
            'status': 'success',
            'levels': list(aggregated.keys()),
            'row_counts': {k: len(v) for k, v in aggregated.items()},
        }
        for level, df in aggregated.items():
            results['stats'][f'{level}_rows'] = len(df)
        
        # Step 5: Final verification
        logger.info("\n[5/5] VERIFICATION: Checking outputs...")
        project_root = Path(__file__).parent.parent.parent
        processed_dir = project_root / config['paths']['processed_data']
        
        expected_files = [
            'demand_overall.parquet',
            'demand_category.parquet',
            'demand_market.parquet',
            'demand_category_market.parquet',
        ]
        
        missing_files = []
        for fname in expected_files:
            fpath = processed_dir / fname
            if fpath.exists():
                df = pd.read_parquet(fpath)
                results['paths'][f'processed_{fname.replace(".parquet", "")}'] = str(fpath)
                logger.info(f"  ✓ {fname}: {len(df)} rows")
            else:
                missing_files.append(fname)
                logger.warning(f"  ✗ {fname}: MISSING")
        
        if missing_files:
            logger.warning(f"Missing output files: {missing_files}")
        
        results['steps']['verification'] = {
            'status': 'success' if not missing_files else 'partial',
            'missing_files': missing_files,
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Processed data saved to: {processed_dir}")
        logger.info(f"Interim data saved to: {interim_path.parent}")
        
        return results
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        results['steps']['error'] = str(e)
        raise


def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DemandCast ETL Pipeline')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip data validation step')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml')
    args = parser.parse_args()
    
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    try:
        results = run_pipeline(config, skip_validation=args.skip_validation)
        print("\nPipeline completed successfully!")
        print(f"Output files:")
        for key, path in results['paths'].items():
            print(f"  {key}: {path}")
        return 0
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())