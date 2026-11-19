#!/usr/bin/env python
"""
Run forecasting experiments: baselines, ETS, XGBoost with backtesting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import yaml
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from evaluation.backtesting import (
    RollingOriginBacktester, evaluate_backtest
)
from evaluation.metrics import (
    evaluate_forecast, compare_models, diebold_mariano_test
)
from models.baselines import get_all_baselines
from models.ets_model import ETSForecaster
from models.xgboost_model import XGBoostForecaster


def main():
    # Load config
    project_root = Path(__file__).parent.parent
    with open(project_root / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processed_dir = project_root / config['paths']['processed_data']
    reports_dir = project_root / config['paths']['reports_dir']
    models_dir = project_root / config['paths']['models_dir']
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processed data: {processed_dir}")
    print(f"Reports: {reports_dir}")
    print(f"Models: {models_dir}")
    
    # Load category-level demand data
    cat_data = pd.read_parquet(processed_dir / "demand_category.parquet")
    cat_data['order_week_start'] = pd.to_datetime(cat_data['order_week_start'])
    cat_data = cat_data.sort_values(['category_id', 'order_week_start'])
    
    # Configuration
    HORIZON = 4
    STEP_SIZE = 4
    MIN_TRAIN = 52
    
    # Build category time series
    cat_series = {}
    for cat_id in cat_data['category_id'].unique():
        cat_df = cat_data[cat_data['category_id'] == cat_id].copy()
        cat_df = cat_df.set_index('order_week_start')['order_item_quantity']
        cat_df = cat_df.asfreq('W-MON').fillna(0)
        if len(cat_df) >= MIN_TRAIN + HORIZON:
            cat_series[cat_id] = cat_df
    
    print(f"Prepared {len(cat_series)} category time series with sufficient data")
    
    # Select top 5 categories by volume
    top_cats = sorted(cat_series.keys(), 
                      key=lambda c: cat_series[c].sum(), 
                      reverse=True)[:5]
    print(f"Top 5 categories: {top_cats}")
    
    # Get all baseline models
    baselines = get_all_baselines(seasonal_period=52)
    
    # Storage
    all_category_results = {}
    
    # Run experiments on each category
    for cat_id in top_cats:
        y = cat_series[cat_id]
        print(f"\n=== Category {cat_id} (n={len(y)}) ===")
        
        backtester = RollingOriginBacktester(
            horizon=HORIZON, 
            step_size=STEP_SIZE, 
            min_train_size=MIN_TRAIN
        )
        
        cat_results = {}
        cat_metrics = {}
        
        # Run baselines
        for name, model in baselines.items():
            try:
                res = backtester.backtest(y, lambda: model)
                cat_results[name] = res
                cat_metrics[name] = evaluate_backtest(res)
                print(f"  {name:20s}: MAE={cat_metrics[name]['mae']:.2f}")
            except Exception as e:
                print(f"  {name:20s}: FAILED - {e}")
        
        # ETS
        try:
            res = backtester.backtest(y, lambda: ETSForecaster(seasonal_period=52))
            cat_results['ets'] = res
            cat_metrics['ets'] = evaluate_backtest(res)
            print(f"  {'ets':20s}: MAE={cat_metrics['ets']['mae']:.2f} (type={res['model'].iloc[0] if 'model' in res.columns else 'N/A'})")
        except Exception as e:
            print(f"  {'ets':20s}: FAILED - {e}")
        
        # XGBoost - disabled for now due to feature mismatch
        # TODO: Fix feature mismatch between train and predict
        cat_results['xgboost'] = None
        cat_metrics['xgboost'] = None
        print(f"  {'xgboost':20s}: SKIPPED (feature mismatch)")
        
        all_category_results[cat_id] = {
            'results': cat_results,
            'metrics': cat_metrics
        }
    
    # Aggregate metrics
    print("\n=== Average Metrics Across Categories ===")
    avg_metrics = {}
    for model_name in ['naive', 'seasonal_naive', 'moving_avg_4', 'moving_avg_12', 
                        'historical_avg', 'ensemble', 'ets']:  # xgboost disabled
        maes = []
        rmses = []
        for cat_id in top_cats:
            m = all_category_results[cat_id]['metrics'].get(model_name)
            if m is not None:
                maes.append(m['mae'])
                rmses.append(m['rmse'])
        if maes:
            avg_metrics[model_name] = {
                'avg_mae': float(np.mean(maes)),
                'std_mae': float(np.std(maes)),
                'avg_rmse': float(np.mean(rmses)),
                'std_rmse': float(np.std(rmses)),
            }
    
    avg_df = pd.DataFrame(avg_metrics).T.round(3)
    avg_df = avg_df.sort_values('avg_mae')
    print(avg_df.to_string())
    
    # Win counts
    print("\n=== Win Counts (lower MAE = win) ===")
    wins = {model: 0 for model in ['ets']}  # xgboost disabled
    baseline_names = ['naive', 'seasonal_naive', 'moving_avg_4', 'moving_avg_12', 
                      'historical_avg', 'ensemble']
    
    for cat_id in top_cats:
        cat_metrics = all_category_results[cat_id]['metrics']
        for model in ['ets']:
            m = cat_metrics.get(model)
            if m is not None:
                model_mae = m['mae']
                for b in baseline_names:
                    b_m = cat_metrics.get(b)
                    if b_m is not None and model_mae < b_m['mae']:
                        wins[model] += 1
    
    print(wins)
    
    # Best model per category
    print("\n=== Best Model Per Category ===")
    best_models = {}
    for cat_id in top_cats:
        cat_metrics = all_category_results[cat_id]['metrics']
        # Filter out None metrics
        valid_metrics = {k: v for k, v in cat_metrics.items() if v is not None}
        best = min(valid_metrics.items(), key=lambda x: x[1]['mae'])
        best_models[cat_id] = best[0]
        print(f"  Cat {cat_id}: {best[0]} (MAE={best[1]['mae']:.2f})")
    
    # Statistical significance tests
    print("\n=== Diebold-Mariano Tests (vs Best Baseline) ===")
    for cat_id in top_cats:
        cat_results = all_category_results[cat_id]['results']
        cat_metrics = all_category_results[cat_id]['metrics']
        
        best_baseline = min(
            [(n, m['mae']) for n, m in cat_metrics.items() if n in baseline_names],
            key=lambda x: x[1]
        )[0]
        
        for model_name in ['ets']:  # xgboost disabled
            if model_name in cat_results and best_baseline in cat_results:
                stat, p = diebold_mariano_test(
                    cat_results[best_baseline]['actual'].values,
                    cat_results[best_baseline]['predicted'].values,
                    cat_results[model_name]['predicted'].values
                )
                sig = "**" if p < 0.05 else ""
                print(f"  Cat {cat_id}: {best_baseline} vs {model_name}: DM={stat:.3f}, p={p:.4f} {sig}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    # Save experiment results
    experiment_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'horizon': HORIZON,
            'step_size': STEP_SIZE,
            'min_train': MIN_TRAIN,
            'categories_tested': [convert(c) for c in top_cats],
        },
        'average_metrics': convert(avg_metrics),
        'wins': convert(wins),
        'best_per_category': convert(best_models),
    }
    
    with open(reports_dir / 'forecasting_experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\nResults saved to {reports_dir / 'forecasting_experiment_results.json'}")
    
    # Train final models on full data
    print("\n=== Final Model Training (Full Data) ===")
    final_models = {}
    
    for cat_id in top_cats:
        y = cat_series[cat_id]
        best_model_name = best_models[cat_id]
        
        if best_model_name == 'ets':
            model = ETSForecaster(seasonal_period=52).fit(y)
        elif best_model_name == 'xgboost':
            model = XGBoostForecaster(horizon=HORIZON, seasonal_period=52).fit(y)
        else:
            model = baselines[best_model_name].fit(y)
        
        final_models[cat_id] = {
            'model': model,
            'model_type': best_model_name,
            'horizon': HORIZON,
            'trained_on': len(y),
            'last_date': str(y.index[-1]),
        }
        
        joblib.dump(model, models_dir / f"model_cat_{cat_id}.joblib")
        print(f"  Cat {cat_id}: {best_model_name} saved")
    
    print(f"\nModels saved to {models_dir}")
    print("\n=== EXPERIMENTS COMPLETE ===")


if __name__ == "__main__":
    main()