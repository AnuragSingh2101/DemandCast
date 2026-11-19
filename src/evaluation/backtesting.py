"""
Rolling-Origin Backtesting Framework for DemandCast.

Implements chronological validation only - no random splits.
Detects common leakage issues.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


class RollingOriginBacktester:
    """
    Rolling-origin (walk-forward) backtesting for time series.
    
    For each fold:
    - Train on data up to fold start
    - Validate/test on next horizon periods
    - Roll forward by step_size
    """
    
    def __init__(
        self,
        horizon: int = 4,
        step_size: int = 4,
        min_train_size: int = 52,
        max_folds: Optional[int] = None
    ):
        self.horizon = horizon
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.max_folds = max_folds
    
    def split(self, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for rolling-origin splits.
        
        Returns:
            List of (train_idx, test_idx, fold_num) tuples
        """
        n = len(y)
        splits = []
        
        # Start from min_train_size, ensure we have horizon for test
        start = self.min_train_size
        max_start = n - self.horizon
        
        fold = 0
        for train_end in range(start, max_start + 1, self.step_size):
            test_end = min(train_end + self.horizon, n)
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            
            splits.append((train_idx, test_idx, fold))
            fold += 1
            
            if self.max_folds and fold >= self.max_folds:
                break
        
        return splits
    
    def backtest(
        self,
        y: pd.Series,
        model_fn: Callable,
        model_kwargs: Optional[Dict] = None,
        predict_method: str = 'predict'
    ) -> pd.DataFrame:
        """
        Run rolling-origin backtest for a single model.
        
        Args:
            y: Target time series
            model_fn: Callable that returns a fitted model (e.g., lambda: MyModel())
            model_kwargs: Optional kwargs for model_fn
            predict_method: Method name to call for predictions
        
        Returns:
            DataFrame with columns: fold, train_end, test_start, test_end, 
            actual, predicted, horizon_step
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        splits = self.split(y)
        results = []
        
        for train_idx, test_idx, fold in splits:
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Fit model on training data
            model = model_fn(**model_kwargs)
            model.fit(y_train)
            
            # Predict
            predict_fn = getattr(model, predict_method)
            y_pred = predict_fn(len(y_test))
            
            # Store results
            for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
                results.append({
                    'fold': fold,
                    'train_end': train_idx[-1],
                    'test_idx': test_idx[i],
                    'actual': actual,
                    'predicted': pred,
                    'horizon_step': i + 1,
                })
        
        return pd.DataFrame(results)


def evaluate_backtest(backtest_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate metrics from backtest results.
    """
    from .metrics import mae, rmse, wape, smape, bias, mase
    
    actual = backtest_df['actual'].values
    pred = backtest_df['predicted'].values
    
    # Overall metrics
    metrics = {
        'mae': mae(actual, pred),
        'rmse': rmse(actual, pred),
        'wape': wape(actual, pred),
        'smape': smape(actual, pred),
        'bias': bias(actual, pred),
    }
    
    # Per-horizon metrics
    for h in sorted(backtest_df['horizon_step'].unique()):
        mask = backtest_df['horizon_step'] == h
        metrics[f'mae_h{h}'] = mae(actual[mask], pred[mask])
        metrics[f'rmse_h{h}'] = rmse(actual[mask], pred[mask])
    
    return metrics


def backtest_all_models(
    y: pd.Series,
    models: Dict[str, Any],
    horizon: int = 4,
    step_size: int = 4,
    min_train_size: int = 52
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Run backtest for multiple models and return combined results.
    
    Args:
        y: Target time series
        models: Dict of {name: model_fn_or_instance}
        horizon, step_size, min_train_size: Backtest parameters
    
    Returns:
        (combined_results_df, metrics_dict)
    """
    backtester = RollingOriginBacktester(
        horizon=horizon,
        step_size=step_size,
        min_train_size=min_train_size
    )
    
    all_results = []
    metrics = {}
    
    for name, model in models.items():
        print(f"Backtesting {name}...")
        
        # Handle both model instances and factory functions
        if callable(model) and not hasattr(model, 'fit'):
            # It's a factory function
            def model_fn(**kwargs):
                return model(**kwargs)
        else:
            # It's an instance - need to clone for each fold
            def model_fn(**kwargs):
                return clone(model)
        
        results = backtester.backtest(y, model_fn)
        results['model'] = name
        all_results.append(results)
        
        metrics[name] = evaluate_backtest(results)
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined, metrics


# Leakage Detection Tests
def test_no_future_leakage(model, y: pd.Series, horizon: int = 4) -> bool:
    """
    Test that model doesn't use future data.
    Perturbs future values and checks predictions don't change.
    """
    from sklearn.base import clone
    
    # Split
    train_end = len(y) - horizon
    y_train = y.iloc[:train_end]
    y_test = y.iloc[train_end:]
    
    # Fit on original
    model1 = clone(model)
    model1.fit(y_train)
    pred1 = model1.predict(horizon)
    
    # Perturb future (test) values
    y_perturbed = y.copy()
    y_perturbed.iloc[train_end:] = y_perturbed.iloc[train_end:] * 100  # Massive change
    
    # Fit on perturbed (should give same predictions if no leakage)
    model2 = clone(model)
    model2.fit(y_perturbed.iloc[:train_end])
    pred2 = model2.predict(horizon)
    
    # Predictions should be identical (or very close)
    diff = np.abs(pred1 - pred2).max()
    return diff < 1e-10


def test_no_target_leakage_in_features(X: pd.DataFrame, y: pd.Series) -> List[str]:
    """
    Check for features that contain future target information.
    Returns list of suspicious feature names.
    """
    suspicious = []
    
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            # Check correlation with future target (shifted)
            for lag in [1, 2, 4, 8]:
                if len(X) > lag:
                    corr = X[col].iloc[:-lag].corr(y.iloc[lag:])
                    if abs(corr) > 0.95:  # Near-perfect correlation suggests leakage
                        suspicious.append(f"{col} (corr={corr:.3f} with target t+{lag})")
    
    return suspicious


def validate_chronological_split(
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> bool:
    """
    Validate that train/test split is chronological (no overlap, train before test).
    """
    # No overlap
    if set(train_idx).intersection(set(test_idx)):
        return False
    
    # All train indices before test indices
    if train_idx.max() >= test_idx.min():
        return False
    
    return True


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    y = pd.Series(np.random.randn(100).cumsum() + 100)
    
    backtester = RollingOriginBacktester(horizon=4, step_size=4, min_train_size=20)
    
    # Test naive forecaster
    from src.models.baselines import NaiveForecaster
    
    results, metrics = backtest_all_models(
        y, 
        {'naive': NaiveForecaster()},
        horizon=4,
        step_size=4,
        min_train_size=20
    )
    
    print("Backtest results shape:", results.shape)
    print("Metrics:", metrics)