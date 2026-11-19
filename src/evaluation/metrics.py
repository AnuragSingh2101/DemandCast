"""
Evaluation Metrics for DemandCast.

All metrics handle edge cases (zeros, NaN) gracefully.
Includes MASE (scaled error) and statistical significance tests.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def _safe_divide(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safe division with default for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, num / den, default)
    return result


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(actual - pred))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - pred) ** 2))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Returns np.inf if actual contains zeros (use wape/smape instead).
    """
    if np.any(actual == 0):
        return np.inf
    return np.mean(np.abs((actual - pred) / actual)) * 100


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (Weighted MAPE).
    Handles zeros in actual by weighting by actual values.
    """
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)) * 100


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Bounded [0, 200], handles zeros.
    """
    denominator = (np.abs(actual) + np.abs(pred)) / 2
    return np.mean(_safe_divide(np.abs(actual - pred), denominator)) * 100


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Mean Error (Bias).
    Positive = over-forecast, Negative = under-forecast.
    """
    return np.mean(pred - actual)


def mase(
    actual: np.ndarray,
    pred: np.ndarray,
    train_actual: Optional[np.ndarray] = None,
    seasonal_period: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    
    Scales MAE by in-sample MAE of naive (or seasonal naive) forecast.
    MASE < 1 means better than naive benchmark.
    MASE > 1 means worse than naive.
    
    Args:
        actual: Test period actuals
        pred: Test period predictions
        train_actual: Training period actuals (required for scaling)
        seasonal_period: Seasonal period for seasonal naive scaling (1 = naive)
    
    Returns:
        MASE value (np.nan if train_actual not provided or scaling fails)
    """
    if train_actual is None or len(train_actual) < seasonal_period + 1:
        return np.nan
    
    # Compute in-sample naive MAE
    if seasonal_period == 1:
        # Naive: y_t = y_{t-1}
        naive_errors = np.abs(train_actual[1:] - train_actual[:-1])
    else:
        # Seasonal naive: y_t = y_{t-seasonal_period}
        if len(train_actual) <= seasonal_period:
            return np.nan
        naive_errors = np.abs(train_actual[seasonal_period:] - train_actual[:-seasonal_period])
    
    scale = np.mean(naive_errors)
    
    if scale == 0:
        return np.nan
    
    return mae(actual, pred) / scale


def mase_from_history(
    actual: np.ndarray,
    pred: np.ndarray,
    history: np.ndarray,
    seasonal_period: int = 1
) -> float:
    """
    Convenience wrapper: history = training data.
    """
    return mase(actual, pred, history, seasonal_period)


# ============================================================
# Prediction Interval Metrics
# ============================================================

def interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """
    Empirical coverage of prediction intervals.
    Returns fraction of actuals within [lower, upper].
    """
    within = (actual >= lower) & (actual <= upper)
    return np.mean(within)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average interval width."""
    return np.mean(upper - lower)


def pinball_loss(
    actual: np.ndarray,
    pred: np.ndarray,
    quantile: float
) -> float:
    """
    Pinball (quantile) loss for quantile forecasts.
    
    Args:
        actual: True values
        pred: Predicted quantile
        quantile: Quantile level (e.g., 0.05, 0.5, 0.95)
    
    Returns:
        Pinball loss
    """
    errors = actual - pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def winkler_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Winkler interval score (proper scoring rule for intervals).
    Lower is better. Penalizes width + missed coverage.
    """
    width = upper - lower
    below = lower - actual
    above = actual - upper
    
    score = width + (2/alpha) * below * (actual < lower) + (2/alpha) * above * (actual > upper)
    return np.mean(score)


# ============================================================
# Statistical Significance Tests
# ============================================================

def diebold_mariano_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss: str = 'mae',
    h: int = 1,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for predictive accuracy.
    
    Tests H0: E[loss(pred1) - loss(pred2)] = 0
    
    Args:
        actual: Actual values
        pred1, pred2: Predictions from two models
        loss: 'mae' or 'mse'
        h: Forecast horizon (for small-sample correction)
        alternative: 'two-sided', 'less', 'greater'
    
    Returns:
        (test_statistic, p_value)
    """
    if loss == 'mae':
        d = np.abs(actual - pred1) - np.abs(actual - pred2)
    elif loss == 'mse':
        d = (actual - pred1)**2 - (actual - pred2)**2
    else:
        raise ValueError(f"Unknown loss: {loss}")
    
    d = d.flatten()
    n = len(d)
    
    if n < 2:
        return np.nan, np.nan
    
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    if d_var == 0:
        return 0.0, 1.0
    
    # Small sample correction (Harvey et al. 1997)
    # DM_stat ~ t_{n-1} under H0
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # Harvey-Leybourne-Newbold correction
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_corrected = dm_stat * correction
    
    # p-value from t-distribution
    df = n - 1
    if alternative == 'two-sided':
        p = 2 * stats.t.sf(abs(dm_stat_corrected), df)
    elif alternative == 'less':
        p = stats.t.sf(dm_stat_corrected, df)
    elif alternative == 'greater':
        p = stats.t.cdf(dm_stat_corrected, df)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return dm_stat_corrected, p


def paired_bootstrap_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    metric_fn: callable = mae,
    n_bootstrap: int = 1000,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Paired bootstrap test on metric differences.
    
    More robust than DM test for small samples or non-normal errors.
    """
    n = len(actual)
    if n < 2:
        return np.nan, np.nan
    
    obs_diff = metric_fn(actual, pred1) - metric_fn(actual, pred2)
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_diff = metric_fn(actual[idx], pred1[idx]) - metric_fn(actual[idx], pred2[idx])
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    if alternative == 'two-sided':
        p = np.mean(np.abs(bootstrap_diffs) >= abs(obs_diff))
    elif alternative == 'less':
        p = np.mean(bootstrap_diffs <= obs_diff)
    elif alternative == 'greater':
        p = np.mean(bootstrap_diffs >= obs_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return obs_diff, p


def holm_correction(p_values: list, alpha: float = 0.05) -> Tuple[list, list]:
    """
    Holm-Bonferroni correction for multiple comparisons.
    
    Returns: (adjusted_p_values, rejected_flags)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    adjusted = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    
    for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
        adjusted[idx] = min(p * (n - i), 1.0)
        if adjusted[idx] < alpha:
            rejected[idx] = True
        else:
            # Once one fails, all subsequent fail (step-down)
            break
    
    return adjusted.tolist(), rejected.tolist()


def benjamini_hochberg_correction(p_values: list, alpha: float = 0.05) -> Tuple[list, list]:
    """
    Benjamini-Hochberg FDR correction.
    
    Returns: (adjusted_p_values, rejected_flags)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    adjusted = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    
    for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
        adjusted[idx] = min(p * n / (i + 1), 1.0)
    
    # Find largest k where p_(k) <= k*alpha/n
    for i in range(n-1, -1, -1):
        if sorted_p[i] <= (i + 1) * alpha / n:
            rejected[sorted_idx[:i+1]] = True
            break
    
    return adjusted.tolist(), rejected.tolist()


# ============================================================
# Comprehensive Evaluation
# ============================================================

def evaluate_forecast(
    actual: np.ndarray,
    pred: np.ndarray,
    train_actual: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    alpha: float = 0.1
) -> dict:
    """
    Comprehensive single-model evaluation.
    
    Returns dict with all metrics.
    """
    metrics = {
        'mae': mae(actual, pred),
        'rmse': rmse(actual, pred),
        'wape': wape(actual, pred),
        'smape': smape(actual, pred),
        'bias': bias(actual, pred),
    }
    
    if train_actual is not None:
        metrics['mase'] = mase(actual, pred, train_actual, seasonal_period)
    
    if lower is not None and upper is not None:
        metrics['coverage'] = interval_coverage(actual, lower, upper)
        metrics['interval_width'] = interval_width(lower, upper)
        metrics['winkler'] = winkler_score(actual, lower, upper, alpha)
        metrics['pinball_lower'] = pinball_loss(actual, lower, alpha/2)
        metrics['pinball_upper'] = pinball_loss(actual, upper, 1 - alpha/2)
    
    return metrics


def compare_models(
    actual: np.ndarray,
    preds: dict,
    train_actual: Optional[np.ndarray] = None,
    seasonal_period: int = 1
) -> pd.DataFrame:
    """
    Compare multiple models and return sorted DataFrame.
    
    Args:
        actual: Actual values
        preds: Dict of {model_name: predictions}
        train_actual: Training data for MASE
        seasonal_period: For MASE
    
    Returns:
        DataFrame with metrics for each model, sorted by MAE
    """
    rows = []
    for name, pred in preds.items():
        m = evaluate_forecast(actual, pred, train_actual, seasonal_period)
        m['model'] = name
        rows.append(m)
    
    df = pd.DataFrame(rows).set_index('model')
    df = df.sort_values('mae')
    return df


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    actual = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
    pred = np.array([11, 11, 12, 12, 14, 15, 15, 19, 18, 20])
    train = np.array([8, 9, 10, 11, 10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
    
    print("MAE:", mae(actual, pred))
    print("RMSE:", rmse(actual, pred))
    print("WAPE:", wape(actual, pred))
    print("sMAPE:", smape(actual, pred))
    print("Bias:", bias(actual, pred))
    print("MASE:", mase(actual, pred, train, seasonal_period=1))
    
    # DM test
    pred2 = np.array([10, 13, 10, 14, 16, 13, 17, 17, 18, 18])
    stat, p = diebold_mariano_test(actual, pred, pred2)
    print(f"DM test: stat={stat:.3f}, p={p:.3f}")
    
    # Bootstrap test
    diff, p = paired_bootstrap_test(actual, pred, pred2)
    print(f"Bootstrap test: diff={diff:.3f}, p={p:.3f}")