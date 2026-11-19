"""
Evaluation Package for DemandCast.

- metrics: MAE, RMSE, WAPE, sMAPE, Bias, MASE, Pinball, Winkler, Diebold-Mariano, Bootstrap
- backtesting: Rolling-origin backtesting framework
"""

from .metrics import (
    mae, rmse, wape, smape, bias, mase, mase_from_history,
    interval_coverage, interval_width, pinball_loss, winkler_score,
    diebold_mariano_test, paired_bootstrap_test,
    holm_correction, benjamini_hochberg_correction,
    evaluate_forecast, compare_models,
)
from .backtesting import (
    RollingOriginBacktester,
    evaluate_backtest,
    backtest_all_models,
    test_no_future_leakage,
    test_no_target_leakage_in_features,
    validate_chronological_split,
)

__all__ = [
    'mae', 'rmse', 'wape', 'smape', 'bias', 'mase', 'mase_from_history',
    'interval_coverage', 'interval_width', 'pinball_loss', 'winkler_score',
    'diebold_mariano_test', 'paired_bootstrap_test',
    'holm_correction', 'benjamini_hochberg_correction',
    'evaluate_forecast', 'compare_models',
    'RollingOriginBacktester',
    'evaluate_backtest',
    'backtest_all_models',
    'test_no_future_leakage',
    'test_no_target_leakage_in_features',
    'validate_chronological_split',
]