"""
Models Package for DemandCast.

- baselines: Naive, Seasonal Naive, Moving Average, Historical Average
- ets_model: Exponential Smoothing (SES, Holt, Holt-Winters)
- xgboost_model: XGBoost with direct multi-step forecasting
"""

from .baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    HistoricalAverageForecaster,
    BaselineEnsemble,
    get_all_baselines,
)
from .ets_model import ETSForecaster, AutoETS
from .xgboost_model import XGBoostForecaster

__all__ = [
    'NaiveForecaster',
    'SeasonalNaiveForecaster',
    'MovingAverageForecaster',
    'HistoricalAverageForecaster',
    'BaselineEnsemble',
    'get_all_baselines',
    'ETSForecaster',
    'AutoETS',
    'XGBoostForecaster',
]