"""
Baseline Models for DemandCast.

Simple, interpretable baselines that any forecast must beat.
All methods are leakage-free (use only historical data available at forecast origin).
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Naive forecast: next period = last observed value.
    Benchmark for non-seasonal data.
    """
    
    def __init__(self):
        self.last_value_ = None
    
    def fit(self, y: pd.Series) -> 'NaiveForecaster':
        """Store the last observed value."""
        self.last_value_ = y.iloc[-1]
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Repeat last value for horizon periods."""
        return np.full(horizon, self.last_value_)
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Seasonal naive forecast: next period = same period last season.
    Benchmark for seasonal data (e.g., weekly data with 52-week seasonality).
    """
    
    def __init__(self, seasonal_period: int = 52):
        self.seasonal_period = seasonal_period
        self.history_ = None
    
    def fit(self, y: pd.Series) -> 'SeasonalNaiveForecaster':
        """Store the full history for seasonal lookup."""
        self.history_ = y.values
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast using seasonal pattern from last year."""
        if len(self.history_) < self.seasonal_period:
            # Fall back to naive if not enough history
            return np.full(horizon, self.history_[-1])
        
        # Get the last seasonal cycle
        last_cycle = self.history_[-self.seasonal_period:]
        
        # Repeat the cycle as needed
        forecasts = np.tile(last_cycle, int(np.ceil(horizon / self.seasonal_period)))[:horizon]
        return forecasts
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)


class MovingAverageForecaster(BaseEstimator, RegressorMixin):
    """
    Simple moving average forecast: next period = average of last k periods.
    """
    
    def __init__(self, window: int = 4):
        self.window = window
        self.history_ = None
    
    def fit(self, y: pd.Series) -> 'MovingAverageForecaster':
        self.history_ = y.values
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast using moving average of last window periods."""
        if len(self.history_) < self.window:
            # Use all available history
            avg = np.mean(self.history_)
        else:
            avg = np.mean(self.history_[-self.window:])
        return np.full(horizon, avg)
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)


class HistoricalAverageForecaster(BaseEstimator, RegressorMixin):
    """
    Historical average: next period = mean of all historical data.
    Simplest possible benchmark.
    """
    
    def __init__(self):
        self.mean_ = None
    
    def fit(self, y: pd.Series) -> 'HistoricalAverageForecaster':
        self.mean_ = y.mean()
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.mean_)
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)


class BaselineEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble of baselines (simple average of predictions).
    Often beats individual baselines.
    """
    
    def __init__(self, seasonal_period: int = 52):
        self.models = {
            'naive': NaiveForecaster(),
            'seasonal_naive': SeasonalNaiveForecaster(seasonal_period),
            'moving_avg_4': MovingAverageForecaster(window=4),
            'moving_avg_12': MovingAverageForecaster(window=12),
            'historical_avg': HistoricalAverageForecaster(),
        }
    
    def fit(self, y: pd.Series) -> 'BaselineEnsemble':
        for model in self.models.values():
            model.fit(y)
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        preds = np.column_stack([m.predict(horizon) for m in self.models.values()])
        return preds.mean(axis=1)
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)


def get_all_baselines(seasonal_period: int = 52) -> dict:
    """Return dictionary of all baseline models."""
    return {
        'naive': NaiveForecaster(),
        'seasonal_naive': SeasonalNaiveForecaster(seasonal_period),
        'moving_avg_4': MovingAverageForecaster(window=4),
        'moving_avg_12': MovingAverageForecaster(window=12),
        'historical_avg': HistoricalAverageForecaster(),
        'ensemble': BaselineEnsemble(seasonal_period),
    }


if __name__ == "__main__":
    # Quick test
    y = pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 20, 22])
    horizon = 4
    
    baselines = get_all_baselines(seasonal_period=4)
    
    for name, model in baselines.items():
        preds = model.fit(y).predict(horizon)
        print(f"{name:20s}: {preds}")