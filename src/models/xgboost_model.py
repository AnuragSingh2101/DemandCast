"""
XGBoost Forecaster for DemandCast.

Direct multi-step forecasting strategy (one model per horizon step).
Uses lag, rolling, and calendar features.
Supports prediction intervals via split-conformal calibration.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import RegressorChain


class FeatureEngineer:
    """
    Creates lag, rolling, and calendar features for time series forecasting.
    All features use only information available at forecast origin (no leakage).
    """
    
    def __init__(
        self,
        lag_periods: List[int] = [1, 2, 4, 8, 12, 26, 52],
        rolling_windows: List[int] = [4, 8, 12, 26],
        seasonal_period: int = 52
    ):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.seasonal_period = seasonal_period
    
    def transform(self, y: pd.Series) -> pd.DataFrame:
        """
        Create features from a single time series.
        
        Returns DataFrame with features aligned to y index.
        """
        df = pd.DataFrame(index=y.index)
        df['target'] = y.values
        
        # Lag features
        for lag in self.lag_periods:
            if lag < len(y):
                df[f'lag_{lag}'] = y.shift(lag)
        
        # Rolling statistics (shifted so window ends at t-1)
        for window in self.rolling_windows:
            if window < len(y):
                df[f'rolling_mean_{window}'] = y.shift(1).rolling(window).mean()
                df[f'rolling_std_{window}'] = y.shift(1).rolling(window).std()
                df[f'rolling_min_{window}'] = y.shift(1).rolling(window).min()
                df[f'rolling_max_{window}'] = y.shift(1).rolling(window).max()
        
        # Expanding statistics (all history up to t-1)
        df['expanding_mean'] = y.shift(1).expanding().mean()
        df['expanding_std'] = y.shift(1).expanding().std()
        
        # Calendar features (from index if datetime)
        if isinstance(y.index, pd.DatetimeIndex):
            df['month'] = y.index.month
            df['quarter'] = y.index.quarter
            df['week_of_year'] = y.index.isocalendar().week
            df['day_of_week'] = y.index.dayofweek
            df['is_month_start'] = y.index.is_month_start.astype(int)
            df['is_month_end'] = y.index.is_month_end.astype(int)
            df['is_quarter_start'] = y.index.is_quarter_start.astype(int)
            df['is_quarter_end'] = y.index.is_quarter_end.astype(int)
        
        # Seasonal indicators
        if self.seasonal_period > 1:
            # Position in seasonal cycle
            df['seasonal_phase'] = (np.arange(len(y)) % self.seasonal_period) / self.seasonal_period
        
        return df
    
    def fit_transform(self, y: pd.Series) -> pd.DataFrame:
        """Fit and transform (no fitting needed, just transform)."""
        return self.transform(y)


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """
    XGBoost forecaster with direct multi-step strategy.
    
    Strategy: One model per horizon step (direct forecasting).
    This avoids recursive error accumulation.
    """
    
    def __init__(
        self,
        horizon: int = 4,
        lag_periods: List[int] = [1, 2, 4, 8, 12, 26, 52],
        rolling_windows: List[int] = [4, 8, 12, 26],
        seasonal_period: int = 52,
        xgb_params: Optional[Dict] = None,
        calibration_size: float = 0.2,
        alpha: float = 0.1,
    ):
        self.horizon = horizon
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.seasonal_period = seasonal_period
        self.xgb_params = xgb_params or {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'early_stopping_rounds': 50,
        }
        self.calibration_size = calibration_size
        self.alpha = alpha
        
        self.feature_engineer_ = FeatureEngineer(
            lag_periods=lag_periods,
            rolling_windows=rolling_windows,
            seasonal_period=seasonal_period
        )
        
        self.models_ = {}  # One model per horizon step
        self.calibration_scores_ = {}  # For conformal intervals
        self.is_fitted_ = False
    
    def _prepare_data(self, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features and target for training."""
        # Ensure y is a pandas Series with proper index
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not isinstance(y.index, pd.DatetimeIndex):
            # Create a simple integer index if no datetime index
            y = pd.Series(y.values, index=pd.RangeIndex(len(y)))
        
        X = self.feature_engineer_.transform(y)
        # Drop rows with NaN (from lags/rolling)
        valid = X.dropna()
        return valid.drop(columns=['target']), valid['target']
    
    def fit(self, y: pd.Series) -> 'XGBoostForecaster':
        """
        Fit one XGBoost model per horizon step.
        
        Uses direct strategy: model_h predicts y[t+h] using features at t.
        """
        X, y_target = self._prepare_data(y)
        
        if len(X) < self.horizon + 10:
            raise ValueError(f"Not enough data after feature engineering: {len(X)} rows")
        
        # Split for calibration (last calibration_size fraction)
        n_cal = int(len(X) * self.calibration_size)
        n_train = len(X) - n_cal
        
        X_train = X.iloc[:n_train]
        y_train = y_target.iloc[:n_train]
        X_cal = X.iloc[n_train:]
        y_cal = y_target.iloc[n_train:]
        
        # Train one model per horizon step
        for h in range(1, self.horizon + 1):
            # Target for this horizon: shift by h
            y_h = y_train.shift(-h).iloc[:-h]  # Remove NaN from shift
            X_h = X_train.iloc[:-h]
            
            # Align calibration targets
            y_cal_h = y_cal.shift(-h).iloc[:-h]
            X_cal_h = X_cal.iloc[:-h]
            
            if len(X_h) < 10:
                continue
            
            # Train model
            model = XGBRegressor(**self.xgb_params)
            
            # Use early stopping with validation set
            eval_set = [(X_cal_h, y_cal_h)] if len(X_cal_h) > 0 else None
            
            if eval_set:
                model.fit(
                    X_h, y_h,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                model.fit(X_h, y_h, verbose=False)
            
            self.models_[h] = model
            
            # Compute conformity scores on calibration set for intervals
            if len(X_cal_h) > 0:
                cal_pred = model.predict(X_cal_h)
                scores = np.abs(y_cal_h - cal_pred)
                self.calibration_scores_[h] = scores
        
        self.is_fitted_ = True
        return self
    
    def predict(self, y: pd.Series, horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate forecasts for horizon steps ahead.
        
        Args:
            y: Historical time series (used for feature engineering)
            horizon: Number of steps to forecast (default: self.horizon)
        
        Returns:
            Array of predictions length horizon
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        if horizon is None:
            horizon = self.horizon
        
        # Ensure y is a pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not isinstance(y.index, pd.DatetimeIndex):
            y = pd.Series(y.values, index=pd.RangeIndex(len(y)))
        
        # Create features for the last available point
        X_all = self.feature_engineer_.transform(y)
        last_features = X_all.drop(columns=['target']).iloc[[-1]]  # Shape: (1, n_features)
        
        preds = []
        for h in range(1, horizon + 1):
            if h in self.models_:
                pred = self.models_[h].predict(last_features)[0]
                preds.append(pred)
            else:
                # Fallback: use last model or naive
                if self.models_:
                    pred = list(self.models_.values())[-1].predict(last_features)[0]
                else:
                    pred = y.iloc[-1]
                preds.append(pred)
        
        return np.array(preds)
    
    def predict_interval(
        self,
        y: pd.Series,
        horizon: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using split-conformal prediction.
        
        Returns:
            (predictions, lower_bounds, upper_bounds)
        """
        if alpha is None:
            alpha = self.alpha
        
        preds = self.predict(y, horizon)
        lower = np.zeros_like(preds)
        upper = np.zeros_like(preds)
        
        for i, h in enumerate(range(1, horizon + 1)):
            if h in self.calibration_scores_:
                scores = self.calibration_scores_[h]
                q = np.quantile(scores, 1 - alpha)
                lower[i] = preds[i] - q
                upper[i] = preds[i] + q
            else:
                # Fallback: use residual std
                std = np.std([s for scores in self.calibration_scores_.values() for s in scores])
                q = stats.norm.ppf(1 - alpha/2) * std
                lower[i] = preds[i] - q
                upper[i] = preds[i] + q
        
        return preds, lower, upper
    
    def fit_predict(self, y: pd.Series, horizon: Optional[int] = None) -> np.ndarray:
        self.fit(y)
        return self.predict(y, horizon)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'horizon': self.horizon,
            'lag_periods': self.lag_periods,
            'rolling_windows': self.rolling_windows,
            'seasonal_period': self.seasonal_period,
            'xgb_params': self.xgb_params,
            'calibration_size': self.calibration_size,
            'alpha': self.alpha,
            'n_models': len(self.models_),
        }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    y = pd.Series(np.random.randn(200).cumsum() + 100, 
                  index=pd.date_range('2020-01-01', periods=200, freq='W-MON'))
    y = y + 10 * np.sin(np.arange(200) * 2 * np.pi / 52)
    
    model = XGBoostForecaster(horizon=4, seasonal_period=52)
    preds = model.fit_predict(y)
    print(f"Predictions: {preds}")
    print(f"Params: {model.get_params()}")
    
    # Test intervals
    preds, lower, upper = model.predict_interval(y)
    print(f"Intervals: lower={lower}, upper={upper}")