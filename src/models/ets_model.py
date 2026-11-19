"""
Exponential Smoothing Models for DemandCast.

Wraps statsmodels' ExponentialSmoothing with a clean sklearn-like interface.
Supports SES, Holt, Holt-Winters with automatic fallback.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt


class ETSForecaster:
    """
    Exponential Smoothing Forecaster with automatic model selection.
    
    Tries models in order of complexity:
    1. Holt-Winters (trend + seasonal)
    2. Holt (trend only)
    3. Simple Exponential Smoothing (SES)
    
    Falls back gracefully if fitting fails.
    """
    
    def __init__(
        self,
        seasonal_period: int = 52,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = 'add',
        damped_trend: bool = True,
        use_boxcox: bool = True,
        remove_bias: bool = True,
        fallback: bool = True
    ):
        self.seasonal_period = seasonal_period
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        self.remove_bias = remove_bias
        self.fallback = fallback
        
        self.model_ = None
        self.fitted_model_ = None
        self.model_type_ = None
        self.params_ = {}
    
    def fit(self, y: pd.Series) -> 'ETSForecaster':
        """
        Fit the best available ETS model.
        
        Attempts in order: HW -> Holt -> SES
        """
        y = np.asarray(y, dtype=float)
        
        # Need at least 2 seasonal periods for HW
        min_hw = 2 * self.seasonal_period
        min_holt = 10
        
        models_to_try = []
        
        if len(y) >= min_hw and self.seasonal_period > 1:
            models_to_try.append(('holt_winters', self._fit_hw))
        
        if len(y) >= min_holt:
            models_to_try.append(('holt', self._fit_holt))
        
        models_to_try.append(('ses', self._fit_ses))
        
        last_error = None
        for model_type, fit_fn in models_to_try:
            try:
                fit_fn(y)
                self.model_type_ = model_type
                return self
            except Exception as e:
                last_error = e
                if not self.fallback:
                    raise
                continue
        
        # If all failed, use naive as last resort
        self.model_type_ = 'naive'
        self.fitted_model_ = None
        self.last_value_ = y[-1]
        
        return self
    
    def _fit_hw(self, y: np.ndarray):
        """Fit Holt-Winters (trend + seasonal)."""
        self.model_ = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_period,
            damped_trend=self.damped_trend
        )
        self.fitted_model_ = self.model_.fit(
            use_boxcox=self.use_boxcox,
            remove_bias=self.remove_bias
        )
        self.params_ = self.fitted_model_.params
    
    def _fit_holt(self, y: np.ndarray):
        """Fit Holt (trend only)."""
        self.model_ = Holt(
            y,
            damped_trend=self.damped_trend,
            trend=self.trend
        )
        self.fitted_model_ = self.model_.fit(
            use_boxcox=self.use_boxcox,
            remove_bias=self.remove_bias
        )
        self.params_ = self.fitted_model_.params
    
    def _fit_ses(self, y: np.ndarray):
        """Fit Simple Exponential Smoothing."""
        self.model_ = SimpleExpSmoothing(y)
        self.fitted_model_ = self.model_.fit(
            use_boxcox=self.use_boxcox,
            remove_bias=self.remove_bias
        )
        self.params_ = self.fitted_model_.params
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts for horizon periods."""
        if self.model_type_ == 'naive' or self.fitted_model_ is None:
            return np.full(horizon, self.last_value_)
        
        try:
            forecast = self.fitted_model_.forecast(horizon)
            return np.asarray(forecast)
        except Exception:
            # Fallback to naive
            return np.full(horizon, self.last_value_ if hasattr(self, 'last_value_') else y[-1])
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)
    
    def get_params(self) -> Dict[str, Any]:
        """Return fitted parameters."""
        return {
            'model_type': self.model_type_,
            'params': self.params_,
            'seasonal_period': self.seasonal_period,
        }


class AutoETS:
    """
    Automated ETS model selection using AIC/BIC.
    
    Tries multiple combinations and selects best by information criterion.
    """
    
    def __init__(
        self,
        seasonal_period: int = 52,
        criterion: str = 'aic',
        trend_options: List[str] = ['add', 'mul', None],
        seasonal_options: List[str] = ['add', 'mul', None],
        damped_options: List[bool] = [True, False],
    ):
        self.seasonal_period = seasonal_period
        self.criterion = criterion
        self.trend_options = trend_options
        self.seasonal_options = seasonal_options
        self.damped_options = damped_options
        
        self.best_model_ = None
        self.best_score_ = np.inf
        self.best_config_ = None
        self.all_results_ = []
    
    def fit(self, y: pd.Series) -> 'AutoETS':
        """Search over model configurations."""
        y = np.asarray(y, dtype=float)
        
        # Generate configurations to try
        configs = []
        for trend in self.trend_options:
            for seasonal in self.seasonal_options:
                for damped in self.damped_options:
                    # Skip invalid combinations
                    if trend is None and damped:
                        continue
                    if seasonal is None and self.seasonal_period > 1:
                        # Allow SES (no seasonal) even if period > 1
                        pass
                    configs.append((trend, seasonal, damped))
        
        for trend, seasonal, damped in configs:
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=self.seasonal_period if seasonal else None,
                    damped_trend=damped
                )
                fitted = model.fit(use_boxcox=True, remove_bias=True)
                
                score = getattr(fitted, self.criterion)
                
                self.all_results_.append({
                    'trend': trend,
                    'seasonal': seasonal,
                    'damped': damped,
                    'score': score,
                    'model': fitted,
                })
                
                if score < self.best_score_:
                    self.best_score_ = score
                    self.best_model_ = fitted
                    self.best_config_ = (trend, seasonal, damped)
                    
            except Exception:
                continue
        
        # Fallback to simple ETS if nothing worked
        if self.best_model_ is None:
            fallback = ETSForecaster(
                seasonal_period=self.seasonal_period,
                fallback=True
            )
            fallback.fit(y)
            self.best_model_ = fallback.fitted_model_
            self.best_config_ = ('fallback', 'fallback', 'fallback')
        
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        if self.best_model_ is None:
            return np.array([])
        return self.best_model_.forecast(horizon)
    
    def fit_predict(self, y: pd.Series, horizon: int) -> np.ndarray:
        self.fit(y)
        return self.predict(horizon)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'best_config': self.best_config_,
            'best_score': self.best_score_,
            'criterion': self.criterion,
            'all_results': self.all_results_,
        }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    y = pd.Series(np.random.randn(100).cumsum() + 100)
    y = y + 10 * np.sin(np.arange(100) * 2 * np.pi / 52)  # Add seasonality
    
    # Test ETSForecaster
    ets = ETSForecaster(seasonal_period=52)
    preds = ets.fit(y).predict(4)
    print(f"ETS ({ets.model_type_}): {preds}")
    print(f"Params: {ets.get_params()}")
    
    # Test AutoETS
    auto = AutoETS(seasonal_period=52)
    preds = auto.fit(y).predict(4)
    print(f"AutoETS ({auto.best_config_}): {preds}")