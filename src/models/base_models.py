import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging

logger = logging.getLogger(__name__)

class MarketingFinancePredictor(BaseEstimator, RegressorMixin):
    """Base class for all marketing-finance prediction models"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the model with time series validation"""
        # Implement time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            self._fit_internal(X_train, y_train)
            
            # Validate
            pred = self.predict(X_val)
            score = mean_squared_error(y_val, pred, squared=False)
            cv_scores.append(score)
        
        self.cv_score = np.mean(cv_scores)
        logger.info(f"Cross-validation RMSE: {self.cv_score:.4f}")
        
        # Final fit on all data
        self._fit_internal(X, y)
        return self
    
    def _fit_internal(self, X: pd.DataFrame, y: pd.Series):
        """Internal fitting method to be overridden"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        predictions = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100,
            'directional_accuracy': np.mean((predictions > 0) == (y > 0))
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def save_model(self, path: str):
        """Save model to disk"""
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load model from disk"""
        return joblib.load(path)
