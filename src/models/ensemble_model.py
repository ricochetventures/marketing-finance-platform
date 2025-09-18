import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import optuna

class AdvancedEnsembleModel(MarketingFinancePredictor):
    """Advanced ensemble model with automatic hyperparameter tuning"""
    
    def __init__(self, optimize_hyperparams: bool = True):
        super().__init__(model_type='ensemble')
        self.optimize_hyperparams = optimize_hyperparams
        self.base_models = {}
        self.model_weights = {}
        self.best_params = {}
        
    def _fit_internal(self, X: pd.DataFrame, y: pd.Series):
        """Fit ensemble of models"""
        
        # Initialize base models
        self.base_models = {
            'lightgbm': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'xgboost': xgb.XGBRegressor(random_state=42, verbosity=0),
            'catboost': CatBoostRegressor(random_state=42, verbose=False),
            'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'extra_trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'ridge': Ridge(random_state=42),
            'elastic_net': ElasticNet(random_state=42)
        }
        
        # Optimize hyperparameters if requested
        if self.optimize_hyperparams:
            self._optimize_all_models(X, y)
        
        # Train all models
        predictions = {}
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y)
            predictions[name] = model.predict(X)
            
            # Calculate feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
        
        # Optimize ensemble weights
        self._optimize_weights(predictions, y)
        
    def _optimize_all_models(self, X: pd.DataFrame, y: pd.Series):
        """Optimize hyperparameters using Optuna"""
        
        def optimize_lightgbm(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            
            model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv,
                                    scoring='neg_mean_squared_error')
            return -scores.mean()
        
        def optimize_xgboost(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
            }
            
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv,
                                    scoring='neg_mean_squared_error')
            return -scores.mean()
        
        # Run optimization for each model
        optimizers = {
            'lightgbm': optimize_lightgbm,
            'xgboost': optimize_xgboost,
        }
        
        for model_name, optimizer in optimizers.items():
            logger.info(f"Optimizing {model_name}...")
            study = optuna.create_study(direction='minimize')
            study.optimize(optimizer, n_trials=50, show_progress_bar=False)
            
            self.best_params[model_name] = study.best_params
            
            # Update model with best params
            if model_name == 'lightgbm':
                self.base_models[model_name] = lgb.LGBMRegressor(
                    **study.best_params, random_state=42, verbosity=-1
                )
            elif model_name == 'xgboost':
                self.base_models[model_name] = xgb.XGBRegressor(
                    **study.best_params, random_state=42, verbosity=0
                )
    
    def _optimize_weights(self, predictions: Dict[str, np.ndarray], y: pd.Series):
        """Optimize ensemble weights using optimization"""
        from scipy.optimize import minimize
        
        def objective(weights):
            weighted_pred = np.zeros_like(y)
            for i, (name, pred) in enumerate(predictions.items()):
                weighted_pred += weights[i] * pred
            return mean_squared_error(y, weighted_pred)
        
        # Initial equal weights
        init_weights = np.ones(len(predictions)) / len(predictions)
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(predictions))]
        
        # Optimize
        result = minimize(objective, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Store optimized weights
        for i, name in enumerate(predictions.keys()):
            self.model_weights[name] = result.x[i]
            
        logger.info(f"Optimized weights: {self.model_weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions"""
        predictions = np.zeros(len(X))
        
        for name, model in self.base_models.items():
            weight = self.model_weights.get(name, 1/len(self.base_models))
            predictions += weight * model.predict(X)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance"""
        importance_df = pd.DataFrame()
        
        for name, importance in self.feature_importance.items():
            importance_df[name] = importance
        
        # Average importance across models
        importance_df['mean_importance'] = importance_df.mean(axis=1)
        
        return importance_df.sort_values('mean_importance', ascending=False)
