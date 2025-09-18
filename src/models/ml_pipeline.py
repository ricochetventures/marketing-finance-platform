import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

class MarketingFinanceMLPipeline:
    def __init__(self, data_dict):
        self.data = data_dict
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def create_features(self, company_name):
        """Create ML features for a specific company"""
        features = pd.DataFrame()
        
        # Get company data
        company_stock = self.data['stock_prices'][company_name] if company_name in self.data['stock_prices'].columns else None
        company_spend = self.data['ad_spend'][self.data['ad_spend']['Company'] == company_name]
        company_agencies = self.data['agencies'][self.data['agencies']['Company'] == company_name]
        company_roi = self.data['roi'][self.data['roi']['Company'] == company_name]
        
        if company_stock is not None:
            # Stock features
            features['returns_7d'] = company_stock.pct_change(7)
            features['returns_30d'] = company_stock.pct_change(30)
            features['volatility_30d'] = company_stock.rolling(30).std()
            features['price_ma_50'] = company_stock.rolling(50).mean()
            features['price_ma_200'] = company_stock.rolling(200).mean()
            
        # Ad spend features
        if not company_spend.empty:
            spend_by_year = company_spend.groupby('Year').agg({
                'Total_Spend': 'sum',
                'Digital_Spend': 'sum',
                'TV_Spend': 'sum'
            })
            features['total_spend'] = spend_by_year['Total_Spend']
            features['digital_ratio'] = spend_by_year['Digital_Spend'] / spend_by_year['Total_Spend']
            features['tv_ratio'] = spend_by_year['TV_Spend'] / spend_by_year['Total_Spend']
            features['spend_growth'] = spend_by_year['Total_Spend'].pct_change()
            
        # Agency features
        if not company_agencies.empty:
            features['agency_changes'] = len(company_agencies)
            features['current_agency_tenure'] = (pd.Timestamp.now() - company_agencies['Start_Date'].iloc[-1]).days / 365
            
        # ROI features
        if not company_roi.empty:
            roi_series = company_roi.set_index('Date')['ROI']
            features['avg_roi'] = roi_series.rolling(4).mean()
            features['roi_trend'] = roi_series.pct_change()
            
        return features
    
    def prepare_training_data(self):
        """Prepare data for model training"""
        all_features = []
        all_targets = []
        
        for company in self.data['companies']:
            features = self.create_features(company)
            if not features.empty:
                # Create targets (future ROI and stock returns)
                if company in self.data['stock_prices'].columns:
                    future_returns = self.data['stock_prices'][company].pct_change(90).shift(-90)
                    
                    # Align features and targets
                    aligned_features = features.loc[future_returns.index]
                    aligned_targets = future_returns.loc[features.index]
                    
                    # Remove NaN values
                    mask = ~(aligned_features.isna().any(axis=1) | aligned_targets.isna())
                    
                    all_features.append(aligned_features[mask])
                    all_targets.append(aligned_targets[mask])
        
        if all_features:
            X = pd.concat(all_features)
            y = pd.concat(all_targets)
            return X, y
        
        return None, None
    
    def train_models(self):
        """Train multiple models for ensemble"""
        X, y = self.prepare_training_data()
        
        if X is None or X.empty:
            print("No training data available")
            return
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train models
        models_config = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                verbosity=0
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
            
            # Evaluate
            score = model.score(X_test_scaled, y_test)
            print(f"{name} R2 Score: {score:.4f}")
        
        # Save models
        self.save_models()
        
    def save_models(self):
        """Save trained models"""
        output_dir = Path('models/saved')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, output_dir / f'{name}_model.pkl')
        
        joblib.dump(self.scalers, output_dir / 'scalers.pkl')
        joblib.dump(self.feature_columns, output_dir / 'feature_columns.pkl')
    
    def predict(self, company_name, scenario):
        """Make predictions for a scenario"""
        features = self.create_features(company_name)
        
        if features.empty:
            return None
        
        # Apply scenario modifications
        if 'new_spend' in scenario:
            features['total_spend'] = scenario['new_spend']
        if 'new_agency' in scenario:
            features['agency_changes'] += 1
            features['current_agency_tenure'] = 0
        
        # Get latest features
        latest_features = features.iloc[-1:][self.feature_columns]
        
        # Scale features
        features_scaled = self.scalers['standard'].transform(latest_features)
        
        # Ensemble prediction
        predictions = []
        for model in self.models.values():
            pred = model.predict(features_scaled)
            predictions.append(pred[0])
        
        return {
            'predicted_return': np.mean(predictions),
            'confidence_interval': (np.percentile(predictions, 25), np.percentile(predictions, 75)),
            'predictions_by_model': predictions
        }

# Train the models
pipeline = MarketingFinanceMLPipeline(data)
pipeline.train_models()
