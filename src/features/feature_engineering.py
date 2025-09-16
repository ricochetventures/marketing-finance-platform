import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for marketing-finance correlation"""
    
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.feature_importance = {}
        
    def create_financial_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced financial features"""
        features = pd.DataFrame(index=stock_data.index)
        
        # Returns at multiple horizons
        for period in [1, 5, 20, 60, 252]:  # Daily, weekly, monthly, quarterly, yearly
            features[f'return_{period}d'] = stock_data['Close'].pct_change(period)
            features[f'volatility_{period}d'] = stock_data['Close'].rolling(period).std()
            
        # Technical indicators
        features['rsi'] = self._calculate_rsi(stock_data['Close'])
        features['macd'], features['macd_signal'] = self._calculate_macd(stock_data['Close'])
        
        # Volume features
        features['volume_ratio'] = stock_data['Volume'] / stock_data['Volume'].rolling(20).mean()
        features['dollar_volume'] = stock_data['Close'] * stock_data['Volume']
        
        # Price features
        features['high_low_ratio'] = stock_data['High'] / stock_data['Low']
        features['close_to_high'] = stock_data['Close'] / stock_data['High']
        
        # Volatility features
        features['parkinson_volatility'] = self._parkinson_volatility(
            stock_data['High'], stock_data['Low']
        )
        features['garman_klass_volatility'] = self._garman_klass_volatility(
            stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close']
        )
        
        return features
    
    def create_marketing_features(self, ad_spend: pd.DataFrame, 
                                 agencies: pd.DataFrame) -> pd.DataFrame:
        """Create marketing-specific features"""
        features = pd.DataFrame()
        
        # Ad spend features
        features['total_spend'] = ad_spend.groupby('Company')['Total_Spend'].transform('sum')
        features['digital_ratio'] = ad_spend['Digital_Spend'] / ad_spend['Total_Spend']
        features['tv_ratio'] = ad_spend['TV_Spend'] / ad_spend['Total_Spend']
        features['print_ratio'] = ad_spend['Print_Spend'] / ad_spend['Total_Spend']
        
        # Spend momentum
        features['spend_growth'] = ad_spend.groupby('Company')['Total_Spend'].pct_change()
        features['spend_acceleration'] = features['spend_growth'].diff()
        
        # Agency features
        features['agency_tenure'] = self._calculate_agency_tenure(agencies)
        features['agency_changes'] = self._count_agency_changes(agencies)
        features['is_holding_company'] = agencies['Agency_Type'] == 'Holding'
        
        # Marketing efficiency
        features['spend_volatility'] = ad_spend.groupby('Company')['Total_Spend'].transform(
            lambda x: x.rolling(4).std()
        )
        
        # Adstock effect (carryover from previous periods)
        features['adstock'] = self._calculate_adstock(ad_spend['Total_Spend'])
        
        return features
    
    def create_interaction_features(self, financial: pd.DataFrame, 
                                   marketing: pd.DataFrame,
                                   economic: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different data sources"""
        features = pd.DataFrame()
        
        # Marketing-Finance interactions
        features['roi_efficiency'] = financial['return_20d'] / (marketing['total_spend'] + 1)
        features['spend_momentum_return'] = marketing['spend_growth'] * financial['return_60d']
        
        # Economic-Marketing interactions
        features['spend_gdp_ratio'] = marketing['total_spend'] / economic['GDP']
        features['sentiment_adjusted_spend'] = (
            marketing['total_spend'] * economic['CONSUMER_SENTIMENT'] / 100
        )
        
        # Agency-Performance interactions
        features['agency_return_impact'] = (
            marketing['agency_changes'] * financial['return_252d']
        )
        
        # Volatility interactions
        features['volatility_spend_ratio'] = (
            financial['volatility_20d'] / (marketing['total_spend'] + 1)
        )
        
        # Lag features for causality
        for lag in [1, 3, 6, 12]:
            features[f'spend_lag_{lag}m'] = marketing['total_spend'].shift(lag * 20)
            features[f'roi_lag_{lag}m'] = financial['return_20d'].shift(lag * 20)
        
        return features
    
    def create_industry_features(self, company_data: pd.DataFrame, 
                                industry_data: pd.DataFrame) -> pd.DataFrame:
        """Create industry-relative performance features"""
        features = pd.DataFrame()
        
        # Industry benchmarks
        industry_means = industry_data.groupby('Industry').mean()
        industry_stds = industry_data.groupby('Industry').std()
        
        # Relative performance
        for metric in ['return', 'volatility', 'spend', 'roi']:
            if metric in company_data.columns:
                features[f'{metric}_vs_industry'] = (
                    company_data[metric] - industry_means[metric]
                ) / industry_stds[metric]
        
        # Industry rank
        features['industry_rank'] = company_data.groupby('Industry')['return'].rank(pct=True)
        
        # Peer comparison
        features['outperformance'] = (
            company_data['return'] > industry_means['return']
        ).astype(int)
        
        return features
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Date components
        features['year'] = df.index.year
        features['quarter'] = df.index.quarter
        features['month'] = df.index.month
        features['week'] = df.index.isocalendar().week
        features['dayofweek'] = df.index.dayofweek
        
        # Cyclical encoding
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
        
        # Holiday effects
        features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        features['is_year_end'] = df.index.is_year_end.astype(int)
        
        # Trend
        features['time_index'] = np.arange(len(df))
        features['time_index_squared'] = features['time_index'] ** 2
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def _parkinson_volatility(self, high: pd.Series, low: pd.Series, 
                             period: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        return np.sqrt(
            np.log(high/low)**2 / (4*np.log(2))
        ).rolling(period).mean()
    
    def _garman_klass_volatility(self, open_: pd.Series, high: pd.Series, 
                                 low: pd.Series, close: pd.Series,
                                 period: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        return np.sqrt(
            0.5 * np.log(high/low)**2 - 
            (2*np.log(2) - 1) * np.log(close/open_)**2
        ).rolling(period).mean()
    
    def _calculate_adstock(self, spend: pd.Series, decay: float = 0.5) -> pd.Series:
        """Calculate advertising adstock (carryover effect)"""
        adstock = pd.Series(index=spend.index, dtype=float)
        adstock.iloc[0] = spend.iloc[0]
        
        for i in range(1, len(spend)):
            adstock.iloc[i] = spend.iloc[i] + decay * adstock.iloc[i-1]
        
        return adstock
    
    def _calculate_agency_tenure(self, agencies: pd.DataFrame) -> pd.Series:
        """Calculate how long each agency has been with the company"""
        tenure = pd.Series(dtype=float)
        
        for idx, row in agencies.iterrows():
            start = row['Start_Date']
            end = row.get('End_Date', pd.Timestamp.now())
            tenure.loc[idx] = (end - start).days / 365.25
        
        return tenure
    
    def _count_agency_changes(self, agencies: pd.DataFrame) -> pd.Series:
        """Count number of agency changes per company"""
        return agencies.groupby('Company')['Agency'].nunique()
