import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketingFinanceDataLoader:
    """Comprehensive data loader for all data sources"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        self.cache_path = self.data_path / "external"
        self.cache_path.mkdir(exist_ok=True)
        
    def load_excel_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Load the provided Excel file with all sheets"""
        logger.info(f"Loading Excel data from {file_path}")
        
        # Read all sheets
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # Process each sheet
        processed_data = {}
        
        # Stock Prices
        if 'Stock_Prices' in excel_data:
            df = excel_data['Stock_Prices']
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            processed_data['stock_prices'] = df
            logger.info(f"Loaded {len(df)} stock price records")
        
        # Ad Spend
        if 'Ad_Spend' in excel_data:
            df = excel_data['Ad_Spend']
            df['Year'] = pd.to_datetime(df['Year'], format='%Y')
            processed_data['ad_spend'] = df
            logger.info(f"Loaded {len(df)} ad spend records")
        
        # Agencies
        if 'Agencies' in excel_data:
            df = excel_data['Agencies']
            df['Start_Date'] = pd.to_datetime(df['Start_Date'])
            if 'End_Date' in df.columns:
                df['End_Date'] = pd.to_datetime(df['End_Date'])
            processed_data['agencies'] = df
            logger.info(f"Loaded {len(df)} agency records")
        
        # Marketing ROI
        if 'Marketing_ROI' in excel_data:
            df = excel_data['Marketing_ROI']
            df['Date'] = pd.to_datetime(df['Date'])
            processed_data['marketing_roi'] = df
            logger.info(f"Loaded {len(df)} ROI records")
        
        return processed_data
    
    def fetch_economic_data(self, start_date: str = "2006-01-01") -> pd.DataFrame:
        """Fetch economic indicators from FRED"""
        from fredapi import Fred
        import os
        
        fred = Fred(api_key=os.getenv('FRED_API_KEY', 'your_fred_key'))
        
        indicators = {
            'GDP': 'GDP',
            'CPI': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
            'CONSUMER_SENTIMENT': 'UMCSENT',
            'INTEREST_RATE': 'DFF',
            'SP500': 'SP500'
        }
        
        economic_data = pd.DataFrame()
        
        for name, series_id in indicators.items():
            try:
                data = fred.get_series(series_id, start_date)
                economic_data[name] = data
                logger.info(f"Fetched {name} data")
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
        
        return economic_data
    
    def fetch_competitor_data(self, companies: List[str]) -> pd.DataFrame:
        """Fetch competitor stock and financial data"""
        competitor_data = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._fetch_company_data, company): company 
                      for company in companies}
            
            for future in futures:
                company = futures[future]
                try:
                    competitor_data[company] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching {company} data: {e}")
        
        return pd.concat(competitor_data, axis=1)
    
    def _fetch_company_data(self, ticker: str) -> pd.DataFrame:
        """Fetch individual company data"""
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist = stock.history(period="max")
        
        # Get financials if available
        try:
            info = stock.info
            financials = stock.financials
            return hist
        except:
            return hist
