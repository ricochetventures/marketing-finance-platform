"""
Company List Manager - Validates publicly traded companies and adds top advertisers
Location: src/data/company_list_manager.py
"""

import yfinance as yf
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompanyListManager:
    """Manage and validate the company list"""
    
    def __init__(self):
        self.top_100_advertisers = [
            # These are the world's top advertisers by spend (2023-2024 data)
            # Tech
            "Amazon", "Alphabet", "Meta", "Apple", "Microsoft", "Samsung", "Alibaba",
            # Automotive
            "Volkswagen", "Toyota", "General Motors", "Ford", "Stellantis", "Honda", 
            "Nissan", "BMW", "Mercedes-Benz", "Hyundai", "Tesla",
            # Retail
            "Walmart", "Target", "Costco", "The Home Depot", "Lowe's", "Kroger",
            # CPG/Food & Beverage
            "Procter & Gamble", "Unilever", "Coca-Cola", "PepsiCo", "Nestlé", 
            "L'Oréal", "Mondelez", "Mars", "Kraft Heinz", "Danone", "General Mills",
            "Kellogg's", "Anheuser-Busch InBev", "Diageo", "Pernod Ricard",
            # Pharma/Healthcare
            "Johnson & Johnson", "Pfizer", "Roche", "Novartis", "Merck", "Sanofi",
            "AstraZeneca", "GlaxoSmithKline", "Bristol-Myers Squibb", "Eli Lilly",
            "AbbVie", "Novo Nordisk",
            # Telecom
            "AT&T", "Verizon", "T-Mobile", "Comcast", "Deutsche Telekom", "Vodafone",
            "Orange", "China Mobile",
            # Financial Services
            "American Express", "JPMorgan Chase", "Bank of America", "Visa", 
            "Mastercard", "Wells Fargo", "Citigroup", "Capital One",
            # Restaurants/QSR
            "McDonald's", "Starbucks", "Yum! Brands", "Restaurant Brands International",
            "Chipotle", "Domino's Pizza",
            # Apparel/Footwear
            "Nike", "Adidas", "Lululemon", "Under Armour", "VF Corporation", "Gap",
            # Entertainment/Media
            "Walt Disney", "Netflix", "Warner Bros Discovery", "Paramount Global",
            "Sony", "Comcast (NBCUniversal)",
            # Consumer Electronics
            "LG Electronics", "Panasonic", "HP Inc", "Dell Technologies",
            # Airlines
            "Delta Air Lines", "American Airlines", "United Airlines", "Southwest Airlines",
            # Other Major Advertisers
            "IKEA", "Booking Holdings", "Expedia", "Airbnb", "Uber", "DoorDash",
            "eBay", "PayPal", "Adobe", "Oracle", "Salesforce", "Intel", "AMD",
            "Qualcomm", "NVIDIA"
        ]
        
        self.ticker_overrides = {
            # Manual mappings for companies with non-obvious tickers
            "Alphabet": "GOOGL",
            "Meta": "META",
            "Nestlé": "NSRGY",
            "L'Oréal": "OR.PA",
            "Coca-Cola": "KO",
            "Procter & Gamble": "PG",
            "Anheuser-Busch InBev": "BUD",
            "Johnson & Johnson": "JNJ",
            "Booking Holdings": "BKNG",
            "Restaurant Brands International": "QSR",
            "Yum! Brands": "YUM",
            "VF Corporation": "VFC",
            "Walt Disney": "DIS",
            "Warner Bros Discovery": "WBD",
            "Paramount Global": "PARA",
            "Mercedes-Benz": "MBG.DE",
            "BMW": "BMW.DE",
            "Adidas": "ADS.DE",
            "Volkswagen": "VOW3.DE",
            "Deutsche Telekom": "DTE.DE",
            "SAP": "SAP",
            "Eli Lilly": "LLY",
            "GlaxoSmithKline": "GSK",
            "Novo Nordisk": "NVO",
            "AbbVie": "ABBV",
            "Kraft Heinz": "KHC",
            "General Mills": "GIS",
            "Kellogg's": "K",
            "Mondelez": "MDLZ",
            "Bristol-Myers Squibb": "BMY",
            "Pernod Ricard": "RI.PA",
            "Dell Technologies": "DELL",
            "HP Inc": "HPQ",
            "Under Armour": "UAA",
            "Lululemon": "LULU",
            "Chipotle": "CMG",
            "Domino's Pizza": "DPZ",
            "Costco": "COST",
            "The Home Depot": "HD",
            "Lowe's": "LOW",
            "Delta Air Lines": "DAL",
            "American Airlines": "AAL",
            "United Airlines": "UAL",
            "Southwest Airlines": "LUV",
            "China Mobile": "CHL",
            "Comcast": "CMCSA",
            "JPMorgan Chase": "JPM",
            "Bank of America": "BAC",
            "Wells Fargo": "WFC"
        }
    
    def validate_ticker(self, ticker: str, company_name: str) -> Optional[Dict]:
        """Validate if ticker exists and return basic info"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if info.get('regularMarketPrice') or info.get('currentPrice'):
                return {
                    'ticker': ticker,
                    'company': company_name,
                    'valid': True,
                    'exchange': info.get('exchange', 'Unknown'),
                    'currency': info.get('currency', 'USD')
                }
            else:
                logger.warning(f"Invalid ticker for {company_name}: {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error validating {ticker} for {company_name}: {e}")
            return None
    
    def find_ticker(self, company_name: str) -> Optional[str]:
        """Find ticker for a company using various strategies"""
        
        # Strategy 1: Check manual overrides
        if company_name in self.ticker_overrides:
            ticker = self.ticker_overrides[company_name]
            if self.validate_ticker(ticker, company_name):
                return ticker
        
        # Strategy 2: Try common patterns
        patterns = [
            company_name,  # Exact name
            company_name.replace(" ", ""),  # No spaces
            company_name.split()[0],  # First word only
            company_name.replace(".", "").replace(",", ""),  # Remove punctuation
        ]
        
        for pattern in patterns:
            # Try as-is
            if self.validate_ticker(pattern.upper(), company_name):
                return pattern.upper()
            
            # Try first 4 characters
            if len(pattern) >= 4:
                short = pattern[:4].upper()
                if self.validate_ticker(short, company_name):
                    return short
        
        logger.warning(f"Could not find ticker for: {company_name}")
        return None
    
    def build_validated_company_list(self) -> List[Dict]:
        """Build list of validated publicly traded companies"""
        
        logger.info("Building validated company list...")
        validated_companies = []
        
        # Use ThreadPoolExecutor for faster validation
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_company = {}
            
            for company in self.top_100_advertisers:
                future = executor.submit(self.process_company, company)
                future_to_company[future] = company
            
            for future in as_completed(future_to_company):
                result = future.result()
                if result:
                    validated_companies.append(result)
                    logger.info(f"✓ {result['company']}: {result['ticker']}")
                time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Validated {len(validated_companies)} companies")
        return validated_companies
    
    def process_company(self, company_name: str) -> Optional[Dict]:
        """Process individual company"""
        ticker = self.find_ticker(company_name)
        
        if ticker:
            validation = self.validate_ticker(ticker, company_name)
            if validation:
                return validation
        
        return None
    
    def save_company_list(self, output_path: str = 'data/processed/companies.json'):
        """Save validated company list to JSON"""
        
        validated = self.build_validated_company_list()
        
        # Remove duplicates by ticker
        seen_tickers = set()
        unique_companies = []
        
        for company in validated:
            if company['ticker'] not in seen_tickers:
                seen_tickers.add(company['ticker'])
                unique_companies.append(company)
        
        # Sort by company name
        unique_companies.sort(key=lambda x: x['company'])
        
        # Save to JSON
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save full details
        with open(output_path, 'w') as f:
            json.dump(unique_companies, f, indent=2)
        
        # Also save simple list for backward compatibility
        simple_list = [c['company'] for c in unique_companies]
        with open(output_path.replace('.json', '_simple.json'), 'w') as f:
            json.dump(simple_list, f, indent=2)
        
        logger.info(f"Saved {len(unique_companies)} unique companies to {output_path}")
        
        return unique_companies


if __name__ == "__main__":
    manager = CompanyListManager()
    companies = manager.save_company_list()
    
    print(f"\n✅ Successfully validated {len(companies)} publicly traded companies")
    print("\nSample companies:")
    for company in companies[:10]:
        print(f"  - {company['company']}: {company['ticker']}")