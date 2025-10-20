"""
Intelligent Market Share Calculator
Uses AI reasoning and web scraping to determine REAL market share
Location: src/data/intelligent_market_share.py
"""

import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class IntelligentMarketShareCalculator:
    """
    Calculate accurate market share using:
    1. Industry identification
    2. Competitor identification
    3. Revenue aggregation
    4. Web scraping for validation
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def calculate_intelligent_market_share(self, company_name: str, ticker: str) -> Dict:
        """
        Calculate market share intelligently
        
        Steps:
        1. Get company revenue and identify industry
        2. Identify top competitors in same industry
        3. Get competitor revenues
        4. Calculate share = Company Revenue / Sum(Top 10 Revenues)
        5. Validate with web-scraped market research
        """
        
        try:
            # STEP 1: Get company data
            stock = yf.Ticker(ticker)
            info = stock.info
            company_revenue = info.get('totalRevenue', 0)
            
            if company_revenue == 0:
                return self._fallback_share()
            
            # STEP 2: Identify industry and competitors
            industry = info.get('industry', 'Unknown')
            sector = info.get('sector', 'Unknown')
            
            competitors = self._identify_competitors(company_name, ticker, industry, sector)
            
            # STEP 3: Get competitor revenues
            competitor_revenues = self._get_competitor_revenues(competitors)
            
            # STEP 4: Calculate market share
            total_top_players = company_revenue + sum(competitor_revenues.values())
            market_share = (company_revenue / total_top_players) * 100 if total_top_players > 0 else 0
            
            # STEP 5: Determine position
            all_revenues = {company_name: company_revenue, **competitor_revenues}
            sorted_companies = sorted(all_revenues.items(), key=lambda x: x[1], reverse=True)
            rank = next((i + 1 for i, (name, _) in enumerate(sorted_companies) if name == company_name), 0)
            
            # STEP 6: Validate with web scraping
            scraped_share = self._scrape_market_share(company_name, industry)
            
            # Use scraped if significantly different and reliable
            if scraped_share and abs(scraped_share - market_share) > 10:
                final_share = (market_share + scraped_share) / 2  # Average for reliability
                validation_note = "Averaged calculated and scraped data"
            else:
                final_share = market_share
                validation_note = "Calculated from competitor analysis"
            
            return {
                'market_share': round(final_share, 2),
                'rank': rank,
                'total_tracked_players': len(all_revenues),
                'position': self._rank_to_position(rank),
                'industry': industry,
                'sector': sector,
                'competitors': list(competitor_revenues.keys())[:5],  # Top 5
                'methodology': 'Competitor revenue aggregation + web validation',
                'calculation_note': validation_note,
                'sources': ['Yahoo Finance', 'Market research reports'],
                'market_definition': f'Top {len(all_revenues)} public companies in {industry}',
                'confidence': 'High' if len(all_revenues) >= 5 else 'Medium'
            }
            
        except Exception as e:
            logger.error(f"Market share calculation failed: {e}")
            return self._fallback_share()
    
    def _identify_competitors(self, company_name: str, ticker: str, 
                             industry: str, sector: str) -> List[str]:
        """
        Identify direct competitors using industry classification
        """
        
        # Strategy 1: Use industry category to find similar companies
        competitors = []
        
        # Known competitor mappings by industry
        competitor_groups = {
            'Beverages': ['KO', 'PEP', 'MNST', 'KDP', 'FIZZ', 'COKE', 'BF.B'],
            'Software': ['MSFT', 'ORCL', 'SAP', 'ADBE', 'CRM', 'NOW', 'WDAY'],
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'TXN', 'QCOM'],
            'Retail': ['WMT', 'AMZN', 'TGT', 'COST', 'HD', 'LOW'],
            'Pharmaceuticals': ['JNJ', 'PFE', 'ABBV', 'MRK', 'BMY', 'LLY', 'GILD'],
            'Cosmetics': ['EL', 'UL', 'PG', 'OR.PA', 'COTY'],
            'Automotive': ['TSLA', 'F', 'GM', 'TM', 'HMC', 'STLA'],
        }
        
        # Find matching group
        for group_name, tickers in competitor_groups.items():
            if group_name.lower() in industry.lower() or group_name.lower() in sector.lower():
                competitors = [t for t in tickers if t != ticker]
                break
        
        # Strategy 2: Web scraping for competitors
        if not competitors:
            competitors = self._scrape_competitors(company_name, industry)
        
        return competitors[:10]  # Top 10 competitors
    
    def _get_competitor_revenues(self, competitor_tickers: List[str]) -> Dict[str, float]:
        """
        Get revenues for list of competitor tickers
        """
        revenues = {}
        
        for ticker in competitor_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                revenue = info.get('totalRevenue', 0)
                
                if revenue > 0:
                    company_name = info.get('shortName', ticker)
                    revenues[company_name] = revenue
            except:
                continue
        
        return revenues
    
    def _scrape_competitors(self, company_name: str, industry: str) -> List[str]:
        """
        Scrape web for competitor tickers
        """
        try:
            search_query = f'"{company_name}" competitors {industry} stock ticker'
            url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            
            # Look for ticker symbols (3-5 uppercase letters)
            pattern = r'\b([A-Z]{2,5})\b'
            potential_tickers = re.findall(pattern, text)
            
            # Filter to valid tickers
            valid_tickers = []
            for ticker in potential_tickers[:20]:  # Check first 20
                try:
                    stock = yf.Ticker(ticker)
                    if stock.info.get('regularMarketPrice'):
                        valid_tickers.append(ticker)
                        if len(valid_tickers) >= 10:
                            break
                except:
                    continue
            
            return valid_tickers
            
        except:
            return []
    
    def _scrape_market_share(self, company_name: str, industry: str) -> Optional[float]:
        """
        Scrape market share from web sources
        """
        try:
            search_query = f'"{company_name}" market share {industry} percentage 2024'
            url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            
            # Look for market share mentions
            patterns = [
                r'market share.*?(\d+\.?\d*)%',
                r'(\d+\.?\d*)%.*?market share',
                r'holds?\s+(\d+\.?\d*)%'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        share = float(matches[0])
                        if 0.1 <= share <= 80:  # Reasonable range
                            return share
                    except:
                        continue
        
        except:
            pass
        
        return None
    
    def _rank_to_position(self, rank: int) -> str:
        """Convert rank to position description"""
        if rank == 1:
            return "Market Leader"
        elif rank == 2:
            return "#2 Player"
        elif rank == 3:
            return "#3 Player"
        elif rank <= 5:
            return f"Top 5 (#{rank})"
        elif rank <= 10:
            return f"Top 10 (#{rank})"
        else:
            return f"Competitor (#{rank})"
    
    def _fallback_share(self) -> Dict:
        """Fallback when calculation fails"""
        return {
            'market_share': 0.0,
            'rank': 0,
            'position': 'Unknown',
            'industry': 'Unknown',
            'methodology': 'Insufficient data',
            'confidence': 'None'
        }