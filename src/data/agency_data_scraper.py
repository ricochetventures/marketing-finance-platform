"""
Agency Data Scraper - Gets current agencies across all categories
Location: src/data/agency_data_scraper.py
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgencyDataScraper:
    """Scrape agency relationships across all marketing categories"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        self.agency_categories = {
            'Creative AOR': ['creative agency', 'advertising agency', 'creative aor'],
            'Media AOR': ['media agency', 'media buying', 'media aor'],
            'Digital/Interactive AOR': ['digital agency', 'interactive agency', 'digital aor'],
            'PR/Communications AOR': ['pr agency', 'public relations', 'communications agency'],
            'Social Media AOR': ['social media agency', 'social aor'],
            'Shopper/Retail Marketing AOR': ['shopper marketing', 'retail marketing agency'],
            'Experiential/Event Marketing AOR': ['experiential agency', 'event marketing'],
            'Content Production AOR': ['content production', 'production company'],
            'CRM/Relationship Marketing AOR': ['crm agency', 'relationship marketing'],
            'Influencer Marketing AOR': ['influencer marketing agency', 'influencer aor'],
            'Performance/Growth Marketing AOR': ['performance marketing', 'growth marketing agency'],
            'Brand Design/Branding AOR': ['brand design agency', 'branding agency']
        }
        
        self.known_agencies = [
            # Holding Companies
            'WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas',
            # Creative Networks
            'BBDO', 'DDB', 'Ogilvy', 'Leo Burnett', 'TBWA', 'Grey', 'McCann', 'FCB',
            'Saatchi & Saatchi', 'Wieden+Kennedy', 'Droga5', 'R/GA',
            # Media Agencies
            'GroupM', 'Zenith', 'Starcom', 'OMD', 'PHD', 'Mindshare', 'MediaCom',
            'Carat', 'Initiative', 'Essence',
            # Digital/Performance
            'Merkle', 'Isobar', 'Digitas', 'Razorfish', 'Sapient', 'VML', 'AKQA',
            # Independent
            'MediaMonks', 'Huge', 'Tool', 'Anomaly', 'Mother', '72andSunny'
        ]
        
        self.cache_dir = Path('data/external/agency_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_agencies(self, company_name: str) -> Dict[str, Dict]:
        """Get agencies across all categories for a company"""
        
        # Check cache first (30-day TTL)
        cache_file = self.cache_dir / f"{company_name.replace(' ', '_')}_agencies.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                cache_date = datetime.fromisoformat(cached['date'])
                if (datetime.now() - cache_date).days < 30:
                    logger.info(f"Using cached agency data for {company_name}")
                    return cached['data']
        
        logger.info(f"Fetching agency data for {company_name}...")
        
        agencies_by_category = {}
        
        for category, search_terms in self.agency_categories.items():
            logger.info(f"  Searching {category}...")
            agency_info = self._search_agency_category(company_name, category, search_terms)
            
            if agency_info['agency'] != 'Unknown':
                agencies_by_category[category] = agency_info
            
            time.sleep(1)  # Rate limiting
        
        # Save to cache
        cache_data = {
            'date': datetime.now().isoformat(),
            'data': agencies_by_category
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Found {len(agencies_by_category)} agency relationships for {company_name}")
        
        return agencies_by_category
    
    def _search_agency_category(self, company_name: str, category: str, 
                                search_terms: List[str]) -> Dict:
        """Search for agency in a specific category"""
        
        # Try multiple search strategies
        strategies = [
            self._search_google_news,
            self._search_adage_database,
            self._search_company_website
        ]
        
        for search_term in search_terms:
            for strategy in strategies:
                result = strategy(company_name, search_term)
                if result['agency'] != 'Unknown':
                    result['category'] = category
                    result['search_term_used'] = search_term
                    return result
        
        return {
		    'agency': 'Data Unavailable',
		    'category': category,
		    'confidence': 'None',
		    'source': 'No reliable source found',
		    'last_updated': None,
		    'methodology': 'Searched Google News, Ad Age database, and company press releases'
		}
    
    def _search_google_news(self, company_name: str, search_term: str) -> Dict:
        """Search Google News for agency announcements"""
        try:
            # Search for recent news about agency appointments
            query = f"{company_name} {search_term} agency appointment OR names OR selects"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for agency names in results
            text = soup.get_text().lower()
            
            for agency in self.known_agencies:
                if agency.lower() in text:
                    # Try to extract date
                    date_pattern = r'(\d{1,2}\s+(?:days?|weeks?|months?)\s+ago|20\d{2})'
                    dates = re.findall(date_pattern, text, re.IGNORECASE)
                    
                    return {
                        'agency': agency,
                        'confidence': 'Medium',
                        'source': 'Google News Search',
                        'last_updated': dates[0] if dates else 'Unknown',
                        'method': 'Web scraping - news articles'
                    }
        
        except Exception as e:
            logger.error(f"Google News search failed: {e}")
        
        return {'agency': 'Unknown', 'confidence': 'None', 'source': 'Search failed'}
    
    def _search_adage_database(self, company_name: str, search_term: str) -> Dict:
        """Search Ad Age database"""
        try:
            query = f"site:adage.com {company_name} {search_term}"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            text = soup.get_text().lower()
            
            for agency in self.known_agencies:
                if agency.lower() in text:
                    return {
                        'agency': agency,
                        'confidence': 'High',
                        'source': 'Ad Age',
                        'last_updated': 'Recent',
                        'method': 'Industry publication search'
                    }
        
        except Exception as e:
            logger.error(f"Ad Age search failed: {e}")
        
        return {'agency': 'Unknown', 'confidence': 'None', 'source': 'Search failed'}
    
    def _search_company_website(self, company_name: str, search_term: str) -> Dict:
        """Search company's own press releases"""
        try:
            query = f"{company_name} press release {search_term}"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            text = soup.get_text()
            
            for agency in self.known_agencies:
                if agency in text:
                    return {
                        'agency': agency,
                        'confidence': 'High',
                        'source': 'Company press release',
                        'last_updated': 'Recent',
                        'method': 'Official company announcement'
                    }
        
        except Exception as e:
            logger.error(f"Company website search failed: {e}")
        
        return {'agency': 'Unknown', 'confidence': 'None', 'source': 'Search failed'}
    
    def format_for_display(self, agencies_dict: Dict[str, Dict]) -> str:
        """Format agency data for display"""
        if not agencies_dict:
            return "No agency data available"
        
        display_lines = []
        
        # Group by agency to avoid repetition
        agencies_by_name = {}
        for category, info in agencies_dict.items():
            agency = info['agency']
            if agency not in agencies_by_name:
                agencies_by_name[agency] = []
            agencies_by_name[agency].append(category)
        
        for agency, categories in agencies_by_name.items():
            display_lines.append(f"**{agency}**")
            display_lines.append(f"  Roles: {', '.join(categories)}")
        
        return "\n".join(display_lines)


if __name__ == "__main__":
    scraper = AgencyDataScraper()
    
    # Test with a few companies
    test_companies = ["Coca-Cola", "Nike", "Apple"]
    
    for company in test_companies:
        print(f"\n{'='*60}")
        print(f"Agency Relationships for {company}")
        print('='*60)
        
        agencies = scraper.get_all_agencies(company)
        print(scraper.format_for_display(agencies))