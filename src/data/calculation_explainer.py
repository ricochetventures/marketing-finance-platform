"""
Calculation Explainer - Provides detailed methodology for every metric
Location: src/data/calculation_explainer.py
"""

from typing import Dict, Optional
from datetime import datetime

class CalculationExplainer:
    """Explains how every number on the platform is calculated"""
    
    def __init__(self):
        self.explanations = {}
    
    def explain_stock_price(self, company_name: str, current_price: float, 
                           change_pct: float, period: str) -> Dict:
        """Explain stock price calculation"""
        return {
            'metric': 'Stock Price',
            'value': f'${current_price:.2f}',
            'change': f'{change_pct:+.1f}%',
            'period': period,
            'data_source': 'Yahoo Finance API (yfinance library)',
            'methodology': [
                '1. Query Yahoo Finance for latest trading data',
                f'2. Extract most recent closing price: ${current_price:.2f}',
                f'3. Compare to price at start of {period}',
                f'4. Calculate percentage change: {change_pct:+.1f}%'
            ],
            'formula': '((Current Price - Start Price) / Start Price) × 100',
            'api_endpoint': f'yf.Ticker(ticker).history(period="{period}")',
            'data_freshness': 'Real-time with 15-minute delay',
            'confidence': 'Very High (99%+)',
            'limitations': [
                'Stock price reflects many factors beyond marketing',
                'Short-term volatility may not reflect marketing impact',
                'Different exchanges may show slight price variations'
            ]
        }
    
    def explain_marketing_roi(self, company_name: str, roi: float, 
                             revenue: float, marketing_spend: float, 
                             calculation_method: str) -> Dict:
        """Explain Marketing ROI calculation"""
        return {
            'metric': 'Marketing ROI',
            'value': f'{roi:.2f}x',
            'data_sources': [
                'Yahoo Finance API for financial data',
                'SEC EDGAR for filings',
                'Company earnings reports',
                'Industry benchmarks from web scraping'
            ],
            'methodology': [
                f'1. Retrieved total revenue: ${revenue/1e9:.2f}B',
                f'2. Estimated marketing spend: ${marketing_spend/1e9:.2f}B',
                '3. Calculated gross profit (Revenue × Gross Margin)',
                '4. Applied formula: Gross Profit ÷ Marketing Spend'
            ],
            'formula': '(Revenue × Gross Margin) ÷ Marketing Spend',
            'calculation_method': calculation_method,
            'assumptions': [
                f'Marketing spend estimated as % of revenue (industry-specific)',
                'Gross margin derived from financial statements',
                'Attribution model accounts for marketing contribution'
            ],
            'confidence': 'Medium (70-80%)',
            'why_medium_confidence': [
                'Marketing spend often not disclosed separately',
                'Estimation required for most companies',
                'Attribution models have inherent uncertainty'
            ],
            'how_to_improve': [
                'Company could disclose actual marketing spend',
                'More frequent earnings call mentions of marketing budget',
                'Industry could standardize reporting'
            ]
        }
    
    def explain_market_share(self, company_name: str, share: float,
                            company_revenue: float, market_size: float,
                            industry: str) -> Dict:
        """Explain market share calculation"""
        return {
            'metric': 'Market Share',
            'value': f'{share:.1f}%',
            'data_sources': [
                f'Company revenue: Yahoo Finance ({company_name})',
                f'Market size: Web scraping from industry reports',
                'Sources: Grand View Research, Statista, IBISWorld, etc.'
            ],
            'methodology': [
                f'1. Retrieved {company_name} revenue: ${company_revenue/1e9:.2f}B',
                f'2. Scraped {industry} market size: ${market_size/1e9:.2f}B',
                '3. Calculated: (Company Revenue ÷ Market Size) × 100',
                f'4. Result: {share:.1f}% market share'
            ],
            'formula': '(Company Revenue ÷ Total Market Size) × 100',
            'industry_definition': f'Global {industry} market',
            'confidence': 'High (85-90%)',
            'limitations': [
                'Market definitions can vary by source',
                'Private competitors not always included',
                'Regional breakdowns may differ',
                'Market size estimates updated annually'
            ],
            'web_scraping_strategy': [
                f'Search query: "{industry} market size 2024"',
                'Extract dollar amounts from reliable sources',
                'Cross-reference multiple reports',
                'Prefer original research over aggregators'
            ]
        }
    
    def explain_digital_marketing_percentage(self, company_name: str, 
                                            digital_pct: float, 
                                            source: str, method: str) -> Dict:
        """Explain digital marketing percentage"""
        return {
            'metric': 'Digital Marketing %',
            'value': f'{digital_pct:.0f}%',
            'data_sources': [source],
            'methodology': method,
            'how_calculated': [
                '1. Search earnings call transcripts for digital spend mentions',
                '2. Search company press releases and investor materials',
                '3. If no direct data, use industry baseline + company adjustment',
                '4. Analyze company digital maturity (tech stack, e-commerce presence)',
                '5. Apply adjustment factor based on company characteristics'
            ],
            'confidence_levels': {
                'High': 'Direct statement from company',
                'Medium': 'Derived from earnings materials',
                'Low': 'Industry estimate with company adjustments'
            },
            'fallback_methodology': [
                'Determine company industry',
                'Scrape industry average digital spend %',
                'Assess company digital maturity (High/Medium/Low)',
                'Apply multiplier: High=1.3x, Medium=1.0x, Low=0.8x industry avg'
            ]
        }
    
    def explain_marketing_efficiency(self, company_name: str, efficiency: float,
                                    revenue: float, marketing_spend: float) -> Dict:
        """Explain marketing efficiency"""
        return {
            'metric': 'Marketing Efficiency',
            'value': f'${efficiency:.2f}',
            'meaning': f'For every $1 spent on marketing, generates ${efficiency:.2f} in revenue',
            'data_sources': [
                'Company revenue from Yahoo Finance',
                'Marketing spend estimation via financial analysis'
            ],
            'methodology': [
                f'1. Total revenue: ${revenue/1e9:.2f}B',
                f'2. Estimated marketing spend: ${marketing_spend/1e9:.2f}B',
                '3. Formula: Revenue ÷ Marketing Spend',
                f'4. Result: ${efficiency:.2f} per $1 spent'
            ],
            'formula': 'Total Revenue ÷ Marketing Spend',
            'interpretation': {
                'excellent': 'Above $10 per $1 spent',
                'good': '$5-10 per $1 spent',
                'average': '$3-5 per $1 spent',
                'poor': 'Below $3 per $1 spent'
            },
            'industry_comparison': 'Compared to industry average efficiency ratio'
        }
    
    def format_for_tooltip(self, explanation: Dict) -> str:
        """Format explanation for tooltip display"""
        tooltip = f"**{explanation['metric']}**\n\n"
        tooltip += f"**Value:** {explanation['value']}\n\n"
        
        if 'data_sources' in explanation:
            tooltip += "**Data Sources:**\n"
            for source in explanation['data_sources']:
                tooltip += f"• {source}\n"
            tooltip += "\n"
        
        if 'methodology' in explanation:
            tooltip += "**How Calculated:**\n"
            if isinstance(explanation['methodology'], list):
                for step in explanation['methodology']:
                    tooltip += f"• {step}\n"
            else:
                tooltip += f"• {explanation['methodology']}\n"
            tooltip += "\n"
        
        if 'formula' in explanation:
            tooltip += f"**Formula:** {explanation['formula']}\n\n"
        
        if 'confidence' in explanation:
            tooltip += f"**Confidence Level:** {explanation['confidence']}\n"
        
        return tooltip
    
    def format_for_small_text(self, explanation: Dict) -> str:
        """Format explanation as small grey text"""
        if 'data_source' in explanation:
            return f"Source: {explanation['data_source']}"
        elif 'data_sources' in explanation:
            return f"Source: {explanation['data_sources'][0]}"
        return "Source: Multiple data sources"