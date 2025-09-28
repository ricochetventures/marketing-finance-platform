# src/api/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="Marketing-Finance AI Platform")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data at startup
try:
    data_path = Path("data/processed")
    data = {}
    
    # Load companies
    companies_file = data_path / "companies.json"
    if companies_file.exists():
        with open(companies_file, 'r') as f:
            data['companies'] = json.load(f)
    else:
        # Create default companies if file doesn't exist
        default_companies = [
            'Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'Meta', 'Netflix', 'Nike',
            'Coca-Cola', 'PepsiCo', 'McDonald\'s', 'Starbucks', 'Disney', 'Walmart', 'Target',
            'Johnson & Johnson', 'Procter & Gamble', 'Unilever', 'L\'Oréal', 'Nestlé'
        ]
        data['companies'] = default_companies
        # Save for future use
        data_path.mkdir(parents=True, exist_ok=True)
        with open(companies_file, 'w') as f:
            json.dump(default_companies, f)
    
    # Load stock prices
    stock_file = data_path / "stock_prices.csv"
    if stock_file.exists():
        data['stock_prices'] = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    
    # Load other data files
    for file_name in ['ad_spend', 'agencies', 'roi']:
        csv_file = data_path / f"{file_name}.csv"
        if csv_file.exists():
            data[file_name] = pd.read_csv(csv_file)
            
    print(f"Loaded data: {list(data.keys())}")
    print(f"Companies count: {len(data.get('companies', []))}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    data = {
        'companies': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'Meta', 'Netflix', 'Nike',
                     'Coca-Cola', 'PepsiCo', 'McDonald\'s', 'Starbucks', 'Disney', 'Walmart', 'Target']
    }

# Industry classifications
INDUSTRY_MAP = {
    'Apple': 'Technology', 'Microsoft': 'Technology', 'Google': 'Technology', 'Meta': 'Technology',
    'Amazon': 'E-commerce', 'Tesla': 'Automotive', 'Netflix': 'Media',
    'Nike': 'Apparel', 'Coca-Cola': 'Beverages', 'PepsiCo': 'Beverages',
    'McDonald\'s': 'Fast Food', 'Starbucks': 'Food Service', 'Disney': 'Entertainment',
    'Walmart': 'Retail', 'Target': 'Retail', 'Johnson & Johnson': 'Healthcare',
    'Procter & Gamble': 'Consumer Goods', 'Unilever': 'Consumer Goods', 'Nestlé': 'Food & Beverage'
}

# Agency performance profiles
AGENCY_PROFILES = {
    'WPP': {'digital_strength': 0.8, 'traditional_strength': 0.9, 'creative_score': 0.85, 'data_analytics': 0.9},
    'Publicis': {'digital_strength': 0.95, 'traditional_strength': 0.7, 'creative_score': 0.8, 'data_analytics': 0.95},
    'Omnicom': {'digital_strength': 0.75, 'traditional_strength': 0.95, 'creative_score': 0.9, 'data_analytics': 0.8},
    'IPG': {'digital_strength': 0.8, 'traditional_strength': 0.8, 'creative_score': 0.85, 'data_analytics': 0.75},
    'Dentsu': {'digital_strength': 0.9, 'traditional_strength': 0.6, 'creative_score': 0.75, 'data_analytics': 0.85},
    'Havas': {'digital_strength': 0.7, 'traditional_strength': 0.7, 'creative_score': 0.8, 'data_analytics': 0.65}
}

class PredictionRequest(BaseModel):
    company: str
    agency: str
    timeframe: int = 36

class ChatMessage(BaseModel):
    message: str
    company: Optional[str] = None

@app.get("/api/companies")
async def get_companies():
    """Get list of all available companies"""
    return {"companies": sorted(data.get('companies', []))}

@app.get("/api/company/{company_name}")
async def get_company_data(company_name: str):
    """Get detailed company data"""
    
    # Base company data with realistic values
    company_data = {
        'current_price': np.random.uniform(80, 300),
        'yearly_change': np.random.uniform(-15, 25),
        'current_agency': np.random.choice(list(AGENCY_PROFILES.keys())),
        'current_roi': np.random.uniform(1.8, 3.2),
        'marketing_efficiency': np.random.uniform(75, 95),
        'digital_ratio': np.random.uniform(55, 85),
        'market_share': np.random.uniform(5, 35),
        'industry': INDUSTRY_MAP.get(company_name, 'Other')
    }
    
    try:
        # Try to load real data if available
        if 'stock_prices' in data and company_name in data['stock_prices'].columns:
            stock = data['stock_prices'][company_name].dropna()
            if len(stock) > 0:
                company_data['current_price'] = float(stock.iloc[-1])
                if len(stock) > 1:
                    company_data['yearly_change'] = float((stock.iloc[-1] / stock.iloc[0] - 1) * 100)
        
        # Add historical data for charts
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        prices = np.random.uniform(company_data['current_price'] * 0.8,
                                 company_data['current_price'] * 1.2, 12)
        
        company_data['stock_history'] = {
            'dates': dates.strftime('%Y-%m-%d').tolist(),
            'prices': prices.tolist()
        }
        
        roi_values = np.random.uniform(1.5, 3.5, 12)
        company_data['roi_history'] = {
            'dates': dates.strftime('%Y-%m-%d').tolist(),
            'values': roi_values.tolist()
        }
    
    except Exception as e:
        print(f"Error processing company data: {e}")
    
    return company_data

@app.post("/api/predict")
async def predict_scenario(request: PredictionRequest):
    """Generate predictions for agency switch scenario"""
    
    # Get agency profile
    agency_profile = AGENCY_PROFILES.get(request.agency, AGENCY_PROFILES['WPP'])
    
    # Calculate prediction based on company industry and agency strengths
    industry = INDUSTRY_MAP.get(request.company, 'Other')
    
    # Industry-specific factors
    industry_factors = {
        'Technology': {'digital_weight': 0.8, 'traditional_weight': 0.2},
        'E-commerce': {'digital_weight': 0.9, 'traditional_weight': 0.1},
        'Beverages': {'digital_weight': 0.4, 'traditional_weight': 0.6},
        'Fast Food': {'digital_weight': 0.6, 'traditional_weight': 0.4},
        'Automotive': {'digital_weight': 0.7, 'traditional_weight': 0.3}
    }
    
    factors = industry_factors.get(industry, {'digital_weight': 0.6, 'traditional_weight': 0.4})
    
    # Calculate agency fit score
    fit_score = (agency_profile['digital_strength'] * factors['digital_weight'] +
                agency_profile['traditional_strength'] * factors['traditional_weight'] +
                agency_profile['creative_score'] * 0.3 +
                agency_profile['data_analytics'] * 0.2) / 1.5
    
    # Generate prediction
    base_return = (fit_score - 0.7) * 0.3  # Convert to return percentage
    base_return = np.clip(base_return, -0.1, 0.2)  # Reasonable bounds
    
    # Generate time series projection
    months = list(range(1, request.timeframe + 1))
    projected_values = []
    cumulative = 100
    
    for month in months:
        # Add seasonality and noise
        seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * month / 12)
        monthly_return = (base_return / 12) * seasonal_factor
        noise = np.random.normal(0, 0.02)
        cumulative *= (1 + monthly_return + noise)
        projected_values.append(cumulative)
    
    confidence_margin = abs(base_return) * 0.5
    
    return {
        'company': request.company,
        'agency': request.agency,
        'predicted_impact': base_return * 100,
        'confidence_interval': [
            (base_return - confidence_margin) * 100,
            (base_return + confidence_margin) * 100
        ],
        'projection': {
            'months': months,
            'values': projected_values
        },
        'recommendation': "Recommended" if base_return > 0.05 else "Moderate" if base_return > 0 else "Not Recommended",
        'fit_score': fit_score,
        'agency_strengths': agency_profile
    }

@app.post("/api/compare-agencies")
async def compare_agencies(company: str = "Apple"):
    """Compare all agencies for a company"""
    
    agencies = list(AGENCY_PROFILES.keys())
    comparisons = []
    
    # Get industry for company
    industry = INDUSTRY_MAP.get(company, 'Other')
    
    for agency in agencies:
        # Use the same prediction logic as above
        agency_profile = AGENCY_PROFILES[agency]
        
        industry_factors = {
            'Technology': {'digital_weight': 0.8, 'traditional_weight': 0.2},
            'E-commerce': {'digital_weight': 0.9, 'traditional_weight': 0.1},
            'Beverages': {'digital_weight': 0.4, 'traditional_weight': 0.6},
            'Fast Food': {'digital_weight': 0.6, 'traditional_weight': 0.4},
            'Automotive': {'digital_weight': 0.7, 'traditional_weight': 0.3}
        }
        
        factors = industry_factors.get(industry, {'digital_weight': 0.6, 'traditional_weight': 0.4})
        
        fit_score = (agency_profile['digital_strength'] * factors['digital_weight'] +
                    agency_profile['traditional_strength'] * factors['traditional_weight'] +
                    agency_profile['creative_score'] * 0.3 +
                    agency_profile['data_analytics'] * 0.2) / 1.5
        
        base_return = (fit_score - 0.7) * 0.3
        predicted_roi = fit_score * 2 + np.random.uniform(0.5, 1.0)  # Convert to ROI
        
        comparisons.append({
            'agency': agency,
            'predicted_roi': predicted_roi,
            'stock_impact': base_return * 100,
            'confidence': fit_score,
            'risk_score': 1 - fit_score,
            'fit_score': fit_score
        })
    
    # Sort by predicted ROI
    comparisons.sort(key=lambda x: x['predicted_roi'], reverse=True)
    
    return {
        'company': company,
        'industry': industry,
        'comparisons': comparisons,
        'best_choice': comparisons[0]['agency'] if comparisons else None
    }

@app.post("/api/chat")
async def chat_with_agent(message: ChatMessage):
    """Enhanced chat with industry-specific insights"""
    
    company = message.company or "this company"
    user_message = message.message.lower()
    
    # Determine response based on message content
    if any(word in user_message for word in ['switch', 'change', 'agency']):
        if any(agency.lower() in user_message for agency in AGENCY_PROFILES.keys()):
            # Specific agency mentioned
            mentioned_agency = None
            for agency in AGENCY_PROFILES.keys():
                if agency.lower() in user_message:
                    mentioned_agency = agency
                    break
            
            if mentioned_agency:
                industry = INDUSTRY_MAP.get(company, 'Other')
                agency_profile = AGENCY_PROFILES[mentioned_agency]
                
                response = f"Based on my analysis, if {company} switches to {mentioned_agency}, "
                
                if industry == 'Technology' and agency_profile['digital_strength'] > 0.85:
                    response += f"this could be highly beneficial. {mentioned_agency} has strong digital capabilities (score: {agency_profile['digital_strength']}) which aligns well with tech industry needs. I predict a 8-15% positive impact on marketing ROI."
                elif industry == 'Beverages' and agency_profile['traditional_strength'] > 0.85:
                    response += f"this could work well. {mentioned_agency} excels in traditional media (score: {agency_profile['traditional_strength']}) which is crucial for beverage brands. Expected impact: 5-12% ROI improvement."
                else:
                    response += f"the fit appears moderate. {mentioned_agency}'s strengths may not perfectly align with {industry} industry needs. Expected impact: 2-8% change in ROI."
                
                return {'narrative': response, 'type': 'agency_analysis'}
    
    elif any(word in user_message for word in ['roi', 'performance', 'predict']):
        industry = INDUSTRY_MAP.get(company, 'Other')
        response = f"For {company} in the {industry} industry, I analyze multiple factors including current agency performance, industry trends, and competitive positioning. "
        response += f"Based on historical data, {industry} companies typically see 15-25% variance in ROI based on agency selection. "
        response += "Would you like me to analyze a specific agency switch scenario?"
        
        return {'narrative': response, 'type': 'performance_analysis'}
    
    else:
        # General response
        response = f"I can help analyze {company}'s marketing strategy and agency relationships. "
        response += "Ask me about specific agency switches, ROI predictions, or industry comparisons. "
        response += "For example: 'What if " + company + " switches to Publicis?' or 'How does " + company + " compare to industry peers?'"
        
        return {'narrative': response, 'type': 'general'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
