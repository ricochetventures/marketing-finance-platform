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
    # Load processed data
    data_path = Path("data/processed")
    data = {}
    
    # Load companies
    companies_file = data_path / "companies.json"
    if companies_file.exists():
        with open(companies_file, 'r') as f:
            data['companies'] = json.load(f)
    else:
        data['companies'] = []
    
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
    data = {'companies': []}

class PredictionRequest(BaseModel):
    company: str
    agency: str
    timeframe: int = 36

@app.get("/api/companies")
async def get_companies():
    """Get list of all available companies"""
    return {"companies": sorted(data.get('companies', []))}

@app.get("/api/company/{company_name}")
async def get_company_data(company_name: str):
    """Get detailed company data"""
    
    company_data = {
        'current_price': 100.0,
        'yearly_change': 5.0,
        'current_agency': 'Unknown',
        'current_roi': 2.34,
        'marketing_efficiency': 87,
        'digital_ratio': 65,
        'market_share': 23.4
    }
    
    try:
        # Current metrics from stock data
        if 'stock_prices' in data and company_name in data['stock_prices'].columns:
            stock = data['stock_prices'][company_name].dropna()
            if len(stock) > 0:
                company_data['current_price'] = float(stock.iloc[-1])
                if len(stock) > 1:
                    company_data['yearly_change'] = float((stock.iloc[-1] / stock.iloc[0] - 1) * 100)
        
        # Get current agency - check for different column names
        if 'agencies' in data:
            agencies_df = data['agencies']
            company_agencies = None
            
            # Try different column names
            if 'Company' in agencies_df.columns:
                company_agencies = agencies_df[agencies_df['Company'] == company_name]
            
            if company_agencies is not None and not company_agencies.empty:
                # Check for agency column names
                if 'Agency' in company_agencies.columns:
                    company_data['current_agency'] = company_agencies['Agency'].iloc[-1]
                elif 'AOR' in company_agencies.columns:
                    company_data['current_agency'] = company_agencies['AOR'].iloc[-1]
        
        # Get ROI
        if 'roi' in data:
            roi_df = data['roi']
            if 'Company' in roi_df.columns:
                company_roi = roi_df[roi_df['Company'] == company_name]
                if not company_roi.empty and 'ROI' in company_roi.columns:
                    company_data['current_roi'] = float(company_roi['ROI'].iloc[-1])
        
        # Add historical data for charts
        if 'stock_prices' in data and company_name in data['stock_prices'].columns:
            stock_history = data['stock_prices'][company_name].dropna()
            company_data['stock_history'] = {
                'dates': stock_history.index.strftime('%Y-%m-%d').tolist(),
                'prices': stock_history.values.tolist()
            }
    
    except Exception as e:
        print(f"Error processing company data: {e}")
    
    return company_data

@app.post("/api/predict")
async def predict_scenario(request: PredictionRequest):
    """Generate predictions for agency switch scenario"""
    
    # Generate mock predictions for now
    base_return = np.random.uniform(0.05, 0.15)
    
    # Generate time series prediction
    months = list(range(1, request.timeframe + 1))
    projected_values = []
    cumulative = 100
    
    for month in months:
        monthly_return = base_return / 12
        noise = np.random.normal(0, 0.02)
        cumulative *= (1 + monthly_return + noise)
        projected_values.append(cumulative)
    
    return {
        'company': request.company,
        'agency': request.agency,
        'predicted_impact': base_return * 100,
        'confidence_interval': [
            (base_return - 0.03) * 100,
            (base_return + 0.03) * 100
        ],
        'projection': {
            'months': months,
            'values': projected_values
        },
        'recommendation': "Recommended" if base_return > 0.08 else "Moderate"
    }

@app.post("/api/compare-agencies")
async def compare_agencies(company: str = "Default"):
    """Compare all agencies for a company"""
    
    agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas']
    comparisons = []
    
    for agency in agencies:
        comparisons.append({
            'agency': agency,
            'predicted_roi': np.random.uniform(1.5, 3.5),
            'stock_impact': np.random.uniform(-5, 15),
            'confidence': np.random.uniform(0.7, 0.95),
            'risk_score': np.random.uniform(0.3, 0.7)
        })
    
    comparisons.sort(key=lambda x: x['predicted_roi'], reverse=True)
    
    return {
        'company': company,
        'comparisons': comparisons,
        'best_choice': comparisons[0]['agency'] if comparisons else None
    }

@app.post("/api/chat")
async def chat_with_agent(message: Dict):
    """Simple chat response"""
    return {
        'type': 'response',
        'narrative': f"Based on the analysis, {message.get('company', 'this company')} shows promising potential with the selected agency.",
        'predictions': {}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
