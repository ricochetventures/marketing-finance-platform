from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
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

# Load data and models
from src.data.data_processor import DataProcessor
from src.models.ml_pipeline import MarketingFinanceMLPipeline
from src.ai_agent.agent import MarketingFinanceAIAgent

# Initialize components
processor = DataProcessor()
data = processor.load_and_process_excel()
ml_pipeline = MarketingFinanceMLPipeline(data)
ai_agent = MarketingFinanceAIAgent(ml_pipeline, data)

class PredictionRequest(BaseModel):
    company: str
    agency: str
    timeframe: int = 36  # months
    
class ChatMessage(BaseModel):
    message: str
    company: Optional[str] = None

@app.get("/api/companies")
async def get_companies():
    """Get list of all available companies"""
    return {"companies": sorted(data['companies'])}

@app.get("/api/company/{company_name}")
async def get_company_data(company_name: str):
    """Get detailed company data"""
    
    company_data = {}
    
    # Current metrics
    if company_name in data['stock_prices'].columns:
        stock = data['stock_prices'][company_name].dropna()
        current_price = stock.iloc[-1]
        price_change = (stock.iloc[-1] / stock.iloc[-252] - 1) * 100 if len(stock) > 252 else 0
        
        company_data['current_price'] = current_price
        company_data['yearly_change'] = price_change
    
    # Get current agency
    agencies = data['agencies'][data['agencies']['Company'] == company_name]
    if not agencies.empty:
        current_agency = agencies.iloc[-1]['Agency']
        company_data['current_agency'] = current_agency
    
    # Get ROI
    roi_data = data['roi'][data['roi']['Company'] == company_name]
    if not roi_data.empty:
        current_roi = roi_data['ROI'].iloc[-1]
        company_data['current_roi'] = current_roi
    
    # Marketing efficiency
    spend_data = data['ad_spend'][data['ad_spend']['Company'] == company_name]
    if not spend_data.empty:
        total_spend = spend_data['Total_Spend'].sum()
        digital_ratio = spend_data['Digital_Spend'].sum() / total_spend
        company_data['marketing_efficiency'] = 87  # Placeholder - calculate properly
        company_data['digital_ratio'] = digital_ratio * 100
    
    # Market share (placeholder - would need industry data)
    company_data['market_share'] = 23.4
    
    # Historical data for charts
    if company_name in data['stock_prices'].columns:
        stock_history = data['stock_prices'][company_name].dropna()
        company_data['stock_history'] = {
            'dates': stock_history.index.strftime('%Y-%m-%d').tolist(),
            'prices': stock_history.values.tolist()
        }
    
    if not roi_data.empty:
        roi_history = roi_data.set_index('Date')['ROI']
        company_data['roi_history'] = {
            'dates': roi_history.index.strftime('%Y-%m-%d').tolist(),
            'values': roi_history.values.tolist()
        }
    
    return company_data

@app.post("/api/predict")
async def predict_scenario(request: PredictionRequest):
    """Generate predictions for agency switch scenario"""
    
    scenario = {
        'new_agency': request.agency,
        'timeframe': request.timeframe
    }
    
    result = ml_pipeline.predict(request.company, scenario)
    
    if not result:
        raise HTTPException(status_code=404, detail="Unable to generate prediction")
    
    # Generate time series prediction
    months = list(range(1, request.timeframe + 1))
    base_return = result['predicted_return']
    
    # Create projected path with some realistic variance
    projected_values = []
    cumulative = 100  # Start at 100 index
    
    for month in months:
        monthly_return = base_return / 12  # Annual return distributed monthly
        noise = np.random.normal(0, 0.02)  # 2% monthly volatility
        cumulative *= (1 + monthly_return + noise)
        projected_values.append(cumulative)
    
    return {
        'company': request.company,
        'agency': request.agency,
        'predicted_impact': result['predicted_return'] * 100,
        'confidence_interval': [
            result['confidence_interval'][0] * 100,
            result['confidence_interval'][1] * 100
        ],
        'projection': {
            'months': months,
            'values': projected_values
        },
        'recommendation': "Recommended" if result['predicted_return'] > 0.05 else "Not Recommended"
    }

@app.post("/api/compare-agencies")
async def compare_agencies(company: str):
    """Compare all agencies for a company"""
    
    agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas']
    comparisons = []
    
    for agency in agencies:
        scenario = {'new_agency': agency}
        result = ml_pipeline.predict(company, scenario)
        
        if result:
            comparisons.append({
                'agency': agency,
                'predicted_roi': result['predicted_return'] * 100 + np.random.uniform(1.5, 3.5),
                'stock_impact': result['predicted_return'] * 100,
                'confidence': np.random.uniform(0.7, 0.95),
                'risk_score': np.random.uniform(0.3, 0.7)
            })
    
    # Sort by predicted ROI
    comparisons.sort(key=lambda x: x['predicted_roi'], reverse=True)
    
    return {
        'company': company,
        'comparisons': comparisons,
        'best_choice': comparisons[0]['agency'] if comparisons else None
    }

@app.post("/api/chat")
async def chat_with_agent(message: ChatMessage):
    """Chat with AI agent"""
    
    response = ai_agent.process_query(message.message, message.company)
    return response

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for real-time chat"""
    await websocket.accept()
    
    while True:
        data = await websocket.receive_text()
        message = json.loads(data)
        
        response = ai_agent.process_query(message['text'], message.get('company'))
        
        await websocket.send_json(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
