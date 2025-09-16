from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import redis
import json

app = FastAPI(title="Marketing-Finance Correlation Platform")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Load models (in production, use model registry)
from src.models.ensemble_model import AdvancedEnsembleModel
from src.models.causal_model import MarketingCausalInference
from src.models.digital_twin import MarketingDigitalTwin

# Initialize models
ensemble_model = AdvancedEnsembleModel.load_model("models/saved/ensemble_model.pkl")
causal_model = MarketingCausalInference()
digital_twin = None  # Initialize per company

class PredictionRequest(BaseModel):
    company: str
    new_agency: Optional[str] = None
    marketing_spend: Optional[float] = None
    time_horizon: int = 12
    include_simulation: bool = True

class CompanyAnalysisRequest(BaseModel):
    company: str
    compare_to_industry: bool = True
    include_recommendations: bool = True

class OptimizationRequest(BaseModel):
    company: str
    budget_constraint: float
    objective: str = "roi"  # roi, revenue, stock_price
    constraints: Dict = {}

@app.get("/")
async def root():
    return {
        "message": "Marketing-Finance Correlation Platform API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/analyze",
            "/optimize",
            "/simulate",
            "/companies",
            "/agencies"
        ]
    }

@app.post("/predict")
async def predict_performance(request: PredictionRequest):
    """Predict company performance with scenario analysis"""
    
    # Check cache
    cache_key = f"prediction:{request.company}:{request.new_agency}:{request.marketing_spend}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Load company data
        company_data = load_company_data(request.company)
        
        # Prepare features
        features = prepare_features(
            company_data,
            new_agency=request.new_agency,
            marketing_spend=request.marketing_spend
        )
        
        # Make predictions
        predictions = {
            'roi': ensemble_model.predict(features)[0],
            'revenue_change': ensemble_model.predict(features)[0] * 100,
            'stock_impact': ensemble_model.predict(features)[0] * 100
        }
        
        # Run simulation if requested
        if request.include_simulation:
            twin = MarketingDigitalTwin(company_data, market_data, {'ensemble': ensemble_model})
            twin.initialize_twin(request.company)
            
            scenario = {}
            if request.new_agency:
                scenario['new_agency'] = request.new_agency
            if request.marketing_spend:
                scenario['spend_change'] = (request.marketing_spend - company_data['Marketing_Spend'].iloc[-1]) / company_data['Marketing_Spend'].iloc[-1]
            
            simulation_results = twin.simulate_scenario(
                scenario,
                time_horizon=request.time_horizon,
                num_simulations=1000
            )
            
            predictions['simulation'] = {
                'expected_roi': simulation_results['roi_mean'],
                'roi_confidence': simulation_results['confidence_intervals']['roi'],
                'expected_revenue': simulation_results['revenue_mean'],
                'revenue_confidence': simulation_results['confidence_intervals']['revenue'],
                'expected_stock': simulation_results['stock_price_mean'],
                'stock_confidence': simulation_results['confidence_intervals']['stock_price']
            }
        
        # Add recommendations
        predictions['recommendations'] = generate_recommendations(predictions)
        
        # Cache result
        redis_client.setex(cache_key, 3600, json.dumps(predictions))
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_company(request: CompanyAnalysisRequest):
    """Comprehensive company analysis with industry comparison"""
    
    try:
        # Load company and industry data
        company_data = load_company_data(request.company)
        industry_data = load_industry_data(company_data['Industry'].iloc[0])
        
        analysis = {
            'company': request.company,
            'current_performance': {
                'roi': calculate_roi(company_data),
                'revenue_growth': calculate_growth(company_data['Revenue']),
                'market_share': company_data['Market_Share'].iloc[-1],
                'stock_performance': calculate_stock_performance(company_data)
            }
        }
        
        if request.compare_to_industry:
            # Industry comparison
            analysis['industry_comparison'] = {
                'roi_percentile': calculate_percentile(
                    analysis['current_performance']['roi'],
                    industry_data['ROI']
                ),
                'growth_percentile': calculate_percentile(
                    analysis['current_performance']['revenue_growth'],
                    industry_data['Growth']
                ),
                'market_position': determine_market_position(company_data, industry_data)
            }
        
        if request.include_recommendations:
            # AI-powered recommendations
            analysis['recommendations'] = {
                'optimal_agency': recommend_agency(company_data, industry_data),
                'optimal_spend': recommend_spend_level(company_data),
                'marketing_mix': recommend_marketing_mix(company_data),
                'action_items': generate_action_items(analysis)
            }
        
        # Causal analysis
        causal_results = causal_model.estimate_agency_change_impact(
            company_data,
            treatment_col='agency_change',
            outcome_col='roi'
        )
        
        analysis['causal_insights'] = {
            'agency_change_impact': causal_results['average_treatment_effect'],
            'confidence': causal_results['ate_confidence_interval'],
            'significant': causal_results['significant']
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_marketing(request: OptimizationRequest):
    """Optimize marketing mix for maximum ROI"""
    
    try:
        # Load company data
        company_data = load_company_data(request.company)
        
        # Initialize digital twin
        twin = MarketingDigitalTwin(company_data, market_data, {'ensemble': ensemble_model})
        twin.initialize_twin(request.company)
        
        # Run optimization
        optimization_results = twin.optimize_marketing_mix(
            constraints={
                'total_budget': request.budget_constraint,
                **request.constraints
            },
            objective=request.objective
        )
        
        # Compare to current state
        current_roi = calculate_roi(company_data)
        expected_improvement = (optimization_results['expected_roi'] - current_roi) / current_roi
        
        results = {
            'optimal_allocation': {
                'digital': optimization_results['optimal_digital_spend'],
                'tv': optimization_results['optimal_tv_spend'],
                'print': optimization_results['optimal_print_spend'],
                'total': optimization_results['optimal_total_spend']
            },
            'expected_performance': {
                'roi': optimization_results['expected_roi'],
                'improvement': expected_improvement * 100
            },
            'recommendations': generate_optimization_recommendations(optimization_results)
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companies")
async def list_companies():
    """Get list of available companies"""
    companies = get_available_companies()
    return {"companies": companies}

@app.get("/agencies")
async def list_agencies():
    """Get list of agencies with performance metrics"""
    agencies = get_agency_performance_metrics()
    return {"agencies": agencies}

# Helper functions
def load_company_data(company: str) -> pd.DataFrame:
    """Load company data from database"""
    # In production, load from database
    return pd.read_csv(f"data/processed/{company}_data.csv")

def load_industry_data(industry: str) -> pd.DataFrame:
    """Load industry benchmark data"""
    return pd.read_csv(f"data/processed/{industry}_industry.csv")

def prepare_features(company_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Prepare features for prediction"""
    # Feature engineering logic
    pass

def generate_recommendations(predictions: Dict) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if predictions['roi'] > 2.0:
        recommendations.append("Strong ROI expected - consider increasing investment")
    elif predictions['roi'] < 1.0:
        recommendations.append("Low ROI expected - review strategy before proceeding")
    
    return recommendations
