import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import simpy
from scipy import stats
import random

@dataclass
class CompanyState:
    """Digital twin state representation"""
    name: str
    current_agency: str
    marketing_spend: float
    revenue: float
    stock_price: float
    market_share: float
    brand_health: float
    customer_satisfaction: float
    
class MarketingDigitalTwin:
    """Digital twin for marketing-finance simulation"""
    
    def __init__(self, company_data: pd.DataFrame, 
                 market_data: pd.DataFrame,
                 ml_models: Dict):
        self.company_data = company_data
        self.market_data = market_data
        self.ml_models = ml_models
        self.state_history = []
        self.current_state = None
        
    def initialize_twin(self, company_name: str) -> CompanyState:
        """Initialize digital twin with current company state"""
        
        company_info = self.company_data[
            self.company_data['Company'] == company_name
        ].iloc[-1]
        
        self.current_state = CompanyState(
            name=company_name,
            current_agency=company_info['Agency'],
            marketing_spend=company_info['Marketing_Spend'],
            revenue=company_info['Revenue'],
            stock_price=company_info['Stock_Price'],
            market_share=company_info['Market_Share'],
            brand_health=company_info.get('Brand_Health', 0.7),
            customer_satisfaction=company_info.get('Customer_Satisfaction', 0.75)
        )
        
        return self.current_state
    
    def simulate_scenario(self, 
                         scenario: Dict,
                         time_horizon: int = 12,
                         num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for a scenario"""
        
        results = {
            'revenue': [],
            'stock_price': [],
            'roi': [],
            'market_share': [],
            'confidence_intervals': {}
        }
        
        for sim in range(num_simulations):
            # Reset to initial state
            sim_state = self.current_state
            sim_results = self._run_single_simulation(
                sim_state, scenario, time_horizon
            )
            
            results['revenue'].append(sim_results['revenue'])
            results['stock_price'].append(sim_results['stock_price'])
            results['roi'].append(sim_results['roi'])
            results['market_share'].append(sim_results['market_share'])
        
        # Calculate statistics
        for metric in ['revenue', 'stock_price', 'roi', 'market_share']:
            values = np.array(results[metric])
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
            results['confidence_intervals'][metric] = (
                np.percentile(values, 5),
                np.percentile(values, 95)
            )
        
        return results
    
    def _run_single_simulation(self, 
                               initial_state: CompanyState,
                               scenario: Dict,
                               time_horizon: int) -> Dict:
        """Run a single simulation path"""
        
        state = initial_state
        results = []
        
        for month in range(time_horizon):
            # Apply scenario changes
            if 'new_agency' in scenario and month == 0:
                state = self._simulate_agency_change(state, scenario['new_agency'])
            
            if 'spend_change' in scenario:
                state.marketing_spend *= (1 + scenario['spend_change'])
            
            # Simulate market dynamics
            market_shock = np.random.normal(0, 0.02)  # 2% volatility
            competitive_pressure = self._simulate_competition(state)
            
            # Update state using ML predictions
            features = self._state_to_features(state)
            
            # Predict outcomes
            predicted_revenue = self.ml_models['revenue'].predict(features)[0]
            predicted_stock = self.ml_models['stock'].predict(features)[0]
            
            # Add stochastic elements
            revenue_noise = np.random.normal(1, 0.05)
            stock_noise = np.random.normal(1, 0.03)
            
            # Update state
            state.revenue = predicted_revenue * revenue_noise
            state.stock_price = predicted_stock * stock_noise * (1 + market_shock)
            state.market_share *= (1 - competitive_pressure)
            
            # Calculate ROI
            roi = (state.revenue - state.marketing_spend) / state.marketing_spend
            
            results.append({
                'month': month,
                'revenue': state.revenue,
                'stock_price': state.stock_price,
                'roi': roi,
                'market_share': state.market_share
            })
        
        # Return final metrics
        final_results = pd.DataFrame(results)
        return {
            'revenue': final_results['revenue'].iloc[-1],
            'stock_price': final_results['stock_price'].iloc[-1],
            'roi': final_results['roi'].mean(),
            'market_share': final_results['market_share'].iloc[-1]
        }
    
    def _simulate_agency_change(self, state: CompanyState, 
                                new_agency: str) -> CompanyState:
        """Simulate the impact of changing agencies"""
        
        # Agency change costs (3-6 months of reduced efficiency)
        transition_cost = state.marketing_spend * 0.15
        
        # Learning curve effect
        efficiency_drop = 0.8  # 20% efficiency drop initially
        
        # Update state
        state.current_agency = new_agency
        state.marketing_spend += transition_cost
        state.brand_health *= efficiency_drop
        
        return state
    
    def _simulate_competition(self, state: CompanyState) -> float:
        """Simulate competitive dynamics"""
        
        # Competitors respond to changes
        if state.marketing_spend > self.market_data['avg_spend'].mean():
            competitive_response = np.random.uniform(0.01, 0.03)
        else:
            competitive_response = np.random.uniform(0, 0.01)
        
        return competitive_response
    
    def _state_to_features(self, state: CompanyState) -> pd.DataFrame:
        """Convert state to ML model features"""
        
        features = pd.DataFrame({
            'marketing_spend': [state.marketing_spend],
            'current_revenue': [state.revenue],
            'market_share': [state.market_share],
            'brand_health': [state.brand_health],
            'customer_satisfaction': [state.customer_satisfaction],
            'agency_is_holding': [1 if 'WPP' in state.current_agency else 0],
            # Add more features as needed
        })
        
        return features
    
    def optimize_marketing_mix(self, 
                               constraints: Dict,
                               objective: str = 'roi') -> Dict:
        """Optimize marketing mix using simulation"""
        
        from scipy.optimize import differential_evolution
        
        def objective_function(params):
            # Unpack parameters
            digital_spend, tv_spend, print_spend = params
            total_spend = digital_spend + tv_spend + print_spend
            
            # Create scenario
            scenario = {
                'digital_ratio': digital_spend / total_spend,
                'tv_ratio': tv_spend / total_spend,
                'print_ratio': print_spend / total_spend,
                'total_spend': total_spend
            }
            
            # Run simulation
            results = self.simulate_scenario(scenario, num_simulations=100)
            
            # Return negative for maximization
            if objective == 'roi':
                return -results['roi_mean']
            elif objective == 'revenue':
                return -results['revenue_mean']
            else:
                return -results['stock_price_mean']
        
        # Set bounds
        bounds = [
            (constraints.get('digital_min', 0), constraints.get('digital_max', 1e8)),
            (constraints.get('tv_min', 0), constraints.get('tv_max', 1e8)),
            (constraints.get('print_min', 0), constraints.get('print_max', 1e8))
        ]
        
        # Optimize
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42
        )
        
        # Extract optimal mix
        digital_opt, tv_opt, print_opt = result.x
        total_opt = digital_opt + tv_opt + print_opt
        
        return {
            'optimal_digital_spend': digital_opt,
            'optimal_tv_spend': tv_opt,
            'optimal_print_spend': print_opt,
            'optimal_total_spend': total_opt,
            'digital_ratio': digital_opt / total_opt,
            'tv_ratio': tv_opt / total_opt,
            'print_ratio': print_opt / total_opt,
            'expected_roi': -result.fun
        }
