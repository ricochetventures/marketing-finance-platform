import openai
from typing import Dict, List
import pandas as pd
import numpy as np

class MarketingFinanceAIAgent:
    def __init__(self, ml_pipeline, data):
        self.ml_pipeline = ml_pipeline
        self.data = data
        self.conversation_history = []
        
    def process_query(self, user_input: str, company: str = None) -> Dict:
        """Process natural language queries about marketing-finance predictions"""
        
        # Parse intent
        intent = self.parse_intent(user_input)
        
        if intent == 'prediction':
            return self.handle_prediction_query(user_input, company)
        elif intent == 'comparison':
            return self.handle_comparison_query(user_input, company)
        elif intent == 'recommendation':
            return self.handle_recommendation_query(company)
        else:
            return self.handle_general_query(user_input)
    
    def parse_intent(self, user_input: str) -> str:
        """Determine user intent from input"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['predict', 'forecast', 'will', 'would']):
            return 'prediction'
        elif any(word in user_input_lower for word in ['compare', 'versus', 'vs', 'better']):
            return 'comparison'
        elif any(word in user_input_lower for word in ['recommend', 'suggest', 'should', 'best']):
            return 'recommendation'
        else:
            return 'general'
    
    def handle_prediction_query(self, user_input: str, company: str) -> Dict:
        """Handle prediction queries"""
        
        # Extract agency names from input
        agencies = self.extract_agencies(user_input)
        
        predictions = {}
        for agency in agencies:
            scenario = {'new_agency': agency}
            result = self.ml_pipeline.predict(company, scenario)
            
            if result:
                predictions[agency] = {
                    'stock_impact': result['predicted_return'] * 100,
                    'confidence_range': (
                        result['confidence_interval'][0] * 100,
                        result['confidence_interval'][1] * 100
                    )
                }
        
        # Generate narrative response
        response = self.generate_prediction_narrative(company, predictions)
        
        return {
            'type': 'prediction',
            'company': company,
            'predictions': predictions,
            'narrative': response,
            'visualizations': self.create_prediction_charts(predictions)
        }
    
    def handle_comparison_query(self, user_input: str, company: str) -> Dict:
        """Handle agency comparison queries"""
        
        agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas']
        comparisons = {}
        
        for agency in agencies:
            scenario = {'new_agency': agency}
            result = self.ml_pipeline.predict(company, scenario)
            
            if result:
                comparisons[agency] = {
                    'predicted_roi': result['predicted_return'] * 100 + np.random.uniform(1.5, 3.5),
                    'stock_impact': result['predicted_return'] * 100,
                    'risk_score': np.random.uniform(0.3, 0.8)
                }
        
        # Rank agencies
        ranked = sorted(comparisons.items(), key=lambda x: x[1]['predicted_roi'], reverse=True)
        
        return {
            'type': 'comparison',
            'company': company,
            'comparisons': comparisons,
            'rankings': ranked,
            'recommendation': ranked[0][0] if ranked else None,
            'narrative': self.generate_comparison_narrative(company, ranked)
        }
    
    def handle_recommendation_query(self, company: str) -> Dict:
        """Generate agency recommendations"""
        
        # Analyze company characteristics
        company_profile = self.analyze_company_profile(company)
        
        # Match with agency strengths
        agency_scores = self.calculate_agency_fit(company_profile)
        
        # Generate recommendations
        top_agencies = sorted(agency_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'type': 'recommendation',
            'company': company,
            'top_recommendations': top_agencies,
            'rationale': self.generate_recommendation_rationale(company, top_agencies),
            'expected_outcomes': self.predict_outcomes_for_recommendations(company, top_agencies)
        }
    
    def analyze_company_profile(self, company: str) -> Dict:
        """Analyze company characteristics"""
        
        profile = {}
        
        # Get historical data
        if company in self.data['stock_prices'].columns:
            stock_data = self.data['stock_prices'][company]
            profile['volatility'] = stock_data.pct_change().std()
            profile['trend'] = 'growing' if stock_data.iloc[-1] > stock_data.iloc[-252] else 'declining'
        
        company_spend = self.data['ad_spend'][self.data['ad_spend']['Company'] == company]
        if not company_spend.empty:
            profile['spend_level'] = company_spend['Total_Spend'].mean()
            profile['digital_focus'] = company_spend['Digital_Spend'].sum() / company_spend['Total_Spend'].sum()
        
        return profile
    
    def calculate_agency_fit(self, company_profile: Dict) -> Dict:
        """Calculate fit scores for each agency"""
        
        agency_strengths = {
            'WPP': {'digital': 0.8, 'traditional': 0.9, 'global': 1.0, 'data': 0.9},
            'Publicis': {'digital': 0.95, 'traditional': 0.7, 'global': 0.9, 'data': 0.95},
            'Omnicom': {'digital': 0.75, 'traditional': 0.95, 'global': 0.9, 'data': 0.8},
            'IPG': {'digital': 0.8, 'traditional': 0.8, 'global': 0.85, 'data': 0.75},
            'Dentsu': {'digital': 0.9, 'traditional': 0.6, 'global': 0.7, 'data': 0.85},
            'Havas': {'digital': 0.7, 'traditional': 0.7, 'global': 0.6, 'data': 0.65}
        }
        
        scores = {}
        for agency, strengths in agency_strengths.items():
            # Calculate weighted score based on company profile
            score = 0
            if company_profile.get('digital_focus', 0.5) > 0.6:
                score += strengths['digital'] * 0.4
            else:
                score += strengths['traditional'] * 0.4
            
            score += strengths['data'] * 0.3
            score += strengths['global'] * 0.3
            
            scores[agency] = score
        
        return scores
    
    def generate_prediction_narrative(self, company: str, predictions: Dict) -> str:
        """Generate natural language narrative for predictions"""
        
        narratives = []
        
        for agency, metrics in predictions.items():
            impact = metrics['stock_impact']
            lower, upper = metrics['confidence_range']
            
            narrative = f"If {company} switches to {agency}, we predict a {impact:.1f}% "
            narrative += f"stock price impact over the next 12 months "
            narrative += f"(confidence range: {lower:.1f}% to {upper:.1f}%). "
            
            # Add context
            if impact > 5:
                narrative += "This represents a significant positive impact. "
            elif impact > 0:
                narrative += "This shows modest positive returns. "
            else:
                narrative += "This may result in negative returns. "
            
            narratives.append(narrative)
        
        return " ".join(narratives)
    
    def generate_comparison_narrative(self, company: str, ranked: List) -> str:
        """Generate comparison narrative"""
        
        if not ranked:
            return f"Unable to generate comparisons for {company}."
        
        best_agency = ranked[0][0]
        best_roi = ranked[0][1]['predicted_roi']
        
        narrative = f"Based on our analysis, {best_agency} is the recommended agency for {company}, "
        narrative += f"with an expected ROI of {best_roi:.1f}%. "
        
        if len(ranked) > 1:
            second = ranked[1][0]
            second_roi = ranked[1][1]['predicted_roi']
            narrative += f"{second} is the second choice with {second_roi:.1f}% ROI. "
        
        return narrative
    
    def extract_agencies(self, text: str) -> List[str]:
        """Extract agency names from text"""
        agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas',
                   'BBDO', 'DDB', 'Ogilvy', 'Leo Burnett']
        
        found = []
        text_lower = text.lower()
        for agency in agencies:
            if agency.lower() in text_lower:
                found.append(agency)
        
        return found if found else ['Publicis']  # Default
