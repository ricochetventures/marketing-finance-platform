"""
Diminishing Returns Hypothesis Integration
Subtly incorporates the core hypothesis into the platform
Location: src/data/hypothesis_insights.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class DiminishingReturnsAnalyzer:
    """
    Analyzes whether additional marketing sophistication provides diminishing returns
    Core hypothesis: There's a point where more data doesn't improve business outcomes
    """
    
    def __init__(self):
        self.optimal_spend_ratio = None
        self.efficiency_curve = None
    
    def analyze_spend_efficiency(self, company_name: str, 
                                 current_roi: float,
                                 current_spend: float,
                                 industry_avg_roi: float) -> Dict:
        """
        Determine if company is past the point of diminishing returns
        """
        
        # Model the efficiency curve
        # As spend increases, ROI typically follows a logarithmic curve
        # Eventually plateaus or declines
        
        spend_levels = np.linspace(current_spend * 0.5, current_spend * 2, 100)
        
        # Model expected ROI at different spend levels
        # Using diminishing returns formula: ROI = a * log(spend) + b
        # Where a and b are derived from current position
        
        a = current_roi / np.log(current_spend) if current_spend > 1 else 1
        b = current_roi - a * np.log(current_spend) if current_spend > 1 else 0
        
        projected_roi = a * np.log(spend_levels) + b
        
        # Find optimal spend (where marginal ROI = 1)
        marginal_roi = np.gradient(projected_roi, spend_levels)
        optimal_idx = np.argmax(marginal_roi < 1) if any(marginal_roi < 1) else -1
        optimal_spend = spend_levels[optimal_idx] if optimal_idx != -1 else current_spend
        
        # Determine if current spend is past optimal
        efficiency_status = self._determine_efficiency_status(
            current_spend, optimal_spend, current_roi, industry_avg_roi
        )
        
        return {
            'current_spend': current_spend,
            'optimal_spend': optimal_spend,
            'efficiency_status': efficiency_status,
            'status_interpretation': self._interpret_status(efficiency_status),
            'recommendation': self._generate_recommendation(efficiency_status, current_spend, optimal_spend),
            'diminishing_returns_insight': self._generate_insight(efficiency_status, current_roi, industry_avg_roi)
        }
    
    def _determine_efficiency_status(self, current_spend: float, optimal_spend: float,
                                    current_roi: float, industry_avg: float) -> str:
        """Determine efficiency status"""
        
        spend_ratio = current_spend / optimal_spend if optimal_spend > 0 else 1
        roi_vs_industry = current_roi / industry_avg if industry_avg > 0 else 1
        
        if spend_ratio < 0.8:
            if roi_vs_industry > 1.2:
                return "HIGHLY_EFFICIENT"  # Under-spending but high ROI
            else:
                return "ROOM_TO_GROW"  # Could invest more
        elif 0.8 <= spend_ratio <= 1.2:
            if roi_vs_industry > 1.0:
                return "OPTIMAL"  # Sweet spot
            else:
                return "EFFICIENT_BUT_LAGGING"  # Efficient but below industry
        else:  # spend_ratio > 1.2
            if roi_vs_industry < 0.9:
                return "DIMINISHING_RETURNS"  # Over-spending with poor returns
            else:
                return "HIGH_SPEND_HIGH_RETURN"  # Spending more but still efficient
    
    def _interpret_status(self, status: str) -> str:
        """Human-readable interpretation"""
        
        interpretations = {
            "HIGHLY_EFFICIENT": "Marketing is highly efficient. Strong ROI with relatively low spend.",
            "ROOM_TO_GROW": "Could increase marketing investment for potentially higher returns.",
            "OPTIMAL": "Marketing spend is near optimal level for maximum efficiency.",
            "EFFICIENT_BUT_LAGGING": "Efficient spend level but ROI below industry average.",
            "DIMINISHING_RETURNS": "May be experiencing diminishing returns. Consider reallocating budget.",
            "HIGH_SPEND_HIGH_RETURN": "High investment is generating strong returns."
        }
        
        return interpretations.get(status, "Status unclear")
    
    def _generate_recommendation(self, status: str, current_spend: float, 
                                optimal_spend: float) -> str:
        """Generate actionable recommendation"""
        
        if status == "DIMINISHING_RETURNS":
            decrease_pct = ((current_spend - optimal_spend) / current_spend) * 100
            return f"Consider reducing spend by ~{decrease_pct:.0f}% and reallocating to higher-ROI channels."
        
        elif status == "ROOM_TO_GROW":
            increase_pct = ((optimal_spend - current_spend) / current_spend) * 100
            return f"Opportunity to increase spend by ~{increase_pct:.0f}% for improved returns."
        
        elif status == "OPTIMAL":
            return "Current spend level is near optimal. Focus on channel mix optimization."
        
        elif status == "HIGHLY_EFFICIENT":
            return "Maintain current efficiency while exploring selective expansion opportunities."
        
        else:
            return "Continue monitoring efficiency and adjust based on performance."
    
    def _generate_insight(self, status: str, current_roi: float, 
                         industry_avg: float) -> str:
        """Generate insight related to diminishing returns hypothesis"""
        
        if status == "DIMINISHING_RETURNS":
            return "⚠️ **Diminishing Returns Detected**: Additional marketing spend may not be generating proportional returns. This aligns with the hypothesis that excessive data sophistication can plateau effectiveness."
        
        elif status == "OPTIMAL":
            return "✅ **Optimal Efficiency Zone**: Current spend level appears to maximize ROI without reaching diminishing returns."
        
        elif status == "HIGHLY_EFFICIENT":
            return "🎯 **High Efficiency**: Strong ROI suggests effective targeting without over-sophistication."
        
        else:
            return "📊 **Efficiency Analysis**: Monitoring for signs of diminishing returns as spend scales."
    
    def calculate_sophistication_score(self, num_data_attributes: int,
                                       roi: float) -> Dict:
        """
        Calculate whether data sophistication correlates with ROI
        Core hypothesis: More attributes ≠ better performance after a threshold
        """
        
        # Model: ROI = f(data_attributes) with diminishing returns
        # Optimal attributes typically in range 2000-4000
        # Beyond that, minimal additional value
        
        optimal_threshold = 3000
        
        if num_data_attributes < optimal_threshold:
            sophistication_score = (num_data_attributes / optimal_threshold) * 100
            status = "BELOW_THRESHOLD"
        else:
            # Diminishing returns kick in
            excess = num_data_attributes - optimal_threshold
            sophistication_score = 100 - (excess / optimal_threshold) * 20  # Penalty for excess
            status = "ABOVE_THRESHOLD"
        
        return {
            'data_attributes': num_data_attributes,
            'sophistication_score': sophistication_score,
            'status': status,
            'threshold': optimal_threshold,
            'interpretation': self._interpret_sophistication(status, num_data_attributes, optimal_threshold),
            'hypothesis_insight': self._generate_sophistication_insight(status)
        }
    
    def _interpret_sophistication(self, status: str, num_attrs: int, threshold: int) -> str:
        """Interpret sophistication level"""
        
        if status == "BELOW_THRESHOLD":
            return f"Using {num_attrs:,} attributes. Could benefit from additional customer insights."
        else:
            excess = num_attrs - threshold
            return f"Using {num_attrs:,} attributes ({excess:,} above optimal threshold). May have reached point of diminishing returns."
    
    def _generate_sophistication_insight(self, status: str) -> str:
        """Generate insight about data sophistication"""
        
        if status == "ABOVE_THRESHOLD":
            return "⚠️ **Sophistication Plateau**: Data suggests diminishing returns from additional customer attributes. Focus on quality over quantity."
        else:
            return "📈 **Growth Opportunity**: Additional customer insights could improve targeting and performance."