from econml.dml import CausalForestDML, LinearDML
from econml.metalearners import TLearner, SLearner, XLearner
from dowhy import CausalModel
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class MarketingCausalInference:
    """Causal inference for marketing impact on financial performance"""
    
    def __init__(self, method: str = 'causal_forest'):
        self.method = method
        self.model = None
        self.treatment_effects = None
        self.confidence_intervals = None
        
    def estimate_agency_change_impact(self,
                                     data: pd.DataFrame,
                                     treatment_col: str = 'agency_change',
                                     outcome_col: str = 'roi',
                                     confounders: List[str] = None) -> Dict:
        """Estimate causal impact of agency changes"""
        
        if confounders is None:
            confounders = ['company_size', 'industry', 'past_performance',
                          'market_conditions', 'competitor_spend']
        
        # Prepare data
        Y = data[outcome_col].values
        T = data[treatment_col].values
        X = data[confounders].values
        
        # Build causal model based on method
        if self.method == 'causal_forest':
            self.model = CausalForestDML(
                model_y=GradientBoostingRegressor(),
                model_t=LogisticRegression(),
                n_estimators=100,
                min_samples_leaf=10,
                random_state=42
            )
        elif self.method == 'linear_dml':
            self.model = LinearDML(
                model_y=GradientBoostingRegressor(),
                model_t=LogisticRegression(),
                random_state=42
            )
        elif self.method == 't_learner':
            self.model = TLearner(
                models=GradientBoostingRegressor()
            )
        
        # Fit model
        self.model.fit(Y, T, X=X)
        
        # Get treatment effects
        self.treatment_effects = self.model.effect(X)
        
        # Get confidence intervals
        lower, upper = self.model.effect_interval(X, alpha=0.05)
        self.confidence_intervals = (lower, upper)
        
        # Calculate average treatment effect
        ate = np.mean(self.treatment_effects)
        ate_ci = (np.mean(lower), np.mean(upper))
        
        # Heterogeneous treatment effects
        hte_analysis = self._analyze_heterogeneous_effects(X, confounders)
        
        return {
            'average_treatment_effect': ate,
            'ate_confidence_interval': ate_ci,
            'individual_effects': self.treatment_effects,
            'heterogeneous_effects': hte_analysis,
            'significant': ate_ci[0] > 0 or ate_ci[1] < 0
        }
    
    def estimate_marketing_spend_elasticity(self,
                                           data: pd.DataFrame,
                                           spend_col: str = 'marketing_spend',
                                           outcome_col: str = 'revenue') -> Dict:
        """Estimate elasticity of revenue to marketing spend"""
        
        # Log transform for elasticity
        data['log_spend'] = np.log(data[spend_col] + 1)
        data['log_revenue'] = np.log(data[outcome_col] + 1)
        
        # Control variables
        controls = ['gdp_growth', 'consumer_sentiment', 'competitor_spend',
                   'seasonality', 'company_size']
        
        X = data[controls].values
        T = data['log_spend'].values.reshape(-1, 1)
        Y = data['log_revenue'].values
        
        # Estimate elasticity using DML
        model = LinearDML(
            model_y=GradientBoostingRegressor(),
            model_t=Ridge(),
            random_state=42
        )
        
        model.fit(Y, T, X=X)
        
        # Get elasticity estimate
        elasticity = model.coef_[0]
        ci_lower, ci_upper = model.coef__interval(alpha=0.05)
        
        # Calculate optimal spend level
        optimal_spend = self._find_optimal_spend(data, elasticity)
        
        return {
            'elasticity': elasticity,
            'confidence_interval': (ci_lower[0], ci_upper[0]),
            'interpretation': f"1% increase in spend → {elasticity:.2f}% increase in revenue",
            'optimal_spend_level': optimal_spend,
            'current_efficiency': self._calculate_efficiency(data, elasticity)
        }
    
    def _analyze_heterogeneous_effects(self, X: np.ndarray,
                                      feature_names: List[str]) -> Dict:
        """Analyze how treatment effects vary by characteristics"""
        
        hte_analysis = {}
        
        for i, feature in enumerate(feature_names):
            # Split by median
            median = np.median(X[:, i])
            below_median = self.treatment_effects[X[:, i] <= median]
            above_median = self.treatment_effects[X[:, i] > median]
            
            hte_analysis[feature] = {
                'below_median_effect': np.mean(below_median),
                'above_median_effect': np.mean(above_median),
                'difference': np.mean(above_median) - np.mean(below_median),
                'significant': stats.ttest_ind(below_median, above_median).pvalue < 0.05
            }
        
        return hte_analysis
    
    def _find_optimal_spend(self, data: pd.DataFrame, elasticity: float) -> float:
        """Find optimal marketing spend level"""
        
        # Use elasticity to find diminishing returns point
        current_spend = data['marketing_spend'].mean()
        revenue = data['revenue'].mean()
        
        # Optimal spend occurs where marginal return = marginal cost
        # Assuming cost of capital is 10%
        cost_of_capital = 0.10
        
        optimal_ratio = elasticity / (1 + cost_of_capital)
        optimal_spend = current_spend * optimal_ratio
        
        return optimal_spend
    
    def _calculate_efficiency(self, data: pd.DataFrame, elasticity: float) -> Dict:
        """Calculate current marketing efficiency"""
        
        current_roi = data['revenue'].sum() / data['marketing_spend'].sum()
        marginal_roi = elasticity * current_roi
        
        return {
            'current_roi': current_roi,
            'marginal_roi': marginal_roi,
            'efficiency_ratio': marginal_roi / current_roi,
            'recommendation': 'Increase spend' if marginal_roi > 1 else 'Decrease spend'
        }
