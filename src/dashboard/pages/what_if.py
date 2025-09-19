import streamlit as st
import pandas as pd
import numpy as np
from src.models.base_model import BaseModel
from src.fairness.metrics import FairnessMetrics

class WhatIfSimulator:
    def __init__(self):
        self.model = BaseModel.load_latest()
        self.metrics = FairnessMetrics()
    
    def render(self):
        st.title("What-If Analysis Simulator")
        
        # Feature adjustment sliders
        st.sidebar.header("Adjust Features")
        features = self._create_feature_sliders()
        
        # Protected attributes selection
        protected_attrs = st.multiselect(
            "Select Protected Attributes",
            ["race", "gender", "age_group"],
            default=["race"]
        )
        
        # Run simulation
        if st.button("Run Simulation"):
            results = self._simulate(features, protected_attrs)
            self._display_results(results)
    
    def _create_feature_sliders(self):
        features = {}
        features['income'] = st.slider("Income", 20000, 200000, 50000)
        features['credit_score'] = st.slider("Credit Score", 300, 850, 650)
        features['loan_amount'] = st.slider("Loan Amount", 10000, 1000000, 200000)
        features['dti_ratio'] = st.slider("DTI Ratio", 0.0, 1.0, 0.4)
        return features
    
    def _simulate(self, features, protected_attrs):
        # Simulation logic here
        pass
    
    def _display_results(self, results):
        # Results visualization here
        pass

if __name__ == "__main__":
    simulator = WhatIfSimulator()
    simulator.render()




    import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

class BiasDetector:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        
    def detect_disparate_impact(self, 
                              data: pd.DataFrame,
                              protected_attributes: List[str],
                              outcome_column: str) -> Dict:
        """Detect disparate impact across multiple protected attributes"""
        results = {}
        
        for attr in protected_attributes:
            impact_ratio = self._calculate_disparate_impact(
                data, attr, outcome_column
            )
            results[attr] = {
                'impact_ratio': impact_ratio,
                'has_disparate_impact': impact_ratio < self.threshold
            }
            
        return results
    
    def detect_geographic_redlining(self,
                                  data: pd.DataFrame,
                                  location_column: str,
                                  outcome_column: str,
                                  demographic_data: pd.DataFrame) -> Dict:
        """Detect geographic redlining patterns"""
        results = {}
        
        # Merge demographic data with lending data
        merged_data = pd.merge(
            data,
            demographic_data,
            on=location_column
        )
        
        # Calculate approval rates by location
        approval_rates = self._calculate_location_approval_rates(
            merged_data, 
            location_column, 
            outcome_column
        )
        
        # Detect suspicious patterns
        suspicious_areas = self._identify_suspicious_areas(
            approval_rates,
            merged_data,
            demographic_data
        )
        
        return suspicious_areas
    
    def detect_intersectional_bias(self,
                                 data: pd.DataFrame,
                                 protected_attributes: List[str],
                                 outcome_column: str) -> Dict:
        """Detect bias patterns across intersecting protected attributes"""
        results = {}
        
        # Generate all possible intersections
        intersections = self._generate_intersections(
            data, 
            protected_attributes
        )
        
        for intersection in intersections:
            impact_ratio = self._calculate_intersectional_impact(
                data,
                intersection,
                outcome_column
            )
            results[str(intersection)] = {
                'impact_ratio': impact_ratio,
                'has_disparate_impact': impact_ratio < self.threshold
            }
            
        return results
    
    def _calculate_disparate_impact(self,
                                  data: pd.DataFrame,
                                  protected_attribute: str,
                                  outcome_column: str) -> float:
        # Implementation details
        pass
    
    def _calculate_location_approval_rates(self,
                                        data: pd.DataFrame,
                                        location_column: str,
                                        outcome_column: str) -> pd.Series:
        # Implementation details
        pass
    
    def _identify_suspicious_areas(self,
                                 approval_rates: pd.Series,
                                 merged_data: pd.DataFrame,
                                 demographic_data: pd.DataFrame) -> Dict:
        # Implementation details
        pass
    
    def _generate_intersections(self,
                              data: pd.DataFrame,
                              protected_attributes: List[str]) -> List[Tuple]:
        # Implementation details
        pass
    
    def _calculate_intersectional_impact(self,
                                       data: pd.DataFrame,
                                       intersection: Tuple,
                                       outcome_column: str) -> float:
        # Implementation details
        pass




import pandas as pd
from typing import Dict, List
from src.fairness.metrics import FairnessMetrics
from src.explainability.shap_analyzer import ShapAnalyzer

class NLGGenerator:
    def __init__(self):
        self.templates = self._load_templates()
        self.metrics = FairnessMetrics()
        self.shap = ShapAnalyzer()
    
    def generate_bias_report(self,
                           data: pd.DataFrame,
                           results: Dict,
                           severity: str) -> str:
        """Generate a comprehensive bias analysis report"""
        sections = []
        
        # Executive Summary
        sections.append(self._generate_executive_summary(results, severity))
        
        # Detailed Findings
        sections.append(self._generate_detailed_findings(results))
        
        # Statistical Evidence
        sections.append(self._generate_statistical_evidence(data, results))
        
        # Recommendations
        sections.append(self._generate_recommendations(results, severity))
        
        return "\n\n".join(sections)
    
    def _load_templates(self) -> Dict:
        return {
            'high_risk': "Critical bias detected in {attribute} with impact ratio of {ratio:.2f}. Immediate action required.",
            'medium_risk': "Potential bias concerns in {attribute} with impact ratio of {ratio:.2f}. Further investigation recommended.",
            'low_risk': "Minor disparities detected in {attribute} with impact ratio of {ratio:.2f}. Monitor for changes.",
            # Add more templates as needed
        }
    
    def _generate_executive_summary(self, results: Dict, severity: str) -> str:
        template = self.templates[f'{severity}_risk']
        summaries = []
        
        for attr, metrics in results.items():
            summary = template.format(
                attribute=attr,
                ratio=metrics['impact_ratio']
            )
            summaries.append(summary)
        
        return "Executive Summary:\n" + "\n".join(summaries)
    
    def _generate_detailed_findings(self, results: Dict) -> str:
        # Implementation details
        pass
    
    def _generate_statistical_evidence(self, 
                                    data: pd.DataFrame,
                                    results: Dict) -> str:
        # Implementation details
        pass
    
    def _generate_recommendations(self, 
                                results: Dict,
                                severity: str) -> str:
        # Implementation details
        pass