import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

class BiasDetector:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        
    def detect_disparate_impact(self, data: pd.DataFrame, protected_attributes: List[str], outcome_column: str) -> Dict:
        results = {}
        for attr in protected_attributes:
            impact_ratio = self._calculate_disparate_impact(data, attr, outcome_column)
            results[attr] = {
                'impact_ratio': impact_ratio,
                'has_disparate_impact': impact_ratio < self.threshold
            }
        return results
    
    def detect_geographic_redlining(self, data: pd.DataFrame, location_column: str, outcome_column: str, demographic_data: pd.DataFrame) -> Dict:
        merged_data = pd.merge(data, demographic_data, on=location_column)
        approval_rates = self._calculate_location_approval_rates(merged_data, location_column, outcome_column)
        suspicious_areas = self._identify_suspicious_areas(approval_rates, merged_data, demographic_data)
        return suspicious_areas
    
    def detect_intersectional_bias(self, data: pd.DataFrame, protected_attributes: List[str], outcome_column: str) -> Dict:
        results = {}
        intersections = self._generate_intersections(data, protected_attributes)
        for intersection in intersections:
            impact_ratio = self._calculate_intersectional_impact(data, intersection, outcome_column)
            results[str(intersection)] = {
                'impact_ratio': impact_ratio,
                'has_disparate_impact': impact_ratio < self.threshold
            }
        return results
    
    def _calculate_disparate_impact(self, data: pd.DataFrame, protected_attribute: str, outcome_column: str) -> float:
        groups = data[protected_attribute].unique()
        rates = data.groupby(protected_attribute)[outcome_column].mean()
        min_rate = rates.min()
        max_rate = rates.max()
        if max_rate == 0:
            return 0.0
        return min_rate / max_rate
    
    def _calculate_location_approval_rates(self, data: pd.DataFrame, location_column: str, outcome_column: str) -> pd.Series:
        return data.groupby(location_column)[outcome_column].mean()
    
    def _identify_suspicious_areas(self, approval_rates: pd.Series, merged_data: pd.DataFrame, demographic_data: pd.DataFrame) -> Dict:
        threshold = approval_rates.mean() - 2 * approval_rates.std()
        suspicious = approval_rates[approval_rates < threshold]
        return suspicious.to_dict()
    
    def _generate_intersections(self, data: pd.DataFrame, protected_attributes: List[str]) -> List[Tuple]:
        from itertools import product
        values = [data[attr].unique() for attr in protected_attributes]
        return list(product(*values))
    
    def _calculate_intersectional_impact(self, data: pd.DataFrame, intersection: Tuple, outcome_column: str) -> float:
        mask = np.ones(len(data), dtype=bool)
        for idx, attr in enumerate(intersection):
            mask &= (data.iloc[:, idx] == attr)
        group_rate = data[mask][outcome_column].mean()
        overall_rate = data[outcome_column].mean()
        if overall_rate == 0:
            return 0.0
        return group_rate / overall_rate