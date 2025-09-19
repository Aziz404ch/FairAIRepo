# src/fairness/metrics.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FairnessMetricResult:
    """Container for fairness metric results."""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    severity: str
    details: Dict[str, Any]
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    regulation: Optional[str] = None


class FairnessMetricsCalculator:
    """
    Calculate various fairness metrics for lending decisions.
    Implements multiple fairness criteria as per regulatory requirements.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the fairness metrics calculator.
        
        Args:
            config: Configuration dictionary with thresholds
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for fairness metrics."""
        return {
            'four_fifths_rule': {
                'threshold': 0.8,
                'marginal_threshold': 0.7,
                'regulation': 'ECOA'
            },
            'demographic_parity': {
                'threshold': 0.1,
                'regulation': 'Fair Housing Act'
            },
            'equal_opportunity': {
                'threshold': 0.1,
                'regulation': 'ECOA'
            },
            'predictive_parity': {
                'threshold': 0.1,
                'regulation': 'ECOA'
            },
            'equalized_odds': {
                'threshold': 0.1,
                'regulation': 'ECOA'
            }
        }
    
    def calculate_all_metrics(self, 
                            df: pd.DataFrame,
                            outcome_col: str,
                            protected_attr: str,
                            reference_group: Optional[str] = None) -> Dict[str, FairnessMetricResult]:
        """
        Calculate all fairness metrics for a protected attribute.
        
        Args:
            df: DataFrame with lending data
            outcome_col: Name of outcome column (e.g., 'approved')
            protected_attr: Name of protected attribute column
            reference_group: Reference group for comparison
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Four-fifths rule (Disparate Impact)
        results['four_fifths_rule'] = self.calculate_disparate_impact(
            df, outcome_col, protected_attr, reference_group
        )
        
        # Demographic Parity
        results['demographic_parity'] = self.calculate_demographic_parity(
            df, outcome_col, protected_attr, reference_group
        )
        
        # Equal Opportunity
        results['equal_opportunity'] = self.calculate_equal_opportunity(
            df, outcome_col, protected_attr, reference_group
        )
        
        # Predictive Parity (if predictions available)
        if 'predicted_probability' in df.columns:
            results['predictive_parity'] = self.calculate_predictive_parity(
                df, outcome_col, protected_attr, reference_group
            )
        
        # Equalized Odds
        results['equalized_odds'] = self.calculate_equalized_odds(
            df, outcome_col, protected_attr, reference_group
        )
        
        return results
    
    def calculate_disparate_impact(self,
                                  df: pd.DataFrame,
                                  outcome_col: str,
                                  protected_attr: str,
                                  reference_group: Optional[str] = None) -> FairnessMetricResult:
        """
        Calculate Disparate Impact (Four-Fifths Rule).
        
        Disparate Impact = P(Y=1|A=minority) / P(Y=1|A=majority)
        Should be >= 0.8 for compliance
        """
        config = self.config['four_fifths_rule']
        
        # Get unique groups
        groups = df[protected_attr].unique()
        
        # Determine reference group
        if reference_group is None:
            # Use group with highest approval rate as reference
            approval_rates = df.groupby(protected_attr)[outcome_col].mean()
            reference_group = approval_rates.idxmax()
        
        # Calculate approval rates
        group_stats = {}
        for group in groups:
            group_data = df[df[protected_attr] == group]
            n = len(group_data)
            approvals = group_data[outcome_col].sum()
            approval_rate = approvals / n if n > 0 else 0
            
            # Calculate confidence interval
            if n > 0:
                se = np.sqrt(approval_rate * (1 - approval_rate) / n)
                ci = (approval_rate - 1.96 * se, approval_rate + 1.96 * se)
            else:
                ci = (0, 0)
            
            group_stats[group] = {
                'n': n,
                'approvals': approvals,
                'approval_rate': approval_rate,
                'confidence_interval': ci
            }
        
        # Calculate disparate impact ratios
        reference_rate = group_stats[reference_group]['approval_rate']
        impact_ratios = {}
        
        for group in groups:
            if group != reference_group and reference_rate > 0:
                ratio = group_stats[group]['approval_rate'] / reference_rate
                impact_ratios[group] = ratio
        
        # Find minimum ratio (worst case)
        min_ratio = min(impact_ratios.values()) if impact_ratios else 1.0
        min_group = min(impact_ratios, key=impact_ratios.get) if impact_ratios else None
        
        # Determine severity
        if min_ratio >= config['threshold']:
            severity = 'Low'
            passed = True
        elif min_ratio >= config['marginal_threshold']:
            severity = 'Medium'
            passed = False
        else:
            severity = 'High'
            passed = False
        
        # Statistical significance test
        if min_group:
            # Chi-square test for independence
            contingency_table = pd.crosstab(
                df[protected_attr].isin([reference_group, min_group]),
                df[outcome_col]
            )
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        else:
            p_value = 1.0
        
        return FairnessMetricResult(
            metric_name='Disparate Impact (Four-Fifths Rule)',
            value=min_ratio,
            threshold=config['threshold'],
            passed=passed,
            severity=severity,
            details={
                'reference_group': reference_group,
                'most_impacted_group': min_group,
                'group_statistics': group_stats,
                'impact_ratios': impact_ratios
            },
            p_value=p_value,
            regulation=config['regulation']
        )
    
    def calculate_demographic_parity(self,
                                    df: pd.DataFrame,
                                    outcome_col: str,
                                    protected_attr: str,
                                    reference_group: Optional[str] = None) -> FairnessMetricResult:
        """
        Calculate Demographic Parity Difference.
        
        DPD = |P(Y=1|A=a) - P(Y=1|A=b)|
        Should be < threshold (e.g., 0.1)
        """
        config = self.config['demographic_parity']
        
        # Calculate approval rates by group
        approval_rates = df.groupby(protected_attr)[outcome_col].mean()
        
        # Determine reference group if not specified
        if reference_group is None:
            reference_group = approval_rates.idxmax()
        
        reference_rate = approval_rates[reference_group]
        
        # Calculate maximum difference
        max_diff = 0
        max_diff_group = None
        
        for group, rate in approval_rates.items():
            if group != reference_group:
                diff = abs(rate - reference_rate)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_group = group
        
        # Determine pass/fail and severity
        passed = max_diff < config['threshold']
        if max_diff < config['threshold']:
            severity = 'Low'
        elif max_diff < config['threshold'] * 2:
            severity = 'Medium'
        else:
            severity = 'High'
        
        return FairnessMetricResult(
            metric_name='Demographic Parity Difference',
            value=max_diff,
            threshold=config['threshold'],
            passed=passed,
            severity=severity,
            details={
                'reference_group': reference_group,
                'reference_rate': reference_rate,
                'max_difference_group': max_diff_group,
                'approval_rates': approval_rates.to_dict()
            },
            regulation=config['regulation']
        )
    
    def calculate_equal_opportunity(self,
                                  df: pd.DataFrame,
                                  outcome_col: str,
                                  protected_attr: str,
                                  reference_group: Optional[str] = None) -> FairnessMetricResult:
        """
        Calculate Equal Opportunity Difference.
        
        EOD = |P(Y_pred=1|Y=1,A=a) - P(Y_pred=1|Y=1,A=b)|
        Focuses on true positive rates being equal across groups.
        """
        config = self.config['equal_opportunity']
        
        # For lending, we'll interpret this as approval rates among qualified applicants
        # Using credit score > 650 as a proxy for "qualified"
        qualified_mask = df['credit_score'] > 650
        qualified_df = df[qualified_mask]
        
        if len(qualified_df) == 0:
            return FairnessMetricResult(
                metric_name='Equal Opportunity Difference',
                value=0,
                threshold=config['threshold'],
                passed=True,
                severity='Low',
                details={'error': 'No qualified applicants found'},
                regulation=config['regulation']
            )
        
        # Calculate approval rates for qualified applicants by group
        qualified_rates = qualified_df.groupby(protected_attr)[outcome_col].mean()
        
        # Determine reference group
        if reference_group is None:
            reference_group = qualified_rates.idxmax()
        
        reference_rate = qualified_rates.get(reference_group, 0)
        
        # Calculate maximum difference
        max_diff = 0
        max_diff_group = None
        
        for group, rate in qualified_rates.items():
            if group != reference_group:
                diff = abs(rate - reference_rate)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_group = group
        
        # Determine pass/fail and severity
        passed = max_diff < config['threshold']
        if max_diff < config['threshold']:
            severity = 'Low'
        elif max_diff < config['threshold'] * 2:
            severity = 'Medium'
        else:
            severity = 'High'
        
        return FairnessMetricResult(
            metric_name='Equal Opportunity Difference',
            value=max_diff,
            threshold=config['threshold'],
            passed=passed,
            severity=severity,
            details={
                'reference_group': reference_group,
                'reference_rate': reference_rate,
                'max_difference_group': max_diff_group,
                'qualified_approval_rates': qualified_rates.to_dict(),
                'qualified_applicants': len(qualified_df)
            },
            regulation=config['regulation']
        )
    
    def calculate_predictive_parity(self,
                                   df: pd.DataFrame,
                                   outcome_col: str,
                                   protected_attr: str,
                                   reference_group: Optional[str] = None) -> FairnessMetricResult:
        """
        Calculate Predictive Parity.
        
        PPD = |P(Y=1|Y_pred=1,A=a) - P(Y=1|Y_pred=1,A=b)|
        Precision should be equal across groups.
        """
        config = self.config['predictive_parity']
        
        # This requires predicted probabilities
        if 'predicted_probability' not in df.columns:
            return FairnessMetricResult(
                metric_name='Predictive Parity',
                value=0,
                threshold=config['threshold'],
                passed=True,
                severity='Low',
                details={'error': 'No predictions available'},
                regulation=config['regulation']
            )
        
        # Create binary predictions
        df['predicted'] = (df['predicted_probability'] > 0.5).astype(int)
        
        # Calculate precision by group
        precisions = {}
        for group in df[protected_attr].unique():
            group_df = df[df[protected_attr] == group]
            predicted_positive = group_df[group_df['predicted'] == 1]
            
            if len(predicted_positive) > 0:
                precision = predicted_positive[outcome_col].mean()
                precisions[group] = precision
        
        if not precisions:
            return FairnessMetricResult(
                metric_name='Predictive Parity',
                value=0,
                threshold=config['threshold'],
                passed=True,
                severity='Low',
                details={'error': 'No positive predictions'},
                regulation=config['regulation']
            )
        
        # Calculate maximum difference
        max_precision = max(precisions.values())
        min_precision = min(precisions.values())
        max_diff = max_precision - min_precision
        
        # Determine pass/fail and severity
        passed = max_diff < config['threshold']
        if max_diff < config['threshold']:
            severity = 'Low'
        elif max_diff < config['threshold'] * 2:
            severity = 'Medium'
        else:
            severity = 'High'
        
        return FairnessMetricResult(
            metric_name='Predictive Parity',
            value=max_diff,
            threshold=config['threshold'],
            passed=passed,
            severity=severity,
            details={
                'precisions_by_group': precisions,
                'max_difference': max_diff
            },
            regulation=config['regulation']
        )
    
    def calculate_equalized_odds(self,
                                df: pd.DataFrame,
                                outcome_col: str,
                                protected_attr: str,
                                reference_group: Optional[str] = None) -> FairnessMetricResult:
        """
        Calculate Equalized Odds.
        
        Both TPR and FPR should be equal across groups.
        """
        config = self.config['equalized_odds']
        
        # Calculate TPR and FPR by group
        group_metrics = {}
        
        for group in df[protected_attr].unique():
            group_df = df[df[protected_attr] == group]
            
            # For approved loans (positive class)
            qualified = group_df[group_df['credit_score'] > 650]
            unqualified = group_df[group_df['credit_score'] <= 650]
            
            tpr = qualified[outcome_col].mean() if len(qualified) > 0 else 0
            fpr = unqualified[outcome_col].mean() if len(unqualified) > 0 else 0
            
            group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate maximum differences
        tpr_values = [m['tpr'] for m in group_metrics.values()]
        fpr_values = [m['fpr'] for m in group_metrics.values()]
        
        max_tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        max_fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        # Combined metric (average of TPR and FPR differences)
        combined_diff = (max_tpr_diff + max_fpr_diff) / 2
        
        # Determine pass/fail and severity
        passed = combined_diff < config['threshold']
        if combined_diff < config['threshold']:
            severity = 'Low'
        elif combined_diff < config['threshold'] * 2:
            severity = 'Medium'
        else:
            severity = 'High'
        
        return FairnessMetricResult(
            metric_name='Equalized Odds',
            value=combined_diff,
            threshold=config['threshold'],
            passed=passed,
            severity=severity,
            details={
                'group_metrics': group_metrics,
                'max_tpr_difference': max_tpr_diff,
                'max_fpr_difference': max_fpr_diff
            },
            regulation=config['regulation']
        )
    
    def calculate_intersectional_bias(self,
                                     df: pd.DataFrame,
                                     outcome_col: str,
                                     protected_attrs: List[str]) -> Dict[str, FairnessMetricResult]:
        """
        Calculate fairness metrics for intersectional groups.
        
        Args:
            df: DataFrame with lending data
            outcome_col: Name of outcome column
            protected_attrs: List of protected attributes to combine
            
        Returns:
            Dictionary of results for intersectional groups
        """
        results = {}
        
        # Create intersectional groups
        df['intersectional_group'] = df[protected_attrs].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        # Calculate metrics for intersectional groups
        intersectional_results = self.calculate_all_metrics(
            df, outcome_col, 'intersectional_group'
        )
        
        # Find most disadvantaged intersectional group
        approval_rates = df.groupby('intersectional_group')[outcome_col].mean()
        worst_group = approval_rates.idxmin()
        worst_rate = approval_rates.min()
        best_rate = approval_rates.max()
        
        # Create summary result
        disparity = best_rate - worst_rate
        
        results['intersectional_summary'] = FairnessMetricResult(
            metric_name='Intersectional Disparity',
            value=disparity,
            threshold=0.2,  # 20% difference threshold
            passed=disparity < 0.2,
            severity='High' if disparity > 0.3 else 'Medium' if disparity > 0.2 else 'Low',
            details={
                'most_disadvantaged_group': worst_group,
                'worst_approval_rate': worst_rate,
                'best_approval_rate': best_rate,
                'all_group_rates': approval_rates.to_dict()
            }
        )
        
        # Add individual metric results
        results.update(intersectional_results)
        
        return results


class BiasScorer:
    """
    Calculate overall bias risk scores based on multiple fairness metrics.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the bias scorer.
        
        Args:
            weights: Weights for different metrics
        """
        self.weights = weights or {
            'four_fifths_rule': 0.4,
            'demographic_parity': 0.2,
            'equal_opportunity': 0.2,
            'predictive_parity': 0.1,
            'equalized_odds': 0.1
        }
    
    def calculate_risk_score(self, 
                            metric_results: Dict[str, FairnessMetricResult]) -> Dict[str, Any]:
        """
        Calculate overall bias risk score.
        
        Args:
            metric_results: Dictionary of fairness metric results
            
        Returns:
            Risk score and details
        """
        scores = []
        weighted_sum = 0
        total_weight = 0
        
        for metric_name, weight in self.weights.items():
            if metric_name in metric_results:
                result = metric_results[metric_name]
                
                # Convert metric to risk score (0-1 scale)
                if metric_name == 'four_fifths_rule':
                    # Lower ratio = higher risk
                    risk = max(0, 1 - result.value / result.threshold)
                else:
                    # Higher difference = higher risk
                    risk = min(1, result.value / (result.threshold * 3))
                
                scores.append({
                    'metric': metric_name,
                    'risk': risk,
                    'weight': weight,
                    'weighted_risk': risk * weight
                })
                
                weighted_sum += risk * weight
                total_weight += weight
        
        # Calculate overall risk score
        overall_risk = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine severity level
        if overall_risk < 0.33:
            severity = 'Low'
            color = 'green'
        elif overall_risk < 0.66:
            severity = 'Medium'
            color = 'yellow'
        else:
            severity = 'High'
            color = 'red'
        
        return {
            'overall_risk_score': overall_risk,
            'severity': severity,
            'color': color,
            'metric_scores': scores,
            'regulatory_compliance': self._check_regulatory_compliance(metric_results),
            'recommendations': self._generate_recommendations(metric_results, overall_risk)
        }
    
    def _check_regulatory_compliance(self, 
                                    metric_results: Dict[str, FairnessMetricResult]) -> Dict[str, Any]:
        """Check compliance with various regulations."""
        compliance = {
            'ECOA': True,
            'Fair_Housing_Act': True,
            'State_Laws': True
        }
        
        violations = []
        
        # Check ECOA compliance (Four-Fifths Rule)
        if 'four_fifths_rule' in metric_results:
            if not metric_results['four_fifths_rule'].passed:
                compliance['ECOA'] = False
                violations.append({
                    'regulation': 'ECOA',
                    'violation': 'Failed Four-Fifths Rule',
                    'severity': metric_results['four_fifths_rule'].severity
                })
        
        # Check Fair Housing Act compliance
        if 'demographic_parity' in metric_results:
            if not metric_results['demographic_parity'].passed:
                compliance['Fair_Housing_Act'] = False
                violations.append({
                    'regulation': 'Fair Housing Act',
                    'violation': 'Demographic Parity violation',
                    'severity': metric_results['demographic_parity'].severity
                })
        
        return {
            'compliant': all(compliance.values()),
            'compliance_status': compliance,
            'violations': violations
        }
    
    def _generate_recommendations(self, 
                                 metric_results: Dict[str, FairnessMetricResult],
                                 risk_score: float) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        if risk_score > 0.66:
            recommendations.append("🚨 URGENT: Immediate review of lending criteria required")
        
        # Check specific metrics
        if 'four_fifths_rule' in metric_results:
            result = metric_results['four_fifths_rule']
            if not result.passed:
                group = result.details.get('most_impacted_group', 'minority groups')
                recommendations.append(
                    f"Review lending criteria that may disadvantage {group} applicants"
                )
        
        if 'demographic_parity' in metric_results:
            result = metric_results['demographic_parity']
            if result.value > result.threshold * 1.5:
                recommendations.append(
                    "Consider implementing bias mitigation techniques in the approval process"
                )
        
        if 'equal_opportunity' in metric_results:
            result = metric_results['equal_opportunity']
            if not result.passed:
                recommendations.append(
                    "Investigate why qualified applicants from different groups have different approval rates"
                )
        
        # Add general recommendations based on risk level
        if risk_score > 0.33:
            recommendations.extend([
                "Conduct detailed analysis of decision factors using SHAP values",
                "Review and potentially retrain models with balanced datasets",
                "Implement regular fairness monitoring and alerts"
            ])
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Load sample data
    import sys
    sys.path.append('..')
    from data.synthetic_generator import SyntheticLendingDataGenerator
    
    # Generate test data
    generator = SyntheticLendingDataGenerator(n_samples=5000)
    df = generator.generate_dataset()
    
    # Initialize calculator
    calculator = FairnessMetricsCalculator()
    
    # Calculate metrics for gender
    print("\n=== Fairness Metrics for Gender ===")
    gender_metrics = calculator.calculate_all_metrics(
        df, 'approved', 'gender', reference_group='Male'
    )
    
    for metric_name, result in gender_metrics.items():
        print(f"\n{result.metric_name}:")
        print(f"  Value: {result.value:.3f}")
        print(f"  Threshold: {result.threshold}")
        print(f"  Passed: {result.passed}")
        print(f"  Severity: {result.severity}")
        if result.p_value:
            print(f"  P-value: {result.p_value:.4f}")
    
    # Calculate metrics for race
    print("\n=== Fairness Metrics for Race ===")
    race_metrics = calculator.calculate_all_metrics(
        df, 'approved', 'race', reference_group='White'
    )
    
    for metric_name, result in race_metrics.items():
        print(f"\n{result.metric_name}:")
        print(f"  Value: {result.value:.3f}")
        print(f"  Passed: {result.passed}")
        print(f"  Severity: {result.severity}")
    
    # Calculate intersectional bias
    print("\n=== Intersectional Analysis ===")
    intersectional_results = calculator.calculate_intersectional_bias(
        df, 'approved', ['gender', 'race']
    )
    
    summary = intersectional_results['intersectional_summary']
    print(f"Intersectional Disparity: {summary.value:.3f}")
    print(f"Most Disadvantaged Group: {summary.details['most_disadvantaged_group']}")
    print(f"Approval Rate: {summary.details['worst_approval_rate']:.2%}")
    
    # Calculate risk score
    print("\n=== Overall Risk Assessment ===")
    scorer = BiasScorer()
    risk_assessment = scorer.calculate_risk_score(gender_metrics)
    
    print(f"Overall Risk Score: {risk_assessment['overall_risk_score']:.2f}")
    print(f"Severity: {risk_assessment['severity']}")
    print(f"Regulatory Compliance: {risk_assessment['regulatory_compliance']['compliant']}")
    
    print("\nRecommendations:")
    for rec in risk_assessment['recommendations']:
        print(f"  • {rec}")