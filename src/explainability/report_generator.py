import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ExplainabilityReportGenerator:
    """Generate comprehensive explainability reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_model_report(self, 
                            model_name: str,
                            shap_analyzer: Any,
                            lime_analyzer: Any,
                            fairness_results: Dict[str, Any],
                            dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive model explainability report."""
        
        logger.info(f"Generating explainability report for {model_name}")
        
        report = {
            'metadata': {
                'model_name': model_name,
                'generated_at': datetime.now().isoformat(),
                'dataset_size': dataset_info.get('size', 'Unknown'),
                'features_count': len(shap_analyzer.feature_names) if shap_analyzer.feature_names else 0
            },
            'executive_summary': self._generate_executive_summary(fairness_results),
            'fairness_analysis': fairness_results,
            'feature_importance': self._analyze_feature_importance(shap_analyzer),
            'bias_analysis': self._analyze_bias_patterns(shap_analyzer, fairness_results),
            'recommendations': self._generate_recommendations(fairness_results),
            'technical_details': self._generate_technical_details(shap_analyzer, lime_analyzer)
        }
        
        return report
    
    def _generate_executive_summary(self, fairness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of fairness findings."""
        summary = {
            'overall_risk_level': 'Unknown',
            'key_findings': [],
            'regulatory_compliance': {},
            'immediate_actions': []
        }
        
        # Extract risk level from bias scorer results
        if 'risk_assessment' in fairness_results:
            risk_data = fairness_results['risk_assessment']
            summary['overall_risk_level'] = risk_data.get('severity', 'Unknown')
            
            # Extract key findings
            for metric_score in risk_data.get('metric_scores', []):
                if metric_score['risk'] > 0.5:
                    summary['key_findings'].append(
                        f"High risk detected in {metric_score['metric']}: "
                        f"Risk score {metric_score['risk']:.2f}"
                    )
            
            # Regulatory compliance
            compliance = risk_data.get('regulatory_compliance', {})
            summary['regulatory_compliance'] = compliance.get('compliance_status', {})
            
            # Immediate actions
            summary['immediate_actions'] = risk_data.get('recommendations', [])[:3]
        
        return summary
    
    def _analyze_feature_importance(self, shap_analyzer: Any) -> Dict[str, Any]:
        """Analyze feature importance from SHAP values."""
        if shap_analyzer.shap_values is None:
            return {'error': 'SHAP values not available'}
        
        importance_df = shap_analyzer.get_feature_importance()
        
        # Categorize features
        feature_categories = {
            'demographic': [],
            'financial': [],
            'geographic': [],
            'loan_specific': [],
            'other': []
        }
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if any(demo in feature.lower() for demo in ['race', 'gender', 'age']):
                feature_categories['demographic'].append((feature, importance))
            elif any(fin in feature.lower() for fin in ['income', 'credit', 'debt', 'employment']):
                feature_categories['financial'].append((feature, importance))
            elif any(geo in feature.lower() for geo in ['region', 'zip', 'urban']):
                feature_categories['geographic'].append((feature, importance))
            elif any(loan in feature.lower() for loan in ['loan', 'amount', 'term', 'interest']):
                feature_categories['loan_specific'].append((feature, importance))
            else:
                feature_categories['other'].append((feature, importance))
        
        return {
            'top_features': importance_df.head(10).to_dict('records'),
            'feature_categories': feature_categories,
            'total_features': len(importance_df)
        }
    
    def _analyze_bias_patterns(self, shap_analyzer: Any, fairness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bias patterns from SHAP and fairness metrics."""
        bias_analysis = {
            'protected_attribute_contributions': {},
            'intersectional_effects': {},
            'bias_severity': 'Low'
        }
        
        # Analyze SHAP contributions for protected attributes
        if hasattr(shap_analyzer, 'analyze_bias_contributions'):
            protected_attrs = ['race', 'gender', 'age']
            bias_contributions = shap_analyzer.analyze_bias_contributions(protected_attrs)
            bias_analysis['protected_attribute_contributions'] = bias_contributions
        
        # Analyze fairness metric results
        if 'metric_results' in fairness_results:
            high_risk_metrics = []
            for metric_name, result in fairness_results['metric_results'].items():
                if hasattr(result, 'severity') and result.severity == 'High':
                    high_risk_metrics.append(metric_name)
            
            if high_risk_metrics:
                bias_analysis['bias_severity'] = 'High'
                bias_analysis['high_risk_metrics'] = high_risk_metrics
        
        return bias_analysis
    
    def _generate_recommendations(self, fairness_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if 'risk_assessment' in fairness_results:
            risk_data = fairness_results['risk_assessment']
            
            # High-level recommendations based on risk level
            risk_level = risk_data.get('severity', 'Low')
            
            if risk_level == 'High':
                recommendations.extend([
                    {
                        'priority': 'Critical',
                        'action': 'Immediate model review and potential suspension',
                        'description': 'High bias risk detected requiring urgent attention',
                        'timeline': 'Immediate (1-3 days)'
                    },
                    {
                        'priority': 'High',
                        'action': 'Bias mitigation implementation',
                        'description': 'Apply bias mitigation techniques such as reweighing or adversarial debiasing',
                        'timeline': 'Short-term (1-2 weeks)'
                    }
                ])
            elif risk_level == 'Medium':
                recommendations.extend([
                    {
                        'priority': 'Medium',
                        'action': 'Enhanced monitoring implementation',
                        'description': 'Implement real-time bias monitoring and alerting',
                        'timeline': 'Medium-term (1 month)'
                    }
                ])
            
            # Add specific recommendations from risk assessment
            for rec in risk_data.get('recommendations', []):
                recommendations.append({
                    'priority': 'Medium',
                    'action': rec,
                    'description': 'Automated recommendation based on fairness analysis',
                    'timeline': 'Medium-term (1-2 months)'
                })
        
        return recommendations
    
    def _generate_technical_details(self, shap_analyzer: Any, lime_analyzer: Any) -> Dict[str, Any]:
        """Generate technical details section."""
        technical_details = {
            'explainability_methods': {},
            'model_interpretability': {},
            'validation_metrics': {}
        }
        
        # SHAP details
        if shap_analyzer.shap_values is not None:
            technical_details['explainability_methods']['shap'] = {
                'method_used': shap_analyzer.model_type,
                'samples_analyzed': shap_analyzer.shap_values.shape[0],
                'features_analyzed': shap_analyzer.shap_values.shape[1],
                'base_value': getattr(shap_analyzer.explainer, 'expected_value', 'N/A')
            }
        
        # LIME details
        if lime_analyzer.explainer is not None:
            technical_details['explainability_methods']['lime'] = {
                'method_used': 'Tabular',
                'categorical_features': len(lime_analyzer.explainer.categorical_features),
                'class_names': lime_analyzer.class_names
            }
        
        return technical_details
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"explainability_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def generate_html_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Generate HTML version of the report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"explainability_report_{timestamp}.html"
        
        html_content = self._create_html_template(report)
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {filepath}")
        return str(filepath)
    
    def _create_html_template(self, report: Dict[str, Any]) -> str:
        """Create HTML template for the report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fair Lending Explainability Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; }}
        .section {{ margin: 20px 0; }}
        .risk-high {{ color: red; font-weight: bold; }}
        .risk-medium {{ color: orange; font-weight: bold; }}
        .risk-low {{ color: green; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Fair Lending Explainability Report</h1>
        <p><strong>Model:</strong> {report['metadata']['model_name']}</p>
        <p><strong>Generated:</strong> {report['metadata']['generated_at']}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p><strong>Overall Risk Level:</strong> 
           <span class="risk-{report['executive_summary']['overall_risk_level'].lower()}">
               {report['executive_summary']['overall_risk_level']}
           </span>
        </p>
        
        <h3>Key Findings:</h3>
        <ul>
            {''.join(f'<li>{finding}</li>' for finding in report['executive_summary']['key_findings'])}
        </ul>
        
        <h3>Immediate Actions Required:</h3>
        <ul>
            {''.join(f'<li>{action}</li>' for action in report['executive_summary']['immediate_actions'])}
        </ul>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <table>
            <tr>
                <th>Priority</th>
                <th>Action</th>
                <th>Description</th>
                <th>Timeline</th>
            </tr>
            {''.join(f'''
            <tr>
                <td>{rec['priority']}</td>
                <td>{rec['action']}</td>
                <td>{rec['description']}</td>
                <td>{rec['timeline']}</td>
            </tr>
            ''' for rec in report.get('recommendations', []))}
        </table>
    </div>
</body>
</html>
        """
        
        return html