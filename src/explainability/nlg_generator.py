# src/explainability/nlg_generator.py
import textwrap
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedNLGGenerator:
    """
    Advanced Natural Language Generation for bias reports with contextual awareness,
    pattern detection, and regulatory compliance mapping.
    """
    
    def __init__(self, wrap_width: int = 80, config: Dict[str, Any] = None):
        self.wrap_width = wrap_width
        self.config = config or {}
        self.templates = self._load_advanced_templates()
        self.pattern_detectors = self._initialize_pattern_detectors()
        
    def _load_advanced_templates(self) -> Dict[str, str]:
        """Load advanced explanation templates with dynamic placeholders."""
        return {
            'executive_summary': 
                "🏦 **FAIR-AI Lending Risk Report**\n\n"
                "**Overall Risk Assessment**: {risk_level} risk (Score: {risk_score:.2f}/1.0)\n"
                "**Regulatory Status**: {compliance_status}\n"
                "**Key Findings**: {key_findings}\n"
                "**Immediate Actions**: {immediate_actions}\n\n"
                "**Analysis Period**: {start_date} to {end_date}\n"
                "**Models Analyzed**: {models_analyzed}\n"
                "**Protected Attributes**: {protected_attributes}",
                
            'metric_explanation':
                "**{metric_name}**: {value:.3f} ({comparison} threshold of {threshold:.3f})\n"
                "**Statistical Significance**: {significance} (p={p_value:.4f})\n"
                "**Regulatory Impact**: {regulation_impact}\n"
                "**Interpretation**: {interpretation}",
                
            'bias_pattern':
                "🔍 **Detected Pattern**: {pattern_type} bias in {attribute}\n"
                "**Affected Groups**: {affected_groups}\n"
                "**Impact Magnitude**: {impact_magnitude}\n"
                "**Confidence**: {confidence_level}\n"
                "**Pattern Details**: {pattern_details}",
                
            'recommendation':
                "🎯 **{priority} Priority**: {action}\n"
                "**Expected Impact**: {expected_impact}\n"
                "**Implementation Timeline**: {timeline}\n"
                "**Resources Required**: {resources}\n"
                "**Regulatory Reference**: {regulation_ref}",
                
            'counterfactual_explanation':
                "🔄 **Counterfactual Analysis**: For {instance_id}\n"
                "**Original Decision**: {original_outcome} ({original_confidence:.1%})\n"
                "**Modified Decision**: {modified_outcome} ({modified_confidence:.1%})\n"
                "**Critical Factors**: {critical_factors}\n"
                "**Sensitivity**: {sensitivity_analysis}"
        }
    
    def _initialize_pattern_detectors(self) -> Dict[str, Any]:
        """Initialize advanced pattern detection algorithms."""
        return {
            'geographic_redlining': self._detect_geographic_redlining,
            'intersectional_bias': self._detect_intersectional_bias,
            'temporal_drift': self._detect_temporal_drift,
            'proxy_discrimination': self._detect_proxy_discrimination
        }
    
    def generate_dynamic_summary(self, analysis_results: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> str:
        """
        Generate dynamic executive summary with contextual awareness.
        """
        context = context or {}
        risk_data = analysis_results.get('risk_assessment', {})
        
        # Detect key patterns
        patterns = self._detect_all_patterns(analysis_results, context)
        key_findings = self._format_patterns_as_findings(patterns)
        
        # Generate compliance status
        compliance_status = self._generate_compliance_status(
            risk_data.get('regulatory_compliance', {})
        )
        
        # Format template
        summary = self.templates['executive_summary'].format(
            risk_level=risk_data.get('severity', 'Unknown'),
            risk_score=risk_data.get('overall_risk_score', 0),
            compliance_status=compliance_status,
            key_findings=key_findings,
            immediate_actions=self._generate_immediate_actions(risk_data),
            start_date=context.get('start_date', 'N/A'),
            end_date=context.get('end_date', 'N/A'),
            models_analyzed=len(analysis_results.get('model_results', {})),
            protected_attributes=", ".join(context.get('protected_attributes', []))
        )
        
        return textwrap.fill(summary, width=self.wrap_width)
    
    def generate_metric_explanations(self, metric_results: Dict[str, Any], 
                                   statistical_results: Dict[str, Any] = None) -> List[str]:
        """
        Generate detailed explanations for each fairness metric with statistical context.
        """
        explanations = []
        statistical_results = statistical_results or {}
        
        for metric_name, result in metric_results.items():
            if not hasattr(result, 'value'):
                continue
                
            # Get statistical context
            stats = statistical_results.get(metric_name, {})
            p_value = getattr(result, 'p_value', stats.get('p_value', 0.05))
            ci = getattr(result, 'confidence_interval', stats.get('confidence_interval', (0, 0)))
            
            explanation = self.templates['metric_explanation'].format(
                metric_name=result.metric_name,
                value=result.value,
                comparison="above" if result.passed else "below",
                threshold=result.threshold,
                significance="Significant" if p_value < 0.05 else "Not Significant",
                p_value=p_value,
                regulation_impact=self._get_regulatory_impact(metric_name, result),
                interpretation=self._interpret_metric(metric_name, result, stats)
            )
            
            explanations.append(textwrap.fill(explanation, width=self.wrap_width))
        
        return explanations
    
    def _detect_all_patterns(self, analysis_results: Dict[str, Any], 
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect all types of bias patterns using specialized detectors.
        """
        patterns = []
        
        for pattern_name, detector in self.pattern_detectors.items():
            try:
                pattern = detector(analysis_results, context)
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pattern_name}: {e}")
        
        return patterns
    
    def _detect_geographic_redlining(self, analysis_results: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect geographic redlining patterns."""
        # Implementation would use spatial analysis and demographic data
        # This is a simplified version
        if 'geographic_analysis' not in analysis_results:
            return None
            
        geo_data = analysis_results['geographic_analysis']
        high_risk_zips = [
            zip_code for zip_code, risk in geo_data.items() 
            if risk.get('risk_score', 0) > 0.7
        ]
        
        if not high_risk_zips:
            return None
            
        return {
            'pattern_type': 'Geographic Redlining',
            'attribute': 'ZIP Code/Region',
            'affected_groups': f"{len(high_risk_zips)} high-risk geographic areas",
            'impact_magnitude': 'High',
            'confidence_level': 'Medium-High',
            'pattern_details': f"Concentrated bias in ZIP codes: {', '.join(high_risk_zips[:5])}{'...' if len(high_risk_zips) > 5 else ''}"
        }
    
    def _detect_intersectional_bias(self, analysis_results: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect intersectional bias patterns."""
        # Implementation would analyze combinations of protected attributes
        if 'intersectional_analysis' not in analysis_results:
            return None
            
        intersectional_data = analysis_results['intersectional_analysis']
        worst_combination = max(
            intersectional_data.items(), 
            key=lambda x: x[1].get('disparity', 0)
        )
        
        if worst_combination[1].get('disparity', 0) < 0.2:
            return None
            
        return {
            'pattern_type': 'Intersectional Bias',
            'attribute': f"{worst_combination[0]}",
            'affected_groups': "Multiple protected attribute combinations",
            'impact_magnitude': 'Variable',
            'confidence_level': 'Medium',
            'pattern_details': f"Highest disparity for {worst_combination[0]} ({worst_combination[1].get('disparity', 0):.2f} disparity score)"
        }
    
    def _generate_compliance_status(self, compliance_data: Dict[str, Any]) -> str:
        """Generate detailed compliance status description."""
        if compliance_data.get('compliant', True):
            return "✅ Fully compliant with all regulatory requirements"
        
        violations = compliance_data.get('violations', [])
        violation_desc = [
            f"{v['regulation']}: {v.get('metric', 'Unknown metric')}" 
            for v in violations
        ]
        
        return f"❌ Non-compliant with {len(violations)} regulations: {', '.join(violation_desc)}"
    
    def _generate_immediate_actions(self, risk_data: Dict[str, Any]) -> str:
        """Generate prioritized immediate actions."""
        recommendations = risk_data.get('recommendations', [])
        if not recommendations:
            return "No immediate actions required"
        
        # Prioritize recommendations
        high_priority = [r for r in recommendations if 'URGENT' in r or 'IMMEDIATE' in r]
        medium_priority = [r for r in recommendations if 'review' in r.lower() or 'investigate' in r.lower()]
        
        actions = []
        if high_priority:
            actions.append(f"High priority: {high_priority[0]}")
        if medium_priority:
            actions.append(f"Medium priority: {medium_priority[0]}")
        
        return "; ".join(actions) if actions else recommendations[0]
    
    def _get_regulatory_impact(self, metric_name: str, result: Any) -> str:
        """Get regulatory impact description for a metric."""
        regulation_map = {
            'four_fifths_rule': 'ECOA Disparate Impact',
            'demographic_parity': 'Fair Housing Act',
            'equal_opportunity': 'ECOA Equal Opportunity',
            'predictive_parity': 'ECOA Predictive Testing',
            'equalized_odds': 'Multiple Regulations'
        }
        
        regulation = regulation_map.get(metric_name, 'Various Regulations')
        status = "Compliant" if result.passed else "Non-compliant"
        
        return f"{regulation} ({status})"
    
    def _interpret_metric(self, metric_name: str, result: Any, stats: Dict[str, Any]) -> str:
        """Generate interpretation for a metric result."""
        interpretations = {
            'four_fifths_rule': 
                f"This indicates {'acceptable' if result.passed else 'potential'} disparate impact, "
                f"with {result.value:.3f} ratio compared to the 0.8 threshold.",
                
            'demographic_parity':
                f"Selection rates differ by {result.value:.3f} across groups, "
                f"{'within' if result.passed else 'exceeding'} acceptable limits.",
                
            'equal_opportunity':
                f"True positive rates vary by {result.value:.3f} across demographic groups, "
                f"{'meeting' if result.passed else 'failing'} equal opportunity standards."
        }
        
        return interpretations.get(metric_name, "Specialized fairness metric requiring further analysis.")
    
    def generate_interactive_report(self, analysis_results: Dict[str, Any], 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete interactive report with multiple sections.
        """
        context = context or {}
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model_version': context.get('model_version', '1.0.0'),
                'analysis_scope': context.get('analysis_scope', 'Full portfolio')
            },
            'executive_summary': self.generate_dynamic_summary(analysis_results, context),
            'detailed_analysis': {
                'metric_explanations': self.generate_metric_explanations(
                    analysis_results.get('metric_results', {})
                ),
                'pattern_detection': self._detect_all_patterns(analysis_results, context),
                'temporal_trends': self._analyze_temporal_trends(analysis_results, context)
            },
            'recommendations': self._generate_prioritized_recommendations(
                analysis_results.get('risk_assessment', {})
            ),
            'compliance_assessment': self._generate_compliance_assessment(
                analysis_results.get('risk_assessment', {}).get('regulatory_compliance', {})
            )
        }
        
        return report
    
    def _analyze_temporal_trends(self, analysis_results: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in bias patterns."""
        # This would typically integrate with time-series data
        return {
            'analysis_performed': False,
            'message': 'Temporal analysis requires historical data not provided in current scope'
        }
    
    def _generate_prioritized_recommendations(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations with resource estimates."""
        recommendations = risk_data.get('recommendations', [])
        prioritized = []
        
        for i, rec in enumerate(recommendations[:5]):  # Top 5 recommendations
            priority = 'High' if i < 2 else 'Medium' if i < 4 else 'Low'
            
            prioritized.append({
                'priority': priority,
                'action': rec,
                'timeline': '1-2 weeks' if priority == 'High' else '2-4 weeks',
                'resources': 'Data science team' if 'model' in rec else 'Compliance team',
                'expected_impact': 'High' if priority == 'High' else 'Medium'
            })
        
        return prioritized
    
    def _generate_compliance_assessment(self, compliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed compliance assessment."""
        return {
            'overall_status': 'Compliant' if compliance_data.get('compliant', True) else 'Non-compliant',
            'violations_count': len(compliance_data.get('violations', [])),
            'violations_details': compliance_data.get('violations', []),
            'recommended_remediation': self._generate_compliance_remediation(compliance_data)
        }
    
    def _generate_compliance_remediation(self, compliance_data: Dict[str, Any]) -> List[str]:
        """Generate compliance remediation steps."""
        if compliance_data.get('compliant', True):
            return ["Maintain current compliance monitoring procedures"]
        
        remediations = []
        for violation in compliance_data.get('violations', []):
            regulation = violation.get('regulation', 'Unknown regulation')
            remediations.append(
                f"Immediate review and documentation for {regulation} violation"
            )
        
        return remediations