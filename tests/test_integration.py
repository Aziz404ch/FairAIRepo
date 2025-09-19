# tests/test_integration.py
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import FAIRAILendingMonitor
from fairness.metrics import FairnessMetricsCalculator
from explainability.nlg_generator import AdvancedNLGGenerator
from utils.monitoring import ContinuousMonitoringSystem

class TestIntegration(unittest.TestCase):
    """Comprehensive integration tests for the FAIR-AI system."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = FAIRAILendingMonitor()
        self.nlg_generator = AdvancedNLGGenerator()
        self.monitoring_system = ContinuousMonitoringSystem()
        
        # Generate test data
        from data.synthetic_generator import SyntheticLendingDataGenerator
        generator = SyntheticLendingDataGenerator(n_samples=1000, random_seed=42)
        self.test_data = generator.generate_dataset()
        
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate data
            self.monitor.generate_synthetic_data(1000)
            
            # Train models
            self.monitor.train_models()
            
            # Analyze fairness
            fairness_results = self.monitor.analyze_fairness()
            
            # Generate explanations
            explanations = self.monitor.generate_explanations(fairness_results)
            
            # Generate report
            report_path = self.monitor.generate_comprehensive_report(fairness_results, explanations)
            
            # Verify report exists
            self.assertTrue(Path(report_path).exists())
            
            # Verify report content
            with open(report_path, 'r') as f:
                report_content = json.load(f)
            
            self.assertIn('metadata', report_content)
            self.assertIn('fairness_analysis', report_content)
            self.assertIn('recommendations', report_content)
    
    def test_nlg_generation(self):
        """Test natural language generation capabilities."""
        # Create test metrics
        from fairness.metrics import FairnessMetricResult
        
        test_metrics = {
            'four_fifths_rule': FairnessMetricResult(
                metric_name='Disparate Impact',
                value=0.75,
                threshold=0.8,
                passed=False,
                severity='High',
                details={}
            )
        }
        
        # Generate summary
        summary = self.nlg_generator.generate_dynamic_summary({
            'risk_assessment': {
                'overall_risk_score': 0.72,
                'severity': 'High',
                'regulatory_compliance': {
                    'compliant': False,
                    'violations': [{'regulation': 'ECOA', 'metric': 'four_fifths_rule'}]
                },
                'recommendations': ['Review model for disparate impact']
            }
        })
        
        self.assertIsInstance(summary, str)
        self.assertIn('High risk', summary)
        self.assertIn('ECOA', summary)
    
    def test_continuous_monitoring(self):
        """Test continuous monitoring system."""
        # Mock data source
        def mock_data_source():
            return self.test_data.sample(100, random_state=42)
        
        # Mock model
        from models.logistic_model import LogisticRegressionModel
        model = LogisticRegressionModel()
        X = self.test_data.drop('approved', axis=1)
        y = self.test_data['approved']
        model.train(X, y)
        
        # Start monitoring
        self.monitoring_system.start_monitoring(
            mock_data_source, model, ['race', 'gender'], interval_minutes=1
        )
        
        # Run a cycle manually
        self.monitoring_system._run_monitoring_cycle(mock_data_source, model, ['race', 'gender'])
        
        # Check for alerts
        self.assertFalse(self.monitoring_system.alert_queue.empty())
        
        # Stop monitoring
        self.monitoring_system.stop_monitoring()
    
    def test_mitigation_techniques(self):
        """Test bias mitigation techniques."""
        from fairness.mitigation import AdvancedBiasMitigator
        from models.logistic_model import LogisticRegressionModel
        
        # Train model
        model = LogisticRegressionModel()
        X = self.test_data.drop('approved', axis=1)
        y = self.test_data['approved']
        model.train(X, y)
        
        # Create mitigator
        mitigator = AdvancedBiasMitigator(model, ['race', 'gender'])
        
        # Test policy simulation
        policy_rules = {
            'credit_score': {'type': 'threshold', 'value': 650, 'direction': 'increase'},
            'annual_income': {'type': 'threshold', 'value': 50000, 'direction': 'increase'}
        }
        
        impact = mitigator.simulate_policy_change(X, policy_rules)
        
        self.assertIn('overall_impact', impact)
        self.assertIn('group_impacts', impact)
        self.assertIn('policy_rules', impact)
        
        # Test counterfactual generation
        instance = X.iloc[0]
        counterfactuals = mitigator.generate_counterfactuals(instance, X, n_counterfactuals=3)
        
        self.assertFalse(counterfactuals.empty)
        self.assertIn('counterfactual_confidence', counterfactuals.columns)
    
    def test_dashboard_integration(self):
        """Test dashboard integration with backend."""
        # This would typically involve testing the Streamlit app
        # For now, we test that the data formatting works correctly
        from dashboard.components.charts import render_risk_gauge
        
        # Test that chart function doesn't crash
        try:
            render_risk_gauge(0.75, 'High')
            chart_works = True
        except:
            chart_works = False
        
        self.assertTrue(chart_works)

if __name__ == '__main__':
    unittest.main()