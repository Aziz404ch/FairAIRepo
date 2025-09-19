import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairness.metrics import FairnessMetricsCalculator, BiasScorer, FairnessMetricResult
from data.synthetic_generator import SyntheticLendingDataGenerator

class TestFairnessMetrics(unittest.TestCase):
    """Test fairness metrics calculation."""
    
    def setUp(self):
        """Set up test data."""
        self.generator = SyntheticLendingDataGenerator(n_samples=1000, random_seed=42)
        self.data = self.generator.generate_dataset()
        self.calculator = FairnessMetricsCalculator()
    
    def test_disparate_impact_calculation(self):
        """Test disparate impact (four-fifths rule) calculation."""
        result = self.calculator.calculate_disparate_impact(
            self.data, 'approved', 'gender', 'Male'
        )
        
        self.assertIsInstance(result, FairnessMetricResult)
        self.assertEqual(result.metric_name, 'Disparate Impact (Four-Fifths Rule)')
        self.assertGreater(result.value, 0)
        self.assertLessEqual(result.value, 1.0)
        self.assertIn(result.severity, ['Low', 'Medium', 'High'])
    
    def test_demographic_parity_calculation(self):
        """Test demographic parity calculation."""
        result = self.calculator.calculate_demographic_parity(
            self.data, 'approved', 'race', 'White'
        )
        
        self.assertIsInstance(result, FairnessMetricResult)
        self.assertEqual(result.metric_name, 'Demographic Parity Difference')
        self.assertGreaterEqual(result.value, 0)
    
    def test_equal_opportunity_calculation(self):
        """Test equal opportunity calculation."""
        result = self.calculator.calculate_equal_opportunity(
            self.data, 'approved', 'gender'
        )
        
        self.assertIsInstance(result, FairnessMetricResult)
        self.assertEqual(result.metric_name, 'Equal Opportunity Difference')
        self.assertGreaterEqual(result.value, 0)
    
    def test_all_metrics_calculation(self):
        """Test calculation of all fairness metrics."""
        results = self.calculator.calculate_all_metrics(
            self.data, 'approved', 'gender', 'Male'
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('four_fifths_rule', results)
        self.assertIn('demographic_parity', results)
        self.assertIn('equal_opportunity', results)
        
        for metric_result in results.values():
            self.assertIsInstance(metric_result, FairnessMetricResult)
    
    def test_intersectional_bias_calculation(self):
        """Test intersectional bias calculation."""
        results = self.calculator.calculate_intersectional_bias(
            self.data, 'approved', ['gender', 'race']
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('intersectional_summary', results)
        
        summary = results['intersectional_summary']
        self.assertIsInstance(summary, FairnessMetricResult)
        self.assertEqual(summary.metric_name, 'Intersectional Disparity')
    
    def test_bias_scorer(self):
        """Test bias risk scoring."""
        scorer = BiasScorer()
        
        # Calculate some metrics first
        metrics = self.calculator.calculate_all_metrics(
            self.data, 'approved', 'gender', 'Male'
        )
        
        risk_assessment = scorer.calculate_risk_score(metrics)
        
        self.assertIn('overall_risk_score', risk_assessment)
        self.assertIn('severity', risk_assessment)
        self.assertIn('regulatory_compliance', risk_assessment)
        self.assertIn('recommendations', risk_assessment)
        
        # Risk score should be between 0 and 1
        self.assertGreaterEqual(risk_assessment['overall_risk_score'], 0)
        self.assertLessEqual(risk_assessment['overall_risk_score'], 1)
        
        # Severity should be valid
        self.assertIn(risk_assessment['severity'], ['Low', 'Medium', 'High'])

class TestBiasDetection(unittest.TestCase):
    """Test bias detection in synthetic data."""
    
    def setUp(self):
        """Set up biased and unbiased datasets."""
        self.generator = SyntheticLendingDataGenerator(n_samples=1000, random_seed=42)
        
        # Generate biased dataset
        self.biased_data = self.generator.generate_dataset(bias_config={
            'gender_bias': True,
            'race_bias': True,
            'age_bias': False,
            'geographic_bias': False,
            'intersectional_bias': False
        })
        
        # Generate unbiased dataset
        self.unbiased_data = self.generator.generate_dataset(bias_config={
            'gender_bias': False,
            'race_bias': False,
            'age_bias': False,
            'geographic_bias': False,
            'intersectional_bias': False
        })
        
        self.calculator = FairnessMetricsCalculator()
        self.scorer = BiasScorer()
    
    def test_bias_detection_gender(self):
        """Test that gender bias is detected in biased dataset."""
        # Biased dataset should show gender bias
        biased_metrics = self.calculator.calculate_all_metrics(
            self.biased_data, 'approved', 'gender', 'Male'
        )
        biased_assessment = self.scorer.calculate_risk_score(biased_metrics)
        
        # Unbiased dataset should show less bias
        unbiased_metrics = self.calculator.calculate_all_metrics(
            self.unbiased_data, 'approved', 'gender', 'Male'
        )
        unbiased_assessment = self.scorer.calculate_risk_score(unbiased_metrics)
        
        # Biased dataset should have higher risk score
        self.assertGreater(
            biased_assessment['overall_risk_score'],
            unbiased_assessment['overall_risk_score']
        )
    
    def test_bias_detection_race(self):
        """Test that race bias is detected in biased dataset."""
        biased_metrics = self.calculator.calculate_all_metrics(
            self.biased_data, 'approved', 'race', 'White'
        )
        
        # Four-fifths rule should fail for biased data
        four_fifths_result = biased_metrics['four_fifths_rule']
        self.assertLess(four_fifths_result.value, 0.8)  # Should fail 80% rule
    
    def test_regulatory_compliance_detection(self):
        """Test regulatory compliance detection."""
        biased_metrics = self.calculator.calculate_all_metrics(
            self.biased_data, 'approved', 'gender', 'Male'
        )
        biased_assessment = self.scorer.calculate_risk_score(biased_metrics)
        
        # Should detect non-compliance
        compliance = biased_assessment['regulatory_compliance']
        self.assertFalse(compliance['compliant'])
        self.assertGreater(len(compliance['violations']), 0)
