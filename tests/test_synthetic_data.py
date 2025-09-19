import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.synthetic_generator import SyntheticLendingDataGenerator

class TestSyntheticDataGenerator(unittest.TestCase):
    """Test synthetic lending data generation."""
    
    def setUp(self):
        """Set up generator."""
        self.generator = SyntheticLendingDataGenerator(n_samples=1000, random_seed=42)
    
    def test_data_generation(self):
        """Test basic data generation."""
        data = self.generator.generate_dataset()
        
        # Check shape
        self.assertEqual(len(data), 1000)
        
        # Check required columns
        required_cols = [
            'application_id', 'race', 'gender', 'age', 'annual_income',
            'credit_score', 'loan_amount', 'approved'
        ]
        
        for col in required_cols:
            self.assertIn(col, data.columns)
    
    def test_demographic_distributions(self):
        """Test demographic distributions match expected values."""
        data = self.generator.generate_dataset()
        
        # Test race distribution (approximately)
        race_counts = data['race'].value_counts(normalize=True)
        self.assertGreater(race_counts['White'], 0.5)  # Should be majority
        self.assertGreater(race_counts['Black'], 0.05)
        
        # Test gender distribution
        gender_counts = data['gender'].value_counts(normalize=True)
        self.assertAlmostEqual(gender_counts['Male'], 0.49, delta=0.1)
        self.assertAlmostEqual(gender_counts['Female'], 0.49, delta=0.1)
    
    def test_financial_correlations(self):
        """Test that financial variables have realistic correlations."""
        data = self.generator.generate_dataset()
        
        # Income and credit score should be positively correlated
        correlation = data['annual_income'].corr(data['credit_score'])
        self.assertGreater(correlation, 0.3)
        
        # Age and employment years should be positively correlated
        correlation = data['age'].corr(data['employment_years'])
        self.assertGreater(correlation, 0.5)
    
    def test_bias_patterns(self):
        """Test that bias patterns are properly implemented."""
        # Generate biased data
        biased_data = self.generator.generate_dataset(bias_config={
            'gender_bias': True,
            'race_bias': True
        })
        
        # Generate unbiased data
        unbiased_data = self.generator.generate_dataset(bias_config={
            'gender_bias': False,
            'race_bias': False
        })
        
        # Check gender bias
        biased_female_rate = biased_data[biased_data['gender'] == 'Female']['approved'].mean()
        biased_male_rate = biased_data[biased_data['gender'] == 'Male']['approved'].mean()
        
        unbiased_female_rate = unbiased_data[unbiased_data['gender'] == 'Female']['approved'].mean()
        unbiased_male_rate = unbiased_data[unbiased_data['gender'] == 'Male']['approved'].mean()
        
        # Biased data should show larger gender gap
        biased_gap = abs(biased_male_rate - biased_female_rate)
        unbiased_gap = abs(unbiased_male_rate - unbiased_female_rate)
        
        self.assertGreater(biased_gap, unbiased_gap)
    
    def test_validation_report(self):
        """Test validation report generation."""
        data = self.generator.generate_dataset()
        report = self.generator.generate_validation_report(data)
        
        # Check report structure
        self.assertIn('total_records', report)
        self.assertIn('approval_rate', report)
        self.assertIn('demographics', report)
        self.assertIn('bias_indicators', report)
        
        # Check values
        self.assertEqual(report['total_records'], 1000)
        self.assertGreater(report['approval_rate'], 0)
        self.assertLess(report['approval_rate'], 1)
    
    def test_data_quality(self):
        """Test data quality checks."""
        data = self.generator.generate_dataset()
        
        # Check for missing values
        self.assertEqual(data.isnull().sum().sum(), 0)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(data['age']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['annual_income']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['credit_score']))
        
        # Check value ranges
        self.assertTrue(all(data['age'] >= 18))
        self.assertTrue(all(data['age'] <= 80))
        self.assertTrue(all(data['credit_score'] >= 300))
        self.assertTrue(all(data['credit_score'] <= 850))
        self.assertTrue(all(data['annual_income'] > 0))

if __name__ == '__main__':
    unittest.main()