import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.logistic_model import LogisticRegressionModel
from models.tree_model import TreeModel
from data.synthetic_generator import SyntheticLendingDataGenerator

class TestModels(unittest.TestCase):
    """Test machine learning models."""
    
    def setUp(self):
        """Set up test data."""
        generator = SyntheticLendingDataGenerator(n_samples=1000, random_seed=42)
        self.data = generator.generate_dataset()
        
        # Prepare features
        feature_cols = [
            'annual_income', 'credit_score', 'employment_years',
            'debt_to_income', 'existing_loans', 'previous_defaults',
            'loan_amount', 'loan_term_months', 'age'
        ]
        categorical_cols = ['race', 'gender', 'age_group', 'region', 'loan_type']
        
        X = self.data[feature_cols + categorical_cols].copy()
        self.X = pd.get_dummies(X, columns=categorical_cols)
        self.y = self.data['approved']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_logistic_regression_model(self):
        """Test logistic regression model training and prediction."""
        model = LogisticRegressionModel()
        
        # Test training
        metrics = model.train(self.X_train, self.y_train)
        
        self.assertTrue(model.is_trained)
        self.assertIn('accuracy', metrics)
        self.assertIn('auc', metrics)
        self.assertGreater(metrics['accuracy'], 0.5)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Test probability prediction
        probabilities = model.predict_proba(self.X_test)
        self.assertEqual(len(probabilities), len(self.X_test))
        self.assertTrue(all(0 <= prob <= 1 for prob in probabilities))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('coefficient', importance.columns)
    
    def test_tree_model(self):
        """Test tree-based model training and prediction."""
        model = TreeModel()
        
        # Test training
        metrics = model.train(self.X_train, self.y_train)
        
        self.assertTrue(model.is_trained)
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0.5)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        import tempfile
        
        model = LogisticRegressionModel()
        model.train(self.X_train, self.y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            model.save_model(f.name)
            
            # Load model
            loaded_model = LogisticRegressionModel.load_model(f.name)
            
            # Test loaded model
            self.assertTrue(loaded_model.is_trained)
            
            # Compare predictions
            original_pred = model.predict(self.X_test)
            loaded_pred = loaded_model.predict(self.X_test)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)