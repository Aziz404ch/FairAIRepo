# models/logistic_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.fairness.metrics import FairnessMetricResult

class LogisticModel:
    """
    Logistic Regression Model for Fair Lending Compliance
    """
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.metric_results = FairnessMetricResult()
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the logistic regression model
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X: pd.DataFrame):
        """
        Predict the target variable
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Evaluate the model
        """
        y_pred = self.predict(X)
        self.metric_results.calculate_all_metrics(X, y, y_pred)
        return self.metric_results


# Example usage
if __name__ == "__main__":
    # Load sample data
    import sys
    sys.path.append('..')
    from data.synthetic_generator import SyntheticLendingDataGenerator
    
    # Generate test data
    generator = SyntheticLendingDataGenerator(n_samples=5000)
    df = generator.generate_dataset()
    
    # Initialize model
    model = LogisticModel()
    
    # Fit model
    model.fit(df.drop('approved', axis=1), df['approved'])
    
    # Evaluate model
    model.evaluate(df.drop('approved', axis=1), df['approved'])
    
    # Print results
    print(model.metric_results)

    # Print recommendation
    print(model.metric_results.recommendation)

    # Print risk score
    print(model.metric_results.risk_score)

    # Print regulatory compliance
    print(model.metric_results.regulatory_compliance)