import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate lending data for completeness and quality."""
    
    def __init__(self):
        self.required_columns = [
            'application_id', 'race', 'gender', 'age', 
            'annual_income', 'credit_score', 'loan_amount', 'approved'
        ]
        
        self.numeric_columns = [
            'age', 'annual_income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_amount', 'loan_term_months'
        ]
        
        self.categorical_columns = [
            'race', 'gender', 'age_group', 'region', 'loan_type'
        ]
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of lending dataframe.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {},
            'recommendations': []
        }
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types and ranges
        self._validate_numeric_columns(df, results)
        self._validate_categorical_columns(df, results)
        self._validate_business_rules(df, results)
        
        # Generate summary
        results['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Generate recommendations
        self._generate_recommendations(df, results)
        
        logger.info(f"Data validation completed. Valid: {results['is_valid']}")
        
        return results
    
    def _validate_numeric_columns(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate numeric columns."""
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            
            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(df[col]):
                results['warnings'].append(f"Column {col} should be numeric")
            
            # Check for reasonable ranges
            if col == 'age':
                invalid_age = df[(df[col] < 18) | (df[col] > 100)]
                if len(invalid_age) > 0:
                    results['warnings'].append(f"Found {len(invalid_age)} records with invalid age")
            
            elif col == 'credit_score':
                invalid_credit = df[(df[col] < 300) | (df[col] > 850)]
                if len(invalid_credit) > 0:
                    results['warnings'].append(f"Found {len(invalid_credit)} records with invalid credit score")
            
            elif col == 'debt_to_income':
                invalid_dti = df[(df[col] < 0) | (df[col] > 2)]
                if len(invalid_dti) > 0:
                    results['warnings'].append(f"Found {len(invalid_dti)} records with unusual debt-to-income ratio")
    
    def _validate_categorical_columns(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate categorical columns."""
        expected_values = {
            'race': ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
            'gender': ['Male', 'Female', 'Non-binary'],
            'loan_type': ['Personal', 'Auto', 'Mortgage', 'Student', 'Business']
        }
        
        for col, expected in expected_values.items():
            if col not in df.columns:
                continue
            
            unexpected_values = df[~df[col].isin(expected)][col].unique()
            if len(unexpected_values) > 0:
                results['warnings'].append(
                    f"Column {col} has unexpected values: {list(unexpected_values)}"
                )
    
    def _validate_business_rules(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate business logic rules."""
        # Rule 1: Approved amount should not exceed loan amount
        if 'approved_amount' in df.columns and 'loan_amount' in df.columns:
            excess_approval = df[df['approved_amount'] > df['loan_amount']]
            if len(excess_approval) > 0:
                results['errors'].append(
                    f"Found {len(excess_approval)} records where approved amount exceeds loan amount"
                )
        
        # Rule 2: Approved loans should have approved_amount > 0
        if 'approved' in df.columns and 'approved_amount' in df.columns:
            invalid_approved = df[(df['approved'] == 1) & (df['approved_amount'] <= 0)]
            if len(invalid_approved) > 0:
                results['warnings'].append(
                    f"Found {len(invalid_approved)} approved loans with zero approved amount"
                )
        
        # Rule 3: Employment years should not exceed age - 16
        if 'employment_years' in df.columns and 'age' in df.columns:
            invalid_employment = df[df['employment_years'] > (df['age'] - 16)]
            if len(invalid_employment) > 0:
                results['warnings'].append(
                    f"Found {len(invalid_employment)} records with implausible employment history"
                )
    
    def _generate_recommendations(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Generate data quality recommendations."""
        if results['summary']['missing_values'] > 0:
            results['recommendations'].append("Consider imputing or removing missing values")
        
        if results['summary']['duplicate_rows'] > 0:
            results['recommendations'].append("Remove duplicate records")
        
        if len(results['warnings']) > 0:
            results['recommendations'].append("Review and clean data quality issues identified in warnings")