import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class LendingDataPreprocessor:
    """Preprocess lending data for machine learning models."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'approved') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessors and transform data.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Fitting and transforming data")
        
        # Separate features and target
        X = df.drop(columns=[target_col] + self._get_id_columns(df))
        y = df[target_col]
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Encode categorical variables
        X = self._encode_categorical_features(X, fit=True)
        
        # Scale numerical features
        X = self._scale_numerical_features(X, fit=True)
        
        self.is_fitted = True
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Preprocessing complete. Features: {len(X.columns)}, Samples: {len(X)}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Transform new data using fitted preprocessors."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming new data")
        
        # Remove ID columns and target if present
        id_cols = self._get_id_columns(df)
        cols_to_drop = id_cols + ([target_col] if target_col and target_col in df.columns else [])
        X = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Apply preprocessing steps
        X = self._handle_missing_values(X, fit=False)
        X = self._encode_categorical_features(X, fit=False)
        X = self._scale_numerical_features(X, fit=False)
        
        return X
    
    def _get_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify ID columns to exclude from features."""
        id_patterns = ['id', 'application_id', 'customer_id', 'loan_id']
        id_columns = []
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in id_patterns):
                id_columns.append(col)
        
        return id_columns
    
    def _handle_missing_values(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values in the data."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        X_processed = X.copy()
        
        # Handle numeric columns
        if len(numeric_columns) > 0:
            if fit:
                self.imputers['numeric'] = SimpleImputer(strategy='median')
                X_processed[numeric_columns] = self.imputers['numeric'].fit_transform(X[numeric_columns])
            else:
                if 'numeric' in self.imputers:
                    X_processed[numeric_columns] = self.imputers['numeric'].transform(X[numeric_columns])
        
        # Handle categorical columns
        if len(categorical_columns) > 0:
            if fit:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                X_processed[categorical_columns] = self.imputers['categorical'].fit_transform(X[categorical_columns])
            else:
                if 'categorical' in self.imputers:
                    X_processed[categorical_columns] = self.imputers['categorical'].transform(X[categorical_columns])
        
        return X_processed
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) == 0:
            return X
        
        X_processed = X.copy()
        
        if fit:
            # Use one-hot encoding for categorical variables
            X_processed = pd.get_dummies(X_processed, columns=categorical_columns, prefix=categorical_columns)
        else:
            # Apply same encoding as training
            for col in categorical_columns:
                if col in X.columns:
                    col_dummies = pd.get_dummies(X[col], prefix=col)
                    
                    # Ensure same columns as training
                    for feature_col in self.feature_names:
                        if feature_col.startswith(f"{col}_") and feature_col not in col_dummies.columns:
                            col_dummies[feature_col] = 0
                    
                    # Remove original column and add dummies
                    X_processed = X_processed.drop(columns=[col])
                    for dummy_col in col_dummies.columns:
                        if dummy_col in self.feature_names:
                            X_processed[dummy_col] = col_dummies[dummy_col]
        
        return X_processed
    
    def _scale_numerical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return X
        
        X_processed = X.copy()
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            X_processed[numeric_columns] = self.scalers['standard'].fit_transform(X[numeric_columns])
        else:
            if 'standard' in self.scalers:
                # Only scale columns that exist in both training and test data
                common_cols = [col for col in numeric_columns if col in self.scalers['standard'].feature_names_in_]
                if common_cols:
                    X_processed[common_cols] = self.scalers['standard'].transform(X[common_cols])
        
        return X_processed
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        return self.feature_names
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform target variable if needed."""
        # For binary classification, no inverse transform needed
        return y_encoded