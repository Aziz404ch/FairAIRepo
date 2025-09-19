from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from .base_model import BaseModel

class TreeModel(BaseModel):
    """Tree-based model (Random Forest) for lending decisions."""
    
    def __init__(self, config: dict = None):
        default_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'tune_hyperparameters': False
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("Random Forest", default_config)
    
    def _create_model(self):
        """Create Random Forest model."""
        if self.config.get('tune_hyperparameters', False):
            return self._create_tuned_model()
        else:
            return RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs']
            )
    
    def _create_tuned_model(self):
        """Create model with hyperparameter tuning."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        return grid_search
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Handle case where model is GridSearchCV
        if hasattr(self.model, 'best_estimator_'):
            importance_scores = self.model.best_estimator_.feature_importances_
        else:
            importance_scores = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_tree_rules(self, max_trees: int = 3) -> list:
        """Extract interpretable rules from the first few trees."""
        if not self.is_trained:
            raise ValueError("Model must be trained to extract rules")
        
        # Get the actual RandomForest model
        if hasattr(self.model, 'best_estimator_'):
            rf_model = self.model.best_estimator_
        else:
            rf_model = self.model
        
        rules = []
        
        for i, tree in enumerate(rf_model.estimators_[:max_trees]):
            tree_rules = self._extract_tree_rules(tree, self.feature_names)
            rules.extend([(f"Tree_{i+1}", rule) for rule in tree_rules])
        
        return rules
    
    def _extract_tree_rules(self, tree, feature_names, max_depth: int = 3):
        """Extract rules from a single decision tree."""
        tree_structure = tree.tree_
        
        def recurse(node, depth, path):
            if depth > max_depth:
                return []
            
            if tree_structure.children_left[node] == tree_structure.children_right[node]:
                # Leaf node
                samples = tree_structure.n_node_samples[node]
                value = tree_structure.value[node][0]
                prediction = "Approved" if value[1] > value[0] else "Rejected"
                confidence = max(value) / sum(value)
                
                return [f"{' AND '.join(path)} → {prediction} (confidence: {confidence:.2f}, samples: {samples})"]
            else:
                # Internal node
                feature = feature_names[tree_structure.feature[node]]
                threshold = tree_structure.threshold[node]
                
                left_path = path + [f"{feature} <= {threshold:.2f}"]
                right_path = path + [f"{feature} > {threshold:.2f}"]
                
                rules = []
                rules.extend(recurse(tree_structure.children_left[node], depth + 1, left_path))
                rules.extend(recurse(tree_structure.children_right[node], depth + 1, right_path))
                
                return rules
        
        return recurse(0, 0, [])