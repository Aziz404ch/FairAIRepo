# src/fairness/mitigation.py
import pandas as pd
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
from scipy import stats

logger = logging.getLogger(__name__)

class AdvancedBiasMitigator:
    """
    Advanced bias mitigation techniques with counterfactual analysis,
    adversarial debiasing, and policy simulation capabilities.
    """
    
    def __init__(self, model: Any, protected_attrs: List[str] = None,
                 config: Dict[str, Any] = None):
        self.model = model
        self.protected_attrs = protected_attrs or ['race', 'gender', 'age_group', 'region']
        self.config = config or {}
        self.fitted_model = None
        self.adversarial_model = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        
    def simulate_policy_change(self, X: pd.DataFrame, policy_rules: Dict[str, Any],
                             outcome_col: str = 'approved') -> Dict[str, Any]:
        """
        Simulate the impact of policy changes on lending decisions.
        """
        baseline_pred = self.model.predict(X)
        baseline_approval = baseline_pred.mean()
        
        # Apply policy rules
        X_modified = X.copy()
        for feature, rule in policy_rules.items():
            if feature in X_modified.columns:
                if rule['type'] == 'threshold':
                    X_modified[feature] = X_modified[feature].apply(
                        lambda x: max(x, rule['value']) if rule['direction'] == 'increase' 
                        else min(x, rule['value'])
                    )
                elif rule['type'] == 'categorical':
                    X_modified[feature] = X_modified[feature].apply(
                        lambda x: rule['mapping'].get(x, x)
                    )
        
        # Predict with modified features
        modified_pred = self.model.predict(X_modified)
        modified_approval = modified_pred.mean()
        
        # Calculate impact by demographic groups
        impact_by_group = {}
        for attr in self.protected_attrs:
            if attr in X.columns:
                impact = {}
                for group in X[attr].unique():
                    group_mask = X[attr] == group
                    group_baseline = baseline_pred[group_mask].mean()
                    group_modified = modified_pred[group_mask].mean()
                    impact[group] = {
                        'baseline_rate': group_baseline,
                        'modified_rate': group_modified,
                        'change': group_modified - group_baseline,
                        'percent_change': (group_modified - group_baseline) / group_baseline * 100 
                        if group_baseline > 0 else 0
                    }
                impact_by_group[attr] = impact
        
        return {
            'overall_impact': {
                'baseline_approval_rate': baseline_approval,
                'modified_approval_rate': modified_approval,
                'absolute_change': modified_approval - baseline_approval,
                'percent_change': (modified_approval - baseline_approval) / baseline_approval * 100 
                if baseline_approval > 0 else 0
            },
            'group_impacts': impact_by_group,
            'policy_rules': policy_rules,
            'affected_applications': sum(baseline_pred != modified_pred)
        }
    
    def generate_counterfactuals(self, instance: pd.Series, X_train: pd.DataFrame,
                               target_class: int = 1, n_counterfactuals: int = 5,
                               proximity_weight: float = 0.5, diversity_weight: float = 0.3,
                               validity_weight: float = 0.2) -> pd.DataFrame:
        """
        Generate diverse counterfactual explanations using optimization.
        """
        # Convert to numpy
        instance_np = instance.values.reshape(1, -1)
        X_train_np = X_train.values
        
        # Initialize SHAP explainer if not already done
        if self.shap_explainer is None:
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(X_train_np, 100, random_state=42)
            )
        
        # Get SHAP values for the instance
        shap_values = self.shap_explainer.shap_values(instance_np)[0]
        
        # Identify most impactful features
        feature_importance = np.abs(shap_values).argsort()[::-1]
        top_features = feature_importance[:10]  # Top 10 features
        
        # Generate counterfactuals by perturbing important features
        counterfactuals = []
        for _ in range(n_counterfactuals * 2):  # Generate extra for diversity
            cf = instance_np.copy()
            
            # Modify features based on importance
            for i, feature_idx in enumerate(top_features):
                # Perturb feature based on its distribution
                feature_values = X_train_np[:, feature_idx]
                if np.issubdtype(feature_values.dtype, np.number):
                    # Numerical feature - perturb within reasonable range
                    std = np.std(feature_values)
                    perturbation = np.random.normal(0, std * 0.5)
                    cf[0, feature_idx] += perturbation
                else:
                    # Categorical feature - change to another category
                    categories = np.unique(feature_values)
                    cf[0, feature_idx] = np.random.choice(categories)
            
            # Check if counterfactual changes the prediction
            cf_pred = self.model.predict(cf)[0]
            cf_proba = self.model.predict_proba(cf)[0][target_class]
            
            if cf_pred == target_class:
                counterfactuals.append({
                    'instance': cf[0],
                    'confidence': cf_proba,
                    'changes': np.abs(cf[0] - instance_np[0]),
                    'proximity': np.linalg.norm(cf[0] - instance_np[0])
                })
        
        # Select diverse counterfactuals
        if not counterfactuals:
            return pd.DataFrame()
        
        # Score counterfactuals
        for cf in counterfactuals:
            cf_score = (
                proximity_weight * (1 / (1 + cf['proximity'])) +
                diversity_weight * self._calculate_diversity(cf, counterfactuals) +
                validity_weight * cf['confidence']
            )
            cf['score'] = cf_score
        
        # Select best counterfactuals
        counterfactuals.sort(key=lambda x: x['score'], reverse=True)
        selected = counterfactuals[:min(n_counterfactuals, len(counterfactuals))]
        
        # Convert to DataFrame
        cf_df = pd.DataFrame([cf['instance'] for cf in selected], columns=X_train.columns)
        cf_df['counterfactual_confidence'] = [cf['confidence'] for cf in selected]
        cf_df['proximity_score'] = [cf['proximity'] for cf in selected]
        cf_df['overall_score'] = [cf['score'] for cf in selected]
        
        return cf_df
    
    def _calculate_diversity(self, counterfactual: Dict[str, Any], 
                           all_counterfactuals: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for a counterfactual."""
        if len(all_counterfactuals) <= 1:
            return 1.0
        
        similarities = []
        for cf in all_counterfactuals:
            if cf is not counterfactual:
                sim = np.dot(counterfactual['instance'], cf['instance']) / (
                    np.linalg.norm(counterfactual['instance']) * np.linalg.norm(cf['instance'])
                )
                similarities.append(sim)
        
        return 1 - np.mean(similarities) if similarities else 1.0
    
    def adversarial_debias(self, X: pd.DataFrame, y: pd.Series, protected_attr: str,
                         epochs: int = 100, batch_size: int = 32, 
                         learning_rate: float = 0.001) -> nn.Module:
        """
        Advanced adversarial debiasing using neural networks with improved stability.
        """
        # Prepare data
        X_np = X.drop(columns=[protected_attr]).values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        
        # Encode protected attribute
        le = LabelEncoder()
        z = le.fit_transform(X[protected_attr])
        z_np = z.astype(np.float32)
        
        # Normalize features
        X_np = self.scaler.fit_transform(X_np)
        
        # Create datasets
        dataset = TensorDataset(
            torch.from_numpy(X_np), 
            torch.from_numpy(y_np), 
            torch.from_numpy(z_np)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define models
        input_dim = X_np.shape[1]
        hidden_dim = 64
        
        # Predictor network
        self.adversarial_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adversary network
        adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(np.unique(z))),
            nn.Softmax(dim=1)
        )
        
        # Optimizers
        optimizer_clf = optim.Adam(self.adversarial_model.parameters(), lr=learning_rate)
        optimizer_adv = optim.Adam(adversary.parameters(), lr=learning_rate * 0.1)
        
        # Loss functions
        loss_clf = nn.BCELoss()
        loss_adv = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            total_clf_loss = 0
            total_adv_loss = 0
            
            for batch_x, batch_y, batch_z in loader:
                # Train adversary
                optimizer_adv.zero_grad()
                
                with torch.no_grad():
                    hidden = self.adversarial_model[0:4](batch_x)  # Get up to first hidden layer
                
                pred_z = adversary(hidden)
                adv_loss = loss_adv(pred_z, batch_z.long())
                adv_loss.backward()
                optimizer_adv.step()
                
                # Train classifier with adversarial debiasing
                optimizer_clf.zero_grad()
                
                # Forward pass through classifier
                hidden = self.adversarial_model[0:4](batch_x)
                pred_y = self.adversarial_model[4:](hidden)
                
                # Classification loss
                clf_loss = loss_clf(pred_y.squeeze(), batch_y)
                
                # Adversarial loss (to minimize)
                with torch.no_grad():
                    pred_z = adversary(hidden)
                
                # Negative gradient for adversarial loss
                adv_clf_loss = -loss_adv(pred_z, batch_z.long())
                
                # Combined loss
                total_loss = clf_loss + self.config.get('adversarial_weight', 0.5) * adv_clf_loss
                total_loss.backward()
                optimizer_clf.step()
                
                total_clf_loss += clf_loss.item()
                total_adv_loss += adv_loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Clf Loss {total_clf_loss/len(loader):.4f}, "
                          f"Adv Loss {total_adv_loss/len(loader):.4f}")
        
        logger.info("Adversarial debiasing completed")
        return self.adversarial_model
    
    def calculate_fairness_improvement(self, X: pd.DataFrame, y: pd.Series,
                                     protected_attr: str, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate improvement in fairness metrics after mitigation.
        """
        from ..fairness.metrics import FairnessMetricsCalculator
        
        # Predict with mitigated model
        if self.adversarial_model is not None:
            X_np = X.drop(columns=[protected_attr]).values.astype(np.float32)
            X_np = self.scaler.transform(X_np)
            X_tensor = torch.from_numpy(X_np)
            
            with torch.no_grad():
                y_pred = self.adversarial_model(X_tensor).numpy().squeeze()
                y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = self.fitted_model.predict(X)
        
        # Calculate metrics
        calculator = FairnessMetricsCalculator()
        mitigated_metrics = calculator.calculate_all_metrics(
            X.assign(approved=y_pred_binary), 'approved', protected_attr
        )
        
        # Calculate improvement
        improvement = {}
        for metric_name, baseline_result in baseline_metrics.items():
            if metric_name in mitigated_metrics:
                mitigated_result = mitigated_metrics[metric_name]
                improvement[metric_name] = {
                    'baseline': baseline_result.value,
                    'mitigated': mitigated_result.value,
                    'improvement': baseline_result.value - mitigated_result.value,
                    'percent_improvement': (
                        (baseline_result.value - mitigated_result.value) / baseline_result.value * 100
                        if baseline_result.value != 0 else 0
                    ),
                    'passed_baseline': baseline_result.passed,
                    'passed_mitigated': mitigated_result.passed
                }
        
        return improvement