import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LIMEAnalyzer:
    """LIME-based local interpretability analyzer."""
    
    def __init__(self, model, feature_names: List[str], class_names: List[str] = None):
        """
        Initialize LIME analyzer.
        
        Args:
            model: Trained model with predict_proba method
            feature_names: List of feature names
            class_names: List of class names (default: ['Rejected', 'Approved'])
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Rejected', 'Approved']
        self.explainer = None
    
    def create_explainer(self, X_train: pd.DataFrame, **kwargs):
        """Create LIME tabular explainer."""
        logger.info("Creating LIME tabular explainer")
        
        # Determine categorical features
        categorical_features = []
        for i, col in enumerate(self.feature_names):
            if any(cat in col for cat in ['race_', 'gender_', 'age_group_', 'region_', 'loan_type_']):
                categorical_features.append(i)
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features,
            verbose=True,
            mode='classification',
            **kwargs
        )
        
        logger.info("LIME explainer created successfully")
    
    def explain_instance(self, instance: pd.Series, num_features: int = 10) -> Any:
        """Explain a single prediction instance."""
        if self.explainer is None:
            raise ValueError("Explainer must be created before explaining instances")
        
        explanation = self.explainer.explain_instance(
            instance.values,
            self.model.predict_proba,
            num_features=num_features
        )
        
        return explanation
    
    def create_explanation_plot(self, explanation, title: str = "LIME Explanation") -> go.Figure:
        """Create plotly visualization of LIME explanation."""
        # Extract explanation data
        exp_data = explanation.as_list()
        
        features = [item[0] for item in exp_data]
        values = [item[1] for item in exp_data]
        
        # Create color based on positive/negative contribution
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        fig = px.bar(
            x=values,
            y=features,
            orientation='h',
            title=title,
            color=values,
            color_continuous_scale='RdBu',
            labels={'x': 'Feature Contribution', 'y': 'Features'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            height=400
        )
        
        return fig
    
    def batch_explain(self, X_test: pd.DataFrame, num_samples: int = 100, 
                     num_features: int = 5) -> List[Dict[str, Any]]:
        """Explain multiple instances and return aggregated results."""
        if self.explainer is None:
            raise ValueError("Explainer must be created before explaining instances")
        
        logger.info(f"Generating LIME explanations for {num_samples} instances")
        
        # Sample instances for explanation
        X_sample = X_test.sample(n=min(len(X_test), num_samples), random_state=42)
        
        explanations = []
        
        for idx, (_, instance) in enumerate(X_sample.iterrows()):
            try:
                explanation = self.explain_instance(instance, num_features)
                
                exp_dict = {
                    'instance_id': idx,
                    'prediction_proba': self.model.predict_proba([instance.values])[0],
                    'features': [item[0] for item in explanation.as_list()],
                    'contributions': [item[1] for item in explanation.as_list()]
                }
                
                explanations.append(exp_dict)
                
            except Exception as e:
                logger.warning(f"Failed to explain instance {idx}: {e}")
                continue
        
        return explanations
    
    def analyze_feature_stability(self, instance: pd.Series, num_samples: int = 100) -> Dict[str, Any]:
        """Analyze stability of feature explanations by running LIME multiple times."""
        if self.explainer is None:
            raise ValueError("Explainer must be created before analyzing stability")
        
        logger.info("Analyzing explanation stability")
        
        all_explanations = []
        
        for _ in range(num_samples):
            try:
                explanation = self.explain_instance(instance, num_features=10)
                exp_dict = {feat: contrib for feat, contrib in explanation.as_list()}
                all_explanations.append(exp_dict)
            except Exception as e:
                logger.warning(f"Failed in stability analysis: {e}")
                continue
        
        # Calculate statistics for each feature
        feature_stats = {}
        all_features = set()
        for exp in all_explanations:
            all_features.update(exp.keys())
        
        for feature in all_features:
            contributions = [exp.get(feature, 0) for exp in all_explanations]
            
            feature_stats[feature] = {
                'mean': np.mean(contributions),
                'std': np.std(contributions),
                'min': np.min(contributions),
                'max': np.max(contributions),
                'appearances': sum(1 for c in contributions if c != 0)
            }
        
        return {
            'feature_statistics': feature_stats,
            'num_explanations': len(all_explanations),
            'stability_score': self._calculate_stability_score(feature_stats)
        }
    
    def _calculate_stability_score(self, feature_stats: Dict[str, Dict]) -> float:
        """Calculate overall stability score (lower variance = higher stability)."""
        variances = [stats['std']**2 for stats in feature_stats.values() if stats['appearances'] > 0]
        
        if not variances:
            return 0.0
        
        # Normalize by mean absolute contribution
        mean_abs_contributions = [abs(stats['mean']) for stats in feature_stats.values()]
        
        if sum(mean_abs_contributions) == 0:
            return 1.0
        
        normalized_variance = np.mean(variances) / np.mean(mean_abs_contributions)
        stability_score = 1 / (1 + normalized_variance)
        
        return stability_score