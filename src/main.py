#!/usr/bin/env python3
"""
FAIR-AI Lending Monitor - Main Application Entry Point
Enterprise-grade fair lending risk monitoring and compliance system.
"""

import sys
import os
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.helpers import ConfigManager
from data.synthetic_generator import SyntheticLendingDataGenerator
from models.logistic_model import LogisticRegressionModel
from models.tree_model import TreeModel
from fairness.metrics import FairnessMetricsCalculator, BiasScorer
from explainability.shap_analyzer import SHAPAnalyzer
from explainability.report_generator import ExplainabilityReportGenerator

logger = setup_logger(__name__)

class FAIRAILendingMonitor:
    """Main application class for FAIR-AI Lending Monitor."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the application with configuration."""
        self.config_manager = ConfigManager(config_dir)
        self.config = self.config_manager.load_config("config")
        self.fairness_config = self.config_manager.load_config("fairness_thresholds")
        
        # Initialize components
        self.data_generator = None
        self.models = {}
        self.fairness_calculator = None
        self.bias_scorer = None
        
        logger.info("FAIR-AI Lending Monitor initialized")
    
    def generate_synthetic_data(self, n_samples: int = None, 
                               bias_config: Dict[str, bool] = None) -> None:
        """Generate synthetic lending data for analysis."""
        n_samples = n_samples or self.config.get('data', {}).get('synthetic', {}).get('n_samples', 10000)
        bias_config = bias_config or self.config.get('data', {}).get('synthetic', {}).get('bias_patterns', {})
        
        logger.info(f"Generating {n_samples} synthetic lending records")
        
        self.data_generator = SyntheticLendingDataGenerator(
            n_samples=n_samples,
            random_seed=self.config.get('data', {}).get('synthetic', {}).get('random_seed', 42)
        )
        
        self.data = self.data_generator.generate_dataset(bias_config)
        
        # Save generated data
        output_path = Path("data/synthetic/generated_data.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path, index=False)
        
        logger.info(f"Synthetic data generated and saved to {output_path}")
    
    def train_models(self) -> None:
        """Train fairness evaluation models."""
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Data must be generated first")
        
        logger.info("Training machine learning models")
        
        # Prepare features
        feature_cols = [
            'annual_income', 'credit_score', 'employment_years',
            'debt_to_income', 'existing_loans', 'previous_defaults',
            'loan_amount', 'loan_term_months', 'age'
        ]
        categorical_cols = ['race', 'gender', 'age_group', 'region', 'loan_type', 'urban_rural']
        
        # Create feature matrix with one-hot encoding
        X = self.data[feature_cols + categorical_cols].copy()
        X = pd.get_dummies(X, columns=categorical_cols)
        y = self.data['approved']
        
        # Train Logistic Regression
        lr_config = self.config.get('models', {}).get('logistic_regression', {})
        self.models['logistic'] = LogisticRegressionModel(lr_config)
        self.models['logistic'].train(X, y)
        
        # Train Tree-based model
        tree_config = self.config.get('models', {}).get('xgboost', {})
        self.models['tree'] = TreeModel(tree_config)
        self.models['tree'].train(X, y)
        
        # Save trained models
        models_dir = Path("models/saved_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f"{name}_model.joblib"
            model.save_model(str(model_path))
        
        logger.info("Models trained and saved successfully")
    
    def analyze_fairness(self) -> Dict[str, Any]:
        """Analyze fairness across all models and protected attributes."""
        if not self.models:
            raise ValueError("Models must be trained first")
        
        logger.info("Analyzing fairness metrics")
        
        # Initialize fairness components
        self.fairness_calculator = FairnessMetricsCalculator(self.fairness_config)
        self.bias_scorer = BiasScorer()
        
        fairness_results = {}
        protected_attrs = self.config.get('fairness', {}).get('protected_attributes', ['race', 'gender', 'age_group'])
        
        for model_name, model in self.models.items():
            model_results = {}
            
            # Calculate metrics for each protected attribute
            for attr in protected_attrs:
                reference_group = self.config.get('fairness', {}).get('reference_groups', {}).get(attr)
                
                attr_metrics = self.fairness_calculator.calculate_all_metrics(
                    self.data, 'approved', attr, reference_group
                )
                model_results[attr] = attr_metrics
            
            # Calculate intersectional bias
            intersectional_results = self.fairness_calculator.calculate_intersectional_bias(
                self.data, 'approved', protected_attrs[:2]  # Use first two attributes
            )
            model_results['intersectional'] = intersectional_results
            
            # Calculate overall risk score
            all_metrics = {}
            for attr_results in model_results.values():
                if isinstance(attr_results, dict):
                    all_metrics.update(attr_results)
            
            risk_assessment = self.bias_scorer.calculate_risk_score(all_metrics)
            model_results['risk_assessment'] = risk_assessment
            
            fairness_results[model_name] = model_results
        
        logger.info("Fairness analysis completed")
        return fairness_results
    
    def generate_explanations(self, fairness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model explanations using SHAP."""
        logger.info("Generating model explanations")
        
        # Prepare features for SHAP analysis
        feature_cols = [
            'annual_income', 'credit_score', 'employment_years',
            'debt_to_income', 'existing_loans', 'previous_defaults',
            'loan_amount', 'loan_term_months', 'age'
        ]
        categorical_cols = ['race', 'gender', 'age_group', 'region', 'loan_type', 'urban_rural']
        
        X = self.data[feature_cols + categorical_cols].copy()
        X = pd.get_dummies(X, columns=categorical_cols)
        
        explanations = {}
        
        for model_name, model in self.models.items():
            try:
                # Determine model type for SHAP
                model_type = 'linear' if model_name == 'logistic' else 'tree'
                
                # Create SHAP analyzer
                shap_analyzer = SHAPAnalyzer(model.model, model_type)
                shap_analyzer.create_explainer(X.sample(1000, random_state=42))
                shap_analyzer.calculate_shap_values(X.sample(500, random_state=42))
                
                # Analyze bias contributions
                bias_contributions = shap_analyzer.analyze_bias_contributions(['race', 'gender', 'age'])
                
                explanations[model_name] = {
                    'feature_importance': shap_analyzer.get_feature_importance().to_dict('records'),
                    'bias_contributions': bias_contributions,
                    'sample_explanations': [
                        shap_analyzer.generate_explanation_text(i) for i in range(min(3, len(X)))
                    ]
                }
                
            except Exception as e:
                logger.warning(f"Failed to generate explanations for {model_name}: {e}")
                explanations[model_name] = {'error': str(e)}
        
        return explanations
    
    def generate_comprehensive_report(self, fairness_results: Dict[str, Any], 
                                    explanations: Dict[str, Any]) -> str:
        """Generate comprehensive fairness and explainability report."""
        logger.info("Generating comprehensive report")
        
        report_generator = ExplainabilityReportGenerator("reports")
        
        dataset_info = {
            'size': len(self.data),
            'approval_rate': self.data['approved'].mean(),
            'features': list(self.data.columns)
        }
        
        report = report_generator.generate_model_report(
            model_name="Combined Analysis",
            shap_analyzer=None,
            lime_analyzer=None,
            fairness_results=fairness_results,
            dataset_info=dataset_info
        )
        
        # Add explanations to report
        report['explainability_analysis'] = explanations
        
        # Save reports in multiple formats
        json_path = report_generator.save_report(report)
        html_path = report_generator.generate_html_report(report)
        
        logger.info(f"Reports generated: {json_path}, {html_path}")
        return json_path
    
    def run_full_analysis(self, n_samples: int = None) -> str:
        """Run complete fair lending analysis pipeline."""
        logger.info("Starting full FAIR-AI analysis pipeline")
        
        try:
            # Step 1: Generate synthetic data
            self.generate_synthetic_data(n_samples)
            
            # Step 2: Train models
            self.train_models()
            
            # Step 3: Analyze fairness
            fairness_results = self.analyze_fairness()
            
            # Step 4: Generate explanations
            explanations = self.generate_explanations(fairness_results)
            
            # Step 5: Generate comprehensive report
            report_path = self.generate_comprehensive_report(fairness_results, explanations)
            
            logger.info("Full analysis pipeline completed successfully")
            return report_path
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise
    
    def run_dashboard(self):
        """Launch the Streamlit dashboard."""
        logger.info("Launching FAIR-AI dashboard")
        
        import subprocess
        import sys
        
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_path), 
                "--server.port=8501",
                "--server.headless=true"
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to launch dashboard: {e}")
            raise

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="FAIR-AI Lending Monitor - Fair lending risk assessment system"
    )
    
    parser.add_argument(
        "command",
        choices=["analyze", "dashboard", "generate-data", "train-models"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate (default: 10000)"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory (default: config)"
    )
    
    parser.add_argument(
        "--bias-config",
        type=str,
        help="JSON string with bias configuration"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)"
    )
    
    args = parser.parse_args()
    
    # Initialize application
    app = FAIRAILendingMonitor(args.config_dir)
    
    try:
        if args.command == "analyze":
            # Run full analysis pipeline
            bias_config = None
            if args.bias_config:
                import json
                bias_config = json.loads(args.bias_config)
            
            report_path = app.run_full_analysis(args.samples)
            print(f"Analysis complete. Report saved to: {report_path}")
            
        elif args.command == "dashboard":
            # Launch interactive dashboard
            app.run_dashboard()
            
        elif args.command == "generate-data":
            # Generate synthetic data only
            app.generate_synthetic_data(args.samples)
            print(f"Generated {args.samples} synthetic lending records")
            
        elif args.command == "train-models":
            # Train models only (requires existing data)
            app.train_models()
            print("Models trained successfully")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()