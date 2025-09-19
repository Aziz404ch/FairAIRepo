import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import pickle
import json
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

class ModelCache:
    """Cache for trained models and results."""
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, pd.DataFrame):
            # Use shape and column hash for DataFrame
            columns_str = ",".join(sorted(data.columns))
            shape_str = f"{data.shape[0]}x{data.shape[1]}"
            key_str = f"{columns_str}_{shape_str}"
        else:
            key_str = str(data)
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def save(self, key: str, data: Any, metadata: Dict = None):
        """Save data to cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump({'data': data, 'metadata': metadata or {}}, f)
            
            logger.debug(f"Saved to cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def load(self, key: str) -> Optional[Tuple[Any, Dict]]:
        """Load data from cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                
                logger.debug(f"Loaded from cache: {key}")
                return cached['data'], cached['metadata']
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        
        return None, {}
    
    def clear(self):
        """Clear all cached files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")

class ConfigManager:
    """Manage configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return {}
        
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self._configs[config_name] = config
            logger.info(f"Loaded config: {config_name}")
            return config
        
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
            return {}
    
    def get_nested_config(self, config_name: str, *keys) -> Any:
        """Get nested configuration value."""
        config = self.load_config(config_name)
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return None
        
        return config

class DataExporter:
    """Export data and results to various formats."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_fairness_report(self, results: Dict[str, Any], 
                              format: str = 'json', 
                              filename: str = None) -> str:
        """Export fairness analysis results."""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fairness_report_{timestamp}"
        
        if format.lower() == 'json':
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            filepath = self.output_dir / f"{filename}.csv"
            # Convert nested results to flat DataFrame
            flattened = self._flatten_fairness_results(results)
            pd.DataFrame(flattened).to_csv(filepath, index=False)
        
        elif format.lower() == 'html':
            filepath = self.output_dir / f"{filename}.html"
            html_content = self._generate_html_report(results)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported fairness report to: {filepath}")
        return str(filepath)
    
    def _flatten_fairness_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested fairness results for CSV export."""
        flattened = []
        
        for model_name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
                
            for attr_name, attr_results in model_results.items():
                if attr_name == 'risk_assessment':
                    continue
                
                if isinstance(attr_results, dict):
                    for metric_name, metric_result in attr_results.items():
                        if hasattr(metric_result, 'value'):
                            flattened.append({
                                'model': model_name,
                                'protected_attribute': attr_name,
                                'metric': metric_result.metric_name,
                                'value': metric_result.value,
                                'threshold': metric_result.threshold,
                                'passed': metric_result.passed,
                                'severity': metric_result.severity,
                                'regulation': getattr(metric_result, 'regulation', 'N/A')
                            })
        
        return flattened
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from fairness results."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fair Lending Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .pass {{ border-left: 5px solid green; }}
        .fail {{ border-left: 5px solid red; }}
        .severity-high {{ background-color: #ffebee; }}
        .severity-medium {{ background-color: #fff3e0; }}
        .severity-low {{ background-color: #e8f5e8; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Fair Lending Analysis Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
        """
        
        for model_name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
                
            html += f"<h2>Model: {model_name.title()}</h2>"
            
            # Risk Assessment Summary
            if 'risk_assessment' in model_results:
                risk_data = model_results['risk_assessment']
                severity = risk_data.get('severity', 'Unknown')
                
                html += f"""
                <div class="metric-card severity-{severity.lower()}">
                    <h3>Risk Assessment</h3>
                    <p><strong>Overall Risk Score:</strong> {risk_data.get('overall_risk_score', 'N/A'):.3f}</p>
                    <p><strong>Severity Level:</strong> {severity}</p>
                    <p><strong>Regulatory Compliance:</strong> {'✅ Compliant' if risk_data.get('regulatory_compliance', {}).get('compliant', False) else '❌ Non-Compliant'}</p>
                </div>
                """
            
            # Fairness Metrics
            for attr_name, attr_results in model_results.items():
                if attr_name == 'risk_assessment' or not isinstance(attr_results, dict):
                    continue
                
                html += f"<h3>Protected Attribute: {attr_name.title()}</h3>"
                html += "<table><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th><th>Severity</th></tr>"
                
                for metric_name, metric_result in attr_results.items():
                    if hasattr(metric_result, 'value'):
                        status_class = 'pass' if metric_result.passed else 'fail'
                        status_text = '✅ Pass' if metric_result.passed else '❌ Fail'
                        
                        html += f"""
                        <tr class="{status_class}">
                            <td>{metric_result.metric_name}</td>
                            <td>{metric_result.value:.3f}</td>
                            <td>{metric_result.threshold:.3f}</td>
                            <td>{status_text}</td>
                            <td>{metric_result.severity}</td>
                        </tr>
                        """
                
                html += "</table>"
        
        html += "</body></html>"
        return html
    
    def export_model_predictions(self, model, X_test: pd.DataFrame, 
                                y_test: pd.Series = None, 
                                filename: str = None) -> str:
        """Export model predictions to CSV."""
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        results_df = X_test.copy()
        results_df['prediction'] = predictions
        
        if pred_proba is not None:
            results_df['prediction_probability'] = pred_proba
        
        if y_test is not None:
            results_df['actual'] = y_test
            results_df['correct'] = predictions == y_test
        
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_predictions_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        results_df.to_csv(filepath, index=False)
        
        logger.info(f"Exported predictions to: {filepath}")
        return str(filepath)

class AlertSystem:
    """System for generating bias detection alerts."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            'high_risk': 0.7,
            'medium_risk': 0.4,
            'low_risk': 0.2
        }
        self.alerts = []
    
    def evaluate_bias_risk(self, risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate bias risk and generate alerts."""
        alerts = []
        
        risk_score = risk_assessment.get('overall_risk_score', 0)
        severity = risk_assessment.get('severity', 'Low')
        
        # Generate risk level alert
        if risk_score >= self.thresholds['high_risk']:
            alerts.append({
                'type': 'HIGH_RISK',
                'priority': 'CRITICAL',
                'message': f"High bias risk detected (score: {risk_score:.2f})",
                'action_required': True,
                'recommendations': risk_assessment.get('recommendations', [])[:3]
            })
        
        elif risk_score >= self.thresholds['medium_risk']:
            alerts.append({
                'type': 'MEDIUM_RISK',
                'priority': 'HIGH',
                'message': f"Medium bias risk detected (score: {risk_score:.2f})",
                'action_required': True,
                'recommendations': risk_assessment.get('recommendations', [])[:2]
            })
        
        # Check regulatory compliance
        compliance = risk_assessment.get('regulatory_compliance', {})
        if not compliance.get('compliant', True):
            violations = compliance.get('violations', [])
            alerts.append({
                'type': 'COMPLIANCE_VIOLATION',
                'priority': 'CRITICAL',
                'message': f"Regulatory compliance violations detected: {len(violations)} issues",
                'action_required': True,
                'details': violations
            })
        
        # Check specific metric failures
        metric_scores = risk_assessment.get('metric_scores', [])
        high_risk_metrics = [m for m in metric_scores if m['risk'] > 0.6]
        
        if high_risk_metrics:
            alerts.append({
                'type': 'METRIC_FAILURE',
                'priority': 'HIGH',
                'message': f"Multiple fairness metrics failing: {len(high_risk_metrics)} metrics",
                'action_required': True,
                'details': [m['metric'] for m in high_risk_metrics]
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_active_alerts(self, priority: str = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by priority."""
        if priority:
            return [alert for alert in self.alerts if alert['priority'] == priority.upper()]
        return self.alerts
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
        logger.info("All alerts cleared")