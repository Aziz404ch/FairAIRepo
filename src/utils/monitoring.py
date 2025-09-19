# src/utils/monitoring.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import time
from threading import Thread
from queue import Queue
import schedule
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class ContinuousMonitoringSystem:
    """
    Advanced continuous monitoring system for fair lending compliance
    with automated scheduling, drift detection, and alerting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.monitoring_jobs = {}
        self.alert_queue = Queue()
        self.is_running = False
        self.monitoring_thread = None
        self.baseline_metrics = {}
        self.drift_detectors = self._initialize_drift_detectors()
        
    def _initialize_drift_detectors(self) -> Dict[str, Any]:
        """Initialize advanced drift detection algorithms."""
        return {
            'statistical_drift': self._detect_statistical_drift,
            'fairness_drift': self._detect_fairness_drift,
            'performance_drift': self._detect_performance_drift,
            'temporal_patterns': self._detect_temporal_patterns
        }
    
    def start_monitoring(self, data_source: Callable, model: Any, 
                       protected_attrs: List[str], interval_minutes: int = 60):
        """
        Start continuous monitoring with scheduled jobs.
        """
        self.is_running = True
        
        # Schedule monitoring jobs
        schedule.every(interval_minutes).minutes.do(
            self._run_monitoring_cycle, data_source, model, protected_attrs
        )
        
        # Start alert processing thread
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Started continuous monitoring with {interval_minutes} minute interval")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                self._process_alerts()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _run_monitoring_cycle(self, data_source: Callable, model: Any,
                            protected_attrs: List[str]):
        """
        Run a complete monitoring cycle.
        """
        try:
            # Get latest data
            recent_data = data_source()
            if recent_data is None or len(recent_data) == 0:
                logger.warning("No data available for monitoring cycle")
                return
            
            # Calculate current metrics
            current_metrics = self._calculate_current_metrics(recent_data, model, protected_attrs)
            
            # Detect drifts and changes
            drifts = self._detect_all_drifts(current_metrics, recent_data)
            
            # Generate alerts for significant drifts
            for drift in drifts:
                if drift['severity'] in ['high', 'critical']:
                    self.alert_queue.put({
                        'type': 'DRIFT_DETECTED',
                        'severity': drift['severity'],
                        'message': f"{drift['type']} detected: {drift['description']}",
                        'details': drift,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Update baseline if configured
            if self.config.get('adaptive_baseline', False):
                self._update_baseline_metrics(current_metrics)
            
            logger.info(f"Completed monitoring cycle. Detected {len(drifts)} drifts.")
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            self.alert_queue.put({
                'type': 'MONITORING_ERROR',
                'severity': 'high',
                'message': f"Monitoring cycle failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
    
    def _calculate_current_metrics(self, data: pd.DataFrame, model: Any,
                                 protected_attrs: List[str]) -> Dict[str, Any]:
        """Calculate current fairness and performance metrics."""
        from ..fairness.metrics import FairnessMetricsCalculator
        
        # Make predictions
        predictions = model.predict(data.drop('approved', axis=1))
        data_with_preds = data.assign(predicted=predictions)
        
        # Calculate fairness metrics
        calculator = FairnessMetricsCalculator()
        fairness_metrics = {}
        for attr in protected_attrs:
            if attr in data.columns:
                fairness_metrics[attr] = calculator.calculate_all_metrics(
                    data_with_preds, 'predicted', attr
                )
        
        # Calculate performance metrics
        performance_metrics = {
            'accuracy': accuracy_score(data['approved'], predictions),
            'precision': precision_score(data['approved'], predictions),
            'recall': recall_score(data['approved'], predictions),
            'f1_score': 2 * (precision_score(data['approved'], predictions) * 
                            recall_score(data['approved'], predictions)) / 
                       (precision_score(data['approved'], predictions) + 
                        recall_score(data['approved'], predictions))
        }
        
        # Calculate statistical properties
        statistical_metrics = {
            'approval_rate': data['approved'].mean(),
            'prediction_rate': predictions.mean(),
            'demographic_breakdown': {
                attr: data[attr].value_counts(normalize=True).to_dict()
                for attr in protected_attrs if attr in data.columns
            }
        }
        
        return {
            'fairness_metrics': fairness_metrics,
            'performance_metrics': performance_metrics,
            'statistical_metrics': statistical_metrics,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(data)
        }
    
    def _detect_all_drifts(self, current_metrics: Dict[str, Any], 
                          recent_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect all types of drifts using specialized detectors."""
        drifts = []
        
        for drift_type, detector in self.drift_detectors.items():
            try:
                drift = detector(current_metrics, recent_data)
                if drift:
                    drifts.append(drift)
            except Exception as e:
                logger.warning(f"Drift detection failed for {drift_type}: {e}")
        
        return drifts
    
    def _detect_statistical_drift(self, current_metrics: Dict[str, Any],
                                recent_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect statistical drift in data distributions."""
        if not self.baseline_metrics:
            return None
        
        baseline_stats = self.baseline_metrics.get('statistical_metrics', {})
        current_stats = current_metrics.get('statistical_metrics', {})
        
        drifts = []
        
        # Check approval rate drift
        baseline_approval = baseline_stats.get('approval_rate', 0)
        current_approval = current_stats.get('approval_rate', 0)
        approval_drift = abs(current_approval - baseline_approval)
        
        if approval_drift > self.config.get('approval_drift_threshold', 0.05):
            drifts.append({
                'metric': 'approval_rate',
                'baseline': baseline_approval,
                'current': current_approval,
                'drift_amount': approval_drift,
                'severity': 'high' if approval_drift > 0.1 else 'medium'
            })
        
        # Check demographic distribution drift
        for attr, current_dist in current_stats.get('demographic_breakdown', {}).items():
            baseline_dist = baseline_stats.get('demographic_breakdown', {}).get(attr, {})
            
            for group, current_prop in current_dist.items():
                baseline_prop = baseline_dist.get(group, 0)
                group_drift = abs(current_prop - baseline_prop)
                
                if group_drift > self.config.get('demographic_drift_threshold', 0.1):
                    drifts.append({
                        'metric': f'demographic_proportion_{attr}_{group}',
                        'baseline': baseline_prop,
                        'current': current_prop,
                        'drift_amount': group_drift,
                        'severity': 'high' if group_drift > 0.15 else 'medium'
                    })
        
        if not drifts:
            return None
        
        return {
            'type': 'statistical_drift',
            'description': f"Statistical drift detected in {len(drifts)} metrics",
            'drifts': drifts,
            'severity': max(d['severity'] for d in drifts),
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_fairness_drift(self, current_metrics: Dict[str, Any],
                             recent_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect drift in fairness metrics."""
        if not self.baseline_metrics:
            return None
        
        baseline_fairness = self.baseline_metrics.get('fairness_metrics', {})
        current_fairness = current_metrics.get('fairness_metrics', {})
        
        drifts = []
        
        for attr, baseline_metrics in baseline_fairness.items():
            if attr not in current_fairness:
                continue
                
            current_attr_metrics = current_fairness[attr]
            
            for metric_name, baseline_result in baseline_metrics.items():
                if (hasattr(baseline_result, 'value') and 
                    metric_name in current_attr_metrics and
                    hasattr(current_attr_metrics[metric_name], 'value')):
                    
                    current_value = current_attr_metrics[metric_name].value
                    baseline_value = baseline_result.value
                    metric_drift = abs(current_value - baseline_value)
                    
                    if metric_drift > self.config.get('fairness_drift_threshold', 0.1):
                        drifts.append({
                            'attribute': attr,
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'drift_amount': metric_drift,
                            'severity': 'high' if metric_drift > 0.15 else 'medium'
                        })
        
        if not drifts:
            return None
        
        return {
            'type': 'fairness_drift',
            'description': f"Fairness drift detected in {len(drifts)} metrics",
            'drifts': drifts,
            'severity': max(d['severity'] for d in drifts),
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_alerts(self):
        """Process and handle alerts from the queue."""
        processed_alerts = []
        
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                self._handle_alert(alert)
                processed_alerts.append(alert)
            except Exception as e:
                logger.error(f"Failed to process alert: {e}")
        
        # Log summary
        if processed_alerts:
            logger.info(f"Processed {len(processed_alerts)} alerts")
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle an individual alert based on its type and severity."""
        alert_handlers = {
            'DRIFT_DETECTED': self._handle_drift_alert,
            'MONITORING_ERROR': self._handle_error_alert,
            'PERFORMANCE_DEGRADATION': self._handle_performance_alert
        }
        
        handler = alert_handlers.get(alert['type'], self._handle_generic_alert)
        handler(alert)
    
    def _handle_drift_alert(self, alert: Dict[str, Any]):
        """Handle drift detection alerts."""
        # Implement specific drift handling logic
        if alert['severity'] == 'critical':
            # Immediate action required
            logger.critical(f"CRITICAL DRIFT: {alert['message']}")
            # TODO: Implement automatic mitigation or escalation
        else:
            logger.warning(f"Drift detected: {alert['message']}")
        
        # Store alert for reporting
        self._store_alert(alert)
    
    def _update_baseline_metrics(self, current_metrics: Dict[str, Any]):
        """Update baseline metrics using exponential smoothing."""
        alpha = self.config.get('baseline_smoothing_factor', 0.1)
        
        if not self.baseline_metrics:
            self.baseline_metrics = current_metrics
            return
        
        # Update fairness metrics with smoothing
        for attr, current_attr_metrics in current_metrics.get('fairness_metrics', {}).items():
            if attr not in self.baseline_metrics['fairness_metrics']:
                self.baseline_metrics['fairness_metrics'][attr] = current_attr_metrics
            else:
                baseline_attr_metrics = self.baseline_metrics['fairness_metrics'][attr]
                for metric_name, current_result in current_attr_metrics.items():
                    if (hasattr(current_result, 'value') and 
                        metric_name in baseline_attr_metrics and
                        hasattr(baseline_attr_metrics[metric_name], 'value')):
                        
                        baseline_value = baseline_attr_metrics[metric_name].value
                        smoothed_value = alpha * current_result.value + (1 - alpha) * baseline_value
                        baseline_attr_metrics[metric_name].value = smoothed_value
        
        # Update other metrics with smoothing
        for metric_type in ['performance_metrics', 'statistical_metrics']:
            if metric_type in current_metrics and metric_type in self.baseline_metrics:
                for metric_name, current_value in current_metrics[metric_type].items():
                    if isinstance(current_value, (int, float)):
                        baseline_value = self.baseline_metrics[metric_type].get(metric_name, current_value)
                        smoothed_value = alpha * current_value + (1 - alpha) * baseline_value
                        self.baseline_metrics[metric_type][metric_name] = smoothed_value
    
    def stop_monitoring(self):
        """Stop the continuous monitoring system."""
        self.is_running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=30)
        
        logger.info("Continuous monitoring stopped")
    
    def get_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a monitoring report for the specified period."""
        # This would typically query a database of stored metrics and alerts
        return {
            'report_period': f"Last {days} days",
            'summary': {
                'total_alerts': 0,  # Would be populated from storage
                'critical_alerts': 0,
                'drifts_detected': 0,
                'performance_issues': 0
            },
            'recommendations': [
                "Continue current monitoring configuration",
                "Review drift detection thresholds quarterly"
            ]
        }