# src/dashboard/pages/advanced_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...fairness.mitigation import AdvancedBiasMitigator
from ...explainability.nlg_generator import AdvancedNLGGenerator
from ...utils.monitoring import ContinuousMonitoringSystem

def show_advanced_analysis():
    """Advanced analysis page with mitigation and monitoring capabilities."""
    st.title("🔄 Advanced Analysis & Mitigation")
    
    if 'data' not in st.session_state or 'models' not in st.session_state:
        st.warning("Please load data and train models first")
        return
    
    data = st.session_state.data
    models = st.session_state.models
    
    # Create tabs for different advanced functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "Policy Simulation", 
        "Counterfactual Analysis", 
        "Bias Mitigation", 
        "Continuous Monitoring"
    ])
    
    with tab1:
        _show_policy_simulation(data, models)
    
    with tab2:
        _show_counterfactual_analysis(data, models)
    
    with tab3:
        _show_bias_mitigation(data, models)
    
    with tab4:
        _show_continuous_monitoring(data, models)

def _show_policy_simulation(data: pd.DataFrame, models: Dict[str, Any]):
    """Show policy simulation interface."""
    st.header("📊 Policy Impact Simulation")
    
    selected_model = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model]
    
    # Policy rule configuration
    st.subheader("Configure Policy Rules")
    
    policy_rules = {}
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for feature in numerical_features:
        if feature not in ['approved', 'predicted']:
            col1, col2, col3 = st.columns(3)
            with col1:
                enabled = st.checkbox(f"Modify {feature}", key=f"policy_{feature}")
            if enabled:
                with col2:
                    direction = st.selectbox("Direction", ["increase", "decrease"], key=f"dir_{feature}")
                with col3:
                    threshold = st.number_input("Threshold", value=data[feature].median(), key=f"thresh_{feature}")
                
                policy_rules[feature] = {
                    'type': 'threshold',
                    'value': threshold,
                    'direction': direction
                }
    
    if st.button("Simulate Policy Impact") and policy_rules:
        mitigator = AdvancedBiasMitigator(model)
        impact = mitigator.simulate_policy_change(data, policy_rules)
        
        # Display results
        st.subheader("Simulation Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline Approval Rate", f"{impact['overall_impact']['baseline_approval_rate']:.1%}")
        with col2:
            st.metric("New Approval Rate", f"{impact['overall_impact']['modified_approval_rate']:.1%}")
        
        st.metric("Change", f"{impact['overall_impact']['percent_change']:+.1f}%")
        
        # Show impact by group
        st.subheader("Impact by Demographic Groups")
        
        for attr, group_impact in impact['group_impacts'].items():
            st.write(f"**{attr.upper()}**")
            impact_df = pd.DataFrame(group_impact).T
            st.dataframe(impact_df.style.format({
                'baseline_rate': '{:.1%}',
                'modified_rate': '{:.1%}',
                'change': '{:+.3f}',
                'percent_change': '{:+.1f}%'
            }))

def _show_counterfactual_analysis(data: pd.DataFrame, models: Dict[str, Any]):
    """Show counterfactual analysis interface."""
    st.header("🔍 Counterfactual Analysis")
    
    selected_model = st.selectbox("Select Model", list(models.keys()), key="cf_model")
    model = models[selected_model]
    
    # Instance selection
    instance_id = st.selectbox("Select Application", data['application_id'].unique())
    instance = data[data['application_id'] == instance_id].iloc[0]
    
    st.subheader("Selected Application")
    st.dataframe(instance)
    
    # Original prediction
    original_pred = model.predict(instance.drop('approved').values.reshape(1, -1))[0]
    original_proba = model.predict_proba(instance.drop('approved').values.reshape(1, -1))[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Decision", "Approved" if original_pred == 1 else "Rejected")
    with col2:
        st.metric("Confidence", f"{original_proba:.1%}")
    
    # Generate counterfactuals
    if st.button("Generate Counterfactuals"):
        mitigator = AdvancedBiasMitigator(model)
        counterfactuals = mitigator.generate_counterfactuals(
            instance.drop('approved'), data.drop('approved', axis=1),
            target_class=1 if original_pred == 0 else 0  # Flip the decision
        )
        
        if not counterfactuals.empty:
            st.subheader("Generated Counterfactuals")
            
            for _, cf in counterfactuals.iterrows():
                # Predict for counterfactual
                cf_pred = model.predict(cf.drop(['counterfactual_confidence', 'proximity_score', 'overall_score']).values.reshape(1, -1))[0]
                cf_proba = model.predict_proba(cf.drop(['counterfactual_confidence', 'proximity_score', 'overall_score']).values.reshape(1, -1))[0][1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Decision", "Approved" if cf_pred == 1 else "Rejected")
                with col2:
                    st.metric("Confidence", f"{cf_proba:.1%}")
                with col3:
                    st.metric("Proximity Score", f"{cf['proximity_score']:.3f}")
                
                # Show changes
                changes = cf.drop(['counterfactual_confidence', 'proximity_score', 'overall_score']) - instance.drop('approved')
                significant_changes = changes[abs(changes) > 0.1]
                
                if not significant_changes.empty:
                    st.write("**Significant Changes:**")
                    for feature, change in significant_changes.items():
                        st.write(f"- {feature}: {instance[feature]} → {cf[feature]} ({change:+.2f})")
                
                st.markdown("---")
        else:
            st.warning("No counterfactuals found that change the decision")

def _show_bias_mitigation(data: pd.DataFrame, models: Dict[str, Any]):
    """Show bias mitigation interface."""
    st.header("⚖️ Bias Mitigation Techniques")
    
    selected_model = st.selectbox("Select Model", list(models.keys()), key="mitigation_model")
    model = models[selected_model]
    
    mitigation_technique = st.selectbox(
        "Select Mitigation Technique",
        ["Adversarial Debiasing", "Reweighting", "Preprocessing", "Postprocessing"]
    )
    
    protected_attribute = st.selectbox(
        "Protected Attribute",
        [attr for attr in ['race', 'gender', 'age_group'] if attr in data.columns]
    )
    
    if st.button("Apply Mitigation"):
        with st.spinner("Applying mitigation technique..."):
            mitigator = AdvancedBiasMitigator(model, [protected_attribute])
            
            if mitigation_technique == "Adversarial Debiasing":
                # Train adversarial model
                mitigated_model = mitigator.adversarial_debias(
                    data.drop('approved', axis=1), data['approved'], protected_attribute
                )
                
                # Store mitigated model
                st.session_state.mitigated_models = st.session_state.get('mitigated_models', {})
                st.session_state.mitigated_models[f"{selected_model}_mitigated"] = mitigated_model
                
                st.success("Adversarial debiasing completed")
            
            # Calculate improvement
            from ...fairness.metrics import FairnessMetricsCalculator
            calculator = FairnessMetricsCalculator()
            
            # Baseline metrics
            baseline_metrics = calculator.calculate_all_metrics(
                data.assign(predicted=model.predict(data.drop('approved', axis=1))),
                'predicted', protected_attribute
            )
            
            # Mitigated metrics
            mitigated_preds = mitigated_model.predict(data.drop('approved', axis=1).values.astype(np.float32))
            mitigated_metrics = calculator.calculate_all_metrics(
                data.assign(predicted=mitigated_preds),
                'predicted', protected_attribute
            )
            
            # Show improvement
            st.subheader("Mitigation Results")
            
            for metric_name, baseline_result in baseline_metrics.items():
                if metric_name in mitigated_metrics:
                    mitigated_result = mitigated_metrics[metric_name]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Baseline {metric_name}", f"{baseline_result.value:.3f}")
                    with col2:
                        st.metric(f"Mitigated {metric_name}", f"{mitigated_result.value:.3f}")
                    with col3:
                        improvement = baseline_result.value - mitigated_result.value
                        st.metric("Improvement", f"{improvement:+.3f}")

def _show_continuous_monitoring(data: pd.DataFrame, models: Dict[str, Any]):
    """Show continuous monitoring interface."""
    st.header("📈 Continuous Monitoring")
    
    if 'monitoring_system' not in st.session_state:
        st.session_state.monitoring_system = ContinuousMonitoringSystem()
    
    monitoring_system = st.session_state.monitoring_system
    
    st.subheader("Monitoring Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        interval = st.number_input("Monitoring Interval (minutes)", min_value=1, value=60)
    with col2:
        selected_model = st.selectbox("Model to Monitor", list(models.keys()))
    
    protected_attributes = st.multiselect(
        "Protected Attributes to Monitor",
        ['race', 'gender', 'age_group'],
        default=['race', 'gender']
    )
    
    if st.button("Start Monitoring"):
        # Mock data source for demonstration
        def data_source():
            return data.sample(100, random_state=np.random.randint(1000))
        
        monitoring_system.start_monitoring(
            data_source, models[selected_model], protected_attributes, interval
        )
        st.success("Monitoring started")
    
    if st.button("Stop Monitoring"):
        monitoring_system.stop_monitoring()
        st.success("Monitoring stopped")
    
    # Show monitoring status
    st.subheader("Monitoring Status")
    
    if monitoring_system.is_running:
        st.success("✅ Monitoring active")
        
        # Show recent alerts
        if not monitoring_system.alert_queue.empty():
            st.subheader("Recent Alerts")
            
            alerts = []
            while not monitoring_system.alert_queue.empty():
                alerts.append(monitoring_system.alert_queue.get_nowait())
            
            for alert in alerts:
                if alert['severity'] == 'critical':
                    st.error(f"🚨 {alert['message']}")
                elif alert['severity'] == 'high':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.info("No recent alerts")
    else:
        st.info("Monitoring not active")
    
    # Generate monitoring report
    if st.button("Generate Monitoring Report"):
        report = monitoring_system.get_monitoring_report(days=7)
        st.subheader("Monitoring Report (Last 7 Days)")
        st.json(report)