import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.synthetic_generator import SyntheticLendingDataGenerator
from fairness.metrics import FairnessMetricsCalculator, BiasScorer
from explainability.shap_analyzer import SHAPAnalyzer
from models.logistic_model import LogisticRegressionModel
from models.tree_model import TreeModel
from utils.logger import setup_logger

# Configure page
st.set_page_config(
    page_title="FAIR-AI Lending Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logger = setup_logger(__name__)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

class FAIRAIDashboard:
    """Main dashboard class for FAIR-AI Lending Monitor."""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.fairness_results = {}
        self.shap_analyzers = {}
    
    def render_header(self):
        """Render dashboard header."""
        st.title("🏦 FAIR-AI Lending Risk Monitor")
        st.markdown("*AI-Powered Fair Lending Compliance & Risk Assessment*")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Loaded" if st.session_state.data_loaded else "⚪ Not Loaded"
            st.metric("Data Status", status)
        
        with col2:
            status = "✅ Trained" if st.session_state.models_trained else "⚪ Not Trained"
            st.metric("Models Status", status)
        
        with col3:
            if hasattr(self, 'risk_level'):
                color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.metric("Risk Level", f"{color.get(self.risk_level, '⚪')} {self.risk_level}")
            else:
                st.metric("Risk Level", "⚪ Unknown")
        
        with col4:
            compliance = "✅ Compliant" if getattr(self, 'is_compliant', None) else "❌ Non-Compliant"
            st.metric("Regulatory Status", compliance)
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.title("🔧 Controls")
        
        # Data Generation Section
        st.sidebar.header("Data Generation")
        
        sample_size = st.sidebar.selectbox(
            "Sample Size",
            [1000, 5000, 10000, 25000, 50000],
            index=2
        )
        
        # Bias Configuration
        st.sidebar.subheader("Bias Patterns")
        bias_config = {}
        bias_config['gender_bias'] = st.sidebar.checkbox("Gender Bias", value=True)
        bias_config['race_bias'] = st.sidebar.checkbox("Race Bias", value=True)
        bias_config['age_bias'] = st.sidebar.checkbox("Age Bias", value=True)
        bias_config['geographic_bias'] = st.sidebar.checkbox("Geographic Bias", value=True)
        bias_config['intersectional_bias'] = st.sidebar.checkbox("Intersectional Bias", value=True)
        
        if st.sidebar.button("Generate Synthetic Data"):
            self.generate_data(sample_size, bias_config)
        
        # Model Training Section
        if st.session_state.data_loaded:
            st.sidebar.header("Model Training")
            
            model_types = st.sidebar.multiselect(
                "Select Models",
                ["Logistic Regression", "Random Forest"],
                default=["Logistic Regression", "Random Forest"]
            )
            
            if st.sidebar.button("Train Models"):
                self.train_models(model_types)
        
        # Quick Demo Scenarios
        st.sidebar.header("Demo Scenarios")
        if st.sidebar.button("Load Geographic Bias Case"):
            self.load_demo_scenario("geographic")
        
        if st.sidebar.button("Load Intersectional Bias Case"):
            self.load_demo_scenario("intersectional")
    
    def generate_data(self, sample_size: int, bias_config: dict):
        """Generate synthetic lending data."""
        with st.spinner("Generating synthetic lending data..."):
            try:
                generator = SyntheticLendingDataGenerator(n_samples=sample_size)
                self.data = generator.generate_dataset(bias_config)
                st.session_state.models_trained = True
                st.success(f"Successfully trained {len(model_types)} models")
                logger.info(f"Trained models: {model_types}")
                
                # Calculate fairness metrics
                self.calculate_fairness_metrics(X, y)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                logger.error(f"Model training failed: {e}")
    
    def calculate_fairness_metrics(self, X: pd.DataFrame, y: pd.Series):
        """Calculate fairness metrics for trained models."""
        try:
            calculator = FairnessMetricsCalculator()
            scorer = BiasScorer()
            
            # Calculate metrics for each protected attribute
            protected_attrs = ['gender', 'race', 'age_group']
            
            for model_name, model in self.models.items():
                model_results = {}
                
                for attr in protected_attrs:
                    # Create binary columns for the attribute
                    attr_data = pd.get_dummies(self.data[attr], prefix=attr)
                    combined_data = pd.concat([self.data, attr_data], axis=1)
                    
                    # Calculate metrics
                    metrics = calculator.calculate_all_metrics(
                        combined_data, 'approved', attr
                    )
                    model_results[attr] = metrics
                
                # Calculate overall risk score
                all_metrics = {}
                for attr_results in model_results.values():
                    all_metrics.update(attr_results)
                
                risk_assessment = scorer.calculate_risk_score(all_metrics)
                model_results['risk_assessment'] = risk_assessment
                
                self.fairness_results[model_name] = model_results
                
                # Update dashboard status
                self.risk_level = risk_assessment['severity']
                self.is_compliant = risk_assessment['regulatory_compliance']['compliant']
        
        except Exception as e:
            st.error(f"Error calculating fairness metrics: {str(e)}")
            logger.error(f"Fairness calculation failed: {e}")
    
    def load_demo_scenario(self, scenario_type: str):
        """Load predefined demo scenarios."""
        if scenario_type == "geographic":
            bias_config = {
                'geographic_bias': True,
                'race_bias': False,
                'gender_bias': False,
                'age_bias': False,
                'intersectional_bias': False
            }
        else:  # intersectional
            bias_config = {
                'geographic_bias': True,
                'race_bias': True,
                'gender_bias': True,
                'age_bias': True,
                'intersectional_bias': True
            }
        
        self.generate_data(10000, bias_config)
        self.train_models(["Logistic Regression", "Random Forest"])
    
    def render_overview_page(self):
        """Render overview page."""
        st.header("📊 Overview Dashboard")
        
        if not st.session_state.data_loaded:
            st.info("👈 Please generate data using the sidebar controls to get started")
            return
        
        # Dataset Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Applications", f"{len(self.data):,}")
        
        with col2:
            approval_rate = self.data['approved'].mean()
            st.metric("Overall Approval Rate", f"{approval_rate:.1%}")
        
        with col3:
            avg_loan_amount = self.data['loan_amount'].mean()
            st.metric("Avg Loan Amount", f"${avg_loan_amount:,.0f}")
        
        with col4:
            avg_credit_score = self.data['credit_score'].mean()
            st.metric("Avg Credit Score", f"{avg_credit_score:.0f}")
        
        # Risk Gauge
        if hasattr(self, 'risk_level'):
            st.subheader("🚨 Bias Risk Assessment")
            
            risk_score = 0.3 if self.risk_level == 'Low' else 0.6 if self.risk_level == 'Medium' else 0.9
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Bias Risk Score"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.33], 'color': "lightgreen"},
                        {'range': [0.33, 0.66], 'color': "yellow"},
                        {'range': [0.66, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Approval Rates by Demographics
        if st.session_state.data_loaded:
            st.subheader("📈 Approval Rates by Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gender approval rates
                gender_rates = self.data.groupby('gender')['approved'].mean().reset_index()
                fig = px.bar(
                    gender_rates, x='gender', y='approved',
                    title="Approval Rates by Gender",
                    labels={'approved': 'Approval Rate', 'gender': 'Gender'}
                )
                fig.update_traces(marker_color='lightblue')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Race approval rates
                race_rates = self.data.groupby('race')['approved'].mean().reset_index()
                fig = px.bar(
                    race_rates, x='race', y='approved',
                    title="Approval Rates by Race",
                    labels={'approved': 'Approval Rate', 'race': 'Race'}
                )
                fig.update_traces(marker_color='lightcoral')
                st.plotly_chart(fig, use_container_width=True)
        
        # Geographic Heatmap
        if 'region' in self.data.columns:
            st.subheader("🗺️ Geographic Analysis")
            
            region_stats = self.data.groupby('region').agg({
                'approved': 'mean',
                'application_id': 'count',
                'annual_income': 'mean'
            }).reset_index()
            
            region_stats.columns = ['Region', 'Approval_Rate', 'Applications', 'Avg_Income']
            
            fig = px.scatter(
                region_stats, x='Avg_Income', y='Approval_Rate',
                size='Applications', color='Region',
                title="Regional Analysis: Income vs Approval Rate",
                labels={
                    'Avg_Income': 'Average Income ($)',
                    'Approval_Rate': 'Approval Rate'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_fairness_analysis(self):
        """Render fairness analysis page."""
        st.header("⚖️ Fairness Analysis")
        
        if not st.session_state.models_trained:
            st.info("Please train models first to see fairness analysis")
            return
        
        # Model Selection
        model_name = st.selectbox(
            "Select Model for Analysis",
            list(self.models.keys()),
            format_func=lambda x: "Logistic Regression" if x == 'logistic' else "Random Forest"
        )
        
        if model_name not in self.fairness_results:
            st.warning("Fairness metrics not available for selected model")
            return
        
        model_results = self.fairness_results[model_name]
        
        # Risk Assessment Summary
        if 'risk_assessment' in model_results:
            risk_data = model_results['risk_assessment']
            
            st.subheader("📊 Overall Risk Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_score = risk_data['overall_risk_score']
                st.metric("Risk Score", f"{risk_score:.2f}")
            
            with col2:
                severity = risk_data['severity']
                color_map = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                st.metric("Severity", f"{color_map[severity]} {severity}")
            
            with col3:
                compliant = risk_data['regulatory_compliance']['compliant']
                status = "✅ Compliant" if compliant else "❌ Non-Compliant"
                st.metric("Regulatory Status", status)
            
            # Recommendations
            if risk_data.get('recommendations'):
                st.subheader("🎯 Recommendations")
                for i, rec in enumerate(risk_data['recommendations'][:5], 1):
                    st.write(f"{i}. {rec}")
        
        # Fairness Metrics by Protected Attribute
        st.subheader("📋 Detailed Fairness Metrics")
        
        protected_attr = st.selectbox(
            "Select Protected Attribute",
            ['gender', 'race', 'age_group']
        )
        
        if protected_attr in model_results:
            attr_results = model_results[protected_attr]
            
            # Create metrics table
            metrics_data = []
            for metric_name, result in attr_results.items():
                if hasattr(result, 'metric_name'):
                    metrics_data.append({
                        'Metric': result.metric_name,
                        'Value': f"{result.value:.3f}",
                        'Threshold': f"{result.threshold:.3f}",
                        'Status': "✅ Pass" if result.passed else "❌ Fail",
                        'Severity': result.severity,
                        'Regulation': getattr(result, 'regulation', 'N/A')
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Visualization of metrics
                fig = go.Figure()
                
                for i, row in metrics_df.iterrows():
                    color = 'red' if row['Status'] == '❌ Fail' else 'green'
                    fig.add_trace(go.Bar(
                        x=[row['Metric']],
                        y=[float(row['Value'])],
                        name=row['Metric'],
                        marker_color=color,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title=f"Fairness Metrics for {protected_attr.title()}",
                    xaxis_title="Metrics",
                    yaxis_title="Values",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_what_if_analysis(self):
        """Render what-if analysis page."""
        st.header("🔮 What-If Analysis")
        
        if not st.session_state.models_trained:
            st.info("Please train models first to run what-if analysis")
            return
        
        st.subheader("Bias Mitigation Scenarios")
        
        # Scenario Selection
        scenario = st.selectbox(
            "Select Mitigation Scenario",
            [
                "Original Model (Baseline)",
                "Remove Protected Attributes",
                "Rebalance Training Data",
                "Adjust Decision Thresholds"
            ]
        )
        
        model_name = st.selectbox(
            "Select Model",
            list(self.models.keys()),
            format_func=lambda x: "Logistic Regression" if x == 'logistic' else "Random Forest"
        )
        
        if st.button("Run What-If Analysis"):
            with st.spinner("Running analysis..."):
                results = self.run_what_if_scenario(scenario, model_name)
                
                if results:
                    st.subheader("📊 Results Comparison")
                    
                    # Before/After comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Before Mitigation**")
                        st.json(results['before'])
                    
                    with col2:
                        st.write("**After Mitigation**")
                        st.json(results['after'])
                    
                    # Improvement metrics
                    st.subheader("📈 Improvement Metrics")
                    improvement_data = []
                    
                    for metric in ['approval_rate', 'bias_score']:
                        before_val = results['before'].get(metric, 0)
                        after_val = results['after'].get(metric, 0)
                        improvement = ((after_val - before_val) / before_val * 100) if before_val != 0 else 0
                        
                        improvement_data.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Before': f"{before_val:.3f}",
                            'After': f"{after_val:.3f}",
                            'Improvement': f"{improvement:+.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(improvement_data), use_container_width=True)
    
    def run_what_if_scenario(self, scenario: str, model_name: str) -> dict:
        """Run what-if analysis scenario."""
        try:
            # Get baseline results
            baseline_metrics = self.fairness_results[model_name]['risk_assessment']
            
            # Simulate different scenarios
            if scenario == "Remove Protected Attributes":
                # Simulate removing protected attributes
                simulated_score = baseline_metrics['overall_risk_score'] * 0.7
            elif scenario == "Rebalance Training Data":
                # Simulate rebalanced training
                simulated_score = baseline_metrics['overall_risk_score'] * 0.6
            elif scenario == "Adjust Decision Thresholds":
                # Simulate threshold adjustment
                simulated_score = baseline_metrics['overall_risk_score'] * 0.8
            else:
                simulated_score = baseline_metrics['overall_risk_score']
            
            return {
                'before': {
                    'bias_score': baseline_metrics['overall_risk_score'],
                    'approval_rate': self.data['approved'].mean(),
                    'compliance': baseline_metrics['regulatory_compliance']['compliant']
                },
                'after': {
                    'bias_score': simulated_score,
                    'approval_rate': self.data['approved'].mean() * 1.02,  # Slight improvement
                    'compliance': simulated_score < 0.5
                }
            }
        
        except Exception as e:
            st.error(f"Error in what-if analysis: {str(e)}")
            return None
    
    def render_explainability(self):
        """Render explainability analysis."""
        st.header("🔍 Model Explainability")
        
        if not st.session_state.models_trained:
            st.info("Please train models first to see explainability analysis")
            return
        
        model_name = st.selectbox(
            "Select Model",
            list(self.shap_analyzers.keys()),
            format_func=lambda x: "Logistic Regression" if x == 'logistic' else "Random Forest"
        )
        
        if model_name not in self.shap_analyzers:
            st.warning("SHAP analysis not available for selected model")
            return
        
        shap_analyzer = self.shap_analyzers[model_name]
        
        # Feature Importance
        st.subheader("📊 Feature Importance")
        
        try:
            importance_df = shap_analyzer.get_feature_importance()
            
            # Top features chart
            top_features = importance_df.head(15)
            fig = px.bar(
                top_features, x='importance', y='feature',
                orientation='h',
                title="Top 15 Most Important Features",
                labels={'importance': 'Mean |SHAP Value|', 'feature': 'Features'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Summary Plot
            st.subheader("🎯 SHAP Summary Plot")
            summary_fig = shap_analyzer.create_summary_plot('bar')
            st.plotly_chart(summary_fig, use_container_width=True)
            
            # Individual Explanation
            st.subheader("🔍 Individual Prediction Explanation")
            
            instance_idx = st.number_input(
                "Select instance index for explanation",
                min_value=0,
                max_value=min(999, len(self.data)-1),
                value=0
            )
            
            explanation_text = shap_analyzer.generate_explanation_text(instance_idx)
            st.text_area("Explanation", explanation_text, height=200)
            
            # Waterfall plot
            waterfall_fig = shap_analyzer.create_waterfall_plot(instance_idx)
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in explainability analysis: {str(e)}")
    
    def render_reports(self):
        """Render reports page."""
        st.header("📄 Reports & Export")
        
        if not st.session_state.models_trained:
            st.info("Please train models first to generate reports")
            return
        
        st.subheader("📊 Compliance Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Report", "Regulatory Compliance"]
        )
        
        model_name = st.selectbox(
            "Select Model for Report",
            list(self.models.keys()),
            format_func=lambda x: "Logistic Regression" if x == 'logistic' else "Random Forest"
        )
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                report_data = self.generate_report(report_type, model_name)
                
                if report_data:
                    st.subheader(f"📋 {report_type}")
                    
                    # Display report content
                    for section, content in report_data.items():
                        st.write(f"**{section.title()}:**")
                        if isinstance(content, dict):
                            st.json(content)
                        elif isinstance(content, list):
                            for item in content:
                                st.write(f"• {item}")
                        else:
                            st.write(content)
                        st.write("---")
        
        # Export Options
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Data (CSV)"):
                csv = self.data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"lending_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Metrics (JSON)"):
                if self.fairness_results:
                    import json
                    metrics_json = json.dumps(self.fairness_results, indent=2, default=str)
                    st.download_button(
                        label="Download Metrics",
                        data=metrics_json,
                        file_name=f"fairness_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("Export Report (PDF)"):
                st.info("PDF export functionality would be implemented here")
    
    def generate_report(self, report_type: str, model_name: str) -> dict:
        """Generate different types of reports."""
        if model_name not in self.fairness_results:
            return None
        
        results = self.fairness_results[model_name]
        
        if report_type == "Executive Summary":
            return {
                "overview": f"Analysis of {model_name} model fairness",
                "risk_level": results['risk_assessment']['severity'],
                "compliance_status": results['risk_assessment']['regulatory_compliance']['compliant'],
                "key_recommendations": results['risk_assessment']['recommendations'][:3]
            }
        
        elif report_type == "Technical Report":
            return {
                "model_details": f"{model_name} model analysis",
                "dataset_size": len(self.data),
                "feature_count": len(self.data.columns),
                "fairness_metrics": {
                    attr: {metric: result.value for metric, result in metrics.items() if hasattr(result, 'value')}
                    for attr, metrics in results.items() if attr != 'risk_assessment'
                }
            }
        
        else:  # Regulatory Compliance
            violations = results['risk_assessment']['regulatory_compliance'].get('violations', [])
            return {
                "ecoa_compliance": results['risk_assessment']['regulatory_compliance']['compliance_status'].get('ECOA', True),
                "fair_housing_compliance": results['risk_assessment']['regulatory_compliance']['compliance_status'].get('Fair_Housing_Act', True),
                "violations_found": violations,
                "remediation_required": len(violations) > 0
            }
    
    def run(self):
        """Run the main dashboard."""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "⚖️ Fairness Analysis", "🔮 What-If Analysis", 
            "🔍 Explainability", "📄 Reports"
        ])
        
        with tab1:
            self.render_overview_page()
        
        with tab2:
            self.render_fairness_analysis()
        
        with tab3:
            self.render_what_if_analysis()
        
        with tab4:
            self.render_explainability()
        
        with tab5:
            self.render_reports()

# Main execution
def main():
    """Main function to run the dashboard."""
    dashboard = FAIRAIDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()data_loaded = True
                st.success(f"Generated {len(self.data)} lending records with configured bias patterns")
                logger.info(f"Generated {len(self.data)} synthetic records")
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
                logger.error(f"Data generation failed: {e}")
    
    def train_models(self, model_types: list):
        """Train selected models."""
        if self.data is None:
            st.error("Please generate data first")
            return
        
        with st.spinner("Training models..."):
            try:
                # Prepare features
                feature_cols = [
                    'annual_income', 'credit_score', 'employment_years',
                    'debt_to_income', 'existing_loans', 'previous_defaults',
                    'loan_amount', 'loan_term_months', 'age'
                ]
                
                categorical_cols = ['race', 'gender', 'age_group', 'region', 'loan_type', 'urban_rural']
                
                # Create feature matrix
                X = self.data[feature_cols + categorical_cols].copy()
                X = pd.get_dummies(X, columns=categorical_cols)
                y = self.data['approved']
                
                # Train selected models
                for model_type in model_types:
                    if model_type == "Logistic Regression":
                        model = LogisticRegressionModel()
                        model.train(X, y)
                        self.models['logistic'] = model
                        
                        # Create SHAP analyzer
                        shap_analyzer = SHAPAnalyzer(model.model, 'linear')
                        shap_analyzer.create_explainer(X)
                        shap_analyzer.calculate_shap_values(X, max_samples=1000)
                        self.shap_analyzers['logistic'] = shap_analyzer
                    
                    elif model_type == "Random Forest":
                        model = TreeModel()
                        model.train(X, y)
                        self.models['tree'] = model
                        
                        # Create SHAP analyzer
                        shap_analyzer = SHAPAnalyzer(model.model, 'tree')
                        shap_analyzer.create_explainer(X)
                        shap_analyzer.calculate_shap_values(X, max_samples=1000)
                        self.shap_analyzers['tree'] = shap_analyzer
                
                st.session_state.models_trained = True
                st.success(f"Successfully trained {len(model_types)} models")
                logger.info(f"Trained models: {model_types}")
                
                # Calculate fairness metrics
                self.calculate_fairness_metrics(X, y)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                logger.error(f"Model training failed: {e}")
                