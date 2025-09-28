# frontend/streamlit_app.py
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Configuration
st.set_page_config(
    page_title="Marketing-Finance AI Platform",
    page_icon="🎯",
    layout="wide"
)

API_URL = "http://localhost:8000/api"

# Custom CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🎯 Marketing-Finance AI Platform")
st.markdown("### Predict Marketing ROI & Stock Impact from Agency Decisions")

# Get companies list
@st.cache_data
def get_companies():
    try:
        response = requests.get(f"{API_URL}/companies", timeout=5)
        if response.status_code == 200:
            return response.json()['companies']
        else:
            return ['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla']  # Fallback
    except:
        return ['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla']  # Fallback

companies = get_companies()

# Sidebar
with st.sidebar:
    st.header("Company Selection")
    
    if companies:
        selected_company = st.selectbox(
            "Select Company:",
            companies,
            index=0
        )
        st.session_state.selected_company = selected_company
        
        # Get company data
        try:
            company_data = requests.get(f"{API_URL}/company/{selected_company}", timeout=5).json()
            
            st.markdown("### Current Status")
            st.metric("Stock Price", f"${company_data.get('current_price', 150.23):.2f}",
                     f"{company_data.get('yearly_change', 12.5):.1f}%")
            st.metric("Current Agency", company_data.get('current_agency', 'WPP'))
            st.metric("Marketing ROI", f"{company_data.get('current_roi', 2.34):.2f}x")
            st.metric("Market Share", f"{company_data.get('market_share', 23.4):.1f}%")
        except:
            st.error("Could not load company data")
    else:
        st.error("No companies available")
        selected_company = None

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "🤖 AI Agent", "📈 Compare Agencies"])

with tab1:
    st.header("📊 Executive Dashboard")
    
    if selected_company:
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current ROI", "2.34x", "+12%")
        with col2:
            st.metric("Marketing Efficiency", "87%", "+5%")
        with col3:
            st.metric("Digital Ratio", "65%", "+8%")
        with col4:
            st.metric("Market Share", "23.4%", "-0.5%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample stock chart
            dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
            prices = np.random.uniform(100, 200, 12)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='Stock Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"{selected_company} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample ROI chart
            roi_data = np.random.uniform(1.5, 3.5, 12)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=roi_data,
                mode='lines+markers',
                name='Marketing ROI',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title="Marketing ROI Trend",
                xaxis_title="Date",
                yaxis_title="ROI",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance Table
        st.subheader("📋 Industry Comparison")
        
        performance_data = pd.DataFrame({
            'Company': [selected_company, 'Competitor A', 'Competitor B', 'Industry Avg'],
            'ROI': [2.34, 2.12, 1.98, 2.15],
            'Revenue Growth': ['+8.5%', '+6.2%', '+4.1%', '+6.3%'],
            'Market Share': ['23.4%', '21.2%', '8.5%', '17.8%'],
            'Current Agency': ['WPP', 'Publicis', 'Omnicom', 'Various']
        })
        
        st.dataframe(
            performance_data,
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    st.header("🔮 Agency Switch Predictions")
    
    if selected_company:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Scenario Configuration")
            
            agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas']
            selected_agency = st.selectbox("Select New Agency:", agencies)
            
            timeframe = st.slider("Prediction Timeframe (months):", 6, 60, 36)
            
            if st.button("Generate Prediction", type="primary"):
                with st.spinner("Running AI prediction model..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={
                                "company": selected_company,
                                "agency": selected_agency,
                                "timeframe": timeframe
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            prediction = response.json()
                            st.session_state.prediction = prediction
                            
                            st.success("✅ Prediction Complete!")
                            
                            # Display metrics
                            col1_metric, col2_metric, col3_metric = st.columns(3)
                            
                            with col1_metric:
                                st.metric(
                                    "Predicted Stock Impact",
                                    f"{prediction['predicted_impact']:.1f}%"
                                )
                            
                            with col2_metric:
                                ci_range = f"{prediction['confidence_interval'][0]:.1f}% - {prediction['confidence_interval'][1]:.1f}%"
                                st.metric("Confidence Interval", ci_range)
                            
                            with col3_metric:
                                st.metric("Recommendation", prediction['recommendation'])
                        else:
                            st.error("Prediction failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            st.subheader("Impact Projection")
            
            # Show projection chart if prediction exists
            if hasattr(st.session_state, 'prediction'):
                prediction = st.session_state.prediction
                
                fig = go.Figure()
                
                # Add projection
                fig.add_trace(go.Scatter(
                    x=prediction['projection']['months'],
                    y=prediction['projection']['values'],
                    mode='lines',
                    name=f'Projected with {selected_agency}',
                    line=dict(color='green', width=3)
                ))
                
                # Add baseline (current performance)
                baseline = [100] * len(prediction['projection']['months'])
                fig.add_trace(go.Scatter(
                    x=prediction['projection']['months'],
                    y=baseline,
                    mode='lines',
                    name='Current Performance',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Performance Projection: {selected_company} with {selected_agency}",
                    xaxis_title="Months",
                    yaxis_title="Performance Index (100 = baseline)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Generate a prediction to see the projection chart")

with tab3:
    st.header("🤖 AI Marketing Advisor")
    
    if selected_company:
        st.markdown(f"### Chat with AI Agent about {selected_company}")
        st.info(f"Ask me anything about {selected_company}'s marketing strategy and agency recommendations!")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")
        
        # Input area
        user_input = st.text_input(
            "Your question:",
            placeholder=f"E.g., If {selected_company} switches to Publicis, what would happen to their stock price?"
        )
        
        if st.button("Send", type="primary"):
            if user_input:
                # Add to history
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
                # Get AI response
                with st.spinner("AI is analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/chat",
                            json={
                                "message": user_input,
                                "company": selected_company
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            ai_response = response.json()
                            response_text = ai_response.get('narrative',
                                f"Based on my analysis of {selected_company}, switching to a different agency could impact performance by 5-15% depending on the agency's expertise in your industry. Would you like me to analyze a specific agency comparison?")
                        else:
                            response_text = "I'm experiencing technical difficulties. Please try again."
                        
                        st.session_state.chat_history.append({
                            'role': 'ai',
                            'content': response_text
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab4:
    st.header("📈 Agency Comparison Analysis")
    
    if selected_company:
        if st.button("Compare All Agencies", type="primary"):
            with st.spinner("Analyzing all agency options..."):
                try:
                    response = requests.post(
                        f"{API_URL}/compare-agencies?company={selected_company}",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        comparison = response.json()
                        st.session_state.comparison = comparison
                        
                        st.success(f"✅ Best Choice: **{comparison['best_choice']}**")
                    else:
                        st.error("Comparison failed")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show comparison results
        if hasattr(st.session_state, 'comparison'):
            comparison = st.session_state.comparison
            
            # Create comparison table
            df = pd.DataFrame(comparison['comparisons'])
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df['agency'],
                    y=df['predicted_roi'],
                    marker_color=['green' if i == 0 else 'blue' for i in range(len(df))],
                    text=df['predicted_roi'].round(1),
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Predicted ROI by Agency",
                    xaxis_title="Agency",
                    yaxis_title="ROI",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['risk_score'],
                    y=df['predicted_roi'],
                    mode='markers+text',
                    text=df['agency'],
                    textposition='top center',
                    marker=dict(size=15, color=df['confidence'], colorscale='Viridis')
                ))
                fig.update_layout(
                    title="Risk vs Return Analysis",
                    xaxis_title="Risk Score",
                    yaxis_title="Predicted ROI",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Comparison")
            st.dataframe(df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("**Marketing-Finance AI Platform** | Powered by Advanced ML & Causal Inference")
