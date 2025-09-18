import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .ai-message {
        background-color: #f5f5f5;
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
    response = requests.get(f"{API_URL}/companies")
    return response.json()['companies']

companies = get_companies()

# Sidebar
with st.sidebar:
    st.header("Company Selection")
    
    selected_company = st.selectbox(
        "Select Company:",
        companies,
        index=0 if not st.session_state.selected_company else companies.index(st.session_state.selected_company)
    )
    
    st.session_state.selected_company = selected_company
    
    # Get company data
    if selected_company:
        company_data = requests.get(f"{API_URL}/company/{selected_company}").json()
        
        st.markdown("### Current Status")
        st.metric("Stock Price", f"${company_data.get('current_price', 0):.2f}",
                 f"{company_data.get('yearly_change', 0):.1f}%")
        st.metric("Current Agency", company_data.get('current_agency', 'Unknown'))
        st.metric("Marketing ROI", f"{company_data.get('current_roi', 2.34):.2f}x")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "🤖 AI Agent", "📈 Compare Agencies"])

with tab1:
    if selected_company and company_data:
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current ROI", f"{company_data.get('current_roi', 2.34):.2f}x", "+12%")
        with col2:
            st.metric("Marketing Efficiency", f"{company_data.get('marketing_efficiency', 87)}%", "+5%")
        with col3:
            st.metric("Digital Ratio", f"{company_data.get('digital_ratio', 65):.0f}%", "+8%")
        with col4:
            st.metric("Market Share", f"{company_data.get('market_share', 23.4):.1f}%", "-0.5%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'stock_history' in company_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(company_data['stock_history']['dates']),
                    y=company_data['stock_history']['prices'],
                    mode='lines',
                    name='Stock Price',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title="Stock Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'roi_history' in company_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(company_data['roi_history']['dates']),
                    y=company_data['roi_history']['values'],
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

with tab2:
    st.header("🔮 Agency Switch Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Configuration")
        
        agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas']
        selected_agency = st.selectbox("Select New Agency:", agencies)
        
        timeframe = st.slider("Prediction Timeframe (months):", 6, 60, 36)
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Running AI prediction model..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "company": selected_company,
                        "agency": selected_agency,
                        "timeframe": timeframe
                    }
                )
                
                if response.status_code == 200:
                    prediction = response.json()
                    
                    st.success("✅ Prediction Complete!")
                    
                    # Display metrics
                    st.metric(
                        "Predicted Stock Impact",
                        f"{prediction['predicted_impact']:.1f}%",
                        f"CI: {prediction['confidence_interval'][0]:.1f}% - {prediction['confidence_interval'][1]:.1f}%"
                    )
                    
                    st.info(f"**Recommendation**: {prediction['recommendation']}")
    
    with col2:
        st.subheader("Impact Projection")
        
        # Show projection chart if prediction exists
        if 'prediction' in locals():
            fig = go.Figure()
            
            # Add projection
            fig.add_trace(go.Scatter(
                x=prediction['projection']['months'],
                y=prediction['projection']['values'],
                mode='lines',
                name='Projected Performance',
                line=dict(color='green', width=3)
            ))
            
            # Add confidence bands
            upper = [v * 1.1 for v in prediction['projection']['values']]
            lower = [v * 0.9 for v in prediction['projection']['values']]
            
            fig.add_trace(go.Scatter(
                x=prediction['projection']['months'] + prediction['projection']['months'][::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"Projected Performance with {selected_agency}",
                xaxis_title="Months",
                yaxis_title="Performance Index",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🤖 AI Marketing Advisor")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        st.markdown("### Chat with AI Agent")
        st.info(f"Ask me anything about {selected_company}'s marketing strategy!")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-message user-message'><b>You:</b> {msg['content']}</div>",
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message ai-message'><b>AI:</b> {msg['content']}</div>",
                           unsafe_allow_html=True)
        
        # Input area
        user_input = st.text_input("Your question:", key="chat_input",
                                   placeholder=f"E.g., If {selected_company} switches to Publicis, what would happen?")
        
        if st.button("Send", key="send_button"):
            if user_input:
                # Add to history
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
                # Get AI response
                with st.spinner("AI is thinking..."):
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": user_input,
                            "company": selected_company
                        }
                    )
                    
                    if response.status_code == 200:
                        ai_response = response.json()
                        st.session_state.chat_history.append({
                            'role': 'ai',
                            'content': ai_response.get('narrative', 'I need more information to answer that.')
                        })
                        st.rerun()

with tab4:
    st.header("📈 Agency Comparison Analysis")
    
    if st.button("Compare All Agencies"):
        with st.spinner("Analyzing all agency options..."):
            response = requests.post(
                f"{API_URL}/compare-agencies",
                json={"company": selected_company}
            )
            
            if response.status_code == 200:
                comparison = response.json()
                
                st.success(f"✅ Best Choice: **{comparison['best_choice']}**")
                
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
                        yaxis_title="ROI (%)",
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
                        yaxis_title="Predicted ROI (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Comparison")
                st.dataframe(
                    df.style.highlight_max(subset=['predicted_roi', 'stock_impact', 'confidence'])
                           .highlight_min(subset=['risk_score']),
                    use_container_width=True
                )

# Run the app
if __name__ == "__main__":
    st.sidebar.success("API Connected ✅" if requests.get(f"{API_URL}/companies").status_code == 200 else "API Disconnected ❌")
