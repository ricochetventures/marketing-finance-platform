import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Initialize session state properly
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

st.set_page_config(page_title="Marketing-Finance Platform", layout="wide")

st.title("🎯 Marketing-Finance AI Platform")

# Test API connection
api_status = "❌ API not running"
try:
    response = requests.get("http://localhost:8000")
    if response.status_code == 200:
        api_status = "✅ API Connected"
except:
    pass

if api_status == "✅ API Connected":
    st.success(api_status)
else:
    st.error(f"{api_status}. Start it with: python test_api.py")

# Simple demo interface
st.subheader("Company Analysis")

companies = ["Coca-Cola", "PepsiCo", "Nike", "Apple", "Microsoft"]
selected_company = st.selectbox("Select Company:", companies)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current ROI", "2.34x", "+12%")
with col2:
    st.metric("Marketing Efficiency", "87%", "+5%")
with col3:
    st.metric("Market Share", "23.4%", "-0.5%")

# Sample chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    y=[2.1, 2.3, 2.2, 2.4, 2.5],
    mode='lines+markers',
    name='ROI Trend'
))
fig.update_layout(title="ROI Performance", height=400)
st.plotly_chart(fig, use_container_width=True)
