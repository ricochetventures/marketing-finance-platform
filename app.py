import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Marketing Finance Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Marketing Finance Platform")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Stock Analysis", "Data Upload"])

if page == "Dashboard":
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", "$1.2M", "+12%")
    with col2:
        st.metric("Active Users", "8,429", "+3%")
    with col3:
        st.metric("Conversion Rate", "3.2%", "+0.5%")
    with col4:
        st.metric("ROI", "142%", "+23%")
    
    # Sample chart
    dates = pd.date_range(start='2024-01-01', periods=30)
    df = pd.DataFrame({
        'Date': dates,
        'Revenue': np.random.randn(30).cumsum() + 100,
        'Users': np.random.randn(30).cumsum() + 50
    })
    
    fig = px.line(df, x='Date', y=['Revenue', 'Users'], 
                  title="Platform Metrics Over Time")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Stock Analysis":
    st.header("Stock Analysis")
    
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
    period = st.selectbox("Select Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
    
    if st.button("Fetch Data"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if not hist.empty:
                st.subheader(f"{ticker} Stock Data")
                
                # Price chart
                fig = px.line(hist, y='Close', title=f"{ticker} Stock Price")
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_vol = px.bar(hist, y='Volume', title=f"{ticker} Trading Volume")
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${hist['Close'][-1]:.2f}")
                with col2:
                    st.metric("High", f"${hist['High'].max():.2f}")
                with col3:
                    st.metric("Low", f"${hist['Low'].min():.2f}")
                
                # Raw data
                with st.expander("View Raw Data"):
                    st.dataframe(hist)
            else:
                st.error(f"No data found for {ticker}")
                
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Data Upload":
    st.header("Data Upload & Analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        # Column selector for visualization
        if len(df.columns) > 1:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis:", df.columns)
            with col2:
                y_axis = st.selectbox("Select Y-axis:", df.columns)
            
            chart_type = st.radio("Chart Type:", ["Line", "Scatter", "Bar"])
            
            if chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis)
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            else:
                fig = px.bar(df, x=x_axis, y=y_axis)
            
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Marketing Finance Platform v1.0 | All systems operational ✅")
