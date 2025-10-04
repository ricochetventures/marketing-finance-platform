# /frontend/templates/app.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Marketing-Finance AI Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

# Sidebar
with st.sidebar:
    st.image("logo.png", width=200)
    st.title("🎯 Marketing-Finance AI")
    
    # Navigation
    page = st.selectbox(
        "Navigate to:",
        ["Dashboard", "Predict & Simulate", "Company Analysis", 
         "Marketing Optimization", "Agency Comparison", "Industry Insights"]
    )
    
    st.markdown("---")
    
    # Company selector
    companies = requests.get(f"{API_URL}/companies").json()["companies"]
    selected_company = st.selectbox("Select Company:", companies)
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🔄 Refresh Data"):
        st.rerun()
    if st.button("📥 Download Report"):
        generate_report(selected_company)

# Main content based on page selection
if page == "Dashboard":
    st.title("📊 Executive Dashboard")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current ROI",
            value="2.34x",
            delta="+12%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Marketing Efficiency",
            value="87%",
            delta="+5%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Stock Performance",
            value="+18.5%",
            delta="+2.3%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Market Share",
            value="23.4%",
            delta="-0.5%",
            delta_color="inverse"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI Trend Chart
        st.subheader("📈 ROI Trend Analysis")
        
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        roi_data = pd.DataFrame({
            'Date': dates,
            'Actual ROI': np.random.uniform(1.8, 2.5, 12),
            'Predicted ROI': np.random.uniform(2.0, 2.7, 12)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roi_data['Date'],
            y=roi_data['Actual ROI'],
            mode='lines+markers',
            name='Actual ROI',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=roi_data['Date'],
            y=roi_data['Predicted ROI'],
            mode='lines+markers',
            name='Predicted ROI',
            line=dict(color='green', width=2, dash='dash')
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Marketing Mix Pie Chart
        st.subheader("📊 Marketing Mix Allocation")
        
        mix_data = pd.DataFrame({
            'Channel': ['Digital', 'TV', 'Print', 'Radio', 'OOH'],
            'Spend': [45, 25, 15, 10, 5]
        })
        
        fig = px.pie(mix_data, values='Spend', names='Channel',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance Table
    st.subheader("📋 Company Performance Comparison")
    
    performance_data = pd.DataFrame({
        'Company': ['Coca-Cola', 'PepsiCo', 'Dr Pepper', 'Monster'],
        'ROI': [2.34, 2.12, 1.98, 2.45],
        'Revenue Growth': ['+8.5%', '+6.2%', '+4.1%', '+12.3%'],
        'Market Share': ['23.4%', '21.2%', '8.5%', '6.8%'],
        'Agency': ['WPP', 'BBDO', 'Publicis', 'Omnicom']
    })
    
    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ROI": st.column_config.NumberColumn(format="%.2f"),
        }
    )

elif page == "Predict & Simulate":
    st.title("🔮 Predictive Analytics & Simulation")
    
    # Scenario Configuration
    st.subheader("Configure Scenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agency selection
        agencies = requests.get(f"{API_URL}/agencies").json()["agencies"]
        new_agency = st.selectbox(
            "Select New Agency:",
            ["Keep Current"] + agencies
        )
        
        # Marketing spend
        current_spend = 1000000  # Get from API
        new_spend = st.slider(
            "Marketing Spend ($M):",
            min_value=0.5,
            max_value=10.0,
            value=current_spend/1000000,
            step=0.1
        )
        
        # Time horizon
        time_horizon = st.slider(
            "Prediction Horizon (months):",
            min_value=3,
            max_value=24,
            value=12
        )
    
    with col2:
        # Additional parameters
        st.markdown("### Simulation Parameters")
        num_simulations = st.number_input(
            "Number of Simulations:",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        confidence_level = st.select_slider(
            "Confidence Level:",
            options=[90, 95, 99],
            value=95
        )
        
        include_competition = st.checkbox("Include Competitive Response", value=True)
        include_economic = st.checkbox("Include Economic Factors", value=True)
    
    # Run Prediction Button
    if st.button("🚀 Run Prediction & Simulation", type="primary"):
        with st.spinner("Running AI predictions and simulations..."):
            
            # Call API
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "company": selected_company,
                    "new_agency": new_agency if new_agency != "Keep Current" else None,
                    "marketing_spend": new_spend * 1000000,
                    "time_horizon": time_horizon,
                    "include_simulation": True
                }
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Display Results
                st.success("✅ Prediction Complete!")
                
                # Key Metrics
                st.subheader("📊 Predicted Outcomes")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Expected ROI",
                        f"{results['simulation']['expected_roi']:.2f}x",
                        f"CI: {results['simulation']['roi_confidence'][0]:.2f} - {results['simulation']['roi_confidence'][1]:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Revenue Impact",
                        f"${results['simulation']['expected_revenue']/1000000:.1f}M",
                        f"±${(results['simulation']['revenue_confidence'][1] - results['simulation']['revenue_confidence'][0])/2000000:.1f}M"
                    )
                
                with col3:
                    st.metric(
                        "Stock Price Impact",
                        f"{results['simulation']['expected_stock']:.1%}",
                        f"±{(results['simulation']['stock_confidence'][1] - results['simulation']['stock_confidence'][0])/2:.1%}"
                    )
                
                # Visualization of Simulation Results
                st.subheader("📈 Simulation Results")
                
                # Create simulation visualization
                months = list(range(1, time_horizon + 1))
                
                # Generate sample paths for visualization
                fig = go.Figure()
                
                # Add confidence bands
                fig.add_trace(go.Scatter(
                    x=months + months[::-1],
                    y=[r * 1.1 for r in range(1, time_horizon + 1)] + 
                      [r * 0.9 for r in range(time_horizon, 0, -1)],
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                
                # Add expected path
                fig.add_trace(go.Scatter(
                    x=months,
                    y=list(range(1, time_horizon + 1)),
                    mode='lines',
                    name='Expected ROI',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title="ROI Projection with Confidence Intervals",
                    xaxis_title="Months",
                    yaxis_title="ROI",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 AI Recommendations")
                for rec in results['recommendations']:
                    st.info(f"• {rec}")

elif page == "Company Analysis":
    st.title("🔍 Company Deep Dive Analysis")
    
    # Analysis Options
    col1, col2, col3 = st.columns(3)
    with col1:
        compare_industry = st.checkbox("Compare to Industry", value=True)
    with col2:
        include_causality = st.checkbox("Include Causal Analysis", value=True)
    with col3:
        include_forecast = st.checkbox("Include Forecast", value=True)
    
    if st.button("🔍 Analyze Company"):
        with st.spinner("Analyzing company performance..."):
            
            # Call API
            response = requests.post(
                f"{API_URL}/analyze",
                json={
                    "company": selected_company,
                    "compare_to_industry": compare_industry,
                    "include_recommendations": True
                }
            )
            
            if response.status_code == 200:
                analysis = response.json()
                
                # Current Performance
                st.subheader("📊 Current Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ROI", f"{analysis['current_performance']['roi']:.2f}x")
                with col2:
                    st.metric("Revenue Growth", f"{analysis['current_performance']['revenue_growth']:.1%}")
                with col3:
                    st.metric("Market Share", f"{analysis['current_performance']['market_share']:.1%}")
                with col4:
                    st.metric("Stock Performance", f"{analysis['current_performance']['stock_performance']:.1%}")
                
                # Industry Comparison
                if compare_industry:
                    st.subheader("🏭 Industry Comparison")
                    
                    # Create radar chart for industry comparison
                    categories = ['ROI', 'Growth', 'Market Share', 'Efficiency', 'Innovation']
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[75, 82, 68, 90, 85],
                        theta=categories,
                        fill='toself',
                        name=selected_company
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[60, 60, 60, 60, 60],
                        theta=categories,
                        fill='toself',
                        name='Industry Average',
                        opacity=0.5
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Causal Insights
                if include_causality:
                    st.subheader("🔬 Causal Analysis Insights")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"""
                        **Agency Change Impact**: {analysis['causal_insights']['agency_change_impact']:.2%}
                        
                        **Confidence Interval**: {analysis['causal_insights']['confidence'][0]:.2%} to {analysis['causal_insights']['confidence'][1]:.2%}
                        
                        **Statistical Significance**: {'Yes ✅' if analysis['causal_insights']['significant'] else 'No ❌'}
                        """)
                    
                    with col2:
                        # Visualization of causal impact
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['No Agency Change', 'Agency Change'],
                            y=[100, 100 + analysis['causal_insights']['agency_change_impact']],
                            marker_color=['blue', 'green']
                        ))
                        fig.update_layout(
                            title="Expected Impact of Agency Change",
                            yaxis_title="Performance Index",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

elif page == "Marketing Optimization":
    st.title("🎯 Marketing Mix Optimization")
    
    # Optimization Parameters
    st.subheader("Set Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input(
            "Total Marketing Budget ($M):",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1
        )
        
        objective = st.selectbox(
            "Optimization Objective:",
            ["Maximize ROI", "Maximize Revenue", "Maximize Stock Price", "Minimize Risk"]
        )
    
    with col2:
        st.markdown("### Channel Constraints")
        
        digital_min = st.slider("Digital Min-Max (%)", 0, 100, (20, 60))
        tv_min = st.slider("TV Min-Max (%)", 0, 100, (10, 40))
        print_min = st.slider("Print Min-Max (%)", 0, 100, (5, 20))
    
    # Advanced Options
    with st.expander("Advanced Options"):
        include_seasonality = st.checkbox("Include Seasonality", value=True)
        include_competition = st.checkbox("Consider Competitive Response", value=True)
        risk_tolerance = st.select_slider(
            "Risk Tolerance:",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
    
    # Run Optimization
    if st.button("🚀 Optimize Marketing Mix", type="primary"):
        with st.spinner("Running optimization algorithm..."):
            
            # Call optimization API
            response = requests.post(
                f"{API_URL}/optimize",
                json={
                    "company": selected_company,
                    "budget_constraint": total_budget * 1000000,
                    "objective": objective.lower().replace(" ", "_"),
                    "constraints": {
                        "digital_min": digital_min[0] / 100 * total_budget * 1000000,
                        "digital_max": digital_min[1] / 100 * total_budget * 1000000,
                        "tv_min": tv_min[0] / 100 * total_budget * 1000000,
                        "tv_max": tv_min[1] / 100 * total_budget * 1000000,
                        "print_min": print_min[0] / 100 * total_budget * 1000000,
                        "print_max": print_min[1] / 100 * total_budget * 1000000
                    }
                }
            )
            
            if response.status_code == 200:
                results = response.json()
                
                st.success("✅ Optimization Complete!")
                
                # Display Optimal Allocation
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Optimal Marketing Mix")
                    
                    optimal_data = pd.DataFrame({
                        'Channel': ['Digital', 'TV', 'Print'],
                        'Current': [2.0, 1.5, 0.5],
                        'Optimal': [
                            results['optimal_allocation']['digital'] / 1000000,
                            results['optimal_allocation']['tv'] / 1000000,
                            results['optimal_allocation']['print'] / 1000000
                        ]
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Current Allocation',
                        x=optimal_data['Channel'],
                        y=optimal_data['Current'],
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        name='Optimal Allocation',
                        x=optimal_data['Channel'],
                        y=optimal_data['Optimal'],
                        marker_color='darkgreen'
                    ))
                    fig.update_layout(
                        barmode='group',
                        yaxis_title="Spend ($M)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("💰 Expected Performance")
                    
                    st.metric(
                        "Expected ROI",
                        f"{results['expected_performance']['roi']:.2f}x",
                        f"+{results['expected_performance']['improvement']:.1f}%"
                    )
                    
                    # Improvement breakdown
                    st.markdown("### Improvement Breakdown")
                    improvements = {
                        'Digital Efficiency': '+12%',
                        'TV Reach': '+8%',
                        'Print Targeting': '+3%',
                        'Channel Synergy': '+5%'
                    }
                    
                    for key, value in improvements.items():
                        st.progress(int(value.strip('%+')), text=f"{key}: {value}")

elif page == "Agency Comparison":
    st.title("🏢 Agency Performance Comparison")
    
    # Get agency list
    agencies_response = requests.get(f"{API_URL}/agencies")
    agencies_data = agencies_response.json()["agencies"]
    
    # Multi-select for agencies
    selected_agencies = st.multiselect(
        "Select Agencies to Compare:",
        options=[a['name'] for a in agencies_data],
        default=[a['name'] for a in agencies_data[:4]]
    )
    
    if selected_agencies:
        # Create comparison matrix
        st.subheader("📊 Agency Performance Matrix")
        
        # Sample data - replace with API call
        comparison_data = pd.DataFrame({
            'Agency': selected_agencies,
            'Avg ROI': np.random.uniform(1.5, 3.0, len(selected_agencies)),
            'Client Retention': np.random.uniform(70, 95, len(selected_agencies)),
            'Innovation Score': np.random.uniform(60, 90, len(selected_agencies)),
            'Cost Efficiency': np.random.uniform(65, 85, len(selected_agencies))
        })
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=comparison_data.iloc[:, 1:].values,
            x=comparison_data.columns[1:],
            y=comparison_data['Agency'],
            colorscale='Viridis',
            text=comparison_data.iloc[:, 1:].values,
            texttemplate='%{text:.1f}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Agency Performance Heatmap",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("📈 Detailed Agency Metrics")
        
        tab1, tab2, tab3 = st.tabs(["Performance", "Client Portfolio", "Specializations"])
        
        with tab1:
            # Performance trends
            months = pd.date_range(start='2023-01-01', periods=12, freq='M')
            
            fig = go.Figure()
            for agency in selected_agencies:
                fig.add_trace(go.Scatter(
                    x=months,
                    y=np.random.uniform(1.5, 3.0, 12),
                    mode='lines+markers',
                    name=agency
                ))
            
            fig.update_layout(
                title="ROI Performance Trend",
                xaxis_title="Date",
                yaxis_title="ROI",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Client portfolio
            st.markdown("### Client Portfolio Analysis")
            
            portfolio_data = pd.DataFrame({
                'Agency': selected_agencies,
                'Fortune 500 Clients': np.random.randint(5, 20, len(selected_agencies)),
                'Avg Client Tenure (years)': np.random.uniform(2, 8, len(selected_agencies)),
                'Total Billings ($M)': np.random.uniform(100, 1000, len(selected_agencies))
            })
            
            st.dataframe(portfolio_data, use_container_width=True, hide_index=True)
        
        with tab3:
            # Specializations
            st.markdown("### Agency Specializations")
            
            specializations = {
                'Digital Marketing': ['WPP', 'Publicis', 'Omnicom'],
                'Creative Excellence': ['BBDO', 'Wieden+Kennedy', 'Droga5'],
                'Data & Analytics': ['Publicis', 'WPP', 'Dentsu'],
                'Performance Marketing': ['Havas', 'S4 Capital', 'Accenture']
            }
            
            for spec, agencies in specializations.items():
                matching = [a for a in selected_agencies if a in agencies]
                if matching:
                    st.markdown(f"**{spec}**: {', '.join(matching)}")

elif page == "Industry Insights":
    st.title("🌐 Industry Intelligence Dashboard")
    
    # Industry selector
    industries = ["Beverages", "Food & Snacks", "Personal Care", "Automotive", "Technology"]
    selected_industry = st.selectbox("Select Industry:", industries)
    
    # Time period selector
    time_period = st.select_slider(
        "Analysis Period:",
        options=["Last Quarter", "Last 6 Months", "Last Year", "Last 2 Years"],
        value="Last Year"
    )
    
    # Industry Overview
    st.subheader(f"📊 {selected_industry} Industry Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Market Size", "$127.3B", "+5.2%")
    with col2:
        st.metric("Avg Marketing Spend", "8.7% of Revenue", "+0.3%")
    with col3:
        st.metric("Digital Share", "62%", "+8%")
    with col4:
        st.metric("Industry ROI", "2.1x", "+0.2x")
    
    # Market Trends
    st.subheader("📈 Market Trends & Dynamics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth trends
        quarters = pd.date_range(start='2022-01-01', periods=8, freq='Q')
        growth_data = pd.DataFrame({
            'Quarter': quarters,
            'Industry Growth': np.random.uniform(3, 7, 8),
            'Marketing Spend Growth': np.random.uniform(5, 10, 8)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=growth_data['Quarter'],
            y=growth_data['Industry Growth'],
            mode='lines+markers',
            name='Industry Growth',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=growth_data['Quarter'],
            y=growth_data['Marketing Spend Growth'],
            mode='lines+markers',
            name='Marketing Spend Growth',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title="Growth Trends",
            yaxis_title="Growth Rate (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Market share distribution
        companies = ['Leader 1', 'Leader 2', 'Leader 3', 'Leader 4', 'Others']
        shares = [28, 22, 15, 12, 23]
        
        fig = px.pie(
            values=shares,
            names=companies,
            title="Market Share Distribution",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Competitive Intelligence
    st.subheader("🎯 Competitive Intelligence")
    
    # Create competitive landscape table
    competitive_data = pd.DataFrame({
        'Company': ['Market Leader', 'Challenger 1', 'Challenger 2', 'Innovator', 'Niche Player'],
        'Market Share': ['28%', '22%', '15%', '12%', '8%'],
        'Marketing ROI': [2.8, 2.3, 2.1, 2.5, 1.9],
        'Primary Agency': ['WPP', 'Publicis', 'Omnicom', 'Independent', 'BBDO'],
        'Digital %': [70, 65, 60, 80, 55],
        'YoY Growth': ['+8%', '+12%', '+5%', '+18%', '+3%']
    })
    
    st.dataframe(
        competitive_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Marketing ROI": st.column_config.NumberColumn(format="%.1fx"),
            "Digital %": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%d%%"
            )
        }
    )
    
    # Industry Best Practices
    st.subheader("💡 Industry Best Practices & Insights")
    
    insights = [
        "🎯 **Digital First**: Top performers allocate 60-70% of budget to digital channels",
        "📊 **Data Integration**: Companies with unified data platforms show 25% higher ROI",
        "🔄 **Agile Marketing**: Quarterly campaign adjustments improve performance by 15%",
        "🤝 **Agency Stability**: 3+ year relationships correlate with 20% better results",
        "📈 **Performance Marketing**: Shift to performance-based models driving efficiency"
    ]
    
    for insight in insights:
        st.info(insight)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Marketing-Finance AI Platform v1.0 | Powered by Advanced ML & Causal Inference
    </div>
    """,
    unsafe_allow_html=True
)
